"""
messages.py — общие структуры сообщений v3.0.

Единственный источник правды для интерфейсов Curiosity / Ratio / Coordinator.
Здесь нет импортов belief / evsi / environment — только чистая арифметика
и структуры данных. Это позволяет Coordinator верифицировать доминирование,
не имея доступа к функциям оценки.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from probes import Probe

MODE_CURIOSITY_SOLO = "CURIOSITY_SOLO"
MODE_RATIO_SOLO = "RATIO_SOLO"
MODE_COMBINED = "COMBINED"
MODE_COMBINED_IGNORE = "COMBINED_IGNORE"

MODES = (
    MODE_CURIOSITY_SOLO,
    MODE_RATIO_SOLO,
    MODE_COMBINED,
    MODE_COMBINED_IGNORE,
)

TIE_EPS = 1e-9


@dataclass(frozen=True)
class CuriosityView:
    """
    Единственный канал восприятия Любопытства.

    Содержит ТОЛЬКО нормированное незнание и маску known.
    Запрещено: mu, theta_true, бюджет, цены, EVSI, оценки Рацио.
    Массивы read-only: запись невозможна физически.
    """

    ignorance: np.ndarray
    known: np.ndarray

    def __post_init__(self) -> None:
        ign = np.array(self.ignorance, dtype=np.float64, copy=True)
        kn = np.array(self.known, dtype=bool, copy=True)
        if ign.ndim != 1 or kn.ndim != 1 or ign.shape != kn.shape:
            raise ValueError("ignorance и known должны быть 1-D одной формы")
        if np.any(~np.isfinite(ign)) or np.any(ign < 0.0) or np.any(ign > 1.0):
            raise ValueError("ignorance должен быть конечным и в [0, 1]")
        ign.flags.writeable = False
        kn.flags.writeable = False
        object.__setattr__(self, "ignorance", ign)
        object.__setattr__(self, "known", kn)

    @property
    def n(self) -> int:
        return int(self.ignorance.shape[0])


def build_curiosity_view(
    sigma2: np.ndarray, known: np.ndarray, sigma2_prior: float = 1.0
) -> CuriosityView:
    """
    Строитель вида. Вызывается Runner'ом (у него есть Belief).
    Любопытство получает уже готовый вид и не знает, откуда он взялся.
    ignorance = clip(sigma2 / sigma2_prior, 0, 1)  — диапазон [0, 1] включительно.
    """
    if sigma2_prior <= 0.0:
        raise ValueError("sigma2_prior должен быть > 0")
    s2 = np.asarray(sigma2, dtype=np.float64)
    ign = np.clip(s2 / float(sigma2_prior), 0.0, 1.0)
    return CuriosityView(ignorance=ign, known=np.asarray(known, dtype=bool))


def predictive_surprise(
    observation: float, mu_pred: float, sigma2_pred: float, s2: float
) -> float:
    """
    S = z^2 / (1 + z^2),  z = (o - mu_pred) / sqrt(sigma2_pred + s2).

    Вызывается Runner'ом ДО обновления Belief. Любопытство получает готовый
    скаляр S и никогда не видит mu. E[S] = 0.344324 при корректной модели.
    """
    denom = float(sigma2_pred) + float(s2)
    if denom <= 0.0:
        return 0.0
    z = (float(observation) - float(mu_pred)) / math.sqrt(denom)
    z2 = z * z
    return float(z2 / (1.0 + z2))


def strictly_dominates(
    cost_p: float,
    evsi_p: float,
    cost_z: float,
    evsi_z: float,
    eps_cost: float = 1e-9,
    eps_evsi: float = 1e-9,
) -> bool:
    """
    §3.10. p строго доминирует z.
    Чистая арифметика: используется и Ratio (при формировании),
    и Coordinator (при верификации). Ни та, ни другая сторона не пересчитывает EVSI.
    """
    weak = (cost_p <= cost_z + eps_cost) and (evsi_p >= evsi_z - eps_evsi)
    if not weak:
        return False
    strict = (cost_p < cost_z - eps_cost) or (evsi_p > evsi_z + eps_evsi)
    return bool(strict)


@dataclass(frozen=True)
class CuriosityProposal:
    parcel_id: int
    interest: float


@dataclass(frozen=True)
class RatioProposal:
    probe: Probe
    cost: float
    evsi: float
    is_counter: bool
    target_parcel_id: int | None
    target_cost: float | None
    target_evsi: float | None


@dataclass(frozen=True)
class CoordinatorDecision:
    action: Probe | None
    reason: str
    verified_dominance: bool

def bayesian_surprise(
    observation: float, mu_pred: float, sigma2_pred: float, s2: float
) -> float:
    """
    BS = KL(posterior || prior) по Итти-Балди, гауссовский случай.

    Реализована ПРЯМАЯ форма KL, чтобы тест мог сверить её с независимым
    замкнутым тождеством BS = IG + a/(2(a+r))*(z^2-1).

    E[BS] = IG точно. Обратная KL(prior||post) НЕ используется:
    у неё E = a/r - IG, что сломало бы калибровку.

    При s2 = 0 posterior — дельта, KL расходится: явный отказ, не молчаливый 0.
    """
    a = float(sigma2_pred)
    r = float(s2)
    if a <= 0.0:
        return 0.0
    if r <= 0.0:
        raise ValueError("BS не определён при s2=0: posterior — дельта, KL расходится")
    post_var = a * r / (a + r)
    post_mu = (float(mu_pred) / a + float(observation) / r) / (1.0 / a + 1.0 / r)
    d = post_mu - float(mu_pred)
    return float(0.5 * math.log(a / post_var) + (post_var + d * d) / (2.0 * a) - 0.5)


def behavioral_novelty(count_before: int) -> float:
    """
    B = ln(N+2) - ln(N+1), где N — число зондов ДАННОГО типа по участку
    до текущего исполнения.

    В E0 FAST-only выполняется B = 2*IG точно, то есть величина
    не является независимым драйвом. Фиксируется тестом.
    """
    n = int(count_before)
    if n < 0:
        raise ValueError("count_before >= 0")
    return float(math.log(n + 2.0) - math.log(n + 1.0))