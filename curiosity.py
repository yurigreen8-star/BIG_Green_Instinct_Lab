"""
curiosity.py — Любопытство (базовая формула) для E0 и E1-A.

Формула привлекательности (НЕ менять):

    A_i = u_i * n_i

где
    n_i = 1 / (1 + v_i)                  — новизна,
    v_i                                    — число предыдущих посещений объекта,
    u_i = sigma2_i_current / s_i          — относительная неопределённость,
    s_i                                    — исходная априорная дисперсия объекта.

В реализации `u_i` приходит уже готовым вектором через `CuriosityView.ignorance`,
который вычисляет Runner: `ignorance = clip(sigma2_current / s_i, 0, 1)`.
Поэтому здесь привлекательность = `view_after.ignorance * novelty`.

После k измерений FAST (шум r):
    sigma2_i(k) = s_i * r / (r + k * s_i)
    u_i(k)      = r / (r + k * s_i)
    A_i(k)      = r / ((r + k * s_i) * (1 + k))

Сглаживание интереса:
    I_i <- (1 - lam) * I_i + lam * A_i,    lam = 0.3

Начальное состояние:
    I_i(0) = 1,  A_i(0) = 1.

Обязательное свойство (проверяется тестом): апостериорная дисперсия не зависит
от значения наблюдения, поэтому последовательность выбора зависит только от
начальных s_i, счётчиков посещений, внутреннего состояния и правил разрешения
ничьих. Это "open-loop относительно наблюдений".

Поддержка среды:
    - скалярный `sigma2_prior` (E0, однородные приоры);
    - векторный `sigma2_prior` (E1-A, неоднородные приоры).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from messages import TIE_EPS, CuriosityProposal, CuriosityView


@dataclass
class CuriosityState:
    """Снимок внутреннего состояния Любопытства для логирования и абляций."""
    interest: np.ndarray
    visit_count: np.ndarray
    last_surprise: np.ndarray
    sigma2_prior: np.ndarray
    lam: float


class Curiosity:
    """
    Любопытство: выбирает объект с максимальным внутренним интересом.

    Не получает доступ к: истинным значениям, наблюдениям, ценам, бюджету,
    EVSI, внутреннему состоянию Ratio. Единственный входной канал —
    `CuriosityView` (нормированное незнание и маска known).
    """

    def __init__(
        self,
        n_parcels: int,
        sigma2_prior=1.0,
        lam: float = 0.3,
        seed: int = 0,
    ) -> None:
        if n_parcels <= 0:
            raise ValueError("n_parcels должен быть > 0")
        if not (0.0 < lam < 1.0):
            raise ValueError("lam должен быть в (0, 1)")

        self._n = int(n_parcels)
        self._lam = float(lam)
        self._seed = int(seed)  # сохранён для совместимости сигнатуры

        # Поддержка скалярного (E0) и векторного (E1-A) sigma2_prior.
        sp = np.asarray(sigma2_prior, dtype=np.float64)
        if sp.ndim == 0:
            sp = np.full(self._n, float(sp))
        if sp.shape != (self._n,):
            raise ValueError(
                "sigma2_prior должен быть скаляром или вектором длины n_parcels"
            )
        if np.any(sp <= 0.0):
            raise ValueError("sigma2_prior должен быть > 0")
        self._sigma2_prior = sp

        # Начальное состояние: I_i(0) = 1, что соответствует A_i(0) = 1.
        self._interest = np.ones(self._n, dtype=np.float64)
        self._visit_count = np.zeros(self._n, dtype=np.int64)
        self._last_surprise = np.full(self._n, np.nan, dtype=np.float64)

    @property
    def n(self) -> int:
        return self._n

    @property
    def sigma2_prior(self) -> np.ndarray:
        """Вектор исходных априорных дисперсий (одинаковых в E0, разных в E1-A)."""
        return self._sigma2_prior

    def propose(
        self, known_mask: np.ndarray, rng: np.random.Generator
    ) -> CuriosityProposal | None:
        """
        Предложение объекта: максимум интереса среди неизвестных.

        Разрешение ничьих: случайный выбор среди объектов с интересом,
        неотличимым от максимума в пределах TIE_EPS. RNG передаётся извне,
        внутреннее состояние не меняется.
        """
        kn = np.asarray(known_mask, dtype=bool)
        if kn.shape != (self._n,):
            raise ValueError("known_mask должен быть 1-D формы (n,)")
        cand = np.flatnonzero(~kn)
        if cand.size == 0:
            return None
        vals = self._interest[cand]
        mx = float(vals.max())
        tied = cand[np.abs(vals - mx) < TIE_EPS]
        chosen = int(rng.choice(tied))
        return CuriosityProposal(
            parcel_id=chosen, interest=float(self._interest[chosen])
        )

    def update_after_observation(
        self, parcel_id: int, view_after: CuriosityView, surprise: float
    ) -> None:
        """
        Обновление интереса ВСЕХ объектов на каждом принятом шаге.

        Порядок операций (соответствует спецификации):
          1. Увеличить счётчик посещений выбранного объекта.
          2. Сохранить неожиданность (лог, не участвует в выборе).
          3. Рассчитать новое u_i, n_i, A_i для всех объектов.
          4. Обновить I_i для всех объектов.

        `view_after` построен из обновлённого Belief; `ignorance` уже содержит
        нормированную неопределённость `sigma2_current / s_i`. Surprise на
        выбор не влияет (формула не зависит от значения наблюдения).
        """
        k = int(parcel_id)
        if not (0 <= k < self._n):
            raise ValueError("parcel_id вне диапазона")
        if view_after.n != self._n:
            raise ValueError("view_after формы (n,)")
        if not (0.0 <= surprise < 1.0):
            raise ValueError("surprise должен быть в [0, 1)")

        # Шаг 1: счётчик посещений выбранного объекта.
        self._visit_count[k] += 1
        # Шаг 2: сохранение неожиданности (только для диагностики).
        self._last_surprise[k] = float(surprise)

        # Шаг 3: новизна и привлекательность для всех объектов.
        novelty = 1.0 / (1.0 + self._visit_count.astype(np.float64))
        attractiveness = view_after.ignorance * novelty

        # Шаг 4: сглаживание интереса для всех объектов.
        self._interest = (
            (1.0 - self._lam) * self._interest + self._lam * attractiveness
        )

    def get_state(self) -> CuriosityState:
        """Снимок состояния (копии массивов, чтобы не дать внешних ссылок)."""
        return CuriosityState(
            interest=self._interest.copy(),
            visit_count=self._visit_count.copy(),
            last_surprise=self._last_surprise.copy(),
            sigma2_prior=self._sigma2_prior.copy(),
            lam=self._lam,
        )