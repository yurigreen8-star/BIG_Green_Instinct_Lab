"""
curiosity_bs.py — C_BS, closed-loop Curiosity. Предрегистрация E2 §3.2.

A_i(0) = I_i(0) = IG_i(0) = 0.5*ln(1 + s_i/r)
После измерения участка i: A_i = BS_i/(1+v_i); A остальных СОХРАНЯЮТСЯ.
I обновляется для всех участков.

Значение наблюдения o в модуль не передаётся: BS считается из (mu0,s0,mu1,s1).
Модуль не получает стоимость, бюджет, EVSI, состояние Ratio, theta_true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TIE_EPS = 1e-9
BS_ROUNDING_TOL = 1e-9
R_DEFAULT = 1.0
LAM_DEFAULT = 0.3


@dataclass
class CuriosityBSState:
    interest: np.ndarray
    attract: np.ndarray
    visit_count: np.ndarray
    last_bs: np.ndarray
    lam: float
    r: float


def bayesian_surprise_kl(mu0: float, s0: float, mu1: float, s1: float) -> float:
    """
    BS = 0.5*[ ln(s0/s1) + (s1 + (mu1-mu0)^2)/s0 - 1 ].
    Определено при s0>0, s1>0. Вырожденный posterior (r=0) не рассматривается.
    """
    if s0 <= 0.0 or s1 <= 0.0:
        raise ValueError("BS требует строго положительных дисперсий")
    d = float(mu1) - float(mu0)
    return float(0.5 * (math.log(s0 / s1) + (s1 + d * d) / s0 - 1.0))


class CuriosityBS:
    def __init__(self, n_parcels: int, s_prior, r: float = R_DEFAULT,
                 lam: float = LAM_DEFAULT) -> None:
        if n_parcels <= 0:
            raise ValueError("n_parcels > 0")
        if r <= 0.0:
            raise ValueError("r > 0")
        if not (0.0 < lam < 1.0):
            raise ValueError("lam в (0,1)")
        sp = np.asarray(s_prior, dtype=np.float64)
        if sp.ndim == 0:
            sp = np.full(n_parcels, float(sp))
        if sp.shape != (n_parcels,) or np.any(sp <= 0.0):
            raise ValueError("s_prior: вектор длины n, все > 0")
        self._n = int(n_parcels)
        self._s_prior = sp
        self._r = float(r)
        self._lam = float(lam)
        self._attract = 0.5 * np.log1p(sp / self._r)
        self._interest = self._attract.copy()
        self._visit_count = np.zeros(self._n, dtype=np.int64)
        self._last_bs = np.full(self._n, np.nan, dtype=np.float64)

    @property
    def n(self) -> int:
        return self._n

    @property
    def s_prior(self) -> np.ndarray:
        return self._s_prior

    def interest_vector(self) -> np.ndarray:
        return self._interest.copy()

    def attract_vector(self) -> np.ndarray:
        return self._attract.copy()

    def ranking(self, rng: np.random.Generator) -> list[int]:
        """
        Полное ранжирование по убыванию I. Ничьи (|dI| < TIE_EPS)
        перемешиваются переданным RNG. Первый элемент — предложение.
        """
        order = np.argsort(-self._interest, kind="stable")
        out: list[int] = []
        i = 0
        while i < len(order):
            j = i + 1
            while (j < len(order)
                   and abs(self._interest[order[j]] - self._interest[order[i]]) < TIE_EPS):
                j += 1
            group = order[i:j]
            if len(group) > 1:
                group = rng.permutation(group)
            out.extend(int(x) for x in group)
            i = j
        return out

    def after_measurement(self, parcel_id: int, mu0: float, s0: float,
                          mu1: float, s1: float) -> float:
        k = int(parcel_id)
        if not (0 <= k < self._n):
            raise ValueError("parcel_id вне диапазона")
        bs = bayesian_surprise_kl(mu0, s0, mu1, s1)
        # Численный guard: KL >= 0 математически. Отрицательное значение
        # допустимо только как ошибка округления; иначе — явный отказ.
        if bs < 0.0:
            if bs >= -BS_ROUNDING_TOL:
                bs = 0.0
            else:
                raise ValueError(f"BS отрицателен вне допуска округления: {bs}")
        self._visit_count[k] += 1
        self._last_bs[k] = bs
        self._attract[k] = bs / (1.0 + float(self._visit_count[k]))
        self._interest = (1.0 - self._lam) * self._interest + self._lam * self._attract
        return bs

    def get_state(self) -> CuriosityBSState:
        return CuriosityBSState(
            interest=self._interest.copy(),
            attract=self._attract.copy(),
            visit_count=self._visit_count.copy(),
            last_bs=self._last_bs.copy(),
            lam=self._lam,
            r=self._r,
        )