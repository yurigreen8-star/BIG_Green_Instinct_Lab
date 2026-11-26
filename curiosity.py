"""
curiosity.py — Любопытство v3.0, вариант B (обновление всех объектов каждый шаг).

Контракт порядка операций (однозначно зафиксирован):
  1. Runner исполняет зонд по участку k и обновляет Belief.
  2. Runner строит CuriosityView из НОВОГО Belief (ignorance_after).
  3. Runner вызывает update_after_observation(k, view_after, surprise).
  4. Внутри: сначала visit_count[k] += 1, затем novelty по НОВОМУ счётчику,
     затем A_i для ВСЕХ i, затем сглаживание I_i для ВСЕХ i.

S_i в A_i не участвует (§3.4) — только логируется.
Модуль не импортирует belief / evsi / probes / environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from messages import TIE_EPS, CuriosityProposal, CuriosityView


@dataclass
class CuriosityState:
    interest: np.ndarray
    visit_count: np.ndarray
    last_surprise: np.ndarray
    sigma2_prior: float
    lam: float


class Curiosity:
    def __init__(
        self,
        n_parcels: int,
        sigma2_prior: float = 1.0,
        lam: float = 0.3,
        seed: int = 0,
    ) -> None:
        if n_parcels <= 0:
            raise ValueError("n_parcels должен быть > 0")
        if sigma2_prior <= 0.0:
            raise ValueError("sigma2_prior должен быть > 0")
        if not (0.0 < lam < 1.0):
            raise ValueError("lam должен быть в (0, 1)")
        self._n = int(n_parcels)
        self._sigma2_prior = float(sigma2_prior)
        self._lam = float(lam)
        self._seed = int(seed)
        self._interest = np.ones(self._n, dtype=np.float64)
        self._visit_count = np.zeros(self._n, dtype=np.int64)
        self._last_surprise = np.full(self._n, np.nan, dtype=np.float64)

    @property
    def n(self) -> int:
        return self._n

    @property
    def sigma2_prior(self) -> float:
        return self._sigma2_prior

    def propose(
        self, known_mask: np.ndarray, rng: np.random.Generator
    ) -> CuriosityProposal | None:
        """
        i* = argmax I_i среди not known. Tie-break: rng.choice среди |I_i - max| < 1e-9.
        Получает только маску known и RNG. Ни вида, ни бюджета, ни цен.
        """
        kn = np.asarray(known_mask, dtype=bool)
        if kn.shape != (self._n,):
            raise ValueError("known_mask формы (n,)")
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
        """Обновление состояния ВСЕХ объектов (вариант B)."""
        k = int(parcel_id)
        if not (0 <= k < self._n):
            raise ValueError("parcel_id вне диапазона")
        if view_after.n != self._n:
            raise ValueError("view_after формы (n,)")
        if not (0.0 <= surprise < 1.0):
            raise ValueError("surprise должен быть в [0, 1)")

        self._visit_count[k] += 1
        self._last_surprise[k] = float(surprise)

        novelty = 1.0 / (1.0 + self._visit_count.astype(np.float64))
        attractiveness = view_after.ignorance * novelty
        self._interest = (
            (1.0 - self._lam) * self._interest + self._lam * attractiveness
        )

    def get_state(self) -> CuriosityState:
        return CuriosityState(
            interest=self._interest.copy(),
            visit_count=self._visit_count.copy(),
            last_surprise=self._last_surprise.copy(),
            sigma2_prior=self._sigma2_prior,
            lam=self._lam,
        )