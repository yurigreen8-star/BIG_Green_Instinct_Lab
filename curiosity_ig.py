"""
curiosity_ig.py — C_IG: изолированная альтернатива Любопытства.

A_i = [IG_i / (max_j IG_j + eps)] * novelty_i

Нормировка обязательна. Делитель — ОБЩИЙ положительный скаляр на шаге,
поэтому ранжирование A совпадает с ранжированием IG_i * novelty_i.

Входной канал тот же, что у базового Curiosity: CuriosityView(ignorance, known).
IG выводится из ignorance: sigma2_i = ignorance_i * sigma2_prior.
Новых данных механизм не получает — mu, бюджет, цены, EVSI недоступны.

probe_noise_var — дисперсия шума ЭТАЛОННОГО прибора (FAST), не цена.
Runner принудительно исполняет FAST, поэтому эталон согласован.

Базовый класс Curiosity не изменяется и не наследуется.
"""

from __future__ import annotations

import numpy as np

from curiosity import CuriosityState
from messages import TIE_EPS, CuriosityProposal, CuriosityView


class CuriosityIG:
    def __init__(
        self,
        n_parcels: int,
        sigma2_prior: float = 1.0,
        lam: float = 0.3,
        seed: int = 0,
        probe_noise_var: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        if n_parcels <= 0:
            raise ValueError("n_parcels > 0")
        if sigma2_prior <= 0.0 or probe_noise_var <= 0.0:
            raise ValueError("sigma2_prior и probe_noise_var > 0")
        if not (0.0 < lam < 1.0):
            raise ValueError("lam в (0,1)")
        if eps <= 0.0:
            raise ValueError("eps > 0")
        self._n = int(n_parcels)
        self._sigma2_prior = float(sigma2_prior)
        self._lam = float(lam)
        self._r = float(probe_noise_var)
        self._eps = float(eps)
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
        k = int(parcel_id)
        if not (0 <= k < self._n):
            raise ValueError("parcel_id вне диапазона")
        if view_after.n != self._n:
            raise ValueError("view_after формы (n,)")
        if not (0.0 <= surprise < 1.0):
            raise ValueError("surprise в [0,1)")

        self._visit_count[k] += 1
        self._last_surprise[k] = float(surprise)

        var = view_after.ignorance * self._sigma2_prior
        info = 0.5 * np.log1p(var / self._r)
        denom = float(info.max()) + self._eps
        novelty = 1.0 / (1.0 + self._visit_count.astype(np.float64))
        attractiveness = (info / denom) * novelty
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