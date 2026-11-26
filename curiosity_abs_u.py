"""
curiosity_abs_u.py — АБЛЯЦИЯ: абсолютная неопределённость.

Отличие от базового Curiosity:
  u_i = sigma_i^2  (абсолютная), а не  sigma_i^2 / s_i (относительная).
Инициализация согласована: I_i(0) = A_i(0) = s_i,
поскольку при абсолютной неопределённости начальная привлекательность
равна исходной дисперсии, а не единице.

Это отдельная диагностическая абляция, не объединяемая с основной проверкой.
"""

from __future__ import annotations

import numpy as np

from curiosity import Curiosity


class CuriosityAbsU(Curiosity):
    """Curiosity с абсолютной неопределённостью (абляция)."""

    def __init__(self, n_parcels, sigma2_prior, lam=0.3, seed=0):
        super().__init__(n_parcels, sigma2_prior=sigma2_prior, lam=lam, seed=seed)
        # Согласованная инициализация: I_i(0) = A_i(0) = s_i
        self._interest = self._sigma2_prior.copy()

    def update_after_observation(self, parcel_id, view_after, surprise):
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
        # Абсолютная неопределённость: sigma2_current = ignorance * sigma2_prior
        abs_uncertainty = view_after.ignorance * self._sigma2_prior
        attractiveness = abs_uncertainty * novelty
        self._interest = (
            (1.0 - self._lam) * self._interest + self._lam * attractiveness
        )