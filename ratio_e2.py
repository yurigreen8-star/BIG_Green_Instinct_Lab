"""
ratio_e2.py — Ratio для E2: аналитический EVSI, таблица и argmax.

Формула EVSI не меняется: используется evsi.evsi_analytic из v3.0.
Ничьи разрешаются отдельным RNG (поток 5), как требует предрегистрация §3.4.

Возвращает таблицу EVSI. Не получает внутренние состояния Curiosity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from belief import Belief
from evsi import evsi_analytic
from probes import Probe

TIE_EPS = 1e-9


@dataclass
class RatioTableE2:
    parcel_ids: np.ndarray
    evsi: np.ndarray
    best: int
    best_evsi: float
    max_evsi: float

    def evsi_of(self, parcel_id: int) -> float:
        idx = np.flatnonzero(self.parcel_ids == int(parcel_id))
        if idx.size == 0:
            raise KeyError(f"участок {parcel_id} отсутствует в таблице")
        return float(self.evsi[int(idx[0])])


def compute_ratio_table(belief: Belief, available: list[Probe],
                        rng: np.random.Generator) -> RatioTableE2:
    if not available:
        raise ValueError("available пуст")
    ids = np.array([int(p.parcel_id) for p in available], dtype=np.int64)
    vals = np.array([float(evsi_analytic(belief, p)) for p in available],
                    dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError("EVSI не конечен")
    mx = float(vals.max())
    tied = np.flatnonzero(np.abs(vals - mx) < TIE_EPS)
    pick = int(rng.choice(tied))
    return RatioTableE2(parcel_ids=ids, evsi=vals,
                        best=int(ids[pick]), best_evsi=float(vals[pick]),
                        max_evsi=mx)