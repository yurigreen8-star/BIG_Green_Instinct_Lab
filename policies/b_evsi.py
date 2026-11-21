"""
B_EVSI — основная политика: максимальный абсолютный EVSI.
B_EVSI_cost — дополнительный вариант: EVSI / cost.

По умолчанию используется аналитический EVSI (точный).
use_mc=True включает MC-оценку с CRN.
"""

from __future__ import annotations

import numpy as np

from belief import Belief
from evsi import compute_evsi_table_analytic, compute_evsi_table_mc
from probes import Probe


class EVSIPolicy:
    name = "B_EVSI"

    def __init__(self, use_mc: bool = False, n_samples: int = 2000):
        self.use_mc = use_mc
        self.n_samples = n_samples
        self.last_diagnostics: dict | None = None

    def _score(self, d: dict) -> float:
        return d["evsi"]

    def select_probe(
        self,
        belief: Belief,
        available: list[Probe],
        budget_remaining: float,
        rng: np.random.Generator,
    ) -> Probe | None:
        if not available:
            self.last_diagnostics = None
            return None

        if self.use_mc:
            table = compute_evsi_table_mc(belief, available, rng, self.n_samples)
        else:
            table = compute_evsi_table_analytic(belief, available)

        for d in table:
            d["score"] = self._score(d)

        best = max(
            table,
            key=lambda d: (
                d["score"],
                -d["probe"].cost,
                d["probe"].probe_type.probe_type_id,
                d["probe"].parcel_id,
            ),
        )

        evsi = best["evsi"]
        se = best["se"]
        ci95 = best["ci95"]
        a7 = bool(se > 0.0 and evsi != 0.0 and ci95 / abs(evsi) > 0.05)

        self.last_diagnostics = {
            "method": best["method"],
            "table": table,
            "selected": best["probe"],
            "selected_evsi": evsi,
            "selected_se": se,
            "selected_ci95": ci95,
            "A7_violated": a7,
        }
        return best["probe"]


class EVSICostPolicy(EVSIPolicy):
    name = "B_EVSI_cost"

    def _score(self, d: dict) -> float:
        c = d["probe"].cost
        return d["evsi"] / c if c > 0 else 0.0