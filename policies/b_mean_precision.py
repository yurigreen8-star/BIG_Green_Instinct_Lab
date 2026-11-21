"""
B_mean_precision — сильный эвристический baseline.
Выбирает наиболее перспективный по mu участок и измеряет его
наиболее точным доступным прибором.
"""

from __future__ import annotations

import numpy as np

from belief import Belief
from probes import Probe


class MeanPrecisionPolicy:
    name = "B_mean_precision"

    def select_probe(
        self,
        belief: Belief,
        available: list[Probe],
        budget_remaining: float,
        rng: np.random.Generator,
    ) -> Probe | None:
        if not available:
            return None
        best_mu = max(float(belief.mu[p.parcel_id]) for p in available)
        candidates = [p for p in available if float(belief.mu[p.parcel_id]) == best_mu]
        return min(
            candidates,
            key=lambda p: (p.noise_var, p.cost, p.probe_type.probe_type_id, p.parcel_id),
        )