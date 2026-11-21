from __future__ import annotations

import numpy as np

from belief import Belief
from evsi import ig
from probes import Probe


class IGPolicy:
    name = "B_IG"

    def select_probe(
        self,
        belief: Belief,
        available: list[Probe],
        budget_remaining: float,
        rng: np.random.Generator,
    ) -> Probe | None:
        if not available:
            return None

        def score(p: Probe) -> float:
            return ig(belief, p) / p.cost if p.cost > 0 else float("inf")

        ordered = sorted(
            available,
            key=lambda p: (-score(p), p.cost, p.probe_type.probe_type_id, p.parcel_id),
        )
        return ordered[0]