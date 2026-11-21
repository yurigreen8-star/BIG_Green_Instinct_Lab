from __future__ import annotations

import numpy as np

from belief import Belief
from probes import Probe


class RandomPolicy:
    name = "B_random"

    def select_probe(
        self,
        belief: Belief,
        available: list[Probe],
        budget_remaining: float,
        rng: np.random.Generator,
    ) -> Probe | None:
        if not available:
            return None
        return available[int(rng.integers(len(available)))]