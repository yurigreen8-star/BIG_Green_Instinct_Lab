from __future__ import annotations

import numpy as np

from belief import Belief
from probes import Probe


class NonePolicy:
    name = "B_none"

    def select_probe(
        self,
        belief: Belief,
        available: list[Probe],
        budget_remaining: float,
        rng: np.random.Generator,
    ) -> Probe | None:
        return None