"""
B_random_without_replacement — контроль случайного покрытия без повторов.

Назначение (флаг 2): проверить, даёт ли Curiosity что-нибудь сверх простого
случайного обхода всех участков дешёвым зондом.

Реализация БЕЗ внутреннего состояния — важно, потому что политики в
экспериментах переиспользуются между эпизодами. «Непосещённый» определяется
по Belief: sigma2 == sigma2_prior. В E0 любое наблюдение строго снижает
sigma2, поэтому критерий точен.

Когда все участки посещены — возвращает None (покрытие завершено).
"""

from __future__ import annotations

import numpy as np

from belief import Belief
from probes import SURVEY_FAST, Probe


class RandomWithoutReplacementPolicy:
    name = "B_random_without_replacement"

    def __init__(self, sigma2_prior: float = 1.0) -> None:
        if sigma2_prior <= 0.0:
            raise ValueError("sigma2_prior должен быть > 0")
        self._prior = float(sigma2_prior)

    def select_probe(
        self,
        belief: Belief,
        available: list[Probe],
        budget_remaining: float,
        rng: np.random.Generator,
    ) -> Probe | None:
        if not available:
            return None
        unvisited = np.flatnonzero(
            (~belief.known) & (belief.sigma2 >= self._prior - 1e-12)
        )
        if unvisited.size == 0:
            return None
        cand = [p for p in available if p.parcel_id in set(unvisited.tolist())]
        if not cand:
            return None
        fast = [p for p in cand if p.probe_type.probe_type_id == SURVEY_FAST.probe_type_id]
        pool = fast if fast else cand
        return pool[int(rng.integers(len(pool)))]