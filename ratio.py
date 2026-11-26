"""
ratio.py — Рацио v3.0. Единственная компетенция: EVSI/cost + строгое доминирование.

Tie-break (утверждено архитектором, нотация спеки исправлена):
    max(key=(score, -cost, probe_type_id, parcel_id))
При равенстве score и cost выбираются НАИБОЛЬШИЕ probe_type_id и parcel_id.
Это в точности поведение существующей B_EVSI_cost; тест B4 это фиксирует.

Не получает CuriosityState / interest / visit_count / last_surprise.
Из CuriosityProposal читает ТОЛЬКО parcel_id; поле interest игнорируется.
"""

from __future__ import annotations

import numpy as np

from belief import Belief
from evsi import evsi_analytic
from messages import CuriosityProposal, RatioProposal, strictly_dominates
from probes import SURVEY_FAST, Probe


class Ratio:
    def __init__(self, eps_cost: float = 1e-9, eps_evsi: float = 1e-9) -> None:
        if eps_cost < 0.0 or eps_evsi < 0.0:
            raise ValueError("eps должны быть >= 0")
        self._eps_cost = float(eps_cost)
        self._eps_evsi = float(eps_evsi)

    def propose(
        self,
        belief: Belief,
        budget_remaining: float,
        available: list[Probe],
        curiosity_proposal: CuriosityProposal | None,
        rng: np.random.Generator,
    ) -> RatioProposal | None:
        if not available:
            return None

        scored = [(float(evsi_analytic(belief, p)), p) for p in available]

        def key(item: tuple[float, Probe]):
            ev, p = item
            score = ev / p.cost if p.cost > 0.0 else float("inf")
            return (score, -p.cost, p.probe_type.probe_type_id, p.parcel_id)

        best_evsi, best_probe = max(scored, key=key)

        target_id: int | None = None
        target_cost: float | None = None
        target_evsi: float | None = None
        is_counter = False

        if curiosity_proposal is not None:
            target = Probe(
                parcel_id=int(curiosity_proposal.parcel_id), probe_type=SURVEY_FAST
            )
            target_id = target.parcel_id
            target_cost = float(target.cost)
            target_evsi = float(evsi_analytic(belief, target))
            is_counter = strictly_dominates(
                best_probe.cost,
                best_evsi,
                target_cost,
                target_evsi,
                self._eps_cost,
                self._eps_evsi,
            )

        return RatioProposal(
            probe=best_probe,
            cost=float(best_probe.cost),
            evsi=float(best_evsi),
            is_counter=bool(is_counter),
            target_parcel_id=target_id,
            target_cost=target_cost,
            target_evsi=target_evsi,
        )