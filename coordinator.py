"""
coordinator.py — Координатор v3.0.

Запрещено и физически отсутствует: импорт evsi / belief / environment.
Координатор НЕ пересчитывает EVSI. Он проверяет структурную целостность
полей RatioProposal и арифметику доминирования по переданным значениям.

Режимы чередования: на каждом ходу действует ровно один механизм,
доминирование не проверяется. RANDOM_ALTERNATION требует rng и без него
падает явно, а не подставляет молча детерминированный порядок.

Единственная константа каталога — SURVEY_FAST: для построения Probe из
parcel_id и для сверки target_cost. Структурная проверка, не сравнение
скаляров между механизмами.
"""

from __future__ import annotations

import numpy as np

from messages import (
    ALTERNATION_MODES,
    MODE_COMBINED,
    MODE_COMBINED_IGNORE,
    MODE_CURIOSITY_SOLO,
    MODE_FIXED_ALT_C_FIRST,
    MODE_FIXED_ALT_R_FIRST,
    MODE_RANDOM_ALT_50_50,
    MODE_RATIO_SOLO,
    MODES,
    CoordinatorDecision,
    CuriosityProposal,
    RatioProposal,
    strictly_dominates,
)
from probes import SURVEY_FAST, Probe


class Coordinator:
    def __init__(self, eps_cost: float = 1e-9, eps_evsi: float = 1e-9) -> None:
        self._eps_cost = float(eps_cost)
        self._eps_evsi = float(eps_evsi)

    @staticmethod
    def _to_probe(cp: CuriosityProposal | None) -> Probe | None:
        if cp is None:
            return None
        return Probe(parcel_id=int(cp.parcel_id), probe_type=SURVEY_FAST)

    def _verify(
        self,
        cp: CuriosityProposal | None,
        rp: RatioProposal | None,
    ) -> bool:
        """
        Структурная + арифметическая верификация контра.
        EVSI берётся из полей rp, самостоятельно не вычисляется.
        """
        if cp is None or rp is None or not rp.is_counter:
            return False
        if rp.target_parcel_id is None or rp.target_cost is None or rp.target_evsi is None:
            return False
        if int(rp.target_parcel_id) != int(cp.parcel_id):
            return False
        if abs(float(rp.cost) - float(rp.probe.cost)) > self._eps_cost:
            return False
        if abs(float(rp.target_cost) - float(SURVEY_FAST.cost)) > self._eps_cost:
            return False
        return strictly_dominates(
            rp.cost,
            rp.evsi,
            rp.target_cost,
            rp.target_evsi,
            self._eps_cost,
            self._eps_evsi,
        )

    def coordinate(
        self,
        curiosity_proposal: CuriosityProposal | None,
        ratio_proposal: RatioProposal | None,
        mode: str,
        budget_remaining: float,
        step: int = 0,
        rng: np.random.Generator | None = None,
    ) -> CoordinatorDecision:
        if mode not in MODES:
            raise ValueError(f"неизвестный режим: {mode}")

        cp_probe = self._to_probe(curiosity_proposal)

        if mode in ALTERNATION_MODES:
            if mode == MODE_FIXED_ALT_C_FIRST:
                c_turn = (int(step) % 2 == 0)
            elif mode == MODE_FIXED_ALT_R_FIRST:
                c_turn = (int(step) % 2 == 1)
            else:
                if rng is None:
                    raise ValueError(
                        "RANDOM_ALTERNATION_50_50 требует rng; "
                        "молчаливый детерминированный fallback запрещён"
                    )
                c_turn = bool(rng.integers(2) == 0)

            if c_turn:
                return CoordinatorDecision(
                    action=cp_probe,
                    reason="curiosity" if cp_probe is not None else "stop_all_known",
                    verified_dominance=False,
                )
            action = ratio_proposal.probe if ratio_proposal is not None else None
            return CoordinatorDecision(
                action=action,
                reason="ratio_solo" if action is not None else "stop_no_affordable",
                verified_dominance=False,
            )

        if mode == MODE_CURIOSITY_SOLO:
            return CoordinatorDecision(
                action=cp_probe,
                reason="curiosity" if cp_probe is not None else "stop_all_known",
                verified_dominance=False,
            )

        if mode == MODE_RATIO_SOLO:
            action = ratio_proposal.probe if ratio_proposal is not None else None
            return CoordinatorDecision(
                action=action,
                reason="ratio_solo" if action is not None else "stop_no_affordable",
                verified_dominance=False,
            )

        if mode == MODE_COMBINED:
            if self._verify(curiosity_proposal, ratio_proposal):
                return CoordinatorDecision(
                    action=ratio_proposal.probe,
                    reason="ratio_counter",
                    verified_dominance=True,
                )
            return CoordinatorDecision(
                action=cp_probe,
                reason="curiosity" if cp_probe is not None else "stop_all_known",
                verified_dominance=False,
            )

        # MODE_COMBINED_IGNORE: чистый контроль, возражения не исполняются.
        return CoordinatorDecision(
            action=cp_probe,
            reason="curiosity" if cp_probe is not None else "stop_all_known",
            verified_dominance=False,
        )