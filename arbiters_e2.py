"""
arbiters_e2.py — DOM, OVR, VETO, MATCH. Предрегистрация E2 §3.5, §8.

Арбитры получают: первое предложение Curiosity, её ранжирование,
таблицу EVSI, режим и параметры. НЕ получают внутренние состояния
Curiosity, Environment, theta_true. EVSI самостоятельно НЕ вычисляют.

Проверка инвариантов доступа вынесена в tests/test_e2.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from ratio_e2 import RatioTableE2

EPS_DEFAULT = 1e-12
RHO_DEFAULT = 0.5

MODE_SOLO = "SOLO"
MODE_RATIO = "RATIO"
MODE_DOM = "DOM"
MODE_OVR = "OVR"
MODE_VETO = "VETO"
MODE_MATCH = "MATCH"


@dataclass
class DecisionE2:
    chosen: int
    chose_ratio: bool
    replaced: bool
    reason: str


def _decide(first: int, chosen: int, chose_ratio: bool, reason: str) -> DecisionE2:
    return DecisionE2(chosen=int(chosen), chose_ratio=bool(chose_ratio),
                      replaced=bool(int(chosen) != int(first)), reason=reason)


def arbitrate(mode: str, first: int, ranking: list[int],
              table: RatioTableE2, match_u: float | None = None,
              q: float | None = None, eps: float = EPS_DEFAULT,
              rho: float = RHO_DEFAULT) -> DecisionE2:
    e_first = table.evsi_of(first)
    best = table.best
    e_best = table.best_evsi

    if mode == MODE_SOLO:
        return _decide(first, first, False, "curiosity_solo")

    if mode == MODE_RATIO:
        return _decide(first, best, True, "ratio_solo")

    if mode == MODE_DOM:
        if e_best > e_first + eps:
            return _decide(first, best, True, "dom_override")
        return _decide(first, first, False, "dom_keep")

    if mode == MODE_OVR:
        if e_first >= rho * e_best:
            return _decide(first, first, False, "ovr_accept")
        return _decide(first, best, True, "ovr_override")

    if mode == MODE_VETO:
        if e_first <= eps and table.max_evsi > eps:
            for pid in ranking:
                if table.evsi_of(int(pid)) > eps:
                    return _decide(first, int(pid), False, "veto_next")
        return _decide(first, first, False, "veto_keep")

    if mode == MODE_MATCH:
        if match_u is None or q is None:
            raise ValueError("MATCH требует match_u и q")
        if not (0.0 <= q <= 1.0):
            raise ValueError("q в [0,1]")
        if match_u < q:
            return _decide(first, best, True, "match_ratio")
        return _decide(first, first, False, "match_curiosity")

    raise ValueError(f"неизвестный режим арбитража: {mode}")