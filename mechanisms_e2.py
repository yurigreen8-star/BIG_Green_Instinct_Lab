"""
mechanisms_e2.py — единый интерфейс C_OL и C_BS для runner_e2.

C_OL: обёртка над существующим Curiosity (формула не меняется).
C_BS: обёртка над CuriosityBS.

Ни один механизм не получает стоимость, бюджет, EVSI, состояние Ratio,
theta_true. Наблюдение передаётся только C_OL для ДИАГНОСТИЧЕСКОГО surprise,
который не входит в мотивационное состояние (инвариант H2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from belief import Belief
from curiosity import Curiosity
from curiosity_bs import CuriosityBS
from messages import build_curiosity_view, predictive_surprise

TIE_EPS = 1e-9


@dataclass
class ProposalE2:
    first: int
    ranking: list[int]
    interest_first: float


class _Base:
    kind: str = "base"

    def propose(self, known_mask: np.ndarray, rng: np.random.Generator) -> ProposalE2:
        raise NotImplementedError

    def after_measurement(self, parcel_id: int, belief_before: Belief,
                          belief_after: Belief, observation: float,
                          noise_var: float) -> None:
        raise NotImplementedError

    def interest_vector(self) -> np.ndarray:
        raise NotImplementedError


class COLD1(_Base):
    """C_OL: существующая open-loop Curiosity."""

    kind = "OL"

    def __init__(self, n_parcels: int, s_prior, lam: float = 0.3) -> None:
        self._c = Curiosity(n_parcels, sigma2_prior=s_prior, lam=lam)
        self._n = int(n_parcels)

    def propose(self, known_mask, rng) -> ProposalE2:
        cp = self._c.propose(np.asarray(known_mask, dtype=bool), rng)
        if cp is None:
            raise RuntimeError("нет кандидатов: в E2 known не ограничивает доступ")
        first = int(cp.parcel_id)
        interest = self._c.get_state().interest
        order = np.argsort(-interest, kind="stable")
        tail: list[int] = []
        i = 0
        while i < len(order):
            j = i + 1
            while (j < len(order)
                   and abs(interest[order[j]] - interest[order[i]]) < TIE_EPS):
                j += 1
            group = [int(x) for x in order[i:j] if int(x) != first]
            if len(group) > 1:
                group = [int(x) for x in rng.permutation(np.array(group))]
            tail.extend(group)
            i = j
        return ProposalE2(first=first, ranking=[first] + tail,
                          interest_first=float(cp.interest))

    def after_measurement(self, parcel_id, belief_before, belief_after,
                          observation, noise_var) -> None:
        k = int(parcel_id)
        s = predictive_surprise(float(observation), float(belief_before.mu[k]),
                                float(belief_before.sigma2[k]), float(noise_var))
        view = build_curiosity_view(belief_after.sigma2, belief_after.known,
                                    self._c.sigma2_prior)
        self._c.update_after_observation(k, view, s)

    def interest_vector(self) -> np.ndarray:
        return self._c.get_state().interest

    def visit_counts(self) -> np.ndarray:
        return self._c.get_state().visit_count


class CBSD1(_Base):
    """C_BS: closed-loop. Наблюдение не читает."""

    kind = "BS"

    def __init__(self, n_parcels: int, s_prior, r: float = 1.0,
                 lam: float = 0.3) -> None:
        self._c = CuriosityBS(n_parcels, s_prior=s_prior, r=r, lam=lam)
        self._n = int(n_parcels)

    def propose(self, known_mask, rng) -> ProposalE2:
        ranking = self._c.ranking(rng)
        first = ranking[0]
        return ProposalE2(first=first, ranking=ranking,
                          interest_first=float(self._c.interest_vector()[first]))

    def after_measurement(self, parcel_id, belief_before, belief_after,
                          observation, noise_var) -> None:
        k = int(parcel_id)
        self._c.after_measurement(
            k,
            float(belief_before.mu[k]), float(belief_before.sigma2[k]),
            float(belief_after.mu[k]), float(belief_after.sigma2[k]),
        )

    def interest_vector(self) -> np.ndarray:
        return self._c.interest_vector()

    def attract_vector(self) -> np.ndarray:
        return self._c.attract_vector()

    def visit_counts(self) -> np.ndarray:
        return self._c.get_state().visit_count

    def last_bs(self) -> np.ndarray:
        return self._c.get_state().last_bs


def make_mechanism(kind: str, n_parcels: int, s_prior, r: float = 1.0,
                   lam: float = 0.3):
    if kind == "OL":
        return COLD1(n_parcels, s_prior, lam=lam)
    if kind == "BS":
        return CBSD1(n_parcels, s_prior, r=r, lam=lam)
    raise ValueError(f"неизвестный тип Curiosity: {kind}")