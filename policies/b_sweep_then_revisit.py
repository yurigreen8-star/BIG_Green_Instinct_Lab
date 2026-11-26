"""
policies/b_sweep_then_revisit.py
Два контроля расписания повторов:

B_sweep_then_random_revisit  : случайная перестановка всех участков,
                               затем равномерные повторы С ВОЗВРАЩЕНИЕМ.
B_sweep_then_reverse_revisit : та же перестановка, затем повторы
                               в обратном порядке по кругу.

Оба используют только FAST => iso-budget с CURIOSITY_SOLO по построению.

Сброс состояния: при свежем Belief (все sigma2 == prior) генерируется новая
перестановка — политика безопасна при переиспользовании между эпизодами.
Число выполненных зондов выводится из Belief, а не из счётчика:
для fast-only sigma2 = prior/(1+v), поэтому v = round(prior/sigma2 - 1) точно.
"""

from __future__ import annotations

import numpy as np

from belief import Belief
from probes import SURVEY_FAST, Probe


class _SweepThenRevisit:
    def __init__(self, revisit_mode: str, sigma2_prior: float = 1.0) -> None:
        if revisit_mode not in ("random", "reverse"):
            raise ValueError("revisit_mode: 'random' | 'reverse'")
        if sigma2_prior <= 0.0:
            raise ValueError("sigma2_prior > 0")
        self._mode = revisit_mode
        self._prior = float(sigma2_prior)
        self._perm: np.ndarray | None = None

    def _visits(self, belief: Belief) -> np.ndarray:
        s2 = np.asarray(belief.sigma2, dtype=float)
        v = np.zeros(s2.shape, dtype=int)
        pos = s2 > 0.0
        v[pos] = np.rint(self._prior / s2[pos] - 1.0).astype(int)
        return v

    def select_probe(
        self,
        belief: Belief,
        available: list[Probe],
        budget_remaining: float,
        rng: np.random.Generator,
    ) -> Probe | None:
        if not available:
            return None
        n = belief.n
        fresh = bool(np.all(np.asarray(belief.sigma2) >= self._prior - 1e-12))
        if fresh or self._perm is None or len(self._perm) != n:
            self._perm = rng.permutation(n)

        total = int(self._visits(belief).sum())
        if total < n:
            target = int(self._perm[total])
        elif self._mode == "reverse":
            j = (total - n) % n
            target = int(self._perm[n - 1 - j])
        else:
            target = int(rng.integers(n))

        for p in available:
            if (p.parcel_id == target
                    and p.probe_type.probe_type_id == SURVEY_FAST.probe_type_id):
                return p
        return None


class SweepThenRandomRevisitPolicy(_SweepThenRevisit):
    name = "B_sweep_then_random_revisit"

    def __init__(self, sigma2_prior: float = 1.0) -> None:
        super().__init__("random", sigma2_prior)


class SweepThenReverseRevisitPolicy(_SweepThenRevisit):
    name = "B_sweep_then_reverse_revisit"

    def __init__(self, sigma2_prior: float = 1.0) -> None:
        super().__init__("reverse", sigma2_prior)