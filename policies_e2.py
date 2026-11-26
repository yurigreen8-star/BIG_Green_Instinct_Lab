"""
policies_e2.py — контрольные политики E2 (§4).

B_IG_FAST            : argmax IG среди FAST, ничьи — RNG.
B_RANDOM             : равномерный выбор участка с возвращением.
B_SWEEP_THEN_RANDOM  : обход всех участков в случайном для сценария порядке,
                       затем случайные повторы с возвращением.

Число посещений выводится из точности: 1/sigma2_i - 1/s_i = k/r.
Это корректно при неоднородных приорах (в отличие от скалярного порога).
"""

from __future__ import annotations

import numpy as np

from belief import Belief
from evsi import ig
from probes import Probe

TIE_EPS = 1e-9


def _by_parcel(available: list[Probe]) -> dict[int, Probe]:
    return {int(p.parcel_id): p for p in available}


class IGFastE2:
    name = "B_IG_FAST"

    def choose(self, belief: Belief, available: list[Probe],
               rng: np.random.Generator) -> int:
        ids = np.array([int(p.parcel_id) for p in available], dtype=np.int64)
        vals = np.array([float(ig(belief, p)) for p in available], dtype=np.float64)
        mx = float(vals.max())
        tied = np.flatnonzero(np.abs(vals - mx) < TIE_EPS)
        return int(ids[int(rng.choice(tied))])


class RandomE2:
    name = "B_RANDOM"

    def choose(self, belief: Belief, available: list[Probe],
               rng: np.random.Generator) -> int:
        ids = [int(p.parcel_id) for p in available]
        return int(ids[int(rng.integers(len(ids)))])


class SweepThenRandomE2:
    name = "B_SWEEP_THEN_RANDOM"

    def __init__(self, s_prior, r: float = 1.0) -> None:
        sp = np.asarray(s_prior, dtype=np.float64)
        if sp.ndim != 1 or np.any(sp <= 0.0):
            raise ValueError("s_prior: вектор, все > 0")
        self._s = sp
        self._r = float(r)
        self._perm: np.ndarray | None = None

    def visit_counts(self, belief: Belief) -> np.ndarray:
        s2 = np.asarray(belief.sigma2, dtype=np.float64)
        if np.any(s2 <= 0.0):
            raise ValueError("E2 FAST-only: sigma2 строго положительна")
        k = self._r * (1.0 / s2 - 1.0 / self._s)
        return np.rint(k).astype(np.int64)

    def choose(self, belief: Belief, available: list[Probe],
               rng: np.random.Generator) -> int:
        n = belief.n
        fresh = bool(np.allclose(belief.sigma2, self._s, rtol=0, atol=1e-12))
        if fresh or self._perm is None or len(self._perm) != n:
            self._perm = rng.permutation(n)
        counts = self.visit_counts(belief)
        pool = _by_parcel(available)
        unvisited = [int(i) for i in self._perm if counts[int(i)] == 0]
        if unvisited:
            for pid in unvisited:
                if pid in pool:
                    return pid
        ids = [int(p.parcel_id) for p in available]
        return int(ids[int(rng.integers(len(ids)))])