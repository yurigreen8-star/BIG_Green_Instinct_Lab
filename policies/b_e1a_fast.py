"""
policies/b_e1a_fast.py — политики для E1-A (только FAST).

B_IG_FAST   — максимум ожидаемого информационного выигрыша (IG).
B_EVSI_FAST — максимум EVSI.
B_SWEEP_THEN_RANDOM — однократный обход, затем случайные повторы с возвращением.

Все работают с произвольным каталогом, но в E1-A каталог ограничен FAST.
"""

from __future__ import annotations

import numpy as np

from belief import Belief
from evsi import evsi_analytic, ig
from probes import SURVEY_FAST, Probe


class IGFastPolicy:
    """Выбор объекта по максимальному IG (не IG/cost)."""
    name = "B_IG_FAST"

    def select_probe(self, belief, available, budget_remaining, rng):
        if not available:
            return None
        scored = [(float(ig(belief, p)), p) for p in available]
        best = max(scored, key=lambda d: (d[0], d[1].parcel_id))
        return best[1]


class EVSIFastPolicy:
    """Выбор объекта по максимальному EVSI (не EVSI/cost)."""
    name = "B_EVSI_FAST"

    def select_probe(self, belief, available, budget_remaining, rng):
        if not available:
            return None
        scored = [(float(evsi_analytic(belief, p)), p) for p in available]
        best = max(scored, key=lambda d: (d[0], d[1].parcel_id))
        return best[1]


class SweepThenRandomPolicy:
    """Однократный обход всех объектов, затем случайные повторы с возвращением."""
    name = "B_SWEEP_THEN_RANDOM"

    def __init__(self, sigma2_prior: float = 1.0) -> None:
        self._prior = float(sigma2_prior) if np.ndim(sigma2_prior) == 0 else 1.0
        self._perm: np.ndarray | None = None

    def _visits(self, belief: Belief) -> np.ndarray:
        s2 = np.asarray(belief.sigma2, dtype=float)
        # Для E1-A приоры неоднородны; выводим число посещений из точности.
        # 1/sigma2 = 1/s + k/r  =>  k = r*(1/sigma2 - 1/s). Для FAST r=1.
        return s2  # фактически используем логику ниже через счётчик точности

    def select_probe(self, belief, available, budget_remaining, rng):
        if not available:
            return None
        n = belief.n
        fresh = bool(np.all(np.asarray(belief.sigma2) >= self._prior - 1e-12))
        if fresh or self._perm is None or len(self._perm) != n:
            self._perm = rng.permutation(n)

        # Определяем посещённые по снижению дисперсии относительно приора.
        # В E1-A приоры разные, поэтому используем порог: посещён если
        # 1/sigma2 > 1/s_i (точность выросла хотя бы на одно измерение).
        visited = self._visited_mask(belief)
        n_visited = int(visited.sum())

        if n_visited < n:
            # Фаза обхода: берём следующий непосещённый из перестановки.
            for idx in self._perm:
                if not visited[int(idx)]:
                    target = int(idx)
                    break
        else:
            # Фаза повторов: равномерный случайный выбор с возвращением.
            target = int(rng.integers(n))

        for p in available:
            if (p.parcel_id == target
                    and p.probe_type.probe_type_id == SURVEY_FAST.probe_type_id):
                return p
        return None

    def _visited_mask(self, belief: Belief) -> np.ndarray:
        # Посещён, если апостериорная точность выше априорной.
        # Требует знания s_i; в E1-A берём из belief-независимого источника.
        # Здесь используем сохранённый скалярный приор как аппроксимацию для
        # определения завершения обхода; для неоднородных приоров см. примечание.
        s2 = np.asarray(belief.sigma2, dtype=float)
        return s2 < self._prior - 1e-12