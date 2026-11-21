"""
environment.py — среда с истинными значениями участков.

ВАЖНО: истинные значения (theta_true) НИКОГДА не должны попадать
в код политик или в Belief. Они доступны только через observe(),
которая возвращает зашумлённое измерение.

Это гарантирует, что политики учатся из реальных (шумных) данных,
а не "подглядывают" за истиной.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from probes import Probe


@dataclass
class Environment:
    """
    Среда: хранит истинные качества участков, генерирует наблюдения.

    Атрибуты:
        theta_true: вектор истинных качеств, форма (n,)
        rng_env:    RNG для генерации theta_true (один раз на эпизод)
        rng_obs:    отдельный RNG для наблюдений
    """

    theta_true: np.ndarray
    rng_obs: np.random.Generator

    @classmethod
    def create(
        cls,
        n_parcels: int,
        seed: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ) -> "Environment":
        """
        Создаёт новую среду с заданным числом участков.

        theta_true генерируется из N(prior_mean, prior_std^2).
        Используются два независимых RNG:
          - rng_env  для генерации theta_true (эпизод)
          - rng_obs  для шума наблюдений
        """
        if n_parcels <= 0:
            raise ValueError(f"n_parcels должен быть > 0, получено {n_parcels}")

        rng_env = np.random.default_rng(seed)
        rng_obs = np.random.default_rng(seed + 1000)

        theta_true = rng_env.normal(
            loc=prior_mean, scale=prior_std, size=n_parcels
        )

        return cls(theta_true=theta_true, rng_obs=rng_obs)

    @property
    def n(self) -> int:
        return self.theta_true.shape[0]

    def true_quality(self, parcel_id: int) -> float:
        """Возвращает истинное качество участка (для вычисления regret)."""
        return float(self.theta_true[parcel_id])

    def observe(self, probe: Probe) -> float:
        """
        Возвращает наблюдение = истинное значение + шум.

        Если noise_var == 0, возвращает точное значение (без шума).
        """
        if not (0 <= probe.parcel_id < self.n):
            raise ValueError(
                f"parcel_id {probe.parcel_id} вне диапазона [0, {self.n})"
            )

        q_true = self.theta_true[probe.parcel_id]

        if probe.noise_var == 0.0:
            return float(q_true)

        noise = self.rng_obs.normal(0.0, np.sqrt(probe.noise_var))
        return float(q_true + noise)

    def optimal_action(self) -> int:
        """Возвращает индекс участка с максимальным истинным качеством."""
        return int(np.argmax(self.theta_true))