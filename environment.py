"""
environment.py — среда с истинными значениями.

CRN-протокол:
    Наблюдение детерминировано ключом
    (episode_seed, parcel_id, probe_type_id, rep_index).
    Устойчивые строковые id, без Python hash().
    Environment.observe принимает rep_index явно.
    Скрытого счётчика наблюдений нет.

theta_true НИКОГДА не передаётся в политики.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from probes import Probe


@dataclass(frozen=True)
class Observation:
    """Observation = Action + rep_index + value."""

    probe: Probe
    rep_index: int
    value: float


@dataclass
class Environment:
    theta_true: np.ndarray
    episode_seed: int

    @classmethod
    def create(
        cls,
        n_parcels: int,
        seed: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ) -> "Environment":
        if n_parcels <= 0:
            raise ValueError(f"n_parcels должен быть > 0, получено {n_parcels}")
        rng_env = np.random.default_rng(seed)
        theta_true = rng_env.normal(loc=prior_mean, scale=prior_std, size=n_parcels)
        return cls(theta_true=theta_true, episode_seed=int(seed))

    @property
    def n(self) -> int:
        return self.theta_true.shape[0]

    def true_quality(self, parcel_id: int) -> float:
        return float(self.theta_true[parcel_id])

    def optimal_action(self) -> int:
        return int(np.argmax(self.theta_true))

    def observe(self, probe: Probe, rep_index: int) -> Observation:
        if rep_index < 0:
            raise ValueError(f"rep_index должен быть >= 0, получено {rep_index}")
        if not (0 <= probe.parcel_id < self.n):
            raise ValueError(f"parcel_id {probe.parcel_id} вне диапазона [0, {self.n})")

        q_true = self.theta_true[probe.parcel_id]

        if probe.noise_var == 0.0:
            value = q_true
        else:
            key = (
                f"{self.episode_seed}|{probe.parcel_id}|"
                f"{probe.probe_type.probe_type_id}|{rep_index}"
            )
            digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
            noise_seed = int.from_bytes(digest, byteorder="little", signed=False)
            rng_obs = np.random.default_rng(noise_seed)
            value = q_true + rng_obs.normal(0.0, np.sqrt(probe.noise_var))

        return Observation(probe=probe, rep_index=rep_index, value=float(value))