"""
environment_e2.py — сценарии E2 и независимые RNG-потоки.

Предрегистрация E2 §1, §2. Существующий Environment не изменяется:
используется его dataclass-конструктор и его CRN-ключ в observe().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from belief import Belief
from environment import Environment

N_PARCELS = 30
R_NOISE = 1.0
COST_FAST = 1.0

STREAM_PRIOR = 0
STREAM_TRUTH = 1
STREAM_NOISE = 2
STREAM_TIE_OL = 3
STREAM_TIE_BS = 4
STREAM_TIE_RATIO = 5
STREAM_MATCH = 6
STREAM_EVPI = 7
N_STREAMS = 8


def spawn_streams(seed: int) -> list:
    """SeedSequence.spawn с целочисленными id. Строки не передаются."""
    return np.random.SeedSequence(int(seed)).spawn(N_STREAMS)


def stream_rng(seed: int, stream_id: int) -> np.random.Generator:
    if not (0 <= stream_id < N_STREAMS):
        raise ValueError("stream_id вне диапазона")
    return np.random.default_rng(spawn_streams(seed)[stream_id])


@dataclass
class ScenarioE2:
    seed: int
    mu_prior: np.ndarray
    s_prior: np.ndarray
    env: Environment
    belief0: Belief

    @property
    def n(self) -> int:
        return int(self.mu_prior.shape[0])


def make_scenario(seed: int, n_parcels: int = N_PARCELS) -> ScenarioE2:
    """
    mu_i ~ N(0,1), s_i ~ U(0.5,2.0)  — поток 0
    q_i  ~ N(mu_i, s_i)              — поток 1
    Начальный Belief: mu=mu_i, sigma2=s_i, known=False.
    """
    if n_parcels <= 0:
        raise ValueError("n_parcels > 0")
    ss = spawn_streams(seed)
    rng_prior = np.random.default_rng(ss[STREAM_PRIOR])
    mu = rng_prior.normal(0.0, 1.0, n_parcels)
    s = rng_prior.uniform(0.5, 2.0, n_parcels)

    rng_truth = np.random.default_rng(ss[STREAM_TRUTH])
    q = rng_truth.normal(mu, np.sqrt(s), n_parcels)

    env = Environment(theta_true=q, episode_seed=int(seed), sigma2_prior=s)
    b0 = Belief(mu=mu.copy(), sigma2=s.copy(), known=np.zeros(n_parcels, dtype=bool))
    return ScenarioE2(seed=int(seed), mu_prior=mu, s_prior=s, env=env, belief0=b0)


def make_scenario_alt_truth(seed: int, variant: int,
                            n_parcels: int = N_PARCELS) -> ScenarioE2:
    """
    Тот же сценарий (те же mu_i, s_i, episode_seed => тот же CRN-шум),
    но иная реализация истины. Только для проверки H1/H2.
    В экспериментальных группах не используется.
    """
    base = make_scenario(seed, n_parcels)
    ss = np.random.SeedSequence([int(seed), int(variant), 991]).spawn(1)
    rng_alt = np.random.default_rng(ss[0])
    q_alt = rng_alt.normal(base.mu_prior, np.sqrt(base.s_prior), n_parcels)
    env = Environment(theta_true=q_alt, episode_seed=int(seed),
                      sigma2_prior=base.s_prior.copy())
    return ScenarioE2(seed=int(seed), mu_prior=base.mu_prior.copy(),
                      s_prior=base.s_prior.copy(), env=env,
                      belief0=base.belief0.copy())