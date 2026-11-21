import numpy as np
import pytest

from belief import Belief
from environment import Environment
from policies.b_evsi import EVSIPolicy
from policies.b_mean_precision import MeanPrecisionPolicy
from policies.b_none import NonePolicy
from policies.b_random import RandomPolicy
from probes import SURVEY_FAST, Probe, build_catalog
from runner import run_episode


def _belief0(n):
    return Belief(mu=np.zeros(n), sigma2=np.ones(n), known=np.zeros(n, dtype=bool))


class AlwaysFirstPolicy:
    name = "always_first"

    def select_probe(self, belief, available, budget_remaining, rng):
        return available[0] if available else None


def test_repeats_allowed_rep_index_increments():
    n = 2
    env = Environment.create(n, seed=42)
    catalog = build_catalog(n)
    res = run_episode(AlwaysFirstPolicy(), _belief0(n), catalog, budget=5, env=env, seed=42, mode="A")
    obs_entries = [e for e in res.log if "observation" in e]
    assert len(obs_entries) == 5
    assert [e["rep_index"] for e in obs_entries] == [0, 1, 2, 3, 4]
    assert all(e["probe_type_id"] == "survey_fast" for e in obs_entries)
    assert res.budget_used == 5


def test_budget_not_exceeded_random():
    n = 5
    env = Environment.create(n, seed=7)
    res = run_episode(RandomPolicy(), _belief0(n), build_catalog(n), budget=30, env=env, seed=7, mode="A")
    assert res.budget_used <= 30
    assert res.budget_remaining >= 0


def test_none_policy_zero_probes():
    n = 3
    env = Environment.create(n, seed=1)
    res = run_episode(NonePolicy(), _belief0(n), build_catalog(n), budget=30, env=env, seed=1, mode="A")
    assert res.n_probes == 0
    assert res.stop_reason == "policy_none"


def test_mode_B_stops_by_evpi_ucb():
    n = 2
    env = Environment.create(n, seed=99)
    res = run_episode(
        EVSIPolicy(),
        _belief0(n),
        build_catalog(n),
        budget=1000,
        env=env,
        seed=99,
        mode="B",
        horizon=5000,
        evpi_n_samples=500,
    )
    assert res.stop_reason == "evpi_resolved"
    assert res.budget_used < 1000


def test_same_seed_same_observations():
    n = 4
    catalog = build_catalog(n)
    env1 = Environment.create(n, seed=5)
    env2 = Environment.create(n, seed=5)
    r1 = run_episode(RandomPolicy(), _belief0(n), catalog, 30, env1, seed=5, mode="A")
    r2 = run_episode(RandomPolicy(), _belief0(n), catalog, 30, env2, seed=5, mode="A")
    o1 = [e["observation"] for e in r1.log if "observation" in e]
    o2 = [e["observation"] for e in r2.log if "observation" in e]
    assert o1 == o2


def test_evsi_policy_runs_and_logs_diagnostics():
    n = 3
    env = Environment.create(n, seed=11)
    res = run_episode(EVSIPolicy(), _belief0(n), build_catalog(n), budget=30, env=env, seed=11, mode="A")
    obs_entries = [e for e in res.log if "observation" in e]
    assert len(obs_entries) > 0
    assert all("evsi" in e for e in obs_entries)
    assert all(e["evsi_method"] == "analytic" for e in obs_entries)