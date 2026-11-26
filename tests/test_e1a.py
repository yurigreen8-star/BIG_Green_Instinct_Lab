"""
tests/test_e1a.py — обязательные тесты E1-A (T1–T10).
Восстановлен: файл выпал из сбора при переходе к E2.
"""

import math

import numpy as np
import pytest

from belief import Belief
from coordinator import Coordinator
from curiosity import Curiosity
from curiosity_abs_u import CuriosityAbsU
from environment import Environment
from evsi import evsi_analytic
from messages import MODE_CURIOSITY_SOLO, MODE_RATIO_SOLO
from policies.b_e1a_fast import EVSIFastPolicy
from probes import build_catalog
from ratio import Ratio
from runner import run_episode
from runner_v3 import run_episode_v3

N = 30


def _fast_catalog(n=N):
    return [p for p in build_catalog(n)
            if p.probe_type.probe_type_id == "survey_fast"]


def _b0(env):
    n = len(env.theta_true)
    return Belief(mu=np.zeros(n), sigma2=env.sigma2_prior.copy(),
                  known=np.zeros(n, dtype=bool))


@pytest.mark.parametrize("s,r,k", [
    (0.5, 1.0, 0), (0.5, 1.0, 1), (0.5, 1.0, 2), (0.5, 1.0, 5),
    (1.0, 1.0, 0), (1.0, 1.0, 1), (1.0, 1.0, 2), (1.0, 1.0, 5),
    (2.0, 1.0, 0), (2.0, 1.0, 1), (2.0, 1.0, 2), (2.0, 1.0, 5),
])
def test_T3_bayes_formulas(s, r, k):
    sigma2_k = s * r / (r + k * s)
    u_k = r / (r + k * s)
    A_k = r / ((r + k * s) * (1 + k))
    b = Belief(mu=np.array([0.0]), sigma2=np.array([s]), known=np.array([False]))
    for _ in range(k):
        b = b.update(0, 0.0, r)
    assert b.sigma2[0] == pytest.approx(sigma2_k, rel=1e-9)
    assert u_k == pytest.approx(sigma2_k / s, rel=1e-9)
    assert A_k == pytest.approx(u_k / (1 + k), rel=1e-9)


def test_T4_ig_vs_relative_var_reduction():
    rng = np.random.default_rng(123)
    for _ in range(50):
        s_vals = rng.uniform(0.1, 3.0, 10)
        r = 1.0
        ig_vals = 0.5 * np.log1p(s_vals / r)
        s_post = s_vals * r / (r + s_vals)
        rel_red = 1.0 - s_post / s_vals
        assert np.argsort(-ig_vals).tolist() == np.argsort(-rel_red).tolist()


def test_T5_three_sided_example():
    mu = np.array([-2.0, 0.9, 1.0])
    s = np.array([2.0, 1.0, 0.5])
    sigma2 = s / (1.0 + s)
    b = Belief(mu=mu, sigma2=sigma2, known=np.zeros(3, dtype=bool))
    probes = _fast_catalog(3)

    ig_vals = np.array([0.5 * math.log1p(sigma2[i] / 1.0) for i in range(3)])
    assert ig_vals[0] == pytest.approx(0.2554, abs=1e-3)
    assert int(np.argmax(ig_vals)) == 0

    u = 1.0 / (1.0 + s)
    A = u * 0.5
    assert A[2] == pytest.approx(0.3333, abs=1e-3)
    assert int(np.argmax(A)) == 2

    evsi_vals = np.array([evsi_analytic(b, p) for p in probes])
    assert evsi_vals[1] == pytest.approx(0.118, abs=1e-2)
    assert int(np.argmax(evsi_vals)) == 1


def _curiosity_sequence(env, seed, budget=35.0):
    res = run_episode_v3(
        mode=MODE_CURIOSITY_SOLO,
        curiosity=Curiosity(N, sigma2_prior=env.sigma2_prior.copy(), lam=0.3, seed=seed),
        ratio=Ratio(), coordinator=Coordinator(),
        belief0=_b0(env), catalog=_fast_catalog(), budget=budget, env=env,
        seed=seed, experiment_mode="A",
    )
    return [e["parcel_id"] for e in res.log if e.get("event") is None]


def test_T1_open_loop():
    seed = 40000
    env_a = Environment.create_e1a(N, seed=seed)
    env_b = Environment.create_e1a(N, seed=seed)
    env_b.theta_true = -env_b.theta_true + 5.0
    assert _curiosity_sequence(env_a, seed) == _curiosity_sequence(env_b, seed)


def test_T2_initial_sweep_permutation():
    for seed in range(40000, 40005):
        env = Environment.create_e1a(N, seed=seed)
        seq = _curiosity_sequence(env, seed, budget=30.0)
        assert sorted(seq[:30]) == list(range(N)), f"seed {seed}"


def test_T6_ratio_equals_evsi_fast():
    for seed in range(40000, 40003):
        env = Environment.create_e1a(N, seed=seed)
        b0, cat = _b0(env), _fast_catalog()
        res_ratio = run_episode_v3(
            mode=MODE_RATIO_SOLO,
            curiosity=Curiosity(N, sigma2_prior=env.sigma2_prior.copy(), lam=0.3, seed=seed),
            ratio=Ratio(), coordinator=Coordinator(),
            belief0=b0, catalog=cat, budget=30.0, env=env,
            seed=seed, experiment_mode="A",
        )
        seq_ratio = [e["parcel_id"] for e in res_ratio.log if e.get("event") is None]
        res_evsi = run_episode(EVSIFastPolicy(), b0, cat, 30.0, env, seed=seed, mode="A")
        seq_evsi = [e["parcel_id"] for e in res_evsi.log if e.get("event") is None]
        assert seq_ratio == seq_evsi, f"seed {seed}"


def test_T7_crn():
    env1 = Environment.create_e1a(N, seed=40000)
    env2 = Environment.create_e1a(N, seed=40000)
    probe = _fast_catalog()[0]
    assert env1.observe(probe, 0).value == env2.observe(probe, 0).value


def test_abs_u_initialization():
    env = Environment.create_e1a(N, seed=40000)
    c = CuriosityAbsU(N, sigma2_prior=env.sigma2_prior.copy(), lam=0.3)
    np.testing.assert_allclose(c.get_state().interest, env.sigma2_prior, rtol=1e-12)


def test_T9_T10_evsi_regression_and_calibration():
    env = Environment.create_e1a(N, seed=40001)
    res = run_episode_v3(
        mode=MODE_RATIO_SOLO,
        curiosity=Curiosity(N, sigma2_prior=env.sigma2_prior.copy(), lam=0.3, seed=40001),
        ratio=Ratio(), coordinator=Coordinator(),
        belief0=_b0(env), catalog=_fast_catalog(), budget=30.0, env=env,
        seed=40001, experiment_mode="A",
    )
    steps = [e for e in res.log if e.get("event") is None]
    for e in steps:
        assert e["evsi_analytic_executed"] is not None
        assert e["evsi_analytic_executed"] >= 0.0
    v_final = float(np.max(res.log[-1]["final_mu"]))
    sum_dv = sum(e["delta_posterior_value"] for e in steps)
    assert sum_dv == pytest.approx(v_final - 0.0, abs=1e-9)