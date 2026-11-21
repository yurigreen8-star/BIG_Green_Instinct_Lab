"""
test_environment.py — среда, CRN-ключ, повторы, независимость от belief.
"""

import numpy as np
import pytest

from environment import Environment
from probes import SURVEY_FAST, SURVEY_PRECISE, SURVEY_STANDARD, Probe, ProbeType


def test_theta_true_is_constant():
    env = Environment.create(n_parcels=10, seed=42)
    original = env.theta_true.copy()
    for i in range(10):
        probe = Probe(parcel_id=i, probe_type=SURVEY_STANDARD)
        _ = env.observe(probe, rep_index=i)
    assert np.array_equal(env.theta_true, original)


def test_exact_measurement_returns_true_value():
    """Зонд с noise_var=0 возвращает ровно истину. В каталоге О2 таких нет."""
    env = Environment.create(n_parcels=5, seed=123)
    exact_type = ProbeType(name="exact_for_test", cost=100.0, noise_var=0.0)
    for i in range(5):
        probe = Probe(parcel_id=i, probe_type=exact_type)
        obs = env.observe(probe, rep_index=0)
        assert obs.value == pytest.approx(env.true_quality(i))


def test_noisy_probes_have_noise():
    env = Environment.create(n_parcels=1, seed=456)
    for probe_type in [SURVEY_FAST, SURVEY_STANDARD, SURVEY_PRECISE]:
        probe = Probe(parcel_id=0, probe_type=probe_type)
        true_val = env.true_quality(0)
        observations = [env.observe(probe, rep_index=k).value for k in range(2000)]
        assert np.isclose(np.mean(observations), true_val, atol=0.1)
        assert np.isclose(np.var(observations), probe_type.noise_var, atol=0.15)


def test_observation_deterministic_key():
    env1 = Environment.create(n_parcels=5, seed=42)
    env2 = Environment.create(n_parcels=5, seed=42)
    p = Probe(parcel_id=2, probe_type=SURVEY_STANDARD)
    assert env1.observe(p, rep_index=3).value == env2.observe(p, rep_index=3).value


def test_rep_index_changes_observation():
    env = Environment.create(n_parcels=5, seed=42)
    p = Probe(parcel_id=2, probe_type=SURVEY_FAST)
    o0 = env.observe(p, rep_index=0).value
    o1 = env.observe(p, rep_index=1).value
    assert o0 != o1


def test_order_independence():
    env = Environment.create(n_parcels=5, seed=42)
    p0 = Probe(parcel_id=0, probe_type=SURVEY_FAST)
    p1 = Probe(parcel_id=1, probe_type=SURVEY_STANDARD)
    a0 = env.observe(p0, rep_index=0).value
    a1 = env.observe(p1, rep_index=0).value

    env2 = Environment.create(n_parcels=5, seed=42)
    b1 = env2.observe(p1, rep_index=0).value
    b0 = env2.observe(p0, rep_index=0).value
    assert a0 == b0
    assert a1 == b1


def test_reproducibility():
    env1 = Environment.create(n_parcels=10, seed=42)
    env2 = Environment.create(n_parcels=10, seed=42)
    assert np.array_equal(env1.theta_true, env2.theta_true)


def test_different_seeds_give_different_theta():
    env1 = Environment.create(n_parcels=10, seed=1)
    env2 = Environment.create(n_parcels=10, seed=2)
    assert not np.array_equal(env1.theta_true, env2.theta_true)


def test_optimal_action():
    env = Environment.create(n_parcels=10, seed=42)
    best = env.optimal_action()
    assert env.theta_true[best] == env.theta_true.max()


def test_invalid_parcel_id():
    env = Environment.create(n_parcels=5, seed=42)
    bad = Probe(parcel_id=99, probe_type=SURVEY_FAST)
    with pytest.raises(ValueError):
        env.observe(bad, rep_index=0)


def test_invalid_rep_index():
    env = Environment.create(n_parcels=5, seed=42)
    p = Probe(parcel_id=0, probe_type=SURVEY_FAST)
    with pytest.raises(ValueError):
        env.observe(p, rep_index=-1)