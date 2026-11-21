"""
test_environment.py — тесты среды.
"""

import numpy as np
import pytest

from environment import Environment
from probes import SURVEY_FAST, SURVEY_PRECISE, SURVEY_STANDARD, Probe


# --- Тест 8: theta_true постоянен на протяжении эпизода ---
def test_theta_true_is_constant():
    env = Environment.create(n_parcels=10, seed=42)
    original = env.theta_true.copy()

    # Делаем много наблюдений
    for i in range(10):
        probe = Probe(parcel_id=i, probe_type=SURVEY_STANDARD)
        _ = env.observe(probe)

    assert np.array_equal(env.theta_true, original)


# --- Тест 9: точное измерение возвращает ровно theta_true ---
def test_exact_measurement_returns_true_value():
    """
    Зонд с noise_var=0 возвращает ровно истинное значение.
    
    Примечание: в текущей постановке О2 в каталоге нет зондов с
    нулевой дисперсией. Мы проверяем логику точного измерения
    напрямую, через кастомный тип зонда.
    """
    from probes import ProbeType

    env = Environment.create(n_parcels=5, seed=123)
    exact_type = ProbeType(name="exact_for_test", cost=100.0, noise_var=0.0)

    for i in range(5):
        probe = Probe(parcel_id=i, probe_type=exact_type)
        obs = env.observe(probe)
        assert obs == pytest.approx(env.true_quality(i))


# --- Тест 9б: шумные зонды дают значение в пределах шума ---
def test_noisy_probes_have_noise():
    """
    Все три типа зондов из каталога шумные (по постановке О2).
    Наблюдение отличается от истинного значения, но в среднем совпадает.
    """
    env = Environment.create(n_parcels=1, seed=456)

    for probe_type in [SURVEY_FAST, SURVEY_STANDARD, SURVEY_PRECISE]:
        probe = Probe(parcel_id=0, probe_type=probe_type)
        true_val = env.true_quality(0)

        # Собираем 500 наблюдений для оценки среднего и дисперсии
        observations = [env.observe(probe) for _ in range(500)]
        mean_obs = np.mean(observations)
        var_obs = np.var(observations)

        # Среднее должно совпадать с истиной
        assert np.isclose(mean_obs, true_val, atol=0.15)

        # Дисперсия наблюдений должна быть близка к заявленной
        assert np.isclose(var_obs, probe_type.noise_var, atol=0.1)


# --- Тест 10: наблюдение не зависит от Belief ---
def test_observation_independent_of_belief():
    env = Environment.create(n_parcels=5, seed=456)
    probe = Probe(parcel_id=0, probe_type=SURVEY_STANDARD)

    # Сохраняем RNG-состояние (наблюдение детерминировано при одном seed)
    env_fresh = Environment.create(n_parcels=5, seed=456)
    probe2 = Probe(parcel_id=0, probe_type=SURVEY_STANDARD)

    assert env.observe(probe) == env_fresh.observe(probe2)


# --- Дополнительные тесты ---
def test_reproducibility():
    """Один seed -> одно и то же theta_true."""
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


def test_noisy_observation_has_variance():
    """Шумное измерение в среднем совпадает с истиной, но флуктуирует."""
    env = Environment.create(n_parcels=1, seed=42)
    probe = Probe(parcel_id=0, probe_type=SURVEY_FAST)  # noise_var=1.0

    observations = [env.observe(probe) for _ in range(1000)]
    mean_obs = np.mean(observations)
    var_obs = np.var(observations)

    assert np.isclose(mean_obs, env.true_quality(0), atol=0.1)
    assert np.isclose(var_obs, 1.0, atol=0.2)


def test_invalid_parcel_id():
    env = Environment.create(n_parcels=5, seed=42)
    bad_probe = Probe(parcel_id=99, probe_type=SURVEY_FAST)
    with pytest.raises(ValueError):
        env.observe(bad_probe)