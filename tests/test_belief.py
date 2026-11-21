"""
test_belief.py — тесты для belief.py.

Тесты 1–7 из спецификации проекта (гейт к EVSI).
"""

import math
import warnings

import numpy as np
import pytest

# Добавляем путь к корню проекта, чтобы импортировать belief
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from belief import EPS_NUM, Belief


# ----------------------------------------------------------------
# Тест 1: одно гауссовское наблюдение, аналитика
# ----------------------------------------------------------------
def test_single_gaussian_observation():
    """mu=0, sigma2=1, s2=1, o=2 -> mu'=1, sigma2'=0.5"""
    belief = Belief(
        mu=np.array([0.0]),
        sigma2=np.array([1.0]),
        known=np.array([False]),
    )

    new_belief = belief.update(parcel_id=0, observation=2.0, noise_var=1.0)

    assert np.isclose(new_belief.mu[0], 1.0)
    assert np.isclose(new_belief.sigma2[0], 0.5)
    assert new_belief.known[0] == False  # noqa: E712


# ----------------------------------------------------------------
# Тест 2: серия наблюдений того же участка (батчевое обновление)
# ----------------------------------------------------------------
def test_batch_of_observations():
    """
    3 наблюдения подряд. Последовательное обновление должно совпасть
    с батчевой формулой:
        tau' = tau + sum_k tau_o,k
        mu'  = (tau*mu + sum_k tau_o,k * o_k) / tau'
    """
    belief = Belief(
        mu=np.array([0.0]),
        sigma2=np.array([1.0]),
        known=np.array([False]),
    )

    observations = [1.0, 2.0, 3.0]
    noise_vars = [1.0, 2.0, 0.5]

    for o, s2 in zip(observations, noise_vars):
        belief = belief.update(parcel_id=0, observation=o, noise_var=s2)

    tau_prior = 1.0 / 1.0
    tau_obs_sum = sum(1.0 / s2 for s2 in noise_vars)
    tau_post = tau_prior + tau_obs_sum
    sigma2_post = 1.0 / tau_post
    mu_post = (
        tau_prior * 0.0
        + sum(o / s2 for o, s2 in zip(observations, noise_vars))
    ) / tau_post

    assert np.isclose(belief.mu[0], mu_post)
    assert np.isclose(belief.sigma2[0], sigma2_post)
    assert belief.known[0] == False  # noqa: E712


# ----------------------------------------------------------------
# Тест 3: точное измерение
# ----------------------------------------------------------------
def test_exact_measurement():
    """s2=0, o=0.7 -> mu'=0.7, sigma2'=0, known=True"""
    belief = Belief(
        mu=np.array([0.0]),
        sigma2=np.array([1.0]),
        known=np.array([False]),
    )

    new_belief = belief.update(parcel_id=0, observation=0.7, noise_var=0.0)

    assert np.isclose(new_belief.mu[0], 0.7)
    assert new_belief.sigma2[0] == 0.0
    assert new_belief.known[0] == True  # noqa: E712


# ----------------------------------------------------------------
# Тест 4: повторный зонд по known -> ValueError
# ----------------------------------------------------------------
def test_repeated_probe_on_known_raises():
    belief = Belief(
        mu=np.array([0.7]),
        sigma2=np.array([0.0]),
        known=np.array([True]),
    )

    with pytest.raises(ValueError, match="Повторное зондирование запрещено"):
        belief.update(parcel_id=0, observation=0.8, noise_var=1.0)


# ----------------------------------------------------------------
# Тест 5: малая положительная дисперсия — не known, но предупреждение
# ----------------------------------------------------------------
def test_small_positive_variance_not_known():
    """
    sigma2=1e-12, s2=1e-12 -> sigma2' = 0.5e-12 < EPS_NUM.
    Ожидание: НЕ known, есть предупреждение.
    """
    belief = Belief(
        mu=np.array([0.0]),
        sigma2=np.array([1e-12]),
        known=np.array([False]),
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new_belief = belief.update(
            parcel_id=0, observation=1.0, noise_var=1e-12
        )

    assert 0 < new_belief.sigma2[0] < EPS_NUM
    assert new_belief.known[0] == False  # noqa: E712
    assert len(w) == 1
    assert "ниже EPS_NUM" in str(w[0].message)


# ----------------------------------------------------------------
# Тест 6: некорректные параметры
# ----------------------------------------------------------------
def test_invalid_parameters():
    # Отрицательная дисперсия при создании
    with pytest.raises(ValueError, match="sigma2 не может быть отрицательной"):
        Belief(
            mu=np.array([0.0]),
            sigma2=np.array([-1.0]),
            known=np.array([False]),
        )

    # Отрицательный noise_var при обновлении
    belief = Belief(
        mu=np.array([0.0]),
        sigma2=np.array([1.0]),
        known=np.array([False]),
    )
    with pytest.raises(ValueError, match="noise_var должен быть >= 0"):
        belief.update(parcel_id=0, observation=1.0, noise_var=-0.5)

    # Нарушение инварианта при создании
    with pytest.raises(ValueError, match="known должен быть эквивалентен"):
        Belief(
            mu=np.array([0.0]),
            sigma2=np.array([1.0]),
            known=np.array([True]),
        )

    # Неверный индекс участка
    with pytest.raises(ValueError, match="parcel_id вне диапазона"):
        belief.update(parcel_id=99, observation=1.0, noise_var=1.0)


# ----------------------------------------------------------------
# Тест 7: независимость участков
# ----------------------------------------------------------------
def test_independence_of_parcels():
    """Наблюдение по участку i не меняет mu, sigma2 для j != i."""
    n = 5
    belief = Belief(
        mu=np.arange(n, dtype=np.float64),
        sigma2=np.ones(n),
        known=np.zeros(n, dtype=bool),
    )

    new_belief = belief.update(parcel_id=2, observation=10.0, noise_var=1.0)

    assert new_belief.mu[2] != belief.mu[2]
    assert new_belief.sigma2[2] != belief.sigma2[2]

    for j in range(n):
        if j != 2:
            assert new_belief.mu[j] == belief.mu[j]
            assert new_belief.sigma2[j] == belief.sigma2[j]
            assert new_belief.known[j] == belief.known[j]


# ----------------------------------------------------------------
# Дополнительный тест: иммутабельность
# ----------------------------------------------------------------
def test_immutability():
    """Обновление не изменяет исходный Belief."""
    belief = Belief(
        mu=np.array([0.0, 1.0]),
        sigma2=np.array([1.0, 1.0]),
        known=np.array([False, False]),
    )

    original_mu = belief.mu.copy()
    original_sigma2 = belief.sigma2.copy()
    original_known = belief.known.copy()

    _ = belief.update(parcel_id=0, observation=5.0, noise_var=1.0)

    assert np.array_equal(belief.mu, original_mu)
    assert np.array_equal(belief.sigma2, original_sigma2)
    assert np.array_equal(belief.known, original_known)


# ----------------------------------------------------------------
# Аналитическое значение для будущего теста 11 (документация)
# ----------------------------------------------------------------
def test_evsi_analytical_value_documented():
    """
    Фиксируем аналитическое значение для будущего теста 11:
    mu1=mu2=0, sigma1^2=sigma2^2=1, зонд по участку 1 с s2=1.
    EVSI = 1 / (2 * sqrt(pi)) ~ 0.28209
    """
    expected = 1.0 / (2.0 * math.sqrt(math.pi))
    assert np.isclose(expected, 0.28209, atol=1e-4)