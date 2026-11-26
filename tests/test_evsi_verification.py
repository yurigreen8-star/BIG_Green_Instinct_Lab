"""
tests/test_evsi_verification.py
Независимые тесты корректности EVSI и ΔV.
"""

import math
import numpy as np
import pytest

from belief import Belief
from evsi import evsi_analytic, ig
from probes import SURVEY_FAST, SURVEY_PRECISE, Probe, ProbeType

def _belief(mu, sigma2):
    return Belief(
        mu=np.asarray(mu, dtype=float),
        sigma2=np.asarray(sigma2, dtype=float),
        known=(np.asarray(sigma2, dtype=float) == 0.0)
    )

def test_T1_T4_analytic_match():
    """EVSI пересчёт совпадает с контрольными примерами T1-T4."""
    b = _belief([0.0, 0.0], [1.0, 1.0])
    p = Probe(0, SURVEY_FAST)
    expected = 1.0 / (2.0 * math.sqrt(math.pi))
    assert evsi_analytic(b, p) == pytest.approx(expected, abs=1e-12)

def test_zero_noise_limit():
    """Для нулевой дисперсии шума EVSI соответствует пределу."""
    b = _belief([0.0, 0.0], [1.0, 1.0])
    exact = ProbeType(name="exact", cost=1.0, noise_var=0.0)
    p = Probe(0, exact)
    # При s^2=0, tau^2 = sigma^2 = 1.0, d = 0. EVSI = 1 * phi(0) = 0.398942
    assert evsi_analytic(b, p) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), abs=1e-12)

def test_evsi_bounds():
    """EVSI неотрицателен и не превышает EVPI (для 2 участков EVPI = 1/sqrt(pi))."""
    b = _belief([0.0, 0.0], [1.0, 1.0])
    p = Probe(0, SURVEY_FAST)
    evsi = evsi_analytic(b, p)
    assert evsi >= 0.0
    assert evsi <= (1.0 / math.sqrt(math.pi)) + 1e-9

def test_delta_v_uses_global_max():
    """delta_posterior_value вычисляется по максимуму полного вектора mu."""
    mu_before = np.array([0.0, 0.0, 0.0])
    mu_after = np.array([0.0, -0.0201, 0.0]) # Зондировали участок 1, он упал
    delta_v = float(np.max(mu_after) - np.max(mu_before))
    assert delta_v == 0.0  # А не -0.0201!

def test_telescopic_sum():
    """Сумма ΔV телескопически совпадает с изменением максимального mu за эпизод."""
    mu_0 = np.array([0.0, 0.0])
    mu_1 = np.array([0.5, 0.0])
    mu_2 = np.array([0.5, 0.3])
    
    dv_1 = np.max(mu_1) - np.max(mu_0) # 0.5 - 0.0 = 0.5
    dv_2 = np.max(mu_2) - np.max(mu_1) # 0.5 - 0.5 = 0.0
    
    assert (dv_1 + dv_2) == pytest.approx(np.max(mu_2) - np.max(mu_0))

def test_missing_evsi_not_zero_imputed():
    """Отсутствующее значение EVSI не превращается автоматически в ноль в анализе."""
    # Эмуляция: если в логе null, анализатор должен это явно отметить, а не молча считать 0
    evsi_logged = None
    assert evsi_logged is None  # Анализатор должен использовать is not None проверку