"""
test_evsi.py — T1–T5, CRN, EVPI.
"""

import math

import numpy as np
import pytest

from belief import Belief
from evsi import (
    compute_evsi_table_mc,
    evpi_mc,
    evsi_analytic,
    evsi_mc_from_samples,
    ig,
)
from probes import SURVEY_FAST, SURVEY_STANDARD, Probe, ProbeType


def _belief(mu, sigma2):
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    return Belief(mu=mu, sigma2=sigma2, known=(sigma2 == 0.0))


def test_T1_two_parcels_analytic():
    b = _belief([0.0, 0.0], [1.0, 1.0])
    p = Probe(0, SURVEY_FAST)  # s2=1
    expected = 1.0 / (2.0 * math.sqrt(math.pi))
    assert evsi_analytic(b, p) == pytest.approx(expected, abs=1e-12)


def test_T2_dominated_parcel_small_evsi():
    b = _belief([0.0, 2.0], [1.0, 1.0])
    p = Probe(0, SURVEY_FAST)
    val = evsi_analytic(b, p)
    assert 0.0 <= val < 0.01


def test_T3_exact_one_probe_leq_evpi_two_unknown():
    b = _belief([0.0, 0.0], [1.0, 1.0])
    exact = ProbeType(name="exact", cost=1.0, noise_var=0.0)
    p = Probe(0, exact)
    evsi_exact = evsi_analytic(b, p)
    assert evsi_exact == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), abs=1e-12)

    # EVPI двух неизвестных N(0,1): E[max(q1,q2)] = 1/sqrt(pi)
    evpi_two = 1.0 / math.sqrt(math.pi)
    assert evsi_exact < evpi_two  # НЕ равенство


def test_T4_known_parcel_zero_evsi():
    b = _belief([0.7, 0.0], [0.0, 1.0])
    p = Probe(0, SURVEY_FAST)
    assert evsi_analytic(b, p) == 0.0


def test_T5_mc_matches_analytic_within_se():
    b = _belief([0.0, 0.1, -0.2], [1.0, 0.5, 2.0])
    p = Probe(0, SURVEY_STANDARD)
    analytic = evsi_analytic(b, p)
    rng = np.random.default_rng(12345)
    table = compute_evsi_table_mc(b, [p], rng, n_samples=20000)
    mc = table[0]
    assert mc["se"] > 0.0
    assert abs(mc["evsi"] - analytic) <= 4.0 * mc["se"] + 1e-12


def test_evsi_leq_evpi_random():
    b = _belief([0.2, -0.1, 0.0], [1.0, 0.7, 1.3])
    rng = np.random.default_rng(7)
    evpi = evpi_mc(b, rng, n_samples=20000)
    for pt in [SURVEY_FAST, SURVEY_STANDARD]:
        p = Probe(1, pt)
        assert evsi_analytic(b, p) <= evpi["mean"] + 4.0 * evpi["se"]


def test_crn_reduces_variance():
    b = _belief([0.0, 0.0], [1.0, 1.0])
    z1 = Probe(0, SURVEY_FAST)
    z2 = Probe(0, SURVEY_STANDARD)
    n = 2000
    diffs_shared = []
    diffs_indep = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        theta = rng.normal(b.mu, np.sqrt(b.sigma2), size=(n, b.n))
        omega = rng.standard_normal(n)
        e1 = evsi_mc_from_samples(b, z1, theta, omega)["evsi"]
        e2 = evsi_mc_from_samples(b, z2, theta, omega)["evsi"]
        diffs_shared.append(e1 - e2)

        omega1 = rng.standard_normal(n)
        omega2 = rng.standard_normal(n)
        e1i = evsi_mc_from_samples(b, z1, theta, omega1)["evsi"]
        e2i = evsi_mc_from_samples(b, z2, theta, omega2)["evsi"]
        diffs_indep.append(e1i - e2i)

    assert np.var(diffs_shared) < np.var(diffs_indep)


def test_evpi_mc_returns_mean_se_ucb95():
    b = _belief([0.0, 0.0], [1.0, 1.0])
    rng = np.random.default_rng(0)
    d = evpi_mc(b, rng, n_samples=1000)
    assert set(["mean", "se", "ucb95", "v0"]).issubset(d.keys())
    assert d["ucb95"] == pytest.approx(d["mean"] + 1.96 * d["se"])


def test_ig_exact_uses_eps():
    b = _belief([0.0, 0.0], [1.0, 1.0])
    exact = ProbeType(name="exact", cost=1.0, noise_var=0.0)
    p = Probe(0, exact)
    assert ig(b, p) > 0.0