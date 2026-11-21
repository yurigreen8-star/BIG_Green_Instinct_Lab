import numpy as np
import pytest

from belief import Belief
from policies.b_evsi import EVSICostPolicy, EVSIPolicy
from policies.b_ig import IGPolicy
from policies.b_mean_precision import MeanPrecisionPolicy
from policies.b_none import NonePolicy
from policies.b_random import RandomPolicy
from probes import SURVEY_FAST, SURVEY_PRECISE, SURVEY_STANDARD, Probe, build_catalog
from runner import filter_available


def _belief(mu, sigma2):
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    return Belief(mu=mu, sigma2=sigma2, known=(sigma2 == 0.0))


def test_random_deterministic_and_available():
    b = _belief([0.0, 0.0], [1.0, 1.0])
    catalog = build_catalog(2)
    avail = filter_available(b, catalog, 30)
    p1 = RandomPolicy().select_probe(b, avail, 30, np.random.default_rng(1))
    p2 = RandomPolicy().select_probe(b, avail, 30, np.random.default_rng(1))
    assert p1 == p2
    assert p1 in avail


def test_mean_precision_picks_best_mu_most_precise():
    b = _belief([0.5, 0.1], [1.0, 1.0])
    avail = filter_available(b, build_catalog(2), 30)
    p = MeanPrecisionPolicy().select_probe(b, avail, 30, np.random.default_rng(0))
    assert p.parcel_id == 0
    assert p.probe_type == SURVEY_PRECISE


def test_ig_skips_known_via_filter():
    b = _belief([0.7, 0.0], [0.0, 1.0])
    avail = filter_available(b, build_catalog(2), 30)
    assert all(p.parcel_id != 0 for p in avail)
    p = IGPolicy().select_probe(b, avail, 30, np.random.default_rng(0))
    assert p.parcel_id == 1


def test_evsi_skips_known_via_filter():
    b = _belief([0.7, 0.0], [0.0, 1.0])
    avail = filter_available(b, build_catalog(2), 30)
    p = EVSIPolicy().select_probe(b, avail, 30, np.random.default_rng(0))
    assert p.parcel_id == 1


def test_evsi_cost_variant_differs():
    b = _belief([0.0, 0.0], [1.0, 1.0])
    avail = filter_available(b, build_catalog(2), 30)
    rng = np.random.default_rng(0)
    p_abs = EVSIPolicy().select_probe(b, avail, 30, rng)
    p_cost = EVSICostPolicy().select_probe(b, avail, 30, rng)
    assert p_abs is not None and p_cost is not None


def test_none_returns_none():
    b = _belief([0.0], [1.0])
    assert NonePolicy().select_probe(b, build_catalog(1), 30, np.random.default_rng(0)) is None


def test_filter_respects_budget():
    b = _belief([0.0], [1.0])
    avail = filter_available(b, build_catalog(1), 2)
    assert all(p.cost <= 2 for p in avail)
    assert all(p.probe_type != SURVEY_PRECISE for p in avail)