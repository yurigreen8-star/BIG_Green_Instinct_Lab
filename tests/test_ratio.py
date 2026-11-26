"""M1–M6, M11–M13, A2, B4, B6."""

import inspect
import math

import numpy as np
import pytest

from belief import Belief
from environment import Environment
from evsi import evsi_analytic
from messages import CuriosityProposal, strictly_dominates
from policies.b_evsi import EVSICostPolicy
from probes import SURVEY_FAST, Probe, ProbeType, build_catalog
from ratio import Ratio
from runner import filter_available


def _b(mu, s2):
    mu = np.asarray(mu, float)
    s2 = np.asarray(s2, float)
    return Belief(mu=mu, sigma2=s2, known=(s2 == 0.0))


def _b0(n):
    return Belief(mu=np.zeros(n), sigma2=np.ones(n), known=np.zeros(n, bool))


def test_M1_evsi_two_parcels():
    assert evsi_analytic(_b([0, 0], [1, 1]), Probe(0, SURVEY_FAST)) == pytest.approx(
        1.0 / (2.0 * math.sqrt(math.pi)), abs=1e-12
    )


def test_M2_evsi_exact():
    exact = ProbeType(name="exact", cost=1.0, noise_var=0.0)
    assert evsi_analytic(_b([0, 0], [1, 1]), Probe(0, exact)) == pytest.approx(
        1.0 / math.sqrt(2.0 * math.pi), abs=1e-12
    )


def test_M3_M5_M6_evsi_edges():
    assert evsi_analytic(_b([0, 0], [1, 1]), Probe(0, SURVEY_FAST)) >= 0.0
    assert evsi_analytic(_b([0.7, 0], [0.0, 1]), Probe(0, SURVEY_FAST)) == 0.0
    assert evsi_analytic(_b([0.0], [1.0]), Probe(0, SURVEY_FAST)) == 0.0


def test_M4_evsi_leq_evpi():
    from evsi import evpi_mc

    b = _b([0.2, -0.1, 0.0], [1.0, 0.7, 1.3])
    d = evpi_mc(b, np.random.default_rng(7), n_samples=20000)
    assert evsi_analytic(b, Probe(1, SURVEY_FAST)) <= d["mean"] + 4.0 * d["se"]


def test_M11_strict_dominance_true():
    assert strictly_dominates(1.0, 0.30, 8.0, 0.20) is True
    assert strictly_dominates(1.0, 0.30, 1.0, 0.20) is True
    assert strictly_dominates(1.0, 0.20, 8.0, 0.20) is True


def test_M12_equal_both_not_strict():
    assert strictly_dominates(1.0, 0.2, 1.0, 0.2) is False


def test_M13_tolerance_half_eps():
    e = 1e-9
    assert strictly_dominates(1.0 + e / 2, 0.2 + e / 2, 1.0, 0.2) is False


def test_A2_no_curiosity_state():
    names = set(inspect.signature(Ratio.propose).parameters)
    for bad in ("curiosity_state", "interest", "visit_count", "last_surprise", "theta_true"):
        assert bad not in names


def test_B6_none_when_empty():
    r = Ratio()
    assert r.propose(_b0(2), 30.0, [], None, np.random.default_rng(0)) is None


def test_B4_ratio_solo_identical_to_B_EVSI_cost():
    """
    Контрольное свойство §2.2. Проверяется на уровне решений на 5 эпизодах.
    Ловит расхождение tie-break (флаг 1).
    """
    n = 30
    catalog = build_catalog(n)
    for seed in range(5):
        env = Environment.create(n, seed=seed)
        b_ref = _b0(n)
        b_new = _b0(n)
        budget_ref = budget_new = 30.0
        policy = EVSICostPolicy(use_mc=False)
        ratio = Ratio()
        rep_ref, rep_new = {}, {}
        rng = np.random.default_rng(seed + 10000)

        for _ in range(40):
            av_ref = filter_available(b_ref, catalog, budget_ref)
            av_new = filter_available(b_new, catalog, budget_new)
            if not av_ref:
                assert not av_new
                break
            p_ref = policy.select_probe(b_ref, av_ref, budget_ref, rng)
            rp = ratio.propose(b_new, budget_new, av_new, None, rng)
            assert rp is not None
            assert rp.probe == p_ref, f"seed={seed}: {rp.probe.name} != {p_ref.name}"

            for belief, probe, counters, tag in (
                (b_ref, p_ref, rep_ref, "ref"),
                (b_new, rp.probe, rep_new, "new"),
            ):
                key = (probe.parcel_id, probe.probe_type.probe_type_id)
                idx = counters.get(key, 0)
                counters[key] = idx + 1
                obs = env.observe(probe, idx)
                new_belief = belief.update(probe.parcel_id, obs.value, probe.noise_var)
                if tag == "ref":
                    b_ref = new_belief
                else:
                    b_new = new_belief
            budget_ref -= p_ref.cost
            budget_new -= rp.probe.cost

        assert budget_ref == pytest.approx(budget_new)
        assert np.allclose(b_ref.mu, b_new.mu)
def test_ratio_ignores_curiosity_interest():
    """Поправка 4: при том же parcel_id изменение interest не меняет решение Ratio."""
    n = 5
    b = _b([0.0, 0.3, -0.2, 0.1, 0.0], [1.0, 0.5, 1.0, 0.5, 1.0])
    catalog = build_catalog(n)
    avail = filter_available(b, catalog, 30.0)
    r = Ratio()
    rng = np.random.default_rng(0)

    base = r.propose(b, 30.0, avail, CuriosityProposal(2, 1.0), rng)
    for val in (0.0, 0.001, 0.5, 0.999, 1e6):
        alt = r.propose(b, 30.0, avail, CuriosityProposal(2, val), rng)
        assert alt == base