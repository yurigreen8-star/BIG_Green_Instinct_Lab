"""
tests/test_e2.py — обязательные тесты E2 (предрегистрация §9).
Экспериментальные прогоны групп здесь не выполняются.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import tokenize
from pathlib import Path

import numpy as np
import pytest

import arbiters_e2
import curiosity_bs
import mechanisms_e2
import ratio_e2
from arbiters_e2 import (MODE_DOM, MODE_MATCH, MODE_OVR, MODE_VETO, arbitrate)
from belief import Belief
from curiosity_bs import CuriosityBS, bayesian_surprise_kl
from environment_e2 import STREAM_EVPI, make_scenario, make_scenario_alt_truth, stream_rng
from evpi_e2 import evpi_mc_e2
from evsi import evsi_analytic, ig
from mechanisms_e2 import CBSD1, COLD1
from policies_e2 import IGFastE2, RandomE2, SweepThenRandomE2
from probes import SURVEY_FAST, build_catalog
from ratio_e2 import compute_ratio_table
from runner_e2 import GROUPS_E2, run_episode_e2
from stats_e2 import (holm, paired_bootstrap_ci, paired_t_ci, spearman_safe,
                      tost_paired, wilcoxon_paired)

N = 30


def fast_catalog(n=N):
    return [p for p in build_catalog(n)
            if p.probe_type.probe_type_id == SURVEY_FAST.probe_type_id]


def _code_only(path: Path) -> str:
    """Токены кода без строк и комментариев: докстринг не нарушает запрет."""
    out = []
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                continue
            out.append(tok.string)
    return " ".join(out)


# ---------- 9.1 математические ----------

@pytest.mark.parametrize("s0", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("r", [0.25, 1.0, 4.0])
def test_bs_kl_matches_ig_kappa_z(s0, r):
    rng = np.random.default_rng(1234)
    for _ in range(200):
        mu0 = float(rng.normal())
        o = float(rng.normal(mu0, math.sqrt(s0 + r)))
        s1 = s0 * r / (s0 + r)
        mu1 = (mu0 / s0 + o / r) / (1.0 / s0 + 1.0 / r)
        kl = bayesian_surprise_kl(mu0, s0, mu1, s1)
        IG = 0.5 * math.log1p(s0 / r)
        kappa = s0 / (s0 + r)
        z = (o - mu0) / math.sqrt(s0 + r)
        assert kl == pytest.approx(IG + 0.5 * kappa * (z * z - 1.0), abs=1e-9)


@pytest.mark.parametrize("s0", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("r", [0.25, 1.0, 4.0])
def test_bs_expectation_equals_ig(s0, r):
    n = 20000
    rng = np.random.default_rng(20261126 + int(100 * s0) + int(10 * r))
    mu0 = 0.31
    obs = rng.normal(mu0, math.sqrt(s0 + r), size=n)
    s1 = s0 * r / (s0 + r)
    vals = np.array([
        bayesian_surprise_kl(mu0, s0,
                             (mu0 / s0 + float(o) / r) / (1.0 / s0 + 1.0 / r), s1)
        for o in obs])
    IG = 0.5 * math.log1p(s0 / r)
    se = float(vals.std(ddof=1) / math.sqrt(n))
    z = (float(vals.mean()) - IG) / se
    assert abs(z) <= 3.0, f"s0={s0} r={r} z={z:.3f}"


@pytest.mark.parametrize("s", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("k", [0, 1, 2, 5])
def test_posterior_analytic(s, k):
    r = 1.0
    b = Belief(mu=np.array([0.0]), sigma2=np.array([s]), known=np.array([False]))
    for _ in range(k):
        b = b.update(0, 0.0, r)
    assert float(b.sigma2[0]) == pytest.approx(s * r / (r + k * s), rel=1e-12)


def test_bs_rejects_nonpositive_variance():
    with pytest.raises(ValueError):
        bayesian_surprise_kl(0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        bayesian_surprise_kl(0.0, 1.0, 0.0, 0.0)


def three_sided_belief():
    return Belief(mu=np.array([-2.0, 0.9, 1.0]),
                  sigma2=np.array([2.0 / 3.0, 0.5, 1.0 / 3.0]),
                  known=np.zeros(3, dtype=bool))


def test_evsi_reference_and_stability():
    b = three_sided_belief()
    vals = {p.parcel_id: evsi_analytic(b, p) for p in fast_catalog(3)}
    assert vals[1] == pytest.approx(0.118, abs=1e-3)
    assert vals[2] == pytest.approx(0.072, abs=1e-3)
    assert max(vals, key=vals.get) == 1
    for v in vals.values():
        assert np.isfinite(v) and v >= 0.0


def test_ratio_picks_argmax():
    sc = make_scenario(52000)
    t = compute_ratio_table(sc.belief0, fast_catalog(), stream_rng(52000, 5))
    assert t.best_evsi == pytest.approx(float(t.evsi.max()), abs=1e-12)
    assert t.evsi_of(t.best) == pytest.approx(t.best_evsi, abs=1e-12)


# ---------- 9.3 трёхсторонний пример ----------

def test_three_sided_ig_prefers_A():
    b = three_sided_belief()
    ig_vals = {p.parcel_id: ig(b, p) for p in fast_catalog(3)}
    assert ig_vals[0] == pytest.approx(0.255, abs=1e-3)
    assert max(ig_vals, key=ig_vals.get) == 0


def test_three_sided_col_prefers_C():
    """Измерения строго A -> B -> C; I(0)=1; все A и все I пересчитываются."""
    s_prior = np.array([2.0, 1.0, 0.5])
    m = COLD1(3, s_prior)
    mus = [np.array([0.0, 0.0, 0.0]),
           np.array([-2.0, 0.0, 0.0]),
           np.array([-2.0, 0.9, 0.0]),
           np.array([-2.0, 0.9, 1.0])]
    s2s = [np.array([2.0, 1.0, 0.5]),
           np.array([2.0 / 3, 1.0, 0.5]),
           np.array([2.0 / 3, 0.5, 0.5]),
           np.array([2.0 / 3, 0.5, 1.0 / 3])]
    for k in (0, 1, 2):
        before = Belief(mu=mus[k].copy(), sigma2=s2s[k].copy(),
                        known=np.zeros(3, dtype=bool))
        after = Belief(mu=mus[k + 1].copy(), sigma2=s2s[k + 1].copy(),
                       known=np.zeros(3, dtype=bool))
        m.after_measurement(k, before, after, 0.0, 1.0)

    st = m._c.get_state()
    assert st.visit_count.tolist() == [1, 1, 1]
    # A после одного измерения каждого: 1/6, 1/4, 1/3
    expected_A = np.array([1.0 / 6, 1.0 / 4, 1.0 / 3])
    u = s2s[3] / s_prior
    np.testing.assert_allclose(u * 0.5, expected_A, atol=1e-12)
    # фактическое ранжирование I
    interest = m.interest_vector()
    assert int(np.argmax(interest)) == 2
    prop = m.propose(np.zeros(3, dtype=bool), np.random.default_rng(7))
    assert prop.first == 2
    assert prop.ranking == [2, 1, 0]


def test_three_sided_arbiters():
    b = three_sided_belief()
    table = compute_ratio_table(b, fast_catalog(3), np.random.default_rng(0))
    first, ranking = 2, [2, 1, 0]
    d_dom = arbitrate(MODE_DOM, first, ranking, table)
    assert d_dom.chosen == 1 and d_dom.replaced
    d_ovr = arbitrate(MODE_OVR, first, ranking, table, rho=0.5)
    assert d_ovr.chosen == 2 and not d_ovr.replaced
    d_veto = arbitrate(MODE_VETO, first, ranking, table)
    assert d_veto.chosen == 2 and not d_veto.replaced


# ---------- 9.2 поведенческие ----------

def _col_trace(sc, steps=35):
    m = COLD1(N, sc.s_prior)
    rng = stream_rng(sc.seed, 3)
    belief = sc.belief0.copy()
    by = {p.parcel_id: p for p in fast_catalog()}
    reps: dict[int, int] = {}
    seq, interests = [], []
    for _ in range(steps):
        prop = m.propose(belief.known, rng)
        seq.append(prop.first)
        interests.append(m.interest_vector())
        p = by[prop.first]
        ri = reps.get(prop.first, 0)
        reps[prop.first] = ri + 1
        obs = sc.env.observe(p, ri)
        before = belief
        belief = belief.update(prop.first, obs.value, p.noise_var)
        m.after_measurement(prop.first, before, belief, obs.value, p.noise_var)
    return seq, interests


def test_c_ol_open_loop():
    base = _col_trace(make_scenario(52001))
    for variant in (1, 2, 3):
        alt = _col_trace(make_scenario_alt_truth(52001, variant))
        assert alt[0] == base[0]
        for a, b_ in zip(alt[1], base[1]):
            np.testing.assert_allclose(a, b_, rtol=0, atol=1e-12)


def test_c_bs_closed_loop_constructed():
    s_prior = np.array([1.0, 1.0])
    c1, c2 = CuriosityBS(2, s_prior), CuriosityBS(2, s_prior)
    bs_small = c1.after_measurement(0, 0.0, 1.0, 0.01, 0.5)
    bs_large = c2.after_measurement(0, 0.0, 1.0, 3.0, 0.5)
    assert bs_large > bs_small
    s1, s2 = c1.get_state(), c2.get_state()
    assert s1.attract[0] != s2.attract[0]
    assert s1.interest[0] != s2.interest[0]
    assert c1.ranking(np.random.default_rng(0))[0] != \
           c2.ranking(np.random.default_rng(0))[0]


def test_c_bs_sticky_attract():
    c = CuriosityBS(3, np.array([1.0, 1.0, 1.0]))
    c.after_measurement(0, 0.0, 1.0, 0.5, 0.5)
    a0 = c.get_state().attract[0]
    c.after_measurement(1, 0.0, 1.0, -0.2, 0.5)
    c.after_measurement(2, 0.0, 1.0, 0.1, 0.5)
    assert c.get_state().attract[0] == pytest.approx(a0, abs=0.0)


def test_c_bs_updates_all_interest():
    c = CuriosityBS(3, np.array([0.5, 1.0, 2.0]))
    before = c.get_state().interest
    c.after_measurement(1, 0.0, 1.0, 1.0, 0.5)
    after = c.get_state().interest
    expected = 0.7 * before + 0.3 * c.get_state().attract
    np.testing.assert_allclose(after, expected, rtol=0, atol=1e-12)


def test_c_bs_state_updates_on_every_executed_action():
    """В том числе на шагах, где действие выбрал Ratio."""
    sc = make_scenario(52002)
    res = run_episode_e2("C_BS_DOM", sc, fast_catalog(), budget=20.0)
    counts = np.zeros(N, dtype=int)
    for e in res.log:
        counts[e["parcel_id"]] += 1
    assert res.metrics["visits"] == counts.tolist()
    assert res.metrics["n_probes"] == 20
    ratio_steps = [e for e in res.log if e["chose_ratio"]]
    assert ratio_steps, "ожидался хотя бы один выбор Ratio при бюджете 20"
    for e in ratio_steps:
        assert e["parcel_id"] == e["ratio_best"]


def test_c_bs_adapter_ignores_observation_value():
    """BS зависит только от (mu0,s0,mu1,s1), не от переданного o."""
    sc = make_scenario(52003)
    b0 = sc.belief0.copy()
    b1 = b0.update(0, 1.0, 1.0)
    m1, m2 = CBSD1(N, sc.s_prior), CBSD1(N, sc.s_prior)
    m1.after_measurement(0, b0, b1, 1.0, 1.0)
    m2.after_measurement(0, b0, b1, 999.0, 1.0)
    np.testing.assert_allclose(m1.interest_vector(), m2.interest_vector(),
                               rtol=0, atol=0.0)


def test_match_replacement_semantics():
    b = three_sided_belief()
    table = compute_ratio_table(b, fast_catalog(3), np.random.default_rng(0))
    first = table.best
    d = arbitrate(MODE_MATCH, first, [first, 1, 0], table, match_u=0.0, q=1.0)
    assert d.chose_ratio is True
    assert d.replaced is False
    assert d.chosen == first


def test_match_requires_u_and_q():
    b = three_sided_belief()
    table = compute_ratio_table(b, fast_catalog(3), np.random.default_rng(0))
    with pytest.raises(ValueError):
        arbitrate(MODE_MATCH, 2, [2, 1, 0], table)


def test_arbiters_work_with_evsi_disabled(monkeypatch):
    import evsi as evsi_module

    def boom(*a, **k):
        raise AssertionError("арбитр не должен вычислять EVSI")

    monkeypatch.setattr(evsi_module, "evsi_analytic", boom)
    monkeypatch.setattr(evsi_module, "evpi_mc", boom)
    b = three_sided_belief()
    table = compute_ratio_table(b, fast_catalog(3), np.random.default_rng(0))
    for mode in (MODE_DOM, MODE_OVR, MODE_VETO):
        arbitrate(mode, 2, [2, 1, 0], table)
    arbitrate(MODE_MATCH, 2, [2, 1, 0], table, match_u=0.1, q=0.5)


def test_no_forbidden_access_in_modules():
    forbidden_imports = {"evsi", "environment", "curiosity", "curiosity_bs",
                         "mechanisms_e2", "belief"}
    src = Path(arbiters_e2.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in forbidden_imports
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden_imports

    checks = {
        arbiters_e2: ("interest", "visit_count", "last_surprise", "last_bs",
                      "theta_true", "evsi_analytic", "evpi_mc"),
        curiosity_bs: ("theta_true", "evsi_analytic", "evpi_mc", "budget"),
        ratio_e2: ("interest", "visit_count", "last_surprise", "last_bs",
                   "theta_true"),
        mechanisms_e2: ("theta_true", "evsi_analytic", "evpi_mc", "budget"),
    }
    for mod, tokens in checks.items():
        code = _code_only(Path(mod.__file__))
        for tok in tokens:
            assert re.search(rf"\b{tok}\b", code) is None, \
                f"{mod.__name__}: запрещённый токен {tok}"


def test_determinism_same_seed():
    sc = make_scenario(52003)
    cat = fast_catalog()
    a = run_episode_e2("C_BS_OVR", sc, cat, budget=10.0)
    b = run_episode_e2("C_BS_OVR", sc, cat, budget=10.0)
    assert a.log == b.log
    assert a.metrics["regret"] == b.metrics["regret"]


def test_crn_independent_of_action_order():
    sc = make_scenario(52004)
    p = fast_catalog()[3]
    o1 = sc.env.observe(p, 0)
    sc2 = make_scenario(52004)
    for other in (0, 1, 2, 5):
        sc2.env.observe(fast_catalog()[other], 0)
    o2 = sc2.env.observe(p, 0)
    assert o1.value == o2.value


def test_repeats_allowed_from_first_step():
    sc = make_scenario(52000)
    cat = fast_catalog()
    assert len([p for p in cat if p.cost <= 45.0]) == N
    res = run_episode_e2("B_RANDOM", sc, cat, budget=45.0, policy=RandomE2())
    assert res.metrics["n_probes"] == 45
    assert res.metrics["budget_used"] == pytest.approx(45.0)


def test_sweep_policy_visits_from_precision():
    sc = make_scenario(52000)
    pol = SweepThenRandomE2(sc.s_prior)
    b = sc.belief0.copy()
    np.testing.assert_array_equal(pol.visit_counts(b), np.zeros(N, dtype=int))
    b = b.update(7, 0.0, 1.0)
    b = b.update(7, 0.0, 1.0)
    v = pol.visit_counts(b)
    assert v[7] == 2
    assert v.sum() == 2


def test_evpi_shared_samples():
    sc = make_scenario(52000)
    b = sc.belief0
    r1 = stream_rng(52000, STREAM_EVPI)
    r2 = stream_rng(52000, STREAM_EVPI)
    a = evpi_mc_e2(b, r1, n_samples=4000)
    c = evpi_mc_e2(b, r2, n_samples=4000)
    assert a == c
    assert a["n_samples"] == 4000 and a["se"] > 0.0


def test_evpi_default_n_in_runner():
    sc = make_scenario(52000)
    res = run_episode_e2("C_OL_SOLO", sc, fast_catalog(), budget=5.0)
    assert res.metrics["evpi"]["n_samples"] == 40000
    assert np.isfinite(res.metrics["evpi"]["mean"])


def test_group_count_sixteen():
    assert len(GROUPS_E2) == 16


def test_metrics_no_nan_inf():
    sc = make_scenario(52001)
    res = run_episode_e2("C_BS_OVR", sc, fast_catalog(), budget=12.0)
    for k, v in res.metrics.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"{k} не конечен"


# ---------- 9.5 статистические ----------

def test_holm_registered_vector():
    p = [0.003984, 0.7399, 0.003984, 0.7399, 0.003984]
    _, adj = holm(p)
    np.testing.assert_allclose(adj, np.array([0.0199, 1.0, 0.0199, 1.0, 0.0199]),
                               atol=1e-4)
    assert np.all(adj >= np.asarray(p) - 1e-12)


def test_holm_adjusted_not_below_raw():
    rng = np.random.default_rng(5)
    for _ in range(50):
        p = rng.uniform(0, 1, size=16)
        _, adj = holm(p)
        assert np.all(adj >= p - 1e-12)
        assert np.all(adj <= 1.0)


def test_tost_synthetic():
    rng = np.random.default_rng(11)
    assert tost_paired(rng.normal(0.0, 0.002, 800), delta=0.01)["equivalent"] is True
    assert tost_paired(rng.normal(0.05, 0.002, 800), delta=0.01)["equivalent"] is False


def test_paired_ci_synthetic():
    rng = np.random.default_rng(12)
    d = rng.normal(0.3, 0.1, size=400)
    ci = paired_t_ci(d)
    assert ci["ci_low"] < 0.3 < ci["ci_high"] and ci["n"] == 400
    bs = paired_bootstrap_ci(d, np.random.default_rng(13), n_boot=1000)
    assert bs["ci_low"] < 0.3 < bs["ci_high"]


def test_wilcoxon_paired():
    rng = np.random.default_rng(14)
    assert wilcoxon_paired(rng.normal(0.5, 0.1, size=100)) < 0.01
    assert wilcoxon_paired(np.zeros(50)) == 1.0


def test_spearman_handles_constant():
    assert spearman_safe(np.ones(10), np.arange(10)) is None
    assert spearman_safe(np.arange(10), np.arange(10)) == pytest.approx(1.0)


def test_pairing_same_scenario_across_groups():
    cat = fast_catalog()
    for seed in (52000, 52001):
        sc = make_scenario(seed)
        a = run_episode_e2("C_OL_DOM", sc, cat, budget=5.0)
        b = run_episode_e2("C_BS_DOM", sc, cat, budget=5.0)
        assert a.seed == b.seed
        assert a.metrics["n_probes"] == b.metrics["n_probes"] == 5