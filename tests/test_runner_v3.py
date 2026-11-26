"""E1 (5 сидов), интеграция, инварианты исполнения, residual-тождество."""

import numpy as np
import pytest

from belief import Belief
from coordinator import Coordinator
from curiosity import Curiosity
from environment import Environment
from messages import (
    MODE_COMBINED,
    MODE_COMBINED_IGNORE,
    MODE_CURIOSITY_SOLO,
    MODE_RATIO_SOLO,
    predictive_surprise,
)
from policies.b_evsi import EVSICostPolicy
from policies.b_random_without_replacement import RandomWithoutReplacementPolicy
from probes import build_catalog
from ratio import Ratio
from runner import run_episode
from runner_v3 import run_episode_v3

N = 30
BUDGET = 30.0


def _b0(n=N):
    return Belief(mu=np.zeros(n), sigma2=np.ones(n), known=np.zeros(n, bool))


def _run(mode, seed, experiment_mode="A", budget=BUDGET):
    env = Environment.create(N, seed=seed)
    return run_episode_v3(
        mode=mode,
        curiosity=Curiosity(N, lam=0.3, seed=seed),
        ratio=Ratio(),
        coordinator=Coordinator(),
        belief0=_b0(),
        catalog=build_catalog(N),
        budget=budget,
        env=env,
        seed=seed,
        experiment_mode=experiment_mode,
    )


def _steps(res):
    return [e for e in res.log if e.get("event") is None]


@pytest.mark.parametrize("mode", [MODE_CURIOSITY_SOLO, MODE_RATIO_SOLO,
                                  MODE_COMBINED, MODE_COMBINED_IGNORE])
def test_E1_all_modes_complete(mode):
    for seed in range(5):
        res = _run(mode, seed)
        assert res.budget_used <= BUDGET + 1e-9
        assert res.budget_remaining >= -1e-9
        assert res.log[-1]["event"] == "stop"
        assert res.n_rejected_no_budget == 0  # ветка недостижима в E0


def test_curiosity_solo_is_fast_sweep():
    """Флаг 2: budget = n, принудительный FAST -> обход без повторов."""
    for seed in range(3):
        res = _run(MODE_CURIOSITY_SOLO, seed)
        s = _steps(res)
        assert res.n_probes == 30
        assert res.n_unique_parcels == 30
        assert res.n_repeats_total == 0
        assert all(e["probe_type_id"] == "survey_fast" for e in s)


def test_sweep_control_matches_coverage():
    """Контроль B_random_without_replacement даёт то же покрытие."""
    for seed in range(3):
        env = Environment.create(N, seed=seed)
        ctrl = run_episode(
            RandomWithoutReplacementPolicy(), _b0(), build_catalog(N),
            BUDGET, env, seed=seed, mode="A",
        )
        res = _run(MODE_CURIOSITY_SOLO, seed)
        assert ctrl.n_probes == res.n_probes == 30
        ctrl_unique = len({e["parcel_id"] for e in ctrl.log if "observation" in e})
        assert ctrl_unique == res.n_unique_parcels == 30


def test_B4_integration_ratio_solo_equals_B_EVSI_cost():
    for seed in range(5):
        env_new = Environment.create(N, seed=seed)
        res = _run(MODE_RATIO_SOLO, seed)
        env_old = Environment.create(N, seed=seed)
        old = run_episode(
            EVSICostPolicy(use_mc=False), _b0(), build_catalog(N),
            BUDGET, env_old, seed=seed, mode="A",
        )
        assert res.n_probes == old.n_probes
        assert res.budget_used == pytest.approx(old.budget_used)
        assert res.regret == pytest.approx(old.regret, abs=1e-12)
        a_new = [e["probe_type_id"] + "@" + str(e["parcel_id"]) for e in _steps(res)]
        a_old = [e["probe_name"] for e in old.log if "observation" in e]
        assert a_new == a_old


def test_ignore_equals_curiosity_solo():
    for seed in range(3):
        a = _run(MODE_COMBINED_IGNORE, seed)
        b = _run(MODE_CURIOSITY_SOLO, seed)
        assert [e["parcel_id"] for e in _steps(a)] == [e["parcel_id"] for e in _steps(b)]
        assert a.regret == pytest.approx(b.regret, abs=1e-12)


def test_executed_parcel_drives_curiosity_update():
    """Поправка 5: visit_count растёт у исполненного участка, не у предложенного."""
    res = _run(MODE_COMBINED, 100)
    for e in _steps(res):
        cp = e["curiosity_proposal"]
        k = e["parcel_id"]
        vc = e["curiosity_state_after"]["visit_count"]
        assert vc[k] >= 1
        if e["coordinator_decision"]["reason"] == "ratio_counter" and cp is not None:
            if cp["parcel_id"] != k:
                prev = e["curiosity_state_after"]["visit_count"][cp["parcel_id"]]
                assert prev == sum(
                    1 for p in _steps(res)[: e["step"]] if p["parcel_id"] == cp["parcel_id"]
                )


def test_surprise_computed_before_update():
    res = _run(MODE_COMBINED, 7)
    for e in _steps(res):
        k = e["parcel_id"]
        s2 = 1.0 if e["probe_type_id"] == "survey_fast" else (
            0.25 if e["probe_type_id"] == "survey_standard" else 0.04
        )
        expected = predictive_surprise(
            e["observation"], e["mu_before"][k], e["sigma2_before"][k], s2
        )
        assert e["surprise"] == pytest.approx(expected, abs=1e-12)
        assert e["curiosity_state_after"]["last_surprise"][k] == pytest.approx(expected)


def test_telescoping_and_residual_identity():
    for mode in (MODE_CURIOSITY_SOLO, MODE_RATIO_SOLO, MODE_COMBINED):
        res = _run(mode, 42)
        s = _steps(res)
        dv = sum(e["delta_posterior_value"] for e in s)
        ev = sum(e["evsi_analytic_executed"] for e in s)
        v_final = float(np.max(res.log[-1]["final_mu"]))
        assert dv == pytest.approx(v_final - 0.0, abs=1e-9)
        l1 = (ev - dv) / len(s)
        l3 = dv - ev
        assert l3 == pytest.approx(-len(s) * l1, abs=1e-9)


def test_determinism():
    a = _run(MODE_COMBINED, 11)
    b = _run(MODE_COMBINED, 11)
    assert a.log == b.log


def test_log_required_fields():
    res = _run(MODE_COMBINED, 3)
    required = {
        "step", "experiment_mode", "budget_mode", "curiosity_proposal",
        "ratio_proposal", "coordinator_decision", "observation",
        "mu_before", "sigma2_before", "mu_after", "sigma2_after",
        "delta_posterior_value", "evsi_analytic_executed", "ig_executed",
        "surprise", "curiosity_state_after", "budget_remaining",
        "budget_used_cumulative", "rep_index",
    }
    for e in _steps(res):
        assert required.issubset(e.keys())