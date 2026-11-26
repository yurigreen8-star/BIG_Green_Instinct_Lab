"""
tests/test_e2_groups.py — прогон ВСЕХ 16 групп E2 на малых бюджетах.

Закрывает дыру: test_e2.py не запускал RATIO_SOLO и не покрывал все группы,
из-за чего ошибка диспетчера режимов не была поймана до smoke.
"""

from __future__ import annotations

import numpy as np
import pytest

from environment_e2 import make_scenario
from policies_e2 import IGFastE2, RandomE2, SweepThenRandomE2
from probes import SURVEY_FAST, build_catalog
from runner_e2 import GROUPS_E2, run_episode_e2

N = 30
SMALL_BUDGETS = (3.0, 5.0)
EVPI_N_SMALL = 200


def fast_catalog():
    return [p for p in build_catalog(N)
            if p.probe_type.probe_type_id == SURVEY_FAST.probe_type_id]


def make_policy(group, sc):
    if group == "B_IG_FAST":
        return IGFastE2()
    if group == "B_RANDOM":
        return RandomE2()
    if group == "B_SWEEP_THEN_RANDOM":
        return SweepThenRandomE2(sc.s_prior)
    return None


@pytest.mark.parametrize("group", list(GROUPS_E2))
@pytest.mark.parametrize("budget", SMALL_BUDGETS)
def test_every_group_runs(group, budget):
    sc = make_scenario(52000)
    res = run_episode_e2(group, sc, fast_catalog(), budget=budget,
                         q_match=0.5, policy=make_policy(group, sc),
                         evpi_n=EVPI_N_SMALL)
    m = res.metrics
    assert m["n_probes"] == int(budget)
    assert m["budget_used"] == pytest.approx(budget)
    assert len(res.log) == int(budget)
    assert m["evpi"]["n_samples"] == EVPI_N_SMALL
    for k, v in m.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"{group}/{k} не конечен"


@pytest.mark.parametrize("group", list(GROUPS_E2))
def test_unique_first_n_uses_executed_actions(group):
    """Метрика считается по исполненным действиям, не по предложениям Curiosity."""
    sc = make_scenario(52001)
    res = run_episode_e2(group, sc, fast_catalog(), budget=5.0,
                         q_match=0.5, policy=make_policy(group, sc),
                         evpi_n=EVPI_N_SMALL)
    executed = [e["parcel_id"] for e in res.log]
    assert res.metrics["executed_parcels"] == executed
    assert res.metrics["n_unique_first_n"] == len(set(executed[:5]))
    assert res.metrics["n_unique_first_n"] >= 1


def test_ratio_solo_executes_ratio_best_every_step():
    sc = make_scenario(52002)
    res = run_episode_e2("RATIO_SOLO", sc, fast_catalog(), budget=6.0,
                         evpi_n=EVPI_N_SMALL)
    for e in res.log:
        assert e["parcel_id"] == e["ratio_best"]
        assert e["curiosity_first"] is None
        assert e["chose_ratio"] is True
        assert e["replaced"] is False
    assert res.metrics["replacement_pct"] == 0.0
    assert res.metrics["ratio_choice_pct"] == 100.0
    assert res.metrics["ratio_action_pct"] == 100.0


@pytest.mark.parametrize("group", list(GROUPS_E2))
def test_intervention_counters_match_log(group):
    sc = make_scenario(52003)
    res = run_episode_e2(group, sc, fast_catalog(), budget=5.0,
                         q_match=0.5, policy=make_policy(group, sc),
                         evpi_n=EVPI_N_SMALL)
    m, log = res.metrics, res.log
    assert m["n_ratio_choices"] == sum(1 for e in log if e["chose_ratio"])
    assert m["n_replacements"] == sum(1 for e in log if e["replaced"])
    assert m["n_ratio_actions"] == sum(
        1 for e in log if e["parcel_id"] == e["ratio_best"])
    n = len(log)
    assert m["ratio_choice_pct"] == pytest.approx(100.0 * m["n_ratio_choices"] / n)
    assert m["replacement_pct"] == pytest.approx(100.0 * m["n_replacements"] / n)
    assert m["ratio_action_pct"] == pytest.approx(100.0 * m["n_ratio_actions"] / n)


def test_replacement_never_exceeds_ratio_choices_plus_veto():
    """Замена возможна только при выборе Ratio или при VETO-обходе ранжирования."""
    sc = make_scenario(52004)
    for group in GROUPS_E2:
        res = run_episode_e2(group, sc, fast_catalog(), budget=6.0,
                             q_match=0.5, policy=make_policy(group, sc),
                             evpi_n=EVPI_N_SMALL)
        for e in res.log:
            if e["replaced"]:
                assert e["chose_ratio"] or e["reason"] == "veto_next", \
                    f"{group}: замена без выбора Ratio и без VETO"


def test_all_groups_deterministic():
    sc = make_scenario(52000)
    cat = fast_catalog()
    for group in GROUPS_E2:
        a = run_episode_e2(group, sc, cat, budget=4.0, q_match=0.5,
                           policy=make_policy(group, sc), evpi_n=EVPI_N_SMALL)
        b = run_episode_e2(group, sc, cat, budget=4.0, q_match=0.5,
                           policy=make_policy(group, sc), evpi_n=EVPI_N_SMALL)
        assert a.log == b.log, group
        assert a.metrics["regret"] == b.metrics["regret"], group