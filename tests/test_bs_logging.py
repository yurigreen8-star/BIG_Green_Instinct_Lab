"""Шаг 1. BS и behavioral novelty в логе пересчитываются независимо."""

import math

import numpy as np
import pytest

from belief import Belief
from coordinator import Coordinator
from curiosity import Curiosity
from environment import Environment
from messages import MODE_COMBINED, MODE_RATIO_SOLO
from probes import SURVEY_FAST, SURVEY_PRECISE, SURVEY_STANDARD, build_catalog
from ratio import Ratio
from runner_v3 import run_episode_v3

N = 30
NOISE = {
    SURVEY_FAST.probe_type_id: SURVEY_FAST.noise_var,
    SURVEY_STANDARD.probe_type_id: SURVEY_STANDARD.noise_var,
    SURVEY_PRECISE.probe_type_id: SURVEY_PRECISE.noise_var,
}


def _run(mode, seed, budget=30.0):
    env = Environment.create(N, seed=seed)
    return run_episode_v3(
        mode=mode,
        curiosity=Curiosity(N, lam=0.3, seed=seed),
        ratio=Ratio(), coordinator=Coordinator(),
        belief0=Belief(mu=np.zeros(N), sigma2=np.ones(N), known=np.zeros(N, bool)),
        catalog=build_catalog(N), budget=budget, env=env, seed=seed,
        experiment_mode="A",
    )


@pytest.mark.parametrize("mode", [MODE_COMBINED, MODE_RATIO_SOLO])
def test_bs_logged_correctly(mode):
    """Сверка прямой KL из лога с замкнутым тождеством, допуск 1e-9."""
    for seed in range(5):
        res = _run(mode, seed)
        for e in res.log:
            if e.get("event") is not None:
                continue
            k = e["parcel_id"]
            a = e["sigma2_before"][k]
            r = NOISE[e["probe_type_id"]]
            mu = e["mu_before"][k]
            o = e["observation"]
            z = (o - mu) / math.sqrt(a + r)
            closed = 0.5 * math.log1p(a / r) + a / (2.0 * (a + r)) * (z * z - 1.0)
            assert e["bs_executed"] == pytest.approx(closed, abs=1e-9)


def test_behavioral_novelty_logged_correctly():
    res = _run(MODE_RATIO_SOLO, 3)
    for e in res.log:
        if e.get("event") is not None:
            continue
        rep = e["rep_index"]
        expect = math.log(rep + 2) - math.log(rep + 1)
        assert e["behavioral_novelty_executed"] == pytest.approx(expect, abs=1e-12)


def test_novelty_equals_two_ig_when_fast_only():
    """В FAST-only режиме B = 2*IG; при смешанных типах — нет."""
    res = _run(MODE_COMBINED, 5)
    steps = [e for e in res.log if e.get("event") is None]
    assert all(e["probe_type_id"] == SURVEY_FAST.probe_type_id for e in steps)
    for e in steps:
        assert e["behavioral_novelty_executed"] == pytest.approx(
            2.0 * e["ig_executed"], abs=1e-12
        )


def test_logging_did_not_change_actions():
    """Диагностика не влияет на выбор: последовательность parcel_id прежняя."""
    a = _run(MODE_COMBINED, 11)
    b = _run(MODE_COMBINED, 11)
    assert [e["parcel_id"] for e in a.log if e.get("event") is None] == \
           [e["parcel_id"] for e in b.log if e.get("event") is None]
    assert a.regret == pytest.approx(b.regret, abs=1e-12)