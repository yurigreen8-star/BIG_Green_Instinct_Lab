"""A4, B5, структурная верификация контра."""

import pytest

import evsi as evsi_module
import ratio as ratio_module
from coordinator import Coordinator
from messages import (
    MODE_COMBINED,
    MODE_COMBINED_IGNORE,
    MODE_CURIOSITY_SOLO,
    MODE_RATIO_SOLO,
    CuriosityProposal,
    RatioProposal,
)
from probes import SURVEY_FAST, SURVEY_STANDARD, Probe


def _cp(i=3, interest=1.0):
    return CuriosityProposal(parcel_id=i, interest=interest)


def _rp(probe, evsi, is_counter=True, t_id=3, t_cost=1.0, t_evsi=0.20, cost=None):
    return RatioProposal(
        probe=probe,
        cost=float(probe.cost if cost is None else cost),
        evsi=float(evsi),
        is_counter=is_counter,
        target_parcel_id=t_id,
        target_cost=t_cost,
        target_evsi=t_evsi,
    )


def test_A4_coordinator_works_with_evsi_disabled(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("Coordinator не должен вызывать EVSI")

    monkeypatch.setattr(evsi_module, "evsi_analytic", boom)
    monkeypatch.setattr(evsi_module, "evpi_mc", boom)
    monkeypatch.setattr(ratio_module, "evsi_analytic", boom)

    rp = _rp(Probe(7, SURVEY_FAST), 0.30)
    d = Coordinator().coordinate(_cp(3), rp, MODE_COMBINED, 30.0)
    assert d.reason == "ratio_counter"
    assert d.verified_dominance is True
    assert d.action.parcel_id == 7


def test_reject_target_parcel_mismatch():
    """Контр заявлен на другой участок, чем предложило Любопытство."""
    rp = _rp(Probe(7, SURVEY_FAST), 0.30, t_id=99)
    d = Coordinator().coordinate(_cp(3), rp, MODE_COMBINED, 30.0)
    assert d.action == Probe(3, SURVEY_FAST)
    assert d.verified_dominance is False
    assert d.reason == "curiosity"


def test_reject_falsified_counter_cost():
    """Заявленная cost не совпадает с фактической стоимостью зонда."""
    rp = _rp(Probe(7, SURVEY_STANDARD), 0.30, cost=1.0)  # реально 3.0
    d = Coordinator().coordinate(_cp(3), rp, MODE_COMBINED, 30.0)
    assert d.action == Probe(3, SURVEY_FAST)
    assert d.verified_dominance is False


def test_reject_falsified_target_cost():
    """Заявленная стоимость цели не равна стоимости FAST."""
    rp = _rp(Probe(7, SURVEY_FAST), 0.30, t_cost=8.0)
    d = Coordinator().coordinate(_cp(3), rp, MODE_COMBINED, 30.0)
    assert d.verified_dominance is False
    assert d.action == Probe(3, SURVEY_FAST)


def test_reject_when_no_curiosity_proposal():
    rp = _rp(Probe(7, SURVEY_FAST), 0.30)
    d = Coordinator().coordinate(None, rp, MODE_COMBINED, 30.0)
    assert d.action is None
    assert d.reason == "stop_all_known"


def test_reject_unverified_arithmetic():
    rp = _rp(Probe(7, SURVEY_FAST), 0.10)  # evsi ниже target
    d = Coordinator().coordinate(_cp(3), rp, MODE_COMBINED, 30.0)
    assert d.verified_dominance is False
    assert d.action == Probe(3, SURVEY_FAST)


def test_curiosity_probe_is_fast():
    d = Coordinator().coordinate(_cp(5), None, MODE_CURIOSITY_SOLO, 30.0)
    assert d.action == Probe(5, SURVEY_FAST)


def test_ratio_solo_uses_base():
    rp = _rp(Probe(9, SURVEY_STANDARD), 0.5, t_evsi=0.9)
    d = Coordinator().coordinate(_cp(3), rp, MODE_RATIO_SOLO, 30.0)
    assert d.action == rp.probe
    assert d.reason == "ratio_solo"


def test_B5_ignore_equals_combined_without_counters():
    c = Coordinator()
    rp = _rp(Probe(7, SURVEY_STANDARD), 0.05, is_counter=False)
    for seed in range(5):
        cp = _cp(seed % 30)
        d1 = c.coordinate(cp, rp, MODE_COMBINED, 30.0)
        d2 = c.coordinate(cp, rp, MODE_COMBINED_IGNORE, 30.0)
        assert d1.action == d2.action
        assert d1.reason == d2.reason


def test_ignore_never_executes_ratio():
    rp = _rp(Probe(7, SURVEY_FAST), 0.90)
    d = Coordinator().coordinate(_cp(3), rp, MODE_COMBINED_IGNORE, 30.0)
    assert d.action == Probe(3, SURVEY_FAST)