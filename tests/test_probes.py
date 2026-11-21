import pytest

from probes import (
    SURVEY_FAST,
    SURVEY_PRECISE,
    SURVEY_STANDARD,
    Probe,
    ProbeType,
    build_catalog,
    probe_key,
)


def test_probe_types_values():
    assert SURVEY_FAST.cost == 1.0
    assert SURVEY_FAST.noise_var == 1.00
    assert SURVEY_STANDARD.cost == 3.0
    assert SURVEY_STANDARD.noise_var == 0.25
    assert SURVEY_PRECISE.cost == 8.0
    assert SURVEY_PRECISE.noise_var == 0.04


def test_probe_type_id_stable():
    assert SURVEY_FAST.probe_type_id == "survey_fast"
    assert SURVEY_STANDARD.probe_type_id == "survey_standard"
    assert SURVEY_PRECISE.probe_type_id == "survey_precise"


def test_probe_type_validation():
    with pytest.raises(ValueError):
        ProbeType(name="bad", cost=-1, noise_var=1)
    with pytest.raises(ValueError):
        ProbeType(name="bad", cost=1, noise_var=-1)


def test_probe_validation():
    with pytest.raises(ValueError):
        Probe(parcel_id=-1, probe_type=SURVEY_FAST)


def test_catalog_size():
    assert len(build_catalog(30)) == 90


def test_catalog_total_cost():
    assert sum(p.cost for p in build_catalog(30)) == 360.0


def test_catalog_contains_all_types():
    catalog = build_catalog(30)
    for parcel_id in range(30):
        parcel_probes = [p for p in catalog if p.parcel_id == parcel_id]
        assert len(parcel_probes) == 3
        assert {p.probe_type.name for p in parcel_probes} == {
            "survey_fast",
            "survey_standard",
            "survey_precise",
        }


def test_probe_key():
    p = Probe(parcel_id=7, probe_type=SURVEY_STANDARD)
    assert probe_key(p) == (7, "survey_standard")


def test_build_catalog_invalid():
    with pytest.raises(ValueError):
        build_catalog(0)
    with pytest.raises(ValueError):
        build_catalog(-5)


def test_probe_immutability():
    probe = Probe(parcel_id=0, probe_type=SURVEY_FAST)
    with pytest.raises(AttributeError):
        probe.parcel_id = 1  # type: ignore