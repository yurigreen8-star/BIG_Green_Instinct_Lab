"""
test_probes.py — тесты каталога зондов.
"""

import pytest

from probes import (
    PROBE_TYPES,
    SURVEY_FAST,
    SURVEY_PRECISE,
    SURVEY_STANDARD,
    Probe,
    ProbeType,
    build_catalog,
)


def test_probe_types_values():
    """Значения параметров из постановки О2."""
    assert SURVEY_FAST.cost == 1.0
    assert SURVEY_FAST.noise_var == 1.00
    assert SURVEY_STANDARD.cost == 3.0
    assert SURVEY_STANDARD.noise_var == 0.25
    assert SURVEY_PRECISE.cost == 8.0
    assert SURVEY_PRECISE.noise_var == 0.04


def test_probe_type_validation():
    """Некорректные параметры ProbeType."""
    with pytest.raises(ValueError):
        ProbeType(name="bad", cost=-1, noise_var=1)
    with pytest.raises(ValueError):
        ProbeType(name="bad", cost=1, noise_var=-1)


def test_probe_validation():
    """Некорректные параметры Probe."""
    with pytest.raises(ValueError):
        Probe(parcel_id=-1, probe_type=SURVEY_FAST)


def test_catalog_size():
    """Каталог 30 участков × 3 зонда = 90."""
    catalog = build_catalog(30)
    assert len(catalog) == 90


def test_catalog_total_cost():
    """Совокупная стоимость каталога = 360."""
    catalog = build_catalog(30)
    total = sum(p.cost for p in catalog)
    assert total == 360.0


def test_catalog_contains_all_types():
    """Каждый участок имеет все три типа зондов."""
    catalog = build_catalog(30)
    for parcel_id in range(30):
        parcel_probes = [p for p in catalog if p.parcel_id == parcel_id]
        assert len(parcel_probes) == 3
        names = {p.probe_type.name for p in parcel_probes}
        assert names == {
            "survey_fast",
            "survey_standard",
            "survey_precise",
        }


def test_build_catalog_invalid():
    with pytest.raises(ValueError):
        build_catalog(0)
    with pytest.raises(ValueError):
        build_catalog(-5)


def test_probe_immutability():
    """Probe — frozen dataclass."""
    probe = Probe(parcel_id=0, probe_type=SURVEY_FAST)
    with pytest.raises(AttributeError):
        probe.parcel_id = 1  # type: ignore