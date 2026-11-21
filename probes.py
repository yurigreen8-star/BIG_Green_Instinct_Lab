"""
probes.py — каталог зондов v0.1.

Постановка О2 (обновлённая):
    30 участков, 3 семейства зондов на каждом.
    Повторы разрешены: 90 зондов — это 90 семейств действий.
    Полный каталог: 90 семейств, совокупная стоимость одного прохода 360.

Типы зондов:
    survey_fast:     cost=1, s^2=1.00
    survey_standard: cost=3, s^2=0.25
    survey_precise:  cost=8, s^2=0.04
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeType:
    """Тип зонда: точность и стоимость. probe_type_id — устойчивый идентификатор."""

    name: str
    cost: float
    noise_var: float
    probe_type_id: str = ""

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError(f"cost должен быть >= 0, получено {self.cost}")
        if self.noise_var < 0:
            raise ValueError(f"noise_var должен быть >= 0, получено {self.noise_var}")
        if not self.probe_type_id:
            object.__setattr__(self, "probe_type_id", self.name)


SURVEY_FAST = ProbeType(name="survey_fast", cost=1.0, noise_var=1.00)
SURVEY_STANDARD = ProbeType(name="survey_standard", cost=3.0, noise_var=0.25)
SURVEY_PRECISE = ProbeType(name="survey_precise", cost=8.0, noise_var=0.04)

PROBE_TYPES = [SURVEY_FAST, SURVEY_STANDARD, SURVEY_PRECISE]


@dataclass(frozen=True)
class Probe:
    """Действие: измерение участка данным типом зонда."""

    parcel_id: int
    probe_type: ProbeType

    def __post_init__(self) -> None:
        if self.parcel_id < 0:
            raise ValueError(f"parcel_id должен быть >= 0, получено {self.parcel_id}")

    @property
    def cost(self) -> float:
        return self.probe_type.cost

    @property
    def noise_var(self) -> float:
        return self.probe_type.noise_var

    @property
    def name(self) -> str:
        return f"{self.probe_type.name}@{self.parcel_id}"


def probe_key(probe: Probe) -> tuple[int, str]:
    """Ключ семейства действия для счётчика повторов."""
    return (probe.parcel_id, probe.probe_type.probe_type_id)


def build_catalog(n_parcels: int = 30) -> list[Probe]:
    if n_parcels <= 0:
        raise ValueError(f"n_parcels должен быть > 0, получено {n_parcels}")
    return [Probe(parcel_id=i, probe_type=pt) for i in range(n_parcels) for pt in PROBE_TYPES]


BUDGET_COMPARISON = 30      # режим А
BUDGET_STOPPING = 1000      # режим Б
EVPI_UPPER_EPSILON = 0.05
HORIZON_DEFAULT = 10000


def _verify_catalog_cost() -> None:
    total = sum(p.cost for p in build_catalog(30))
    assert total == 360.0, f"Каталог стоит {total}, ожидалось 360"


_verify_catalog_cost()