"""
probes.py — каталог зондов для v0.1.

Решение архитектора О2:
    30 участков, 3 зонда на каждом.
    Полный каталог: 90 зондов, совокупная стоимость 360.

Типы зондов:
    survey_fast:     cost=1, s^2=1.00
    survey_standard: cost=3, s^2=0.25
    survey_precise:  cost=8, s^2=0.04
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeType:
    """Тип зонда: точность и стоимость."""
    name: str
    cost: float
    noise_var: float

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError(f"cost должен быть >= 0, получено {self.cost}")
        if self.noise_var < 0:
            raise ValueError(
                f"noise_var должен быть >= 0, получено {self.noise_var}"
            )


# Каталог типов зондов (постановка О2)
SURVEY_FAST = ProbeType(name="survey_fast", cost=1.0, noise_var=1.00)
SURVEY_STANDARD = ProbeType(name="survey_standard", cost=3.0, noise_var=0.25)
SURVEY_PRECISE = ProbeType(name="survey_precise", cost=8.0, noise_var=0.04)

PROBE_TYPES = [SURVEY_FAST, SURVEY_STANDARD, SURVEY_PRECISE]


@dataclass(frozen=True)
class Probe:
    """Конкретный зонд: участок + тип измерения."""
    parcel_id: int
    probe_type: ProbeType

    def __post_init__(self) -> None:
        if self.parcel_id < 0:
            raise ValueError(
                f"parcel_id должен быть >= 0, получено {self.parcel_id}"
            )

    @property
    def cost(self) -> float:
        return self.probe_type.cost

    @property
    def noise_var(self) -> float:
        return self.probe_type.noise_var

    @property
    def name(self) -> str:
        return f"{self.probe_type.name}@{self.parcel_id}"


def build_catalog(n_parcels: int = 30) -> list[Probe]:
    """Строит полный каталог: по 3 зонда на каждый участок."""
    if n_parcels <= 0:
        raise ValueError(f"n_parcels должен быть > 0, получено {n_parcels}")
    catalog = []
    for i in range(n_parcels):
        for probe_type in PROBE_TYPES:
            catalog.append(Probe(parcel_id=i, probe_type=probe_type))
    return catalog


# Бюджеты (постановка О2)
BUDGET_COMPARISON = 30    # Режим А: сравнение политик
BUDGET_STOPPING = 360     # Режим Б: самостоятельная остановка

# Критерий остановки режима Б
EVPI_UPPER_EPSILON = 0.05


# Проверка: общая стоимость каталога должна быть 360
def _verify_catalog_cost() -> None:
    catalog = build_catalog(30)
    total = sum(p.cost for p in catalog)
    assert total == 360.0, f"Каталог стоит {total}, ожидалось 360"


_verify_catalog_cost()