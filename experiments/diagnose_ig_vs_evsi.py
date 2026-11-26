"""
Диагностическое сравнение IG и EVSI (задание архитектора, п.5).

Обе политики получают один и тот же Belief и множество допустимых пар.
Сравниваются: IG absolute, IG/cost, EVSI absolute, EVSI/cost.

Два варианта:
1. Однородные priors, затем состояния после первых измерений
2. Неоднородные priors

Это диагностический тест, не реализация самостоятельного инстинкта.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief import Belief
from evsi import evsi_analytic, ig
from probes import SURVEY_FAST, SURVEY_PRECISE, SURVEY_STANDARD, build_catalog

N = 30
RES = Path(__file__).parent.parent / "results"


def _rank(candidates, key_fn):
    scored = [(key_fn(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def _snapshot(belief, catalog):
    return {
        "mu": belief.mu.tolist(),
        "sigma2": belief.sigma2.tolist(),
        "known": belief.known.tolist(),
    }


def diagnose_homogeneous():
    """Однородные priors, затем состояния после первых измерений."""
    belief0 = Belief(mu=np.zeros(N), sigma2=np.ones(N), known=np.zeros(N, bool))
    catalog = build_catalog(N)
    
    rng = np.random.default_rng(42)
    belief = belief0.copy()
    for i in range(10):
        obs = float(rng.normal(0.0, math.sqrt(1.0 + SURVEY_FAST.noise_var)))
        belief = belief.update(i, obs, SURVEY_FAST.noise_var)
    
    available = [p for p in catalog if not belief.known[p.parcel_id]]
    
    ig_abs = _rank(available, lambda p: ig(belief, p))
    ig_cost = _rank(available, lambda p: ig(belief, p) / p.cost if p.cost > 0 else 0)
    evsi_abs = _rank(available, lambda p: evsi_analytic(belief, p))
    evsi_cost = _rank(available, lambda p: evsi_analytic(belief, p) / p.cost if p.cost > 0 else 0)
    
    return {
        "condition": "homogeneous priors, 10 observations",
        "snapshot": _snapshot(belief, catalog),
        "ranking": {
            "ig_absolute": [p.parcel_id for p in ig_abs[:5]],
            "ig_per_cost": [p.parcel_id for p in ig_cost[:5]],
            "evsi_absolute": [p.parcel_id for p in evsi_abs[:5]],
            "evsi_per_cost": [p.parcel_id for p in evsi_cost[:5]],
        },
        "note": "IG и EVSI могут давать разное ранжирование, "
                "поскольку EVSI учитывает инструментальную ценность для финального решения, "
                "а IG измеряет только информационный выигрыш.",
    }


def diagnose_heterogeneous():
    """Неоднородные priors."""
    rng = np.random.default_rng(123)
    mu = rng.normal(0.0, 0.5, N)
    sigma2 = rng.uniform(0.5, 2.0, N)
    belief = Belief(mu=mu, sigma2=sigma2, known=np.zeros(N, bool))
    catalog = build_catalog(N)
    
    available = [p for p in catalog if not belief.known[p.parcel_id]]
    
    ig_abs = _rank(available, lambda p: ig(belief, p))
    ig_cost = _rank(available, lambda p: ig(belief, p) / p.cost if p.cost > 0 else 0)
    evsi_abs = _rank(available, lambda p: evsi_analytic(belief, p))
    evsi_cost = _rank(available, lambda p: evsi_analytic(belief, p) / p.cost if p.cost > 0 else 0)
    
    return {
        "condition": "heterogeneous priors (random mu and sigma2)",
        "snapshot": _snapshot(belief, catalog),
        "ranking": {
            "ig_absolute": [p.parcel_id for p in ig_abs[:5]],
            "ig_per_cost": [p.parcel_id for p in ig_cost[:5]],
            "evsi_absolute": [p.parcel_id for p in evsi_abs[:5]],
            "evsi_per_cost": [p.parcel_id for p in evsi_cost[:5]],
        },
        "note": "При неоднородных priors различие между IG и EVSI становится более выраженным: "
                "EVSI предпочитает участки с высоким mu (лидеры), IG — участки с высокой неопределённостью.",
    }


def main():
    RES.mkdir(exist_ok=True)
    
    hom = diagnose_homogeneous()
    het = diagnose_heterogeneous()
    
    out = {
        "status": "diagnostic",
        "note": "Это диагностический тест, не реализация самостоятельного инстинкта. "
                "Показывает, что IG и EVSI могут давать разное ранжирование.",
        "homogeneous": hom,
        "heterogeneous": het,
    }
    
    (RES / "ig_vs_evsi_diagnostic.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    
    L = []
    W = L.append
    W("ДИАГНОСТИЧЕСКОЕ СРАВНЕНИЕ IG и EVSI")
    W("Статус: диагностический тест, не реализация инстинкта")
    W("")
    W("ОДНОРОДНЫЕ PRIORS (10 наблюдений):")
    W(f"  Top-5 по IG absolute:   {hom['ranking']['ig_absolute']}")
    W(f"  Top-5 по IG/cost:       {hom['ranking']['ig_per_cost']}")
    W(f"  Top-5 по EVSI absolute: {hom['ranking']['evsi_absolute']}")
    W(f"  Top-5 по EVSI/cost:     {hom['ranking']['evsi_per_cost']}")
    W("")
    W("НЕОДНОРОДНЫЕ PRIORS:")
    W(f"  Top-5 по IG absolute:   {het['ranking']['ig_absolute']}")
    W(f"  Top-5 по IG/cost:       {het['ranking']['ig_per_cost']}")
    W(f"  Top-5 по EVSI absolute: {het['ranking']['evsi_absolute']}")
    W(f"  Top-5 по EVSI/cost:     {het['ranking']['evsi_per_cost']}")
    W("")
    W("ВЫВОД: IG и EVSI могут давать разное ранжирование.")
    W("EVSI учитывает инструментальную ценность для финального решения,")
    W("IG измеряет только информационный выигрыш.")
    W("Это не доказывает превосходство одного критерия над другим,")
    W("а показывает их семантическое различие.")
    
    txt = "\n".join(L)
    (RES / "ig_vs_evsi_conclusion.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nСохранено: {RES}/ig_vs_evsi_diagnostic.json, ig_vs_evsi_conclusion.txt")


if __name__ == "__main__":
    main()