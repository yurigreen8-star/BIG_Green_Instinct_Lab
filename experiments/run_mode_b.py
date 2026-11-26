"""
experiments/run_mode_b.py
Режим Б: самостоятельная остановка по UCB95(EVPI) <= 0.05.
Сравниваем стоимость достижения порога точности.
Сохраняет полные per-step логи в raw JSON.
"""

import sys
from pathlib import Path
import json
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief import Belief
from environment import Environment
from probes import build_catalog, BUDGET_STOPPING, EVPI_UPPER_EPSILON
from runner import run_episode

from policies.b_random import RandomPolicy
from policies.b_ig import IGPolicy
from policies.b_mean_precision import MeanPrecisionPolicy
from policies.b_evsi import EVSIPolicy, EVSICostPolicy


def get_belief0(n: int) -> Belief:
    return Belief(
        mu=np.zeros(n),
        sigma2=np.ones(n),
        known=np.zeros(n, dtype=bool),
    )


def run_mode_b_experiment(n_scenarios: int = 50, n_parcels: int = 30):
    print(f"🚀 Режим Б: {n_scenarios} сценариев, бюджет {BUDGET_STOPPING}")
    print(f"   Критерий остановки: UCB95(EVPI) <= {EVPI_UPPER_EPSILON}")
    print("⏳ Запуск...\n")

    policies = {
        "B_random": RandomPolicy(),
        "B_IG": IGPolicy(),
        "B_mean_prec": MeanPrecisionPolicy(),
        "B_EVSI": EVSIPolicy(use_mc=False),
        "B_EVSI_cost": EVSICostPolicy(use_mc=False),
    }

    catalog = build_catalog(n_parcels)
    raw_data = {
        name: {
            "budget_used": [],
            "n_probes": [],
            "regret": [],
            "stop_reason": [],
            "pcs": [],
            "logs": [],
        }
        for name in policies
    }

    for seed in range(n_scenarios):
        env = Environment.create(n_parcels=n_parcels, seed=seed)
        belief0 = get_belief0(n_parcels)

        for name, policy in policies.items():
            res = run_episode(
                policy=policy,
                belief0=belief0,
                catalog=catalog,
                budget=BUDGET_STOPPING,
                env=env,
                seed=seed,
                mode="B",
                evpi_n_samples=2000,
                evpi_epsilon=EVPI_UPPER_EPSILON,
            )
            raw_data[name]["budget_used"].append(res.budget_used)
            raw_data[name]["n_probes"].append(res.n_probes)
            raw_data[name]["regret"].append(res.regret)
            raw_data[name]["stop_reason"].append(res.stop_reason)
            raw_data[name]["pcs"].append(1 if res.pcs else 0)
            raw_data[name]["logs"].append(res.log)

    # 1. Сводная статистика
    print("=" * 100)
    print(f"{'Политика':<20} | {'Ср. Бюджет':<12} | {'Мед. Бюджет':<12} | "
          f"{'Ср. Зондов':<12} | {'Ср. Regret':<12} | {'PCS (%)':<8} | "
          f"{'Остановка по EVPI'}")
    print("-" * 100)

    summary = {}
    for name in policies:
        budget = raw_data[name]["budget_used"]
        probes = raw_data[name]["n_probes"]
        regret = raw_data[name]["regret"]
        pcs = raw_data[name]["pcs"]
        stop_reasons = raw_data[name]["stop_reason"]

        evpi_stops = sum(1 for r in stop_reasons if r == "evpi_resolved")
        evpi_pct = 100.0 * evpi_stops / n_scenarios

        summary[name] = {
            "mean_budget": float(np.mean(budget)),
            "median_budget": float(np.median(budget)),
            "mean_probes": float(np.mean(probes)),
            "mean_regret": float(np.mean(regret)),
            "pcs_pct": float(100.0 * np.mean(pcs)),
            "evpi_stop_pct": evpi_pct,
        }

        print(f"{name:<20} | {summary[name]['mean_budget']:<12.1f} | "
              f"{summary[name]['median_budget']:<12.1f} | "
              f"{summary[name]['mean_probes']:<12.1f} | "
              f"{summary[name]['mean_regret']:<12.4f} | "
              f"{summary[name]['pcs_pct']:<8.1f} | "
              f"{evpi_pct:<8.1f}%")
    print("=" * 100)

    # 2. Сравнение стоимости (только для тех, кто остановился по EVPI)
    print("\n📊 Сравнение стоимости (только сценарии с остановкой по EVPI):")
    print("-" * 100)

    common_indices = []
    for i in range(n_scenarios):
        if all(raw_data[name]["stop_reason"][i] == "evpi_resolved"
               for name in policies):
            common_indices.append(i)

    if len(common_indices) < 10:
        print(f"⚠️ Мало общих сценариев ({len(common_indices)}). "
              f"Пропускаем сравнение.")
    else:
        print(f"Общих сценариев: {len(common_indices)}")

        target = "B_EVSI_cost"
        competitors = ["B_IG", "B_mean_prec", "B_EVSI"]

        print(f"\n{'Сравнение':<40} | {'Разница бюджета':<15} | "
              f"{'p-value':<15} | {'Значимо?'}")
        print("-" * 100)

        for comp in competitors:
            target_budget = [
                raw_data[target]["budget_used"][i] for i in common_indices
            ]
            comp_budget = [
                raw_data[comp]["budget_used"][i] for i in common_indices
            ]

            stat, p_val = stats.wilcoxon(
                target_budget, comp_budget, alternative='two-sided'
            )
            diff = np.mean(target_budget) - np.mean(comp_budget)

            is_sig = "✅ ДА" if p_val < 0.05 else "❌ НЕТ"
            print(f"{target} vs {comp:<25} | {diff:<15.1f} | "
                  f"{p_val:<15.4e} | {is_sig}")

    print("=" * 100)

    # 3. Сохранение
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    export_data = {
        "mode": "B",
        "n_scenarios": n_scenarios,
        "budget_limit": BUDGET_STOPPING,
        "evpi_epsilon": EVPI_UPPER_EPSILON,
        "summary": summary,
    }

    out_file = results_dir / "mode_b_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    # Сырые данные с полными логами
    raw_out_file = results_dir / "mode_b_raw.json"
    raw_export = {}
    for name in policies:
        raw_export[name] = {
            "budget_used": raw_data[name]["budget_used"],
            "n_probes": raw_data[name]["n_probes"],
            "regret": raw_data[name]["regret"],
            "stop_reason": raw_data[name]["stop_reason"],
            "pcs": raw_data[name]["pcs"],
            "logs": raw_data[name]["logs"],
        }

    with open(raw_out_file, "w", encoding="utf-8") as f:
        json.dump(raw_export, f, indent=2)

    print(f"\n💾 Результаты сохранены в:")
    print(f"   - {out_file}")
    print(f"   - {raw_out_file}")
    print("✅ Эксперимент завершён.")


if __name__ == "__main__":
    run_mode_b_experiment(n_scenarios=50, n_parcels=30)