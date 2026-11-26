"""
experiments/run_100_scenarios.py
Полноценный сравнительный эксперимент: 100 сценариев, статистический анализ.
"""

import sys
from pathlib import Path
import json
import numpy as np
from scipy import stats

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from belief import Belief
from environment import Environment
from probes import build_catalog, BUDGET_COMPARISON
from runner import run_episode

from policies.b_none import NonePolicy
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


def run_full_experiment(n_scenarios: int = 100, n_parcels: int = 30):
    print(f"🚀 Запуск: {n_scenarios} сценариев, {n_parcels} участков, бюджет {BUDGET_COMPARISON}")
    print("⏳ Это займёт около 10-15 секунд...\n")

    policies = {
        "B_none": NonePolicy(),
        "B_random": RandomPolicy(),
        "B_IG": IGPolicy(),
        "B_mean_prec": MeanPrecisionPolicy(),
        "B_EVSI": EVSIPolicy(use_mc=False),
        "B_EVSI_cost": EVSICostPolicy(use_mc=False),
    }

    catalog = build_catalog(n_parcels)
    
    # Хранилище сырых данных: {policy_name: {metric: [values]}}
    raw_data = {name: {"regret": [], "pcs": [], "n_probes": [], "budget_used": []} 
                for name in policies}

    for seed in range(n_scenarios):
        env = Environment.create(n_parcels=n_parcels, seed=seed)
        belief0 = get_belief0(n_parcels)

        for name, policy in policies.items():
            res = run_episode(
                policy=policy,
                belief0=belief0,
                catalog=catalog,
                budget=BUDGET_COMPARISON,
                env=env,
                seed=seed,
                mode="A",
            )
            raw_data[name]["regret"].append(res.regret)
            raw_data[name]["pcs"].append(1 if res.pcs else 0)
            raw_data[name]["n_probes"].append(res.n_probes)
            raw_data[name]["budget_used"].append(res.budget_used)

    # 1. Сводная статистика
    print("=" * 85)
    print(f"{'Политика':<15} | {'Ср. Regret':<10} | {'Медиана':<10} | {'Стд. откл.':<10} | {'PCS (%)':<8} | {'Ср. зондов':<10}")
    print("-" * 85)

    summary = {}
    for name in policies:
        reg = raw_data[name]["regret"]
        pcs = raw_data[name]["pcs"]
        probes = raw_data[name]["n_probes"]
        
        summary[name] = {
            "mean_regret": float(np.mean(reg)),
            "median_regret": float(np.median(reg)),
            "std_regret": float(np.std(reg)),
            "pcs_pct": float(100.0 * np.mean(pcs)),
            "mean_probes": float(np.mean(probes)),
            "raw_regret": reg, # сохраняем для тестов
        }
        
        print(f"{name:<15} | {summary[name]['mean_regret']:<10.4f} | "
              f"{summary[name]['median_regret']:<10.4f} | {summary[name]['std_regret']:<10.4f} | "
              f"{summary[name]['pcs_pct']:<8.1f} | {summary[name]['mean_probes']:<10.1f}")
    print("=" * 85)

    # 2. Статистическое сравнение (Wilcoxon signed-rank test)
    # Сравниваем B_EVSI_cost со всеми остальными (кроме себя и B_none)
    print("\n📊 Парные статистические сравнения (Wilcoxon signed-rank test, alpha=0.05)")
    print("Нулевая гипотеза (H0): медианы regret двух политик равны.")
    print("-" * 85)
    
    target = "B_EVSI_cost"
    competitors = ["B_random", "B_IG", "B_mean_prec", "B_EVSI"]
    
    p_values = []
    comparisons = []
    
    for comp in competitors:
        # Wilcoxon для парных выборок (одни и те же seeds)
        stat, p_val = stats.wilcoxon(
            raw_data[target]["regret"], 
            raw_data[comp]["regret"],
            alternative='less' # проверяем, что target < comp
        )
        p_values.append(p_val)
        comparisons.append((target, comp, p_val))

       # Поправка Холма-Бонферрони (ручная реализация, без statsmodels)
    def holm_bonferroni(p_values, alpha=0.05):
        m = len(p_values)
        indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[indices]
        reject = np.zeros(m, dtype=bool)
        pvals_corrected = np.zeros(m)
        
        for i, idx in enumerate(indices):
            threshold = alpha / (m - i)
            pvals_corrected[idx] = min(sorted_p[i] * (m - i), 1.0)
            if sorted_p[i] <= threshold:
                reject[idx] = True
            else:
                # Все последующие тоже не отвергаются
                for j in range(i, m):
                    reject[indices[j]] = False
                break
        
        return reject, pvals_corrected

    reject, pvals_corrected = holm_bonferroni(p_values, alpha=0.05)

    # 3. Сохранение результатов
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Убираем сырые данные из JSON для читаемости, оставляем только сводку
    export_data = {
        "n_scenarios": n_scenarios,
        "n_parcels": n_parcels,
        "budget": BUDGET_COMPARISON,
        "summary": {k: {mk: mv for mk, mv in v.items() if mk != "raw_regret"} for k, v in summary.items()},
        "statistical_tests": [
            {"target": t, "competitor": c, "p_raw": float(p), "p_holm": float(pc), "significant": bool(sig)}
            for (t, c, p), pc, sig in zip(comparisons, pvals_corrected, reject)
        ]
    }
    
    out_file = results_dir / "exp_100_scenarios_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
    
        # Сохраняем сырые данные с полными логами
    raw_out_file = results_dir / "h2_results_raw.json"  # (или соответствующее имя)
    
    # Формируем структуру с обратной совместимостью
    raw_export = {}
    for name in policies:
        raw_export[name] = {
            "regret": raw_data[name]["regret"],
            "pcs": raw_data[name]["pcs"],
            "n_probes": raw_data[name]["n_probes"],
            "budget_used": raw_data[name]["budget_used"],
            "stop_reason": raw_data[name]["stop_reason"],
            "logs": raw_data[name]["logs"],  # <-- НОВОЕ ПОЛЕ
        }
        
    with open(raw_out_file, "w", encoding="utf-8") as f:
        json.dump(raw_export, f, indent=2)

    print(f"\n💾 Результаты сохранены в:\n   - {out_file}\n   - {raw_out_file}")
    print("✅ Эксперимент успешно завершён.")


if __name__ == "__main__":
    run_full_experiment(n_scenarios=100, n_parcels=30)