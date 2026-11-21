"""
experiments/run_h2.py
Строгая проверка гипотезы H2 на свежих сидах (100-199).
Двусторонний тест Уилкоксона, поправка Холма-Бонферрони.
"""

import sys
from pathlib import Path
import json
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief import Belief
from environment import Environment
from probes import build_catalog, BUDGET_COMPARISON
from runner import run_episode

from policies.b_evsi import EVSIPolicy, EVSICostPolicy
from policies.b_ig import IGPolicy
from policies.b_mean_precision import MeanPrecisionPolicy


class PreciseDiversePolicy:
    """Exploratory baseline: покупает точные зонды на разных участках, пока есть бюджет."""
    name = "B_precise_diverse"
    
    def select_probe(self, belief, available, budget_remaining, rng):
        if not available:
            return None
        # Фильтруем только точные зонды
        precise_probes = [p for p in available if p.probe_type.name == "survey_precise"]
        if precise_probes:
            # Берем первый доступный (на наименьшем parcel_id для детерминизма)
            return precise_probes[0]
        # Если точных нет или не хватает бюджета, ничего не делаем (или можно fallback на fast)
        return None


def get_belief0(n: int) -> Belief:
    return Belief(
        mu=np.zeros(n),
        sigma2=np.ones(n),
        known=np.zeros(n, dtype=bool),
    )


def run_h2_experiment(n_scenarios: int = 100, n_parcels: int = 30, seed_start: int = 100):
    print(f"🚀 H2 Experiment: {n_scenarios} сценариев (seeds {seed_start}-{seed_start+n_scenarios-1})")
    print(f"   Участков: {n_parcels}, Бюджет: {BUDGET_COMPARISON}")
    print("⏳ Запуск...\n")

    policies = {
        "B_EVSI_cost": EVSICostPolicy(use_mc=False),
        "B_IG": IGPolicy(),
        "B_mean_prec": MeanPrecisionPolicy(),
        "B_EVSI": EVSIPolicy(use_mc=False),
        "B_precise_diverse": PreciseDiversePolicy(), # exploratory
    }

    catalog = build_catalog(n_parcels)
    raw_data = {name: {"regret": [], "pcs": [], "n_probes": [], "budget_used": []} for name in policies}

    for seed in range(seed_start, seed_start + n_scenarios):
        env = Environment.create(n_parcels=n_parcels, seed=seed)
        belief0 = get_belief0(n_parcels)

        for name, policy in policies.items():
            res = run_episode(
                policy=policy, belief0=belief0, catalog=catalog,
                budget=BUDGET_COMPARISON, env=env, seed=seed, mode="A",
            )
            raw_data[name]["regret"].append(res.regret)
            raw_data[name]["pcs"].append(1 if res.pcs else 0)
            raw_data[name]["n_probes"].append(res.n_probes)
            raw_data[name]["budget_used"].append(res.budget_used)

    # 1. Сводная статистика
    print("=" * 95)
    print(f"{'Политика':<20} | {'Ср. Regret':<10} | {'Медиана':<10} | {'Стд. откл.':<10} | {'PCS (%)':<8} | {'Ср. зондов':<10}")
    print("-" * 95)

    summary = {}
    target = "B_EVSI_cost"
    competitors = ["B_IG", "B_mean_prec", "B_EVSI"]
    
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
        }
        print(f"{name:<20} | {summary[name]['mean_regret']:<10.4f} | "
              f"{summary[name]['median_regret']:<10.4f} | {summary[name]['std_regret']:<10.4f} | "
              f"{summary[name]['pcs_pct']:<8.1f} | {summary[name]['mean_probes']:<10.1f}")
    print("=" * 95)

    # 2. Статистическое сравнение (Двусторонний тест Уилкоксона)
    print("\n📊 Парные сравнения (Двусторонний Wilcoxon signed-rank test, alpha=0.05)")
    print("H0: медианы regret целевой и конкурирующей политик равны.")
    print("-" * 95)
    
    p_values = []
    comparisons = []
    
    for comp in competitors:
        # alternative='two-sided' - строгий двусторонний тест
        stat, p_val = stats.wilcoxon(
            raw_data[target]["regret"], 
            raw_data[comp]["regret"],
            alternative='two-sided'
        )
        p_values.append(p_val)
        comparisons.append((target, comp, p_val))

    # Поправка Холма-Бонферрони
    def holm_bonferroni(p_vals, alpha=0.05):
        m = len(p_vals)
        indices = np.argsort(p_vals)
        sorted_p = np.array(p_vals)[indices]
        reject = np.zeros(m, dtype=bool)
        pvals_corrected = np.zeros(m)
        for i, idx in enumerate(indices):
            threshold = alpha / (m - i)
            pvals_corrected[idx] = min(sorted_p[i] * (m - i), 1.0)
            if sorted_p[i] <= threshold:
                reject[idx] = True
            else:
                for j in range(i, m):
                    reject[indices[j]] = False
                break
        return reject, pvals_corrected

    reject, pvals_corrected = holm_bonferroni(p_values, alpha=0.05)

    print(f"{'Сравнение':<40} | {'p-value (raw)':<15} | {'p-value (Holm)':<15} | {'Значимо?'}")
    print("-" * 95)
    
    all_significant = True
    for i, (t, c, p_raw) in enumerate(comparisons):
        p_corr = pvals_corrected[i]
        is_sig = "✅ ДА" if reject[i] else "❌ НЕТ"
        if not reject[i]:
            all_significant = False
        print(f"{t} vs {c:<25} | {p_raw:<15.4e} | {p_corr:<15.4e} | {is_sig}")
    print("=" * 95)

    # 3. Вердикт
    print("\n🏁 ВЕРДИКТ ПО H2:")
    if all_significant:
        # Дополнительная проверка направления эффекта
        target_med = summary[target]["median_regret"]
        direction_ok = all(target_med < summary[comp]["median_regret"] for comp in competitors)
        if direction_ok:
            print("✅ ГИПОТЕЗА H2 ПОДТВЕРЖДЕНА. B_EVSI_cost статистически значимо лучше всех конкурентов.")
        else:
            print("⚠️ РАЗЛИЧИЯ ЗНАЧИМЫ, НО НАПРАВЛЕНИЕ ЭФФЕКТА НЕ СООТВЕТСТВУЕТ ОЖИДАНИЯМ. H2 НЕ ПОДТВЕРЖДЕНА.")
    else:
        print("❌ ГИПОТЕЗА H2 НЕ ПОДТВЕРЖДЕНА. Нет статистически значимого превосходства над всеми конкурентами.")

    # 4. Сохранение
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    export_data = {
        "h2_preregistered": True,
        "seed_range": [seed_start, seed_start + n_scenarios - 1],
        "n_scenarios": n_scenarios,
        "summary": summary,
        "statistical_tests": [
            {"target": t, "competitor": c, "p_raw": float(p), "p_holm": float(pc), "significant": bool(sig)}
            for (t, c, p), pc, sig in zip(comparisons, pvals_corrected, reject)
        ],
        "h2_confirmed": bool(all_significant)
    }
    
    out_file = results_dir / "h2_results_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
        
    raw_out_file = results_dir / "h2_results_raw.json"
    with open(raw_out_file, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    print(f"\n💾 Результаты сохранены в:\n   - {out_file}\n   - {raw_out_file}")


if __name__ == "__main__":
    run_h2_experiment(n_scenarios=100, n_parcels=30, seed_start=100)