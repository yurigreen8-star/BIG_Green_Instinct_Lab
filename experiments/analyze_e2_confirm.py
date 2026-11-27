"""
experiments/analyze_e2_confirm.py — агрегация и парные тесты этапа D.

Читает четыре raw-файла e2_c{1,2,3,4}_raw.json, строит парные разности
по сидам (CRN гарантирует парность), применяет Holm-коррекцию к 16
зарегистрированным контрастам, TOST для эквивалентности.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from stats_e2 import holm, paired_t_ci, tost_paired

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"

PRIMARY_CONTRASTS = [
    ("C_OL_DOM", "RATIO_SOLO"),
    ("C_BS_DOM", "RATIO_SOLO"),
    ("C_OL_OVR", "RATIO_SOLO"),
    ("C_BS_OVR", "RATIO_SOLO"),
    ("MATCH_C_OL_DOM", "C_OL_DOM"),
    ("MATCH_C_BS_DOM", "C_BS_DOM"),
    ("MATCH_C_OL_OVR", "C_OL_OVR"),
    ("MATCH_C_BS_OVR", "C_BS_OVR"),
    ("C_OL_DOM", "C_OL_SOLO"),
    ("C_BS_DOM", "C_BS_SOLO"),
    ("C_OL_OVR", "C_OL_SOLO"),
    ("C_BS_OVR", "C_BS_SOLO"),
    ("MATCH_C_OL_DOM", "B_IG_FAST"),
    ("MATCH_C_BS_DOM", "B_IG_FAST"),
    ("MATCH_C_OL_OVR", "B_IG_FAST"),
    ("MATCH_C_BS_OVR", "B_IG_FAST"),
]

BUDGETS = [45, 90]
DELTA_TOST = 0.01


def load_all_raw():
    """Загружает все 4 raw-файла и сливает в единый датасет по (group, budget, seed)."""
    data = {}
    for i in range(1, 5):
        path = RES / f"e2_c{i}_raw.json"
        if not path.exists():
            raise SystemExit(f"нет {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key, group_data in raw.items():
            group, budget = key.split("|")
            budget = int(budget)
            if (group, budget) not in data:
                data[(group, budget)] = {}
            for metrics in group_data["metrics"]:
                seed = metrics["seed"]
                if seed in data[(group, budget)]:
                    print(f"WARNING: seed {seed} дублируется в {group}|{budget}")
                data[(group, budget)][seed] = metrics
    return data


def paired_diff(data, g1, g2, budget, metric="regret"):
    """Парная разность: g1 - g2 для всех общих сидов."""
    d1 = data.get((g1, budget), {})
    d2 = data.get((g2, budget), {})
    seeds = sorted(set(d1.keys()) & set(d2.keys()))
    if not seeds:
        return None
    diffs = np.array([d1[s][metric] - d2[s][metric] for s in seeds])
    return seeds, diffs


def run_tests(data, contrasts, budgets, alpha=0.05, delta_tost=0.01):
    """Выполняет все тесты для списка контрастов и бюджетов."""
    results = []
    for g1, g2 in contrasts:
        for budget in budgets:
            res = paired_diff(data, g1, g2, budget, "regret")
            if res is None:
                continue
            seeds, diffs = res
            n = len(diffs)
            ci = paired_t_ci(diffs, alpha=alpha)
            tost = tost_paired(diffs, delta=delta_tost, alpha=alpha)
            p_raw = wilcoxon_paired(diffs)
            results.append({
                "contrast": f"{g1} vs {g2}",
                "budget": budget,
                "n": n,
                "mean_diff": ci["mean"],
                "sd_diff": ci["sd"],
                "se_diff": ci["se"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "p_raw": p_raw,
                "tost_p": tost["p_tost"],
                "tost_equivalent": tost["equivalent"],
                "ci90_low": tost["ci90_low"],
                "ci90_high": tost["ci90_high"],
            })
    return results


def wilcoxon_paired(diffs):
    """Wilcoxon signed-rank test, two-sided."""
    d = np.asarray(diffs, dtype=np.float64)
    if np.allclose(d, 0.0):
        return 1.0
    return float(stats.wilcoxon(d, alternative="two-sided").pvalue)


def main():
    print("Загрузка данных из e2_c{1,2,3,4}_raw.json...")
    data = load_all_raw()
    print(f"Загружено: {len(data)} групп×бюджетов")

    # Проверка, что все сиды на месте
    seeds_per_budget = {b: set() for b in BUDGETS}
    for (g, b), seeds_dict in data.items():
        seeds_per_budget[b].update(seeds_dict.keys())
    for b in BUDGETS:
        expected = set(range(51000, 52000))
        missing = expected - seeds_per_budget[b]
        extra = seeds_per_budget[b] - expected
        if missing:
            print(f"WARNING: бюджет {b}: отсутствуют сиды {sorted(missing)[:10]}...")
        if extra:
            print(f"WARNING: бюджет {b}: лишние сиды {sorted(extra)[:10]}...")
        print(f"Бюджет {b}: {len(seeds_per_budget[b])} уникальных сидов")

    print("\n" + "=" * 120)
    print("Парные тесты (16 зарегистрированных контрастов × 2 бюджета = 32 теста)")
    print("=" * 120)

    results = run_tests(data, PRIMARY_CONTRASTS, BUDGETS, alpha=0.05, delta_tost=DELTA_TOST)

    # Holm-коррекция
    pvals = [r["p_raw"] for r in results]
    _, adj = holm(pvals, alpha=0.05)
    for i, r in enumerate(results):
        r["p_holm"] = adj[i]
        r["significant_holm"] = bool(adj[i] < 0.05)

    # Вывод таблицы
    print(f"{'Контраст':<40}{'B':>4}{'n':>6}{'Δ mean':>10}{'95% CI':>20}{'p_raw':>10}{'p_Holm':>10}{'H0 откл':>8}{'TOST экв':>9}")
    print("-" * 120)
    for r in results:
        ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        sig = "ДА" if r["significant_holm"] else "нет"
        equiv = "ДА" if r["tost_equivalent"] else "нет"
        print(f"{r['contrast']:<40}{r['budget']:>4}{r['n']:>6}{r['mean_diff']:10.5f}{ci_str:>20}"
              f"{r['p_raw']:10.4f}{r['p_holm']:10.4f}{sig:>8}{equiv:>9}")

    # Сохранение в JSON
    output = {
        "n_contrasts": len(PRIMARY_CONTRASTS),
        "n_budgets": len(BUDGETS),
        "n_tests": len(results),
        "delta_tost": DELTA_TOST,
        "alpha": 0.05,
        "seeds_range": [51000, 52000],
        "results": results,
    }
    out_path = RES / "e2_confirm_analysis.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nСохранено: {out_path}")

    # Краткая сводка
    n_sig = sum(1 for r in results if r["significant_holm"])
    n_equiv = sum(1 for r in results if r["tost_equivalent"])
    print(f"\nИтого: {n_sig}/{len(results)} значимых различий (Holm), {n_equiv}/{len(results)} эквивалентностей (TOST)")


if __name__ == "__main__":
    main()