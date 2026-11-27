"""
experiments/analyze_median.py — медианный анализ C_BS vs C_OL.

Считает попарные медианы разностей по сидам для всех комбинаций
арбитров (DOM, OVR, SOLO) и обоих бюджетов. Это разведочный анализ,
дополняющий основной отчёт.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"

CONTRASTS = [
    ("C_BS_DOM", "C_OL_DOM"),
    ("C_BS_OVR", "C_OL_OVR"),
    ("C_BS_SOLO", "C_OL_SOLO"),
]
BUDGETS = [45, 90]


def load_all():
    data = {}
    for i in range(1, 5):
        raw = json.loads((RES / f"e2_c{i}_raw.json").read_text(encoding="utf-8"))
        for key, gd in raw.items():
            group, budget = key.split("|")
            data.setdefault((group, int(budget)), {})
            for m in gd["metrics"]:
                data[(group, int(budget))][m["seed"]] = m
    return data


def main():
    data = load_all()
    print(f"{'Контраст':<35}{'B':>4}{'n':>6}{'median_diff':>12}{'IQR':>12}{'p_wilcox':>12}")
    print("-" * 85)
    for g1, g2 in CONTRASTS:
        for b in BUDGETS:
            d1 = data.get((g1, b), {})
            d2 = data.get((g2, b), {})
            seeds = sorted(set(d1) & set(d2))
            diffs = np.array([d1[s]["regret"] - d2[s]["regret"] for s in seeds])
            med = float(np.median(diffs))
            q1, q3 = np.percentile(diffs, [25, 75])
            iqr = q3 - q1
            from scipy.stats import wilcoxon
            try:
                p = float(wilcoxon(diffs, alternative="two-sided").pvalue)
            except Exception:
                p = 1.0
            print(f"{g1} vs {g2:<15}{b:>4}{len(seeds):>6}{med:12.5f}{iqr:12.5f}{p:12.5f}")


if __name__ == "__main__":
    main()