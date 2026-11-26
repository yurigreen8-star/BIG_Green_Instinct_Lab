"""
experiments/run_v3_main.py — основной прогон v3.0.

4 режима v3 + 7 контролей v0.1 на одних сидах.
Три residual-метрики: L1 (пошаговое), L3 (сумма по эпизоду), mean_diff.
Тождество L3 = -k*L1 проверяется assert'ом для каждой группы.
mean_diff тождественно L1 при одинаковой агрегации — в печать не выводится.

Факторный контраст I_CR — ТОЛЬКО для аддитивных метрик.
Для regret — парные сравнения, без знаковой интерпретации контраста.

Сиды 0-99 использовались в v0.1 => прогон помечен EXPLORATORY.
Расписания (фиксированное/случайное чередование) в v3.0 отсутствуют.
"""

import sys
from pathlib import Path
import json
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief import Belief
from coordinator import Coordinator
from curiosity import Curiosity
from environment import Environment
from messages import (
    MODE_COMBINED, MODE_COMBINED_IGNORE, MODE_CURIOSITY_SOLO, MODE_RATIO_SOLO,
)
from policies.b_evsi import EVSIPolicy, EVSICostPolicy
from policies.b_ig import IGPolicy
from policies.b_mean_precision import MeanPrecisionPolicy
from policies.b_none import NonePolicy
from policies.b_random import RandomPolicy
from policies.b_random_without_replacement import RandomWithoutReplacementPolicy
from probes import build_catalog
from ratio import Ratio
from runner import run_episode
from runner_v3 import run_episode_v3

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--budget", type=float, default=30.0)
_ap.add_argument("--n-seeds", type=int, default=100)
_ap.add_argument("--seed-start", type=int, default=0)
_ap.add_argument("--tag", type=str, default="main")
_args = _ap.parse_args()

N = 30
BUDGET = _args.budget
SEEDS = range(_args.seed_start, _args.seed_start + _args.n_seeds)
TAG = _args.tag
BOOT_RNG = np.random.default_rng(20261123)
N_BOOT = 2000

V3_MODES = {
    "CURIOSITY_SOLO": MODE_CURIOSITY_SOLO,
    "RATIO_SOLO": MODE_RATIO_SOLO,
    "COMBINED": MODE_COMBINED,
    "COMBINED_IGNORE": MODE_COMBINED_IGNORE,
}

V01_POLICIES = {
    "B_none": NonePolicy,
    "B_random": RandomPolicy,
    "B_IG": IGPolicy,
    "B_mean_prec": MeanPrecisionPolicy,
    "B_EVSI": lambda: EVSIPolicy(use_mc=False),
    "B_EVSI_cost": lambda: EVSICostPolicy(use_mc=False),
    "B_random_without_replacement": RandomWithoutReplacementPolicy,
}


def belief0():
    return Belief(mu=np.zeros(N), sigma2=np.ones(N), known=np.zeros(N, dtype=bool))


def steps(log):
    return [e for e in log if e.get("event") is None]


def field(e, *names):
    """Совместимость схем лога v0.1 и v3. Отсутствующее -> None, не 0."""
    for n in names:
        if n in e and e[n] is not None:
            return e[n]
    return None


def gini(counts):
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * x.sum()))


def episode_metrics(log):
    s = steps(log)
    stop = log[-1]
    ev, ig_, dv = [], [], []
    for e in s:
        v = field(e, "evsi_analytic_executed", "evsi_analytic_selected")
        g = field(e, "ig_executed", "ig_selected")
        if v is None or g is None:
            raise ValueError("в логе нет EVSI/IG — нуль не подставляется")
        ev.append(float(v))
        ig_.append(float(g))
        dv.append(float(np.max(e["mu_after"]) - np.max(e["mu_before"])))

    att = np.zeros(N, dtype=int)
    for k, c in stop.get("attention_distribution", {}).items():
        att[int(k)] = int(c)

    final_s2 = np.asarray(stop["final_sigma2"], dtype=float)
    entropy = 0.5 * float(np.sum(np.log(1.0 / final_s2[final_s2 > 0.0])))

    return {
        "n_steps": len(s),
        "sum_evsi": sum(ev),
        "sum_ig": sum(ig_),
        "sum_dv": sum(dv),
        "regret": float(stop["final_regret"]),
        "pcs": int(bool(stop["final_pcs"])),
        "budget_used": float(stop["budget_used_cumulative"]),
        "n_unique": int(stop.get("n_unique_parcels", len(np.nonzero(att)[0]))),
        "n_repeats": int(stop.get("n_repeats_total", 0)),
        "gini": gini(att),
        "ig_entropy_absdiff": abs(sum(ig_) - entropy),
        "n_counter_proposed": int(stop.get("n_ratio_counter_proposed", 0)),
        "n_counter_executed": int(stop.get("n_ratio_counter_executed", 0)),
        "stop_reason": stop["stop_reason"],
    }


def holm(pvals, alpha=0.05):
    """
    Стандартные Holm-adjusted p-values: монотонны, >= сырых, заполнены для всех.
    Исправлено: прежняя версия выходила по break и оставляла corr=0
    для записей после первого незначимого p.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    corr = np.zeros(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        corr[idx] = min(running, 1.0)
    reject = corr <= alpha
    return reject, corr


def paired_bootstrap_contrast(cr, c, r, zero, n_boot=N_BOOT):
    """Парный бутстрэп по эпизодам: все четыре режима на одних сидах."""
    a = np.asarray(cr, dtype=float)
    b = np.asarray(c, dtype=float)
    d = np.asarray(r, dtype=float)
    e = np.asarray(zero, dtype=float)
    n = len(a)
    point = float(a.mean() - b.mean() - d.mean() + e.mean())
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = BOOT_RNG.integers(0, n, n)
        vals[i] = a[idx].mean() - b[idx].mean() - d[idx].mean() + e[idx].mean()
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    print(f"v3.0 MAIN (EXPLORATORY, seeds {SEEDS.start}-{SEEDS.stop - 1}), "
          f"n={N}, budget={BUDGET}")
    catalog = build_catalog(N)
    raw, per_ep = {}, {}

    for name, mode in V3_MODES.items():
        recs, logs = [], []
        for seed in SEEDS:
            env = Environment.create(N, seed=seed)
            res = run_episode_v3(
                mode=mode, curiosity=Curiosity(N, lam=0.3, seed=seed),
                ratio=Ratio(), coordinator=Coordinator(), belief0=belief0(),
                catalog=catalog, budget=BUDGET, env=env, seed=seed,
                experiment_mode="A",
            )
            recs.append(episode_metrics(res.log))
            logs.append(res.log)
        per_ep[name] = recs
        raw[name] = {"logs": logs, **{k: [r[k] for r in recs] for k in recs[0]}}
        print(f"   {name}: {len(recs)} эпизодов")

    for name, factory in V01_POLICIES.items():
        recs, logs = [], []
        for seed in SEEDS:
            env = Environment.create(N, seed=seed)
            res = run_episode(factory(), belief0(), catalog, BUDGET, env,
                              seed=seed, mode="A")
            recs.append(episode_metrics(res.log))
            logs.append(res.log)
        per_ep[name] = recs
        raw[name] = {"logs": logs, **{k: [r[k] for r in recs] for k in recs[0]}}
        print(f"   {name}: {len(recs)} эпизодов")

    print("\n" + "=" * 115)
    print(f"{'Группа':<30} {'regret':>8} {'PCS%':>6} {'probes':>7} {'uniq':>6} "
          f"{'Gini':>6} {'L1':>12} {'L3':>12} {'IG-entr':>9} {'ctr':>6}")
    print("-" * 115)
    summary = {}
    for name, recs in per_ep.items():
        T = sum(r["n_steps"] for r in recs)
        E = len(recs)
        if T > 0:
            mean_evsi = sum(r["sum_evsi"] for r in recs) / T
            mean_dv = sum(r["sum_dv"] for r in recs) / T
            l1 = mean_evsi - mean_dv
            l3 = float(np.mean([r["sum_dv"] - r["sum_evsi"] for r in recs]))
            md = l1
            k = T / E
            assert abs(l3 + k * l1) < 1e-9, f"{name}: тождество L3=-k*L1 нарушено"
        else:
            l1 = l3 = md = None
        summary[name] = {
            "mean_regret": float(np.mean([r["regret"] for r in recs])),
            "median_regret": float(np.median([r["regret"] for r in recs])),
            "pcs_pct": 100.0 * float(np.mean([r["pcs"] for r in recs])),
            "mean_probes": float(np.mean([r["n_steps"] for r in recs])),
            "mean_unique": float(np.mean([r["n_unique"] for r in recs])),
            "mean_repeats": float(np.mean([r["n_repeats"] for r in recs])),
            "mean_budget": float(np.mean([r["budget_used"] for r in recs])),
            "mean_gini": float(np.mean([r["gini"] for r in recs])),
            "mean_ig_total": float(np.mean([r["sum_ig"] for r in recs])),
            "max_ig_entropy_absdiff": float(max(r["ig_entropy_absdiff"] for r in recs)),
            "total_steps": T,
            "n_episodes": E,
            "stepwise_residual_mean": l1,
            "episode_telescoping_residual_mean": l3,
            "mean_diff": md,
            "mean_counter_proposed": float(np.mean([r["n_counter_proposed"] for r in recs])),
            "mean_counter_executed": float(np.mean([r["n_counter_executed"] for r in recs])),
            "stop_reasons": {s: sum(1 for r in recs if r["stop_reason"] == s)
                             for s in {r["stop_reason"] for r in recs}},
        }
        s = summary[name]
        fmt = lambda v: "         n/a" if v is None else f"{v:12.6f}"
        print(f"{name:<30} {s['mean_regret']:8.4f} {s['pcs_pct']:6.1f} "
              f"{s['mean_probes']:7.2f} {s['mean_unique']:6.2f} {s['mean_gini']:6.3f} "
              f"{fmt(l1)} {fmt(l3)} {s['max_ig_entropy_absdiff']:9.2e} "
              f"{s['mean_counter_executed']:6.2f}")
    print("=" * 115)
    print("mean_diff == L1 тождественно (одинаковая агрегация); значение в summary.json")

    target = "COMBINED"
    comps = ["CURIOSITY_SOLO", "RATIO_SOLO", "COMBINED_IGNORE",
             "B_EVSI_cost", "B_random_without_replacement"]
    print(f"\nRegret: парный двусторонний Wilcoxon, {target} vs контроли (Holm)")
    print("-" * 96)
    pvals, rows = [], []
    a = np.asarray([r["regret"] for r in per_ep[target]])
    for c in comps:
        b = np.asarray([r["regret"] for r in per_ep[c]])
        if np.allclose(a - b, 0.0):
            p, note = 1.0, "идентичны"
        else:
            p, note = float(stats.wilcoxon(a, b, alternative="two-sided").pvalue), ""
        pvals.append(p)
        rows.append((c, float(a.mean() - b.mean()), p, note))
    reject, corr = holm(pvals)
    print(f"{'Контроль':<32} {'dregret':>10} {'p_raw':>12} {'p_holm':>12} {'значимо':>8}")
    for i, (c, d, p, note) in enumerate(rows):
        print(f"{c:<32} {d:10.4f} {p:12.4e} {corr[i]:12.4e} "
              f"{'ДА' if reject[i] else 'НЕТ':>8} {note}")
    print("=" * 96)

    print("\nФакторный контраст I_CR = Y_CR - Y_C - Y_R + Y_0 (Y_0 = B_none)")
    print("   Только аддитивные метрики. Для regret НЕ применяется.")
    print("   I_CR != 0 не доказывает эмерджентность: механизмы замещающие.")
    print("-" * 96)
    contrasts = {}
    for metric in ("n_steps", "n_unique", "sum_ig", "budget_used"):
        pt, lo, hi = paired_bootstrap_contrast(
            [r[metric] for r in per_ep["COMBINED"]],
            [r[metric] for r in per_ep["CURIOSITY_SOLO"]],
            [r[metric] for r in per_ep["RATIO_SOLO"]],
            [r[metric] for r in per_ep["B_none"]],
        )
        contrasts[metric] = {"I_CR": pt, "ci95": [lo, hi]}
        print(f"{metric:<20} I_CR = {pt:10.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print("=" * 96)

    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)

    summary_path = out / f"v3_{TAG}_summary.json"
    raw_path = out / f"v3_{TAG}_raw.json"
    raw_path.write_text(json.dumps(raw, separators=(",", ":")), encoding="utf-8")

    summary_path = out / f"v3_{TAG}_summary.json"
    summary_path.write_text(json.dumps({
        "status": "EXPLORATORY",
        "seed_range": [SEEDS.start, SEEDS.stop - 1],
        "n_parcels": N, "budget": BUDGET,
        "lambda_curiosity": 0.3,
        "summary": summary,
        "regret_paired_tests": [
            {"target": target, "control": c, "delta": d,
             "p_raw": p, "p_holm": float(corr[i]), "significant": bool(reject[i]),
             "note": note}
            for i, (c, d, p, note) in enumerate(rows)
        ],
        "factorial_contrast_additive_only": contrasts,
        "limitations": [
            "Сиды 0-99 использовались в v0.1 — прогон exploratory, не confirmatory.",
            "Фиксированное и случайное чередование в v3.0 не реализованы; "
            "отличие COMBINED от расписаний не проверено.",
            "I_CR не применяется к regret: механизмы замещающие, контраст неаддитивен.",
            "Curiosity solo в Mode A = sweep 30xFAST (доказано тестом); "
            "сравнение с B_random_without_replacement обязательно.",
            "mean_diff тождественно stepwise_residual_mean при одинаковой агрегации.",
        ],
    }, indent=2), encoding="utf-8")

    mb = raw_path.stat().st_size / 1e6
    print(f"\nresults/v3_main_raw.json ({mb:.1f} MB), results/v3_main_summary.json")


if __name__ == "__main__":
    main()