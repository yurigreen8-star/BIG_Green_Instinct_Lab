"""
experiments/run_e2.py — драйвер E2. Предрегистрация §10, §11.

Этапы: smoke (52000-52004), design (50000-50099), confirm (51000-51999).
Подтверждающий этап требует зафиксированного q_V.json и запрещает
пересечение с уже израсходованными сидами.

Интерпретация regret/EVPI на этапах smoke и design не производится.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment_e2 import make_scenario
from policies_e2 import IGFastE2, RandomE2, SweepThenRandomE2
from probes import SURVEY_FAST, build_catalog
from runner_e2 import (EXPLORATORY_GROUPS, GROUPS_E2, PRIMARY_V,
                       run_episode_e2)

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"

STAGES = {"smoke": (52000, 5), "design": (50000, 100), "confirm": (51000, 1000)}
FORBIDDEN_CONFIRM = [(0, 199), (5000, 5099), (20000, 20099),
                     (40100, 40199), (50000, 50099)]
BUDGETS = (45.0, 90.0)
N_PARCELS = 30
Q_SMOKE_PLACEHOLDER = 0.5


def git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "UNAVAILABLE"


def fast_catalog():
    return [p for p in build_catalog(N_PARCELS)
            if p.probe_type.probe_type_id == SURVEY_FAST.probe_type_id]


def make_policy(group, scenario):
    if group == "B_IG_FAST":
        return IGFastE2()
    if group == "B_RANDOM":
        return RandomE2()
    if group == "B_SWEEP_THEN_RANDOM":
        return SweepThenRandomE2(scenario.s_prior)
    return None


def q_for(group, budget, stage, q_table):
    if not group.startswith("MATCH_"):
        return None, "not_applicable"
    if stage == "confirm":
        # Убираем префикс MATCH_ для поиска в q_V.json
        matched_v = group.replace("MATCH_", "")
        key = f"{matched_v}|{int(budget)}"
        if q_table is None or key not in q_table:
            raise SystemExit(f"confirm требует q_V.json с ключом {key}")
        return float(q_table[key]["q"]), "q_V.json"
    return Q_SMOKE_PLACEHOLDER, f"placeholder_{stage}"


def finite_check(obj, path="root"):
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += finite_check(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += finite_check(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        if not np.isfinite(obj):
            bad.append(path)
    return bad


def aggregate(records):
    def col(name):
        return np.array([r[name] for r in records], dtype=np.float64)

    sp = [r["spearman_s_repeats"] for r in records
          if r["spearman_s_repeats"] is not None]
    ev = np.array([r["evpi"]["mean"] for r in records], dtype=np.float64)
    es = np.array([r["evpi"]["se"] for r in records], dtype=np.float64)
    reg = col("regret")
    return {
        "n_episodes": len(records),
        "mean_regret": float(reg.mean()),
        "median_regret": float(np.median(reg)),
        "sd_regret": float(reg.std(ddof=1)) if len(reg) > 1 else 0.0,
        "pcs_pct": 100.0 * float(col("pcs").mean()),
        "mean_probes": float(col("n_probes").mean()),
        "mean_budget_used": float(col("budget_used").mean()),
        "mean_unique": float(col("n_unique").mean()),
        "mean_repeats": float(col("n_repeats").mean()),
        "mean_entropy_visits": float(col("entropy_visits").mean()),
        "mean_gini_visits": float(col("gini_visits").mean()),
        "mean_spearman_s_repeats": float(np.mean(sp)) if sp else None,
        "n_spearman_defined": len(sp),
        "mean_replacement_pct": float(col("replacement_pct").mean()),
        "mean_ratio_choice_pct": float(col("ratio_choice_pct").mean()),
        "mean_ratio_action_pct": float(col("ratio_action_pct").mean()),
        "mean_frac_max_evsi_tiny_pct": float(col("frac_max_evsi_tiny_pct").mean()),
        "mean_evpi": float(ev.mean()),
        "sd_evpi": float(ev.std(ddof=1)) if len(ev) > 1 else 0.0,
        "mean_evpi_mc_se": float(es.mean()),
        "evpi_n_samples": int(records[0]["evpi"]["n_samples"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS))
    ap.add_argument("--groups", type=str, nargs="+", default=list(GROUPS_E2))
    ap.add_argument("--evpi-n", type=int, default=40000)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--seed-start", type=int, default=None)
    ap.add_argument("--q-file", type=str, default=None)
    ap.add_argument("--full-log", action="store_true", default=False)
    args = ap.parse_args()

    start, count = STAGES[args.stage]
    start = args.seed_start if args.seed_start is not None else start
    count = args.n_seeds if args.n_seeds is not None else count
    seeds = list(range(start, start + count))
    tag = args.tag or f"e2_{args.stage}"

    if args.stage == "confirm":
        for lo, hi in FORBIDDEN_CONFIRM:
            hit = [s for s in seeds if lo <= s <= hi]
            if hit:
                raise SystemExit(f"confirm: запрещённые сиды {hit[:5]} из {lo}-{hi}")

    q_table = None
    if args.q_file:
        q_table = json.loads(Path(args.q_file).read_text(encoding="utf-8"))
    elif args.stage == "confirm":
        raise SystemExit("confirm требует --q-file q_V.json")

    for g in args.groups:
        if g not in GROUPS_E2:
            raise SystemExit(f"неизвестная группа {g}")

    config = {
        "stage": args.stage, "tag": tag,
        "seed_start": start, "n_seeds": count,
        "budgets": list(args.budgets), "groups": list(args.groups),
        "n_parcels": N_PARCELS, "evpi_n": args.evpi_n,
        "lambda": 0.3, "rho": 0.5, "eps": 1e-12, "delta_tost": 0.01,
        "git_head": git("rev-parse", "HEAD"),
        "prereg_blob": git("rev-parse", "HEAD:PREREGISTRATION_E2.md"),
        "prereg_commit": git("log", "-1", "--format=%H", "--",
                             "PREREGISTRATION_E2.md"),
        "prereg_dirty": git("status", "--short", "PREREGISTRATION_E2.md"),
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
    }
    if config["prereg_dirty"]:
        raise SystemExit("предрегистрация имеет незакоммиченные изменения")

    print(f"E2 stage={args.stage} seeds={start}-{start + count - 1} "
          f"budgets={args.budgets} groups={len(args.groups)}")
    print(f"HEAD={config['git_head'][:12]} prereg={config['prereg_blob'][:12]}")

    catalog = fast_catalog()
    assert len(catalog) == N_PARCELS
    raw, summary, records_index = {}, {}, {}

    for budget in args.budgets:
        for group in args.groups:
            recs, logs = [], []
            for seed in seeds:
                sc = make_scenario(seed, N_PARCELS)
                q, q_src = q_for(group, budget, args.stage, q_table)
                res = run_episode_e2(group, sc, catalog, budget=float(budget),
                                     q_match=q, policy=make_policy(group, sc),
                                     evpi_n=args.evpi_n)
                m = dict(res.metrics)
                m["q_match"] = q
                m["q_source"] = q_src
                recs.append(m)
                log = res.log
                if not args.full_log and group not in EXPLORATORY_GROUPS:
                    log = [{k: v for k, v in e.items() if k != "curiosity_ranking"}
                           for e in log]
                logs.append(log)
            key = f"{group}|{int(budget)}"
            summary[key] = aggregate(recs)
            raw[key] = {"metrics": recs, "logs": logs}
            records_index[key] = recs
            print(f"   {key}: {len(recs)} эпизодов")

    checks = {
        "budget_and_probes": True,
        "no_nan_inf": True,
        "evpi_n_correct": True,
        "reproducible": True,
        "details": [],
    }
    for key, recs in records_index.items():
        budget = float(key.split("|")[1])
        for r in recs:
            if r["n_probes"] != int(budget) or abs(r["budget_used"] - budget) > 1e-9:
                checks["budget_and_probes"] = False
                checks["details"].append(f"{key} seed={r['seed']}: бюджет/шаги")
            if r["evpi"]["n_samples"] != args.evpi_n:
                checks["evpi_n_correct"] = False
            bad = finite_check(r)
            if bad:
                checks["no_nan_inf"] = False
                checks["details"].append(f"{key} seed={r['seed']}: {bad[:3]}")

    sc = make_scenario(seeds[0], N_PARCELS)
    g0 = args.groups[0]
    b0 = float(args.budgets[0])
    q0, _ = q_for(g0, b0, args.stage, q_table)
    a = run_episode_e2(g0, sc, catalog, budget=b0, q_match=q0,
                       policy=make_policy(g0, sc), evpi_n=args.evpi_n)
    b = run_episode_e2(g0, sc, catalog, budget=b0, q_match=q0,
                       policy=make_policy(g0, sc), evpi_n=args.evpi_n)
    checks["reproducible"] = (a.log == b.log)

    RES.mkdir(exist_ok=True)
    (RES / f"{tag}_config.json").write_text(json.dumps(config, indent=2),
                                            encoding="utf-8")
    (RES / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    (RES / f"{tag}_raw.json").write_text(json.dumps(raw, separators=(",", ":")),
                                         encoding="utf-8")
    (RES / f"{tag}_checks.json").write_text(json.dumps(checks, indent=2),
                                            encoding="utf-8")

    print("\n" + "=" * 118)
    print(f"{'Группа|бюджет':<30}{'regret':>9}{'PCS%':>7}{'probes':>8}"
          f"{'uniq':>7}{'repl%':>8}{'ratio%':>8}{'EVPI':>10}{'MCse':>9}")
    print("-" * 118)
    for key, s in summary.items():
        print(f"{key:<30}{s['mean_regret']:9.4f}{s['pcs_pct']:7.1f}"
              f"{s['mean_probes']:8.2f}{s['mean_unique']:7.2f}"
              f"{s['mean_replacement_pct']:8.2f}{s['mean_ratio_choice_pct']:8.2f}"
              f"{s['mean_evpi']:10.5f}{s['mean_evpi_mc_se']:9.5f}")
    print("=" * 118)
    print(f"Проверки: budget/probes={checks['budget_and_probes']} "
          f"finite={checks['no_nan_inf']} evpi_n={checks['evpi_n_correct']} "
          f"reproducible={checks['reproducible']}")
    if checks["details"]:
        print("Проблемы:", checks["details"][:10])
    mb = (RES / f"{tag}_raw.json").stat().st_size / 1e6
    print(f"\nresults/{tag}_raw.json ({mb:.1f} MB), _summary.json, "
          f"_config.json, _checks.json")
    if args.stage in ("smoke", "design"):
        print("Этап не предназначен для интерпретации regret/EVPI.")


if __name__ == "__main__":
    main()