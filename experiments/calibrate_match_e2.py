"""
experiments/calibrate_match_e2.py — калибровка MATCH. Предрегистрация E2 §8.

Цель: подобрать q так, чтобы ФАКТИЧЕСКАЯ доля замен MATCH совпала с фактической
долей замен соответствующего V. q и доля замен — разные величины.

Сеточный поиск, без предположения монотонности r(q). Бисекция не применяется.
Броски MATCH не зависят от q (поток 6 от seed сценария), поэтому кривая r(q)
детерминирована на фиксированном наборе пилотных сценариев.

Экономия без изменения метода: MATCH_C_OL_DOM и MATCH_C_OL_OVR — одна и та же
политика (C_OL + MATCH), различаются только целевой частотой. Кривая r(q)
считается один раз на (тип Curiosity, бюджет), затем q выбирается под каждую цель.

EVPI в калибровке не используется: измеряется только доля замен.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment_e2 import make_scenario
from probes import SURVEY_FAST, build_catalog
from runner_e2 import run_episode_e2

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"

PILOT_SEEDS = list(range(50000, 50100))
BUDGETS = (45.0, 90.0)
MECHS = ("OL", "BS")
V_GROUPS = ("C_OL_DOM", "C_BS_DOM", "C_OL_OVR", "C_BS_OVR")

BENCHMARK = {
    ("C_OL_DOM", 45): 0.91, ("C_OL_DOM", 90): 0.72,
    ("C_BS_DOM", 45): 0.89, ("C_BS_DOM", 90): 0.71,
    ("C_OL_OVR", 45): 0.91, ("C_OL_OVR", 90): 0.88,
    ("C_BS_OVR", 45): 0.93, ("C_BS_OVR", 90): 0.91,
}
BENCH_TOL = 0.05


def git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "UNAVAILABLE"


def fast_catalog(n=30):
    return [p for p in build_catalog(n)
            if p.probe_type.probe_type_id == SURVEY_FAST.probe_type_id]


def mech_of(group: str) -> str:
    return "BS" if "_BS_" in group else "OL"


def curve_group(mech: str) -> str:
    """Любая MATCH-группа данного типа: политика идентична."""
    return f"MATCH_C_{mech}_DOM"


def measure(mech, budget, q, seeds, catalog, evpi_n):
    group = curve_group(mech)
    rep, steps = 0, 0
    per_seed = []
    for seed in seeds:
        sc = make_scenario(seed)
        res = run_episode_e2(group, sc, catalog, budget=float(budget),
                             q_match=float(q), evpi_n=evpi_n)
        rep += res.metrics["n_replacements"]
        steps += res.metrics["n_probes"]
        per_seed.append(res.metrics["n_replacements"])
    return {"q": round(float(q), 6), "r": rep / steps,
            "n_replacements": rep, "n_steps": steps,
            "per_seed_replacements": per_seed}


def pick_min(cands, target):
    """Минимум |r - target|; при равенстве — меньшее q."""
    best = None
    for c in sorted(cands, key=lambda d: d["q"]):
        gap = abs(c["r"] - target)
        if best is None or gap < best[0] - 1e-15:
            best = (gap, c)
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design-summary", type=str,
                    default="results/e2_design_summary.json")
    ap.add_argument("--q-step", type=float, default=0.01)
    ap.add_argument("--refine-step", type=float, default=0.001)
    ap.add_argument("--refine-window", type=float, default=0.01)
    ap.add_argument("--evpi-n", type=int, default=200)
    ap.add_argument("--tag", type=str, default="e2_calib")
    ap.add_argument("--selftest", action="store_true", default=False)
    ap.add_argument("--resume", action="store_true", default=False)
    args = ap.parse_args()

    seeds = PILOT_SEEDS[:5] if args.selftest else PILOT_SEEDS
    q_grid = ([0.0, 0.5, 1.0] if args.selftest
              else [round(x * args.q_step, 6)
                    for x in range(int(round(1.0 / args.q_step)) + 1)])
    budgets = (45.0,) if args.selftest else BUDGETS

    spath = ROOT / args.design_summary
    if not spath.exists():
        raise SystemExit(f"нет {spath}: сначала выполни этап C1")
    design = json.loads(spath.read_text(encoding="utf-8"))

    targets = {}
    for g in V_GROUPS:
        for b in budgets:
            key = f"{g}|{int(b)}"
            if key not in design:
                raise SystemExit(f"в проектной сводке нет {key}")
            targets[(g, int(b))] = design[key]["mean_replacement_pct"] / 100.0

    catalog = fast_catalog()
    cache: dict[tuple, dict] = {}
    curves: dict[str, list] = {}
    t0 = time.time()

    partial = RES / f"{args.tag}_grid_partial.json"
    if args.resume and partial.exists():
        prev = json.loads(partial.read_text(encoding="utf-8"))
        for k, rows in prev.items():
            mech, bs = k.split("|")
            for row in rows:
                cache[(mech, int(bs), round(row["q"], 6))] = row
        print(f"Возобновление: загружено {len(cache)} точек из {partial.name}")

    print(f"Калибровка MATCH: seeds {seeds[0]}-{seeds[-1]}, "
          f"сетка шаг {args.q_step}, точек {len(q_grid)}, "
          f"бюджеты {budgets}, EVPI N={args.evpi_n}")
    print("Целевые доли замен (из проектного прогона):")
    for (g, bi), tg in targets.items():
        bn = BENCHMARK.get((g, bi))
        dv = None if bn is None else tg - bn
        flag = "" if dv is None or abs(dv) <= BENCH_TOL else "  ВНЕ ДОПУСКА"
        bs = "n/a" if bn is None else f"{bn:.2f}"
        ds = "n/a" if dv is None else f"{dv:+.4f}"
        print(f"   {g:<12} B={bi:<3} r_target={tg:.4f} эталон={bs} "
              f"откл={ds}{flag}")
    if args.selftest:
        print("РЕЖИМ SELFTEST: проверка работоспособности скрипта, "
              "не зарегистрированная калибровка")

    total_coarse = len(MECHS) * len(budgets) * len(q_grid)
    done_coarse = 0
    for mech in MECHS:
        for b in budgets:
            rows = []
            for i, q in enumerate(q_grid):
                ck = (mech, int(b), round(q, 6))
                if ck not in cache:
                    cache[ck] = measure(mech, b, q, seeds, catalog, args.evpi_n)
                rows.append(cache[ck])
                done_coarse += 1
                if (i + 1) % 10 == 0 or i + 1 == len(q_grid):
                    el = time.time() - t0
                    eta = el / max(done_coarse, 1) * (total_coarse - done_coarse)
                    print(f"   C_{mech} B={int(b)}: {i+1}/{len(q_grid)} "
                          f"q={q:.2f} r={rows[-1]['r']:.4f} "
                          f"[{el/60:.1f} мин, ETA {eta/60:.1f} мин]")
            curves[f"{mech}|{int(b)}"] = rows
            RES.mkdir(exist_ok=True)
            partial.write_text(json.dumps(curves, separators=(",", ":")),
                               encoding="utf-8")

    results, report_rows = {}, []
    for g in V_GROUPS:
        for b in budgets:
            bi = int(b)
            mech = mech_of(g)
            rows = curves[f"{mech}|{bi}"]
            target = targets[(g, bi)]
            coarse, _ = pick_min(rows, target)

            refined = []
            if not args.selftest:
                lo = max(0.0, coarse["q"] - args.refine_window)
                hi = min(1.0, coarse["q"] + args.refine_window)
                npts = int(round((hi - lo) / args.refine_step)) + 1
                print(f"   уточнение {g} B={bi}: q в [{lo:.3f}, {hi:.3f}], "
                      f"{npts} точек")
                for k in range(npts):
                    q = round(lo + k * args.refine_step, 6)
                    ck = (mech, bi, q)
                    if ck not in cache:
                        cache[ck] = measure(mech, b, q, seeds, catalog,
                                            args.evpi_n)
                    refined.append(cache[ck])
            pool = rows + refined
            best, gap = pick_min(pool, target)

            bench = BENCHMARK.get((g, bi))
            dev = None if bench is None else round(target - bench, 6)
            flag = (bench is not None and abs(dev) > BENCH_TOL)
            r_max = max(x["r"] for x in pool)
            unreachable = bool(target > r_max + 1e-12)

            results[f"{g}|{bi}"] = {
                "curiosity_type": mech,
                "coordinator_type": "MATCH",
                "matched_V": g,
                "budget": bi,
                "r_target": target,
                "q": best["q"],
                "r_match_at_q": best["r"],
                "abs_gap": gap,
                "coarse_q": coarse["q"],
                "coarse_r": coarse["r"],
                "r_match_max_on_grid": r_max,
                "target_above_reachable_max": unreachable,
                "n_refine_points": len(refined),
                "seeds": seeds,
                "q_step": args.q_step,
                "refine_step": args.refine_step,
                "refine_window": args.refine_window,
                "evpi_n_used": args.evpi_n,
                "rng_config": {
                    "mechanism": "numpy SeedSequence(seed).spawn(8)",
                    "match_coin_stream": 6,
                    "tie_ol_stream": 3, "tie_bs_stream": 4,
                    "ratio_tie_stream": 5, "evpi_stream": 7,
                    "coin_independent_of_q": True,
                },
                "benchmark_author": bench,
                "target_minus_benchmark": dev,
                "benchmark_deviation_over_tol": bool(flag),
                "git_head": git("rev-parse", "HEAD"),
                "prereg_blob": git("rev-parse", "HEAD:PREREGISTRATION_E2.md"),
                "python": platform.python_version(),
                "numpy": np.__version__, "scipy": scipy.__version__,
            }
            report_rows.append((g, bi, target, bench, dev, best["q"],
                                best["r"], gap, flag, unreachable))

    mono = {}
    for k, rows in curves.items():
        rr = [x["r"] for x in sorted(rows, key=lambda d: d["q"])]
        d = np.diff(rr)
        signs = np.sign(d[np.abs(d) > 1e-12])
        mono[k] = {
            "n_sign_changes": int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0,
            "non_decreasing": bool(np.all(d >= -1e-12)),
            "r_at_q0": rr[0], "r_at_q1": rr[-1],
        }

    RES.mkdir(exist_ok=True)
    (RES / f"{args.tag}_grid.json").write_text(
        json.dumps({"curves": curves, "monotonicity": mono}, indent=2),
        encoding="utf-8")

    if not args.selftest:
        (RES / "q_V.json").write_text(json.dumps(results, indent=2),
                                      encoding="utf-8")

    print("\n" + "=" * 118)
    print(f"{'Группа':<14}{'B':>4}{'r_target':>10}{'эталон':>9}{'откл':>9}"
          f"{'q':>8}{'r_MATCH':>10}{'|gap|':>9}{'допуск':>8}{'недостиж':>10}")
    print("-" * 118)
    for g, bi, tg, bn, dv, q, rm, gap, flag, unreach in report_rows:
        bs = "     n/a" if bn is None else f"{bn:9.2f}"
        ds = "     n/a" if dv is None else f"{dv:+9.4f}"
        print(f"{g:<14}{bi:>4}{tg:10.4f}{bs}{ds}{q:8.3f}{rm:10.4f}"
              f"{gap:9.5f}{'ВНЕ' if flag else 'ок':>8}"
              f"{'ДА' if unreach else 'нет':>10}")
    print("=" * 118)
    print("Монотонность r(q):")
    for k, v in mono.items():
        print(f"   {k}: смен знака={v['n_sign_changes']}, "
              f"неубывающая={v['non_decreasing']}, "
              f"r(0)={v['r_at_q0']:.4f}, r(1)={v['r_at_q1']:.4f}")
    print(f"\nВремя: {(time.time()-t0)/60:.1f} мин")
    if args.selftest:
        print("SELFTEST: q_V.json НЕ записан.")
    else:
        print("Записано: results/q_V.json, results/e2_calib_grid.json")
        print("Этап C не предназначен для интерпретации regret/EVPI.")
        print("q_V.json фиксируется отдельным коммитом до этапа D.")


if __name__ == "__main__":
    main()