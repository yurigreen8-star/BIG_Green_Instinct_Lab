"""
experiments/run_e1a.py — эксперимент E1-A.

Среда: неоднородные приоры, только FAST.
Этапы: smoke (5 сидов), разведочный (100 сидов). Подтверждающий — отдельно.
"""

import sys
from pathlib import Path
import json
import argparse
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief import Belief
from coordinator import Coordinator
from curiosity import Curiosity
from curiosity_abs_u import CuriosityAbsU
from environment import Environment
from messages import (
    MODE_COMBINED, MODE_COMBINED_IGNORE, MODE_CURIOSITY_SOLO, MODE_RATIO_SOLO,
    MODE_FIXED_ALT_C_FIRST, MODE_FIXED_ALT_R_FIRST, MODE_RANDOM_ALT_50_50,
)
from policies.b_e1a_fast import IGFastPolicy, EVSIFastPolicy, SweepThenRandomPolicy
from policies.b_random import RandomPolicy
from probes import Probe, ProbeType, build_catalog
from ratio import Ratio
from runner import run_episode
from runner_v3 import run_episode_v3

_ap = argparse.ArgumentParser(description="E1-A: неоднородные приоры, FAST-only.")
_ap.add_argument("--budget", type=float, default=45.0)
_ap.add_argument("--n-seeds", type=int, default=5)
_ap.add_argument("--seed-start", type=int, default=40000)
_ap.add_argument("--tag", type=str, default="e1a_smoke")
_ap.add_argument("--fast-only", action="store_true", default=True)


def fast_catalog(n_parcels: int):
    """Каталог, ограниченный одним типом FAST (экспериментальное ограничение E1-A)."""
    full = build_catalog(n_parcels)
    return [p for p in full if p.probe_type.probe_type_id == "survey_fast"]


def belief0_e1a(env: Environment) -> Belief:
    n = len(env.theta_true)
    return Belief(
        mu=np.zeros(n),
        sigma2=env.sigma2_prior.copy(),
        known=np.zeros(n, dtype=bool),
    )


def main():
    args = _ap.parse_args()
    budget = args.budget
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    tag = args.tag
    N = 30

    print(f"E1-A (seeds {seeds.start}-{seeds.stop - 1}), n={N}, budget={budget}, FAST-only")

    catalog = fast_catalog(N)
    assert all(p.probe_type.probe_type_id == "survey_fast" for p in catalog), \
        "E1-A: каталог должен содержать только FAST"

    v3_modes = {
        "CURIOSITY_SOLO": MODE_CURIOSITY_SOLO,
        "RATIO_SOLO": MODE_RATIO_SOLO,
        "COMBINED": MODE_COMBINED,
        "COMBINED_IGNORE": MODE_COMBINED_IGNORE,
        "FIXED_ALTERNATION_C_FIRST": MODE_FIXED_ALT_C_FIRST,
        "FIXED_ALTERNATION_R_FIRST": MODE_FIXED_ALT_R_FIRST,
        "RANDOM_ALTERNATION_50_50": MODE_RANDOM_ALT_50_50,
    }

    results = {name: [] for name in v3_modes}
    results["B_IG_FAST"] = []
    results["B_EVSI_FAST"] = []
    results["B_RANDOM"] = []
    results["B_SWEEP_THEN_RANDOM"] = []
    results["CURIOSITY_ABS_U"] = []

    for seed in seeds:
        env = Environment.create_e1a(N, seed=seed)
        b0 = belief0_e1a(env)
        sp_vec = env.sigma2_prior.copy()

        # v3-режимы
        for name, mode in v3_modes.items():
            res = run_episode_v3(
                mode=mode,
                curiosity=Curiosity(N, sigma2_prior=sp_vec, lam=0.3, seed=seed),
                ratio=Ratio(),
                coordinator=Coordinator(),
                belief0=b0,
                catalog=catalog,
                budget=budget,
                env=env,
                seed=seed,
                experiment_mode="A",
            )
            results[name].append(_episode_metrics(res.log, env))

        # Абляция
        res = run_episode_v3(
            mode=MODE_CURIOSITY_SOLO,
            curiosity=CuriosityAbsU(N, sigma2_prior=sp_vec, lam=0.3, seed=seed),
            ratio=Ratio(),
            coordinator=Coordinator(),
            belief0=b0,
            catalog=catalog,
            budget=budget,
            env=env,
            seed=seed,
            experiment_mode="A",
        )
        results["CURIOSITY_ABS_U"].append(_episode_metrics(res.log, env))

        # v0.1-совместимые FAST-политики через старый раннер
        for pname, factory in [
            ("B_IG_FAST", IGFastPolicy),
            ("B_EVSI_FAST", EVSIFastPolicy),
            ("B_RANDOM", RandomPolicy),
            ("B_SWEEP_THEN_RANDOM", lambda: SweepThenRandomPolicy()),
        ]:
            res = run_episode(factory(), b0, catalog, budget, env, seed=seed, mode="A")
            results[pname].append(_episode_metrics(res.log, env))

        print(f"   seed {seed}: обработан")

    # Агрегация
    summary = {}
    for name, recs in results.items():
        regrets = np.array([r["regret"] for r in recs])
        summary[name] = {
            "mean_regret": float(regrets.mean()),
            "median_regret": float(np.median(regrets)),
            "sd_regret": float(regrets.std(ddof=1)),
            "mean_pcs": float(np.mean([r["pcs"] for r in recs])),
            "mean_probes": float(np.mean([r["n_steps"] for r in recs])),
            "mean_unique": float(np.mean([r["n_unique"] for r in recs])),
            "mean_repeats": float(np.mean([r["n_repeats"] for r in recs])),
            "n_episodes": len(recs),
        }

    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    (out / f"{tag}_summary.json").write_text(
        json.dumps({"budget": budget, "seed_range": [seeds.start, seeds.stop - 1],
                    "n_parcels": N, "summary": summary}, indent=2),
        encoding="utf-8")
    (out / f"{tag}_raw.json").write_text(
        json.dumps(results, separators=(",", ":")), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"{'Политика':<28}{'regret':>9}{'PCS%':>7}{'probes':>8}{'uniq':>7}{'repeats':>9}")
    print("-" * 90)
    for name, s in summary.items():
        print(f"{name:<28}{s['mean_regret']:9.4f}{100*s['mean_pcs']:7.1f}"
              f"{s['mean_probes']:8.2f}{s['mean_unique']:7.2f}{s['mean_repeats']:9.2f}")
    print("=" * 90)
    print(f"\nresults/{tag}_summary.json, results/{tag}_raw.json")


def _episode_metrics(log, env):
    steps = [e for e in log if e.get("event") is None]
    stop = log[-1]
    att = np.zeros(len(env.theta_true), dtype=int)
    for k, c in stop.get("attention_distribution", {}).items():
        att[int(k)] = int(c)
    return {
        "n_steps": len(steps),
        "regret": float(stop["final_regret"]),
        "pcs": int(bool(stop["final_pcs"])),
        "n_unique": int(stop.get("n_unique_parcels", len(np.nonzero(att)[0]))),
        "n_repeats": int(stop.get("n_repeats_total", 0)),
        "budget_used": float(stop["budget_used_cumulative"]),
        "visits": att.tolist(),
    }


if __name__ == "__main__":
    main()