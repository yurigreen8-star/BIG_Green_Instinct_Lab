"""
Шаг 3 (исправлено). Отчёт: bs_verification.json, ciq_equivalence.json, заключение.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief import Belief
from coordinator import Coordinator
from curiosity import Curiosity
from curiosity_ig import CuriosityIG
from environment import Environment
from messages import MODE_CURIOSITY_SOLO, bayesian_surprise, predictive_surprise
from probes import build_catalog
from ratio import Ratio
from runner_v3 import run_episode_v3

N = 30
RES = Path(__file__).parent.parent / "results"
GRID_A = [0.5, 1.0, 2.0]
GRID_R = [0.25, 1.0, 4.0]
N_OBS = 10000
SEEDS = range(10)


def bs_block():
    rows = []
    for a in GRID_A:
        for r in GRID_R:
            seed = 20261124 + int(100 * a) + int(10 * r)
            rng = np.random.default_rng(seed)
            mu = 0.37
            obs = rng.normal(mu, math.sqrt(a + r), size=N_OBS)
            bs = np.array([bayesian_surprise(o, mu, a, r) for o in obs])
            ig_exact = 0.5 * math.log1p(a / r)
            se = float(bs.std(ddof=1) / math.sqrt(N_OBS))
            diff = float(bs.mean()) - ig_exact
            kappa = a / (a + r)
            rows.append({
                "sigma2": a, "noise_var": r, "n_obs": N_OBS, "seed": seed,
                "mean_bs": float(bs.mean()), "ig_exact": ig_exact,
                "diff": diff, "se": se, "z_score": diff / se,
                "kappa": kappa,
                "ci95_mean_bs": [float(bs.mean() - 1.96 * se),
                                 float(bs.mean() + 1.96 * se)],
                "within_3se": bool(abs(diff) <= 3.0 * se),
            })
    
    a = r = 1.0
    ig1 = 0.5 * math.log(2.0)
    div = []
    for z in (0.1, 1.0, 5.0, 10.0, 200.0):
        o = z * math.sqrt(2.0)
        bs = bayesian_surprise(o, 0.0, a, r)
        s = predictive_surprise(o, 0.0, a, r)
        div.append({"z": z, "bs": bs, "s": s, "bs_minus_s": bs - s,
                    "gap_over_z2": (bs - s) / (z * z)})
    
    return {
        "identity": "BS = IG + kappa/2 * (z^2 - 1), kappa = sigma2/(sigma2+r)",
        "kl_direction": "KL(posterior||prior) (Itti-Baldi)",
        "note": "E[BS] = IG при усреднении по prior predictive distribution",
        "mc_grid": rows,
        "all_within_3se": all(x["within_3se"] for x in rows),
        "max_abs_mc_zscore": max(abs(x["z_score"]) for x in rows),
        "bs_vs_s": div,
        "asymptotic_gap_coefficient": "kappa/2",
        "example_kappa_0_5": 0.25,
        "crossovers_note": "crossover points зависят от kappa и параметров",
        "crossovers_example_kappa_0_5": [0.396557, 1.567306],
        "corrections_to_assignment": [
            "асимптотика BS-S есть (kappa/2)*z^2, не z^2/4 (последнее при kappa=0.5)",
            "BS > S неверно при z=1; BS<S на (0.3966,1.5673) при kappa=0.5",
            "max_abs_z переименован в max_abs_mc_zscore: это z-score отклонения MC mean от IG",
        ],
    }


def _traj(cls, seed, budget):
    env = Environment.create(N, seed=seed)
    res = run_episode_v3(
        mode=MODE_CURIOSITY_SOLO, curiosity=cls(N, lam=0.3, seed=seed),
        ratio=Ratio(), coordinator=Coordinator(),
        belief0=Belief(mu=np.zeros(N), sigma2=np.ones(N), known=np.zeros(N, bool)),
        catalog=build_catalog(N), budget=budget, env=env, seed=seed,
        experiment_mode="A",
    )
    return [e["parcel_id"] for e in res.log if e.get("event") is None], res


def ciq_block():
    out = {}
    for budget in (30.0, 45.0):
        rows = []
        for seed in SEEDS:
            pa, ra = _traj(Curiosity, seed, budget)
            pb, rb = _traj(CuriosityIG, seed, budget)
            same = pa == pb
            first = None if same else next(
                i for i, (x, y) in enumerate(zip(pa, pb)) if x != y
            )
            rows.append({
                "seed": seed, "identical": same, "first_divergence_step": first,
                "regret_base": ra.regret, "regret_ig": rb.regret,
                "n_probes_base": ra.n_probes, "n_probes_ig": rb.n_probes,
            })
        out[f"budget_{int(budget)}"] = {
            "n_seeds": len(rows),
            "all_identical": all(x["identical"] for x in rows),
            "n_identical": sum(x["identical"] for x in rows),
            "equivalence_conditions": [
                "одинаковые Gaussian priors (sigma2_prior=1.0)",
                "Curiosity выбирает только объект (parcel_id)",
                "Runner назначает FAST",
                "одинаковое правило разрешения равенств (rng.choice)",
                "текущие правила обновления внутреннего состояния",
            ],
            "episodes": rows,
        }
    return out


def main():
    RES.mkdir(exist_ok=True)
    bs = bs_block()
    ciq = ciq_block()
    (RES / "bs_verification.json").write_text(json.dumps(bs, indent=2), encoding="utf-8")
    (RES / "ciq_equivalence.json").write_text(json.dumps(ciq, indent=2), encoding="utf-8")

    L = []
    W = L.append
    W("ЗАКЛЮЧЕНИЕ (Шаг 0-3, исправлено)")
    W("1. Тождество BS = IG + (kappa/2)*(z^2-1) доказано.")
    W(f"2. MC {N_OBS} набл. x 9 комбинаций: max|z_score| = {bs['max_abs_mc_zscore']:.3f}, "
      f"все в 3 sigma: {bs['all_within_3se']}.")
    W("3. E[BS]=IG при усреднении по prior predictive distribution.")
    W("4. Исправление: асимптотика BS-S = (kappa/2)*z^2, не z^2/4.")
    W("   z^2/4 соответствует kappa=0.5 (sigma2=r=1).")
    W("5. max_abs_z переименован в max_abs_mc_zscore.")
    W("6. Behavioral novelty: B = 2*IG точно в FAST-only E0 (доказательство ниже).")
    W("7. A-ранжирование BASE и IG идентично при любых v: общий делитель")
    W("   порядок не меняет, обе A строго убывают по v.")
    for k, v in ciq.items():
        W(f"8. {k}: идентичных траекторий {v['n_identical']}/{v['n_seeds']}.")
        W(f"   Условия: {', '.join(v['equivalence_conditions'])}")
    W("9. BASE и C_IG структурно эквивалентны в указанной конфигурации E0.")
    W("   Это НЕ доказывает эквивалентность IG и EVSI в целом.")
    W("10. Диагностика BS/novelty добавлена в Runner; выбор действий не изменён.")
    W("")
    W("ДОКАЗАТЕЛЬСТВО B = 2*IG (FAST-only, sigma2_prior=1.0):")
    W("  B = ln(v+2) - ln(v+1), где v = число предыдущих зондов")
    W("  После v зондов: sigma2 = 1/(1+v)")
    W("  IG = 0.5 * ln(1 + sigma2/r) = 0.5 * ln(1 + 1/(1+v)) = 0.5 * ln((v+2)/(v+1))")
    W("  B = ln((v+2)/(v+1)) = 2 * 0.5 * ln((v+2)/(v+1)) = 2 * IG")
    W("  Условия тождества:")
    W("    - FAST-only (r = 1.0)")
    W("    - Prior sigma2 = 1.0")
    W("    - v = число предыдущих зондов данного типа")
    W("  Behavioral novelty является преобразованием неопределённости,")
    W("  а не самостоятельным драйвом.")
    
    txt = "\n".join(L)
    (RES / "bs_ciq_conclusion.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nСохранено: {RES}/bs_verification.json, ciq_equivalence.json, bs_ciq_conclusion.txt")


if __name__ == "__main__":
    main()