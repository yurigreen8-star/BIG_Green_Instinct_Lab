"""
experiments/compare_policies.py
Первый сравнительный эксперимент: 10 сценариев, Режим А (бюджет 30).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from belief import Belief
from environment import Environment
from probes import build_catalog, BUDGET_COMPARISON
from runner import run_episode

from policies.b_none import NonePolicy
from policies.b_random import RandomPolicy
from policies.b_ig import IGPolicy
from policies.b_mean_precision import MeanPrecisionPolicy
from policies.b_evsi import EVSIPolicy


def get_belief0(n: int) -> Belief:
    return Belief(
        mu=np.zeros(n),
        sigma2=np.ones(n),
        known=np.zeros(n, dtype=bool),
    )


def run_experiment(n_scenarios: int = 10, n_parcels: int = 30):
    print(f"🚀 Запуск эксперимента: {n_scenarios} сценариев, {n_parcels} участков, бюджет {BUDGET_COMPARISON}")
    print("-" * 80)

    policies = [
        ("B_none", NonePolicy()),
        ("B_random", RandomPolicy()),
        ("B_IG", IGPolicy()),
        ("B_mean_prec", MeanPrecisionPolicy()),
        ("B_EVSI", EVSIPolicy(use_mc=False)),  # Аналитический EVSI
    ]

    # Хранилище результатов: {policy_name: {metric: [values]}}
    results = {name: {"regret": [], "n_probes": [], "budget_used": [], "pcs": []} 
               for name, _ in policies}

    catalog = build_catalog(n_parcels)

    for seed in range(n_scenarios):
        env = Environment.create(n_parcels=n_parcels, seed=seed)
        belief0 = get_belief0(n_parcels)

        for name, policy in policies:
            res = run_episode(
                policy=policy,
                belief0=belief0,
                catalog=catalog,
                budget=BUDGET_COMPARISON,
                env=env,
                seed=seed,
                mode="A",
            )
            results[name]["regret"].append(res.regret)
            results[name]["n_probes"].append(res.n_probes)
            results[name]["budget_used"].append(res.budget_used)
            results[name]["pcs"].append(1 if res.pcs else 0)

    # Печатаем сводную таблицу
    print(f"{'Политика':<15} | {'Ср. Regret':<10} | {'Ср. Зондов':<10} | {'Ср. Бюджет':<10} | {'PCS (%)':<8}")
    print("-" * 80)

    for name, _ in policies:
        avg_regret = np.mean(results[name]["regret"])
        avg_probes = np.mean(results[name]["n_probes"])
        avg_budget = np.mean(results[name]["budget_used"])
        pcs_pct = 100.0 * np.mean(results[name]["pcs"])

        print(f"{name:<15} | {avg_regret:<10.4f} | {avg_probes:<10.1f} | {avg_budget:<10.1f} | {pcs_pct:<8.1f}")

    print("-" * 80)
    print("✅ Эксперимент завершён.")
    print("💡 Смотри на 'Ср. Regret': чем меньше, тем лучше политика находит истинно лучший участок.")
    print("💡 PCS (%) = Probability of Correct Selection (доля случаев, когда выбран абсолютно лучший участок).")


if __name__ == "__main__":
    run_experiment(n_scenarios=10, n_parcels=30)