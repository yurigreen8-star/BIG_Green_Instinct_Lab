"""
experiments/compare_with_cost.py
Сравнение B_EVSI vs B_EVSI_cost.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from belief import Belief
from environment import Environment
from probes import build_catalog, BUDGET_COMPARISON
from runner import run_episode
from policies.b_evsi import EVSIPolicy, EVSICostPolicy
from policies.b_ig import IGPolicy

def get_belief0(n: int) -> Belief:
    return Belief(
        mu=np.zeros(n),
        sigma2=np.ones(n),
        known=np.zeros(n, dtype=bool),
    )

def run_test(n_scenarios: int = 10, n_parcels: int = 30):
    print(f"🔍 Сравнение: B_EVSI vs B_EVSI_cost vs B_IG")
    print(f"   {n_scenarios} сценариев, {n_parcels} участков, бюджет {BUDGET_COMPARISON}")
    print("-" * 80)

    policies = [
        ("B_EVSI (abs)", EVSIPolicy(use_mc=False)),
        ("B_EVSI_cost", EVSICostPolicy(use_mc=False)),
        ("B_IG", IGPolicy()),
    ]

    results = {name: {"regret": [], "pcs": []} for name, _ in policies}
    catalog = build_catalog(n_parcels)

    for seed in range(n_scenarios):
        env = Environment.create(n_parcels=n_parcels, seed=seed)
        belief0 = get_belief0(n_parcels)
        for name, policy in policies:
            res = run_episode(policy, belief0, catalog, BUDGET_COMPARISON, env, seed, mode="A")
            results[name]["regret"].append(res.regret)
            results[name]["pcs"].append(1 if res.pcs else 0)

    print(f"{'Политика':<20} | {'Ср. Regret':<10} | {'PCS (%)':<8}")
    print("-" * 80)
    for name, _ in policies:
        avg_reg = np.mean(results[name]["regret"])
        pcs_pct = 100.0 * np.mean(results[name]["pcs"])
        print(f"{name:<20} | {avg_reg:<10.4f} | {pcs_pct:<8.1f}")

if __name__ == "__main__":
    run_test()