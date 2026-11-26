"""
experiments/verify_h2_and_sample.py
1. Генерирует results/sample_log.json (1 сценарий, seed 100)
2. Проверяет агрегаты H2 для B_EVSI_cost (seeds 100-199)
"""
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief import Belief
from environment import Environment
from probes import build_catalog, BUDGET_COMPARISON
from runner import run_episode
from policies.b_evsi import EVSICostPolicy


def get_belief0(n: int) -> Belief:
    return Belief(
        mu=np.zeros(n),
        sigma2=np.ones(n),
        known=np.zeros(n, dtype=bool),
    )


def main():
    n_parcels = 30
    catalog = build_catalog(n_parcels)
    policy = EVSICostPolicy(use_mc=False)
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

        # 1. Генерация sample_log.json с таймстампом
    from datetime import datetime
    
    print("📝 Генерация results/sample_log_<timestamp>.json...")
    env_sample = Environment.create(n_parcels=n_parcels, seed=100)
    belief0 = get_belief0(n_parcels)
    res_sample = run_episode(
        policy, belief0, catalog, BUDGET_COMPARISON, env_sample, seed=100, mode="A"
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"sample_log_{timestamp}.json"
    
    with open(results_dir / log_filename, "w") as f:
        json.dump(res_sample.log, f, indent=1)
    print(f"   Сохранено {len(res_sample.log)} записей в {log_filename}\n")

    # 2. Проверка агрегатов H2
    print("🔍 Проверка агрегатов H2 (B_EVSI_cost, seeds 100-199)...")
    regrets = []
    for seed in range(100, 200):
        env = Environment.create(n_parcels=n_parcels, seed=seed)
        belief0 = get_belief0(n_parcels)
        res = run_episode(
            policy, belief0, catalog, BUDGET_COMPARISON, env, seed=seed, mode="A"
        )
        regrets.append(res.regret)
    
    mean_regret = float(np.mean(regrets))
    print(f"   Получено: {mean_regret:.4f}")
    print(f"   Ожидалось: 0.3999")
    
    if abs(mean_regret - 0.3999) < 0.0001:
        print("   ✅ СОВПАДЕНИЕ! Логирование не внесло сайд-эффектов в математику.")
    else:
        print("   ❌ РАСХОЖДЕНИЕ! Проверь runner.py на наличие багов.")


if __name__ == "__main__":
    main()