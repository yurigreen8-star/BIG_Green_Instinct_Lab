"""
experiments/test_rng_isolation.py — проверка детерминированности.

Один и тот же сид в разных вызовах должен давать побитово идентичный лог.
"""

from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from environment_e2 import make_scenario
from probes import SURVEY_FAST, build_catalog
from runner_e2 import run_episode_e2
from policies_e2 import RandomE2


def main():
    seed = 51000
    sc = make_scenario(seed)
    cat = [p for p in build_catalog(30)
           if p.probe_type.probe_type_id == SURVEY_FAST.probe_type_id]

    for group, policy_cls in [("B_RANDOM", RandomE2)]:
        r1 = run_episode_e2(group, sc, cat, budget=45.0,
                            policy=policy_cls(), evpi_n=200)
        r2 = run_episode_e2(group, sc, cat, budget=45.0,
                            policy=policy_cls(), evpi_n=200)
        same_log = r1.log == r2.log
        same_metrics = r1.metrics["regret"] == r2.metrics["regret"]
        same_parcels = r1.metrics["executed_parcels"] == r2.metrics["executed_parcels"]
        print(f"{group}: log={same_log}, regret={same_metrics}, parcels={same_parcels}")
        if not (same_log and same_metrics and same_parcels):
            raise AssertionError(f"Детерминированность нарушена для {group}")

    print("OK: все проверки детерминированности пройдены")


if __name__ == "__main__":
    main()