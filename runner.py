"""
runner.py — общий последовательный цикл.

Режим А: бюджет 30, без самостоятельной остановки по EVSI/EVPI.
Режим Б: бюджет 1000, остановка по UCB95(EVPI) <= 0.05,
         либо бюджет/горизонт/каталог.

Повторы разрешены. rep_counters ведёт Runner.
Environment.observe получает rep_index явно.
Политики получают только Belief, available, budget, свой RNG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from belief import Belief
from environment import Environment
from evsi import evpi_mc
from probes import (
    BUDGET_COMPARISON,
    BUDGET_STOPPING,
    EVPI_UPPER_EPSILON,
    HORIZON_DEFAULT,
    Probe,
    probe_key,
)


def filter_available(belief: Belief, catalog: list[Probe], budget: float) -> list[Probe]:
    return [
        p
        for p in catalog
        if (not belief.known[p.parcel_id]) and p.cost <= budget
    ]


@dataclass
class EpisodeResult:
    belief: Belief
    log: list[dict]
    budget_remaining: float
    budget_used: float
    n_probes: int
    chosen_action: int
    true_best: int
    regret: float
    pcs: bool
    stop_reason: str


def run_episode(
    policy,
    belief0: Belief,
    catalog: list[Probe],
    budget: float,
    env: Environment,
    seed: int = 0,
    mode: str = "A",
    horizon: int = HORIZON_DEFAULT,
    evpi_n_samples: int = 2000,
    evpi_epsilon: float = EVPI_UPPER_EPSILON,
) -> EpisodeResult:
    belief = belief0.copy()
    budget_remaining = float(budget)
    rep_counters: dict[tuple[int, str], int] = {}
    log: list[dict] = []

    rng_policy = np.random.default_rng(seed + 10000)
    rng_evpi = np.random.default_rng(seed + 20000)
    stop_reason = "horizon"

    for step in range(horizon):
        available = filter_available(belief, catalog, budget_remaining)
        if not available:
            stop_reason = "no_available_probes"
            break

        evpi_info = None
        if mode == "B":
            evpi_info = evpi_mc(belief, rng_evpi, evpi_n_samples)
            if evpi_info["ucb95"] <= evpi_epsilon:
                stop_reason = "evpi_resolved"
                log.append({"step": step, "event": "stop", "reason": stop_reason, "evpi": evpi_info})
                break

        probe = policy.select_probe(belief, available, budget_remaining, rng_policy)
        if probe is None:
            stop_reason = "policy_none"
            break

        key = probe_key(probe)
        rep_index = rep_counters.get(key, 0)
        rep_counters[key] = rep_index + 1

        obs = env.observe(probe, rep_index)
        belief = belief.update(probe.parcel_id, obs.value, probe.noise_var)
        budget_remaining -= probe.cost

        entry = {
            "step": step,
            "parcel_id": probe.parcel_id,
            "probe_type_id": probe.probe_type.probe_type_id,
            "probe_name": probe.name,
            "rep_index": rep_index,
            "cost": probe.cost,
            "observation": obs.value,
            "budget_remaining": budget_remaining,
            "mu_after": float(belief.mu[probe.parcel_id]),
            "sigma2_after": float(belief.sigma2[probe.parcel_id]),
        }

        diag = getattr(policy, "last_diagnostics", None)
        if diag is not None:
            entry.update(
                {
                    "evsi": diag["selected_evsi"],
                    "evsi_se": diag["selected_se"],
                    "evsi_ci95": diag["selected_ci95"],
                    "evsi_method": diag["method"],
                    "A7_violated": diag["A7_violated"],
                }
            )

        if evpi_info is not None:
            entry["evpi"] = evpi_info

        log.append(entry)

    chosen = int(np.argmax(belief.mu))
    true_best = env.optimal_action()
    regret = float(env.theta_true[true_best] - env.theta_true[chosen])
    n_probes = sum(1 for e in log if "observation" in e)

    return EpisodeResult(
        belief=belief,
        log=log,
        budget_remaining=budget_remaining,
        budget_used=float(budget) - budget_remaining,
        n_probes=n_probes,
        chosen_action=chosen,
        true_best=true_best,
        regret=regret,
        pcs=(chosen == true_best),
        stop_reason=stop_reason,
    )