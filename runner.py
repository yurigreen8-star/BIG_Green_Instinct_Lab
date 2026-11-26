"""
runner.py — общий последовательный цикл с полным когнитивным логированием.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Optional

import numpy as np

from belief import Belief
from environment import Environment, Observation
from evsi import evpi_mc, ig, evsi_analytic
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
    budget_used_cumulative = 0.0
    rep_counters: dict[tuple[int, str], int] = {}
    log: list[dict] = []
    
    policy_name = getattr(policy, "name", "UnknownPolicy")

    rng_policy = np.random.default_rng(seed + 10000)
    rng_evpi = np.random.default_rng(seed + 20000)
    stop_reason = "horizon"

    for step in range(horizon):
        available = filter_available(belief, catalog, budget_remaining)
        if not available:
            stop_reason = "no_affordable_probe"
            break

        evpi_info = None
        if mode == "B":
            evpi_info = evpi_mc(belief, rng_evpi, evpi_n_samples)
            if evpi_info["ucb95"] <= evpi_epsilon:
                stop_reason = "evpi_resolved"
                break

        probe = policy.select_probe(belief, available, budget_remaining, rng_policy)
        if probe is None:
            stop_reason = "policy_none"
            break

        # --- Когнитивные метрики ДО шага ---
        mu_before = belief.mu.tolist()
        sigma2_before = belief.sigma2.tolist()
        
        ig_selected = float(ig(belief, probe))
        evsi_analytic_selected = float(evsi_analytic(belief, probe))
        
        # Контрфактический EVSI-оптимум
        evsi_best_alt_parcel = None
        evsi_best_alt_score = -1.0
        if len(available) > 1:
            best_p = max(available, key=lambda p: evsi_analytic(belief, p))
            evsi_best_alt_parcel = best_p.parcel_id
            evsi_best_alt_score = float(evsi_analytic(belief, best_p))

        # --- Исполнение ---
        key = probe_key(probe)
        rep_index = rep_counters.get(key, 0)
        rep_counters[key] = rep_index + 1

        obs: Observation = env.observe(probe, rep_index)
        
        belief = belief.update(probe.parcel_id, obs.value, probe.noise_var)
        
        cost = probe.cost
        budget_remaining -= cost
        budget_used_cumulative += cost

        # --- Когнитивные метрики ПОСЛЕ шага ---
        mu_after = belief.mu.tolist()
        sigma2_after = belief.sigma2.tolist()

        # Сборка записи шага
        step_entry = {
            "step": step,
            "policy_name": policy_name,
            "mode": mode,
            "parcel_id": probe.parcel_id,
            "probe_type_id": probe.probe_type.probe_type_id,
            "probe_name": probe.name,
            "rep_index": rep_index,
            "cost": cost,
            "observation": obs.value,
            "mu_before": mu_before,
            "sigma2_before": sigma2_before,
            "mu_after": mu_after,
            "sigma2_after": sigma2_after,
            "ig_selected": ig_selected,
            "evsi_analytic_selected": evsi_analytic_selected,
            "evsi_method_used": getattr(policy, "use_mc", False) and "mc" or "analytic",
            "evsi_mc_selected": None,  # Заполняется ниже, если политика предоставила
            "evsi_mc_se": None,
            "evsi_best_alternative_parcel": evsi_best_alt_parcel,
            "evsi_best_alternative_score": evsi_best_alt_score,
            "budget_remaining": budget_remaining,
            "budget_used_cumulative": budget_used_cumulative,
        }

        # Если политика хочет добавить свои MC-диагностики
        diag = getattr(policy, "last_diagnostics", None)
        if diag is not None and diag.get("method") == "mc":
            step_entry["evsi_mc_selected"] = diag.get("selected_evsi")
            step_entry["evsi_mc_se"] = diag.get("selected_se")
            step_entry["A7_violated"] = diag.get("A7_violated")

        if evpi_info is not None:
            step_entry["evpi_mean"] = evpi_info["mean"]
            step_entry["evpi_se"] = evpi_info["se"]
            step_entry["evpi_ucb95"] = evpi_info["ucb95"]

        log.append(step_entry)

    # --- Событие остановки ---
    chosen = int(np.argmax(belief.mu))
    true_best = env.optimal_action()
    regret = float(env.theta_true[true_best] - env.theta_true[chosen])
    pcs = (chosen == true_best)
    
    # Статистика по логу
    parcel_counts = Counter(entry["parcel_id"] for entry in log if "parcel_id" in entry)
    n_unique_parcels = len(parcel_counts)
    n_repeats_total = sum(count - 1 for count in parcel_counts.values() if count > 1)

    stop_entry = {
        "step": len(log),
        "event": "stop",
        "policy_name": policy_name,
        "stop_reason": stop_reason,
        "budget_used_cumulative": budget_used_cumulative,
        "final_mu": belief.mu.tolist(),
        "final_sigma2": belief.sigma2.tolist(),
        "final_regret": regret,
        "final_pcs": pcs,
        "final_evpi": evpi_info if mode == "B" else None,
        "n_unique_parcels": n_unique_parcels,
        "n_repeats_total": n_repeats_total,
        "attention_distribution": dict(parcel_counts),
    }
    log.append(stop_entry)

    n_probes = sum(1 for e in log if "observation" in e)

    return EpisodeResult(
        belief=belief,
        log=log,
        budget_remaining=budget_remaining,
        budget_used=budget_used_cumulative,
        n_probes=n_probes,
        chosen_action=chosen,
        true_best=true_best,
        regret=regret,
        pcs=pcs,
        stop_reason=stop_reason,
    )