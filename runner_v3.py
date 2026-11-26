"""
runner_v3.py — цикл v3.0 (Curiosity / Ratio / Coordinator) + полный event log.
...докстринг...
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from belief import Belief
from environment import Environment
from evsi import evpi_mc, evsi_analytic, ig
from curiosity import Curiosity
from curiosity_ig import CuriosityIG
from messages import (
    MODES,
    bayesian_surprise,
    behavioral_novelty,
    build_curiosity_view,
    predictive_surprise,
)
from probes import Probe
from runner import filter_available

@dataclass
class EpisodeResultV3:
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
    n_curiosity_executed: int
    n_ratio_counter_proposed: int
    n_ratio_counter_executed: int
    n_rejected_no_budget: int
    n_unique_parcels: int
    n_repeats_total: int
    attention_distribution: dict = field(default_factory=dict)


def _json_float(x):
    xf = float(x)
    return None if not np.isfinite(xf) else xf


def run_episode_v3(
    mode: str,
    curiosity,
    ratio,
    coordinator,
    belief0: Belief,
    catalog: list[Probe],
    budget: float,
    env: Environment,
    seed: int,
    experiment_mode: str = "A",
    horizon: int = 10000,
    evpi_n_samples: int = 2000,
    evpi_epsilon: float = 0.05,
    curiosity_type: str = "base",
) -> EpisodeResultV3:
    if mode not in MODES:
        raise ValueError(f"неизвестный режим: {mode}")
    if experiment_mode not in ("A", "B"):
        raise ValueError("experiment_mode должен быть 'A' или 'B'")

    if curiosity_type not in ("base", "ig"):
        raise ValueError("curiosity_type: 'base' | 'ig'")
    if curiosity is None:
        cls = Curiosity if curiosity_type == "base" else CuriosityIG
        curiosity = cls(belief0.n, sigma2_prior=1.0, lam=0.3, seed=seed)
    belief = belief0.copy()
    budget_remaining = float(budget)
    budget_used = 0.0
    rep_counters: dict[tuple[int, str], int] = {}
    attention: Counter = Counter()
    log: list[dict] = []

    rng_curiosity = np.random.default_rng([seed, 20000])
    rng_ratio = np.random.default_rng([seed, 30000])
    rng_evpi = np.random.default_rng([seed, 40000])

    n_curiosity_executed = 0
    n_counter_proposed = 0
    n_counter_executed = 0
    n_rejected = 0
    n_probes = 0
    stop_reason = "horizon_reached"
    evpi_info = None

    for step in range(horizon):
        available = filter_available(belief, catalog, budget_remaining)
        if not available:
            stop_reason = "budget_exhausted" if budget_remaining <= 0.0 else "no_affordable"
            break

        evpi_info = None
        if experiment_mode == "B":
            evpi_info = evpi_mc(belief, rng_evpi, evpi_n_samples)
            if evpi_info["ucb95"] <= evpi_epsilon:
                stop_reason = "evpi_resolved"
                break

        cp = rp = decision = None
        rejects_this_step = 0

        for _attempt in range(2):
            cp = curiosity.propose(belief.known, rng_curiosity)
            rp = ratio.propose(belief, budget_remaining, available, cp, rng_ratio)
            decision = coordinator.coordinate(cp, rp, mode, budget_remaining)

            if decision.action is None:
                break
            if decision.action.cost <= budget_remaining:
                break

            rejects_this_step += 1
            n_rejected += 1
            log.append({
                "step": step,
                "event": "rejected_no_budget",
                "experiment_mode": mode,
                "budget_mode": experiment_mode,
                "proposed_by": "ratio" if decision.reason == "ratio_counter" else "curiosity",
                "proposed_probe": decision.action.name,
                "proposed_cost": float(decision.action.cost),
                "budget_remaining": float(budget_remaining),
            })
            decision = None

        if decision is None or decision.action is None:
            stop_reason = "rejected_no_budget_twice" if rejects_this_step >= 2 else "satiated"
            break

        action = decision.action
        k = int(action.parcel_id)

        if rp is not None and rp.is_counter:
            n_counter_proposed += 1
        if decision.reason == "curiosity":
            n_curiosity_executed += 1
        elif decision.reason == "ratio_counter":
            n_counter_executed += 1

        key = (k, action.probe_type.probe_type_id)
        rep_index = rep_counters.get(key, 0)
        rep_counters[key] = rep_index + 1

        mu_before = belief.mu.copy()
        sigma2_before = belief.sigma2.copy()
        mu_pred = float(belief.mu[k])
        sigma2_pred = float(belief.sigma2[k])

        obs = env.observe(action, rep_index)

        surprise = predictive_surprise(obs.value, mu_pred, sigma2_pred, action.noise_var)
        evsi_executed = float(evsi_analytic(belief, action))
        ig_executed = float(ig(belief, action))
        bs_executed = float(
            bayesian_surprise(obs.value, mu_pred, sigma2_pred, action.noise_var)
        )
        behavioral_novelty_executed = float(behavioral_novelty(rep_index))

        belief = belief.update(k, obs.value, action.noise_var)

        view_after = build_curiosity_view(
            belief.sigma2, belief.known, curiosity.sigma2_prior
        )
        curiosity.update_after_observation(k, view_after, surprise)

        budget_remaining -= float(action.cost)
        budget_used += float(action.cost)
        n_probes += 1
        attention[k] += 1

        delta_v = float(belief.mu.max() - mu_before.max())
        st = curiosity.get_state()

        log.append({
            "step": step,
            "experiment_mode": mode,
            "budget_mode": experiment_mode,
            "curiosity_proposal": (
                None if cp is None
                else {"parcel_id": int(cp.parcel_id), "interest": float(cp.interest)}
            ),
            "ratio_proposal": (
                None if rp is None
                else {
                    "probe": rp.probe.name,
                    "cost": float(rp.cost),
                    "evsi": float(rp.evsi),
                    "is_counter": bool(rp.is_counter),
                    "target_parcel_id": rp.target_parcel_id,
                    "target_cost": rp.target_cost,
                    "target_evsi": rp.target_evsi,
                }
            ),
            "coordinator_decision": {
                "reason": decision.reason,
                "verified_dominance": bool(decision.verified_dominance),
                "executed_probe": action.name,
                "executed_cost": float(action.cost),
            },
            "parcel_id": k,
            "probe_type_id": action.probe_type.probe_type_id,
            "observation": float(obs.value),
            "mu_before": mu_before.tolist(),
            "sigma2_before": sigma2_before.tolist(),
            "mu_after": belief.mu.tolist(),
            "sigma2_after": belief.sigma2.tolist(),
            "delta_posterior_value": delta_v,
            "evsi_analytic_executed": evsi_executed,
            "ig_executed": ig_executed,
            "bs_executed": bs_executed,
            "behavioral_novelty_executed": behavioral_novelty_executed,
            "curiosity_type": curiosity_type,
            "surprise": float(surprise),
            "curiosity_state_after": {
                "interest": st.interest.tolist(),
                "visit_count": st.visit_count.tolist(),
                "last_surprise": [_json_float(x) for x in st.last_surprise],
            },
            "budget_remaining": float(budget_remaining),
            "budget_used_cumulative": float(budget_used),
            "rep_index": rep_index,
            "evpi": evpi_info,
        })

    chosen = int(np.argmax(belief.mu))
    true_best = env.optimal_action()
    regret = float(env.theta_true[true_best] - env.theta_true[chosen])
    n_unique = len(attention)

    log.append({
        "step": len(log),
        "event": "stop",
        "experiment_mode": mode,
        "budget_mode": experiment_mode,
        "stop_reason": stop_reason,
        "budget_used_cumulative": float(budget_used),
        "n_probes": n_probes,
        "final_mu": belief.mu.tolist(),
        "final_sigma2": belief.sigma2.tolist(),
        "final_regret": regret,
        "final_pcs": bool(chosen == true_best),
        "final_evpi": evpi_info,
        "n_curiosity_interventions": n_curiosity_executed,
        "n_ratio_counter_executed": n_counter_executed,
        "n_ratio_counter_proposed": n_counter_proposed,
        "n_rejected_no_budget": n_rejected,
        "n_unique_parcels": n_unique,
        "n_repeats_total": n_probes - n_unique,
        "attention_distribution": dict(attention),
    })

    return EpisodeResultV3(
        belief=belief,
        log=log,
        budget_remaining=budget_remaining,
        budget_used=budget_used,
        n_probes=n_probes,
        chosen_action=chosen,
        true_best=true_best,
        regret=regret,
        pcs=bool(chosen == true_best),
        stop_reason=stop_reason,
        n_curiosity_executed=n_curiosity_executed,
        n_ratio_counter_proposed=n_counter_proposed,
        n_ratio_counter_executed=n_counter_executed,
        n_rejected_no_budget=n_rejected,
        n_unique_parcels=n_unique,
        n_repeats_total=n_probes - n_unique,
        attention_distribution=dict(attention),
    )