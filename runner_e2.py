"""
runner_e2.py — цикл эпизода E2. Предрегистрация §1, §3, §6.

Доступность: все участки на всех шагах, повторы разрешены с первого шага,
known_mask НЕ ограничивает доступ (§1.1). Фильтр только по стоимости.

Три метрики вмешательства (§6.3) считаются РАЗДЕЛЬНО:
  ratio_choice_pct  — арбитр принял решение в пользу предложения Ratio;
  replacement_pct   — исполненное действие != первого предложения Curiosity;
  ratio_action_pct  — исполненный участок совпал с предложением Ratio
                      (может совпадать и тогда, когда арбитр оставил Curiosity).

n_unique_first_n считается по ИСПОЛНЕННЫМ действиям, не по предложениям.

runner_v3.py и runner.py не изменяются.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arbiters_e2 import (EPS_DEFAULT, MODE_MATCH, MODE_RATIO, MODE_SOLO,
                         RHO_DEFAULT, arbitrate)
from belief import Belief
from environment_e2 import (STREAM_EVPI, STREAM_MATCH, STREAM_TIE_BS,
                            STREAM_TIE_OL, STREAM_TIE_RATIO, ScenarioE2,
                            stream_rng)
from evpi_e2 import evpi_mc_e2
from mechanisms_e2 import make_mechanism
from probes import Probe
from ratio_e2 import compute_ratio_table
from stats_e2 import entropy_counts, gini_counts, spearman_safe

GROUPS_E2 = {
    "RATIO_SOLO":            (None, MODE_RATIO),
    "C_OL_SOLO":             ("OL", MODE_SOLO),
    "C_BS_SOLO":             ("BS", MODE_SOLO),
    "C_OL_DOM":              ("OL", "DOM"),
    "C_BS_DOM":              ("BS", "DOM"),
    "C_OL_OVR":              ("OL", "OVR"),
    "C_BS_OVR":              ("BS", "OVR"),
    "C_OL_VETO":             ("OL", "VETO"),
    "C_BS_VETO":             ("BS", "VETO"),
    "MATCH_C_OL_DOM":        ("OL", MODE_MATCH),
    "MATCH_C_BS_DOM":        ("BS", MODE_MATCH),
    "MATCH_C_OL_OVR":        ("OL", MODE_MATCH),
    "MATCH_C_BS_OVR":        ("BS", MODE_MATCH),
    "B_IG_FAST":             (None, "IG"),
    "B_RANDOM":              (None, "RANDOM"),
    "B_SWEEP_THEN_RANDOM":   (None, "SWEEP"),
}

PRIMARY_V = ["C_OL_DOM", "C_BS_DOM", "C_OL_OVR", "C_BS_OVR"]
EXPLORATORY_GROUPS = ["C_OL_VETO", "C_BS_VETO"]
POLICY_MODES = ("IG", "RANDOM", "SWEEP")


@dataclass
class EpisodeE2:
    group: str
    seed: int
    budget: float
    log: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def available_e2(belief: Belief, catalog: list[Probe],
                 budget_remaining: float) -> list[Probe]:
    """Только фильтр по стоимости. known не ограничивает доступ (§1.1)."""
    return [p for p in catalog if p.cost <= budget_remaining]


def run_episode_e2(group: str, scenario: ScenarioE2, catalog: list[Probe],
                   budget: float, mech_kind: str | None = None,
                   arb_mode: str | None = None, q_match: float | None = None,
                   policy=None, eps: float = EPS_DEFAULT,
                   rho: float = RHO_DEFAULT, lam: float = 0.3,
                   evpi_n: int = 40000, first_n: int = 30) -> EpisodeE2:
    if group not in GROUPS_E2:
        raise ValueError(f"неизвестная группа: {group}")
    default_kind, default_mode = GROUPS_E2[group]
    mech_kind = default_kind if mech_kind is None else mech_kind
    arb_mode = default_mode if arb_mode is None else arb_mode

    seed = scenario.seed
    n = scenario.n
    belief = scenario.belief0.copy()
    env = scenario.env
    budget_remaining = float(budget)

    rng_tie = stream_rng(seed, STREAM_TIE_BS if mech_kind == "BS" else STREAM_TIE_OL)
    rng_ratio = stream_rng(seed, STREAM_TIE_RATIO)
    rng_match = stream_rng(seed, STREAM_MATCH)
    rng_evpi = stream_rng(seed, STREAM_EVPI)

    mech = None
    if mech_kind is not None:
        mech = make_mechanism(mech_kind, n, scenario.s_prior, lam=lam)

    by_parcel = {int(p.parcel_id): p for p in catalog}
    rep_counters: dict[int, int] = {}
    visits = np.zeros(n, dtype=np.int64)
    log: list[dict] = []

    n_steps = 0
    n_ratio_choices = 0
    n_replacements = 0
    n_ratio_actions = 0
    n_max_evsi_tiny = 0
    budget_used = 0.0
    executed: list[int] = []

    while True:
        avail = available_e2(belief, catalog, budget_remaining)
        if not avail:
            break

        prop = mech.propose(belief.known, rng_tie) if mech is not None else None
        table = compute_ratio_table(belief, avail, rng_ratio)

        # Бросок MATCH генерируется КАЖДЫЙ шаг во всех группах: поток 6
        # не зависит от режима и числа действий других политик (CRN, §8.2).
        match_u = float(rng_match.random())

        if table.max_evsi < 1e-12:
            n_max_evsi_tiny += 1

        if arb_mode in POLICY_MODES:
            if policy is None:
                raise ValueError("для контрольной политики нужен policy")
            chosen = int(policy.choose(belief, avail, rng_tie))
            chose_ratio = False
            reason = arb_mode.lower()
            replaced = bool(prop is not None and chosen != prop.first)
        elif arb_mode == MODE_RATIO and prop is None:
            # RATIO_SOLO не использует Curiosity: заменa не определена.
            chosen = int(table.best)
            chose_ratio = True
            replaced = False
            reason = "ratio_solo"
        else:
            if prop is None:
                raise ValueError(f"режим {arb_mode} требует Curiosity")
            dec = arbitrate(arb_mode, prop.first, prop.ranking, table,
                            match_u=match_u, q=q_match, eps=eps, rho=rho)
            chosen, chose_ratio, replaced, reason = (
                dec.chosen, dec.chose_ratio, dec.replaced, dec.reason)

        probe = by_parcel[chosen]
        rep_index = rep_counters.get(chosen, 0)
        rep_counters[chosen] = rep_index + 1

        b_before = belief
        mu0 = float(belief.mu[chosen])
        s0 = float(belief.sigma2[chosen])
        obs = env.observe(probe, rep_index)
        belief = belief.update(chosen, obs.value, probe.noise_var)

        if mech is not None:
            mech.after_measurement(chosen, b_before, belief,
                                   float(obs.value), float(probe.noise_var))

        visits[chosen] += 1
        executed.append(chosen)
        n_steps += 1
        budget_used += float(probe.cost)
        budget_remaining -= float(probe.cost)
        if chose_ratio:
            n_ratio_choices += 1
        if replaced:
            n_replacements += 1
        if chosen == int(table.best):
            n_ratio_actions += 1

        log.append({
            "step": n_steps - 1,
            "parcel_id": chosen,
            "rep_index": rep_index,
            "curiosity_first": None if prop is None else prop.first,
            "curiosity_ranking": None if prop is None else list(prop.ranking),
            "ratio_best": int(table.best),
            "evsi_first": None if prop is None else table.evsi_of(prop.first),
            "evsi_best": table.best_evsi,
            "max_evsi": table.max_evsi,
            "match_u": match_u,
            "chose_ratio": bool(chose_ratio),
            "replaced": bool(replaced),
            "executed_equals_ratio_best": bool(chosen == int(table.best)),
            "reason": reason,
            "observation": float(obs.value),
            "mu_before": mu0, "sigma2_before": s0,
            "mu_after": float(belief.mu[chosen]),
            "sigma2_after": float(belief.sigma2[chosen]),
            "budget_remaining": float(budget_remaining),
        })

    chosen_final = int(np.argmax(belief.mu))
    true_best = env.optimal_action()
    regret = float(env.theta_true[true_best] - env.theta_true[chosen_final])
    evpi = evpi_mc_e2(belief, rng_evpi, n_samples=evpi_n)
    repeats = np.maximum(visits - 1, 0)
    denom = max(n_steps, 1)

    metrics = {
        "group": group, "seed": int(seed), "budget": float(budget),
        "regret": regret,
        "pcs": int(chosen_final == true_best),
        "budget_used": budget_used,
        "n_probes": n_steps,
        "n_unique": int(np.count_nonzero(visits)),
        "n_repeats": int(n_steps - np.count_nonzero(visits)),
        "n_unique_first_n": len(set(executed[:first_n])),
        "first_n": int(first_n),
        "executed_parcels": executed,
        "visits": visits.tolist(),
        "repeats": repeats.tolist(),
        "entropy_visits": entropy_counts(visits),
        "gini_visits": gini_counts(visits),
        "spearman_s_repeats": spearman_safe(scenario.s_prior, repeats),
        "frac_max_evsi_tiny_pct": 100.0 * n_max_evsi_tiny / denom,
        "n_ratio_choices": n_ratio_choices,
        "n_replacements": n_replacements,
        "n_ratio_actions": n_ratio_actions,
        "ratio_choice_pct": 100.0 * n_ratio_choices / denom,
        "replacement_pct": 100.0 * n_replacements / denom,
        "ratio_action_pct": 100.0 * n_ratio_actions / denom,
        "evpi": evpi,
        "final_mu": belief.mu.tolist(),
        "final_sigma2": belief.sigma2.tolist(),
    }
    return EpisodeE2(group=group, seed=int(seed), budget=float(budget),
                     log=log, metrics=metrics)