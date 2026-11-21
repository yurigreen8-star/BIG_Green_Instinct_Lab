"""
evsi.py — аналитический EVSI, MC-EVSI, IG и EVPI.

Линейная полезность U(a, theta) = q_a.
Один зонд по участку i, гауссовский приор и гауссовский шум.

Аналитическая формула:
    tau^2 = sigma_i^4 / (sigma_i^2 + s^2)   (при s^2=0: tau^2 = sigma_i^2)
    M     = max_{j != i} mu_j
    delta = |mu_i - M|
    z     = delta / tau
    EVSI  = tau * phi(z) - delta * Phi(-z)

Частный случай mu_i = M, sigma_i^2 = s^2 = 1:
    EVSI = 1 / (2 * sqrt(pi)) ≈ 0.28209479
"""

from __future__ import annotations

import math

import numpy as np

from belief import Belief
from probes import Probe

EPS_EXACT = 1e-6  # технический предел для IG при s^2=0, не физическое значение


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def evsi_analytic(belief: Belief, probe: Probe) -> float:
    n = belief.n
    if n <= 1:
        return 0.0
    i = probe.parcel_id
    sig2 = float(belief.sigma2[i])
    if sig2 <= 0.0:
        return 0.0

    mu_i = float(belief.mu[i])
    M = float(np.max(np.delete(belief.mu, i)))
    s2 = float(probe.noise_var)

    tau2 = sig2 if s2 == 0.0 else sig2 * sig2 / (sig2 + s2)
    if tau2 <= 0.0:
        return 0.0

    tau = math.sqrt(tau2)
    delta = abs(mu_i - M)
    z = delta / tau
    value = tau * _phi(z) - delta * _Phi(-z)
    return max(0.0, float(value))


def ig(belief: Belief, probe: Probe) -> float:
    sig2 = float(belief.sigma2[probe.parcel_id])
    if sig2 <= 0.0:
        return 0.0
    s2 = float(probe.noise_var)
    if s2 == 0.0:
        s2 = EPS_EXACT
    return 0.5 * math.log1p(sig2 / s2)


def _sample_belief(belief: Belief, rng: np.random.Generator, n_samples: int) -> np.ndarray:
    return rng.normal(
        loc=belief.mu,
        scale=np.sqrt(belief.sigma2),
        size=(n_samples, belief.n),
    )


def evsi_mc_from_samples(
    belief: Belief,
    probe: Probe,
    theta_samples: np.ndarray,
    omega: np.ndarray,
) -> dict:
    i = probe.parcel_id
    sig2 = float(belief.sigma2[i])
    mu_i = float(belief.mu[i])
    s2 = float(probe.noise_var)

    o = theta_samples[:, i] + math.sqrt(s2) * omega
    if s2 == 0.0:
        mu_post = o
    else:
        mu_post = (sig2 * o + s2 * mu_i) / (sig2 + s2)

    if belief.n > 1:
        M = float(np.max(np.delete(belief.mu, i)))
        base = max(mu_i, M)
    else:
        M = -math.inf
        base = mu_i

    V0 = np.maximum(mu_post, M)
    evsi = float(V0.mean() - base)
    se = float(V0.std(ddof=1) / math.sqrt(len(V0))) if len(V0) > 1 else 0.0
    return {"evsi": evsi, "se": se, "ci95": 1.96 * se}


def compute_evsi_table_mc(
    belief: Belief,
    available: list[Probe],
    rng: np.random.Generator,
    n_samples: int = 2000,
) -> list[dict]:
    theta_samples = _sample_belief(belief, rng, n_samples)
    omega = rng.standard_normal(n_samples)  # общий omega внутри шага
    table = []
    for probe in available:
        d = evsi_mc_from_samples(belief, probe, theta_samples, omega)
        d["probe"] = probe
        d["method"] = "mc"
        table.append(d)
    return table


def compute_evsi_table_analytic(belief: Belief, available: list[Probe]) -> list[dict]:
    return [
        {
            "probe": p,
            "evsi": evsi_analytic(belief, p),
            "se": 0.0,
            "ci95": 0.0,
            "method": "analytic",
        }
        for p in available
    ]


def evpi_mc(belief: Belief, rng: np.random.Generator, n_samples: int = 2000) -> dict:
    theta = _sample_belief(belief, rng, n_samples)
    mx = theta.max(axis=1)
    v0 = float(belief.mu.max())
    mean = float(mx.mean() - v0)
    se = float(mx.std(ddof=1) / math.sqrt(n_samples)) if n_samples > 1 else 0.0
    return {"mean": mean, "se": se, "ucb95": mean + 1.96 * se, "v0": v0}