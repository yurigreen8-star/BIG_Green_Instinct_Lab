"""
stats_e2.py — единая процедура Holm и парные методы.
Предрегистрация E2 §7.3, §9.5.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def holm(pvals, alpha: float = 0.05):
    """
    Стандартные Holm-adjusted p: монотонны, >= исходных, заполнены для всех.
    Одна функция для всего зарегистрированного семейства.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    adj = np.zeros(m, dtype=np.float64)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj <= alpha, adj


def paired_t_ci(d, alpha: float = 0.05) -> dict:
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n < 2:
        raise ValueError("нужно >= 2 парных наблюдений")
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    return {"n": int(n), "mean": mean, "sd": sd, "se": se,
            "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se,
            "alpha": float(alpha)}


def paired_bootstrap_ci(d, rng: np.random.Generator, n_boot: int = 4000,
                        alpha: float = 0.05) -> dict:
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n < 2:
        raise ValueError("нужно >= 2 парных наблюдений")
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {"ci_low": lo, "ci_high": hi, "n_boot": int(n_boot),
            "n": int(n), "alpha": float(alpha)}


def wilcoxon_paired(d) -> float:
    d = np.asarray(d, dtype=np.float64)
    if np.allclose(d, 0.0):
        return 1.0
    return float(stats.wilcoxon(d, alternative="two-sided").pvalue)


def tost_paired(d, delta: float = 0.01, alpha: float = 0.05) -> dict:
    """
    TOST: H01 mean <= -delta, H02 mean >= delta.
    Эквивалентность при p_tost < alpha (то же, что 90% CI внутри [-delta, delta]).
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n < 2:
        raise ValueError("нужно >= 2 парных наблюдений")
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        p1 = 0.0 if mean > -delta else 1.0
        p2 = 0.0 if mean < delta else 1.0
    else:
        t1 = (mean + delta) / se
        t2 = (mean - delta) / se
        p1 = float(stats.t.sf(t1, n - 1))
        p2 = float(stats.t.cdf(t2, n - 1))
    p = max(p1, p2)
    ci = paired_t_ci(d, alpha=2 * alpha)
    return {"p_tost": p, "p_lower": p1, "p_upper": p2, "delta": float(delta),
            "equivalent": bool(p < alpha),
            "ci90_low": ci["ci_low"], "ci90_high": ci["ci_high"], "n": int(n)}


def gini_counts(counts) -> float:
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * x.sum()))


def entropy_counts(counts) -> float:
    x = np.asarray(counts, dtype=np.float64)
    tot = x.sum()
    if tot <= 0:
        return 0.0
    p = x[x > 0] / tot
    return float(-np.sum(p * np.log(p)))


def spearman_safe(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    res = stats.spearmanr(a, b)
    val = float(res.statistic)
    return None if not np.isfinite(val) else val