"""
evpi_e2.py — первичная метрика EVPI(b_T). Предрегистрация E2 §6.1.

EVPI(b_T) = E[max_i q_i | b_T] - max_i E[q_i | b_T]

Monte Carlo, N=40000 по умолчанию, пакетная генерация для ограничения памяти.
Общие MC-выборки для всех политик одного сценария обеспечиваются тем, что RNG
создаётся из потока 7 только от seed сценария и не зависит от числа действий
политики.
"""

from __future__ import annotations

import numpy as np

from belief import Belief

N_SAMPLES_DEFAULT = 40000
BATCH_DEFAULT = 2000


def evpi_mc_e2(belief: Belief, rng: np.random.Generator,
               n_samples: int = N_SAMPLES_DEFAULT,
               batch: int = BATCH_DEFAULT) -> dict:
    """
    Возвращает оценку EVPI, стандартную ошибку MC и число выборок.
    Пакетно: за один раз генерируется не более `batch` строк.
    """
    if n_samples <= 0 or batch <= 0:
        raise ValueError("n_samples и batch должны быть > 0")
    mu = np.asarray(belief.mu, dtype=np.float64)
    s2 = np.asarray(belief.sigma2, dtype=np.float64)
    if mu.ndim != 1 or mu.shape != s2.shape:
        raise ValueError("mu и sigma2 должны быть 1-D одной формы")
    if np.any(s2 < 0.0):
        raise ValueError("sigma2 >= 0")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(s2)):
        raise ValueError("mu и sigma2 конечны")

    sd = np.sqrt(s2)
    base = float(mu.max())
    n = int(mu.shape[0])

    chunks = []
    left = int(n_samples)
    while left > 0:
        b = min(int(batch), left)
        z = rng.standard_normal((b, n))
        q = mu + sd * z
        chunks.append(q.max(axis=1) - base)
        left -= b

    vals = np.concatenate(chunks)
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(vals.size))
    return {"mean": mean, "se": se, "n_samples": int(n_samples)}