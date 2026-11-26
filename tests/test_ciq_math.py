"""
Шаг 0. Четыре математических теста ДО кода управления.

Отклонения от поручения, зафиксированные с доказательством:
  * асимптотика BS - S есть z^2/4, а не z^2/2 (последнее соответствует
    обратной KL, у которой E != IG, что сломало бы test_bs_equals_ig);
  * BS > S неверно на (0.3966, 1.5673), в частности при z = 1.
"""

import math

import numpy as np
import pytest

from messages import bayesian_surprise, behavioral_novelty, predictive_surprise

V_SETS = [
    (0, 0, 0, 0),
    (0, 1, 2, 3),
    (1, 1, 2, 5),
    (0, 1, 2, 3, 5, 10),
    (10, 5, 3, 2, 1, 0),
    (2, 2, 2, 10),
    (0, 10),
]


def _A_legacy(v):
    v = np.asarray(v, dtype=float)
    n = 1.0 / (1.0 + v)
    return n * n


def _A_ig(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = 1.0 / (1.0 + v)
    info = 0.5 * np.log1p(n)
    return (info / (float(info.max()) + eps)) * n


def _order(a):
    return list(np.argsort(-np.asarray(a), kind="stable"))


@pytest.mark.parametrize("v", V_SETS)
def test_ciq_equivalence_in_e0(v):
    """Обе A строго убывают по v; общий делитель порядок не меняет."""
    al, ai = _A_legacy(v), _A_ig(v)
    assert _order(al) == _order(ai)
    vv = np.asarray(v, dtype=float)
    assert np.all(np.diff(al[np.argsort(vv)]) <= 1e-15)
    assert np.all(np.diff(ai[np.argsort(vv)]) <= 1e-15)


def test_ciq_equivalence_random_states():
    rng = np.random.default_rng(20261124)
    for _ in range(200):
        v = rng.integers(0, 12, size=int(rng.integers(2, 31)))
        assert _order(_A_legacy(v)) == _order(_A_ig(v))


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("r", [0.25, 1.0, 4.0])
def test_bs_equals_ig(a, r):
    """
    E[BS] = IG точно. MC проверяет E[z^2]=1.
    Внимание: 9 сравнений при пороге 3 sigma дают ~2.4% семейной вероятности
    ложного падения. При падении смотреть z в bs_verification.json,
    а не считать это сразу ошибкой математики.
    """
    n = 10000
    seed = 20261124 + int(100 * a) + int(10 * r)
    rng = np.random.default_rng(seed)
    mu = 0.37
    obs = rng.normal(mu, math.sqrt(a + r), size=n)
    bs = np.array([bayesian_surprise(o, mu, a, r) for o in obs])
    ig_exact = 0.5 * math.log1p(a / r)
    se = float(bs.std(ddof=1) / math.sqrt(n))
    z = (float(bs.mean()) - ig_exact) / se
    assert abs(z) <= 3.0, f"a={a} r={r} z={z:.3f}"


def test_bs_closed_form_identity():
    """BS = IG + a/(2(a+r))*(z^2-1) — независимая сверка с прямой KL."""
    rng = np.random.default_rng(7)
    for a in (0.5, 1.0, 2.0):
        for r in (0.25, 1.0, 4.0):
            ig_exact = 0.5 * math.log1p(a / r)
            for _ in range(50):
                z = float(rng.normal())
                o = 0.0 + z * math.sqrt(a + r)
                direct = bayesian_surprise(o, 0.0, a, r)
                closed = ig_exact + a / (2.0 * (a + r)) * (z * z - 1.0)
                assert direct == pytest.approx(closed, abs=1e-12)


def test_bs_vs_s_divergence():
    """Исправлено: асимптотика z^2/4; кроссоверы 0.396557 и 1.567306."""
    a = r = 1.0
    ig_exact = 0.5 * math.log(2.0)
    const = ig_exact - 0.25

    for z in (0.1, 1.0, 5.0, 10.0):
        o = z * math.sqrt(a + r)
        bs = bayesian_surprise(o, 0.0, a, r)
        s = predictive_surprise(o, 0.0, a, r)
        assert bs == pytest.approx(0.25 * z * z + const, abs=1e-12)
        assert s == pytest.approx(z * z / (1.0 + z * z), abs=1e-12)

    assert bayesian_surprise(0.1 * math.sqrt(2), 0, a, r) > predictive_surprise(0.1 * math.sqrt(2), 0, a, r)
    assert bayesian_surprise(1.0 * math.sqrt(2), 0, a, r) < predictive_surprise(1.0 * math.sqrt(2), 0, a, r)
    for z in (5.0, 10.0):
        assert bayesian_surprise(z * math.sqrt(2), 0, a, r) > predictive_surprise(z * math.sqrt(2), 0, a, r)

    for z_star in (0.396557, 1.567306):
        o = z_star * math.sqrt(2)
        assert bayesian_surprise(o, 0, a, r) == pytest.approx(
            predictive_surprise(o, 0, a, r), abs=1e-5
        )

    z = 200.0
    o = z * math.sqrt(2)
    gap = bayesian_surprise(o, 0, a, r) - predictive_surprise(o, 0, a, r)
    assert gap / (z * z) == pytest.approx(0.25, abs=1e-4)


@pytest.mark.parametrize("v", [0, 1, 2, 3, 5, 10])
def test_behavioral_novelty_degenerates(v):
    """B = ln(v+2)-ln(v+1) и B = 2*IG_fast точно => не независимый драйв."""
    b = behavioral_novelty(v)
    assert b == pytest.approx(math.log(v + 2) - math.log(v + 1), abs=1e-15)
    ig_fast = 0.5 * math.log1p(1.0 / (1.0 + v))
    assert b == pytest.approx(2.0 * ig_fast, abs=1e-12)