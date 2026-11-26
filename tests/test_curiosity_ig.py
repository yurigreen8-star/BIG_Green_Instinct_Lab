"""Шаг 2. C_IG изолирован; траектории BASE vs IG в E0."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

import curiosity_ig as cig_module
from belief import Belief
from coordinator import Coordinator
from curiosity import Curiosity
from curiosity_ig import CuriosityIG
from environment import Environment
from messages import MODE_CURIOSITY_SOLO, build_curiosity_view
from probes import build_catalog
from ratio import Ratio
from runner_v3 import run_episode_v3

N = 30


def _b0():
    return Belief(mu=np.zeros(N), sigma2=np.ones(N), known=np.zeros(N, bool))


def _parcels(cls, seed, budget):
    env = Environment.create(N, seed=seed)
    res = run_episode_v3(
        mode=MODE_CURIOSITY_SOLO,
        curiosity=cls(N, lam=0.3, seed=seed),
        ratio=Ratio(), coordinator=Coordinator(), belief0=_b0(),
        catalog=build_catalog(N), budget=budget, env=env, seed=seed,
        experiment_mode="A",
    )
    return [e["parcel_id"] for e in res.log if e.get("event") is None], res


@pytest.mark.parametrize("budget", [30.0, 45.0])
def test_ciq_same_trajectory_in_e0(budget):
    """
    budget=30: идентичность доказана (непосещённые — неподвижная точка).
    budget=45: равенство A-ранжирования НЕ влечёт равенства EMA-траекторий.
    При расхождении — ЭСКАЛАЦИЯ, не подгонка.
    """
    for seed in range(5):
        pa, ra = _parcels(Curiosity, seed, budget)
        pb, rb = _parcels(CuriosityIG, seed, budget)
        if pa != pb:
            first = next(i for i, (x, y) in enumerate(zip(pa, pb)) if x != y)
            pytest.fail(
                f"РАСХОЖДЕНИЕ budget={budget} seed={seed} шаг={first}: "
                f"base={pa[first]} ig={pb[first]}; "
                f"regret base={ra.regret:.6f} ig={rb.regret:.6f}"
            )


def test_flag_selects_class():
    env = Environment.create(N, seed=1)
    res = run_episode_v3(
        mode=MODE_CURIOSITY_SOLO, curiosity=None, curiosity_type="ig",
        ratio=Ratio(), coordinator=Coordinator(), belief0=_b0(),
        catalog=build_catalog(N), budget=30.0, env=env, seed=1,
        experiment_mode="A",
    )
    steps = [e for e in res.log if e.get("event") is None]
    assert all(e["curiosity_type"] == "ig" for e in steps)
    assert res.n_probes == 30


def test_default_is_base():
    env = Environment.create(N, seed=1)
    res = run_episode_v3(
        mode=MODE_CURIOSITY_SOLO, curiosity=None,
        ratio=Ratio(), coordinator=Coordinator(), belief0=_b0(),
        catalog=build_catalog(N), budget=30.0, env=env, seed=1,
        experiment_mode="A",
    )
    steps = [e for e in res.log if e.get("event") is None]
    assert all(e["curiosity_type"] == "base" for e in steps)


def test_ig_normalization_bounded():
    c = CuriosityIG(N)
    rng = np.random.default_rng(0)
    s2 = np.ones(N)
    for _ in range(40):
        p = c.propose(np.zeros(N, bool), rng)
        k = p.parcel_id
        s2[k] = 1.0 / (1.0 / s2[k] + 1.0)
        c.update_after_observation(k, build_curiosity_view(s2, np.zeros(N, bool)), 0.3)
    st = c.get_state()
    assert np.all(st.interest >= 0.0)
    assert np.all(st.interest <= 1.0 + 1e-12)


def test_ig_no_forbidden_inputs():
    for fn in (CuriosityIG.__init__, CuriosityIG.propose,
               CuriosityIG.update_after_observation):
        names = set(inspect.signature(fn).parameters)
        for bad in ("budget", "budget_remaining", "cost", "available",
                    "catalog", "evsi", "mu", "belief", "theta_true"):
            assert bad not in names

    src = Path(cig_module.__file__).read_text(encoding="utf-8")
    forbidden = {"belief", "evsi", "environment", "runner", "runner_v3", "probes"}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for al in node.names:
                assert al.name.split(".")[0] not in forbidden
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden


def test_ig_excludes_known():
    c = CuriosityIG(3)
    p = c.propose(np.array([True, True, False]), np.random.default_rng(0))
    assert p.parcel_id == 2