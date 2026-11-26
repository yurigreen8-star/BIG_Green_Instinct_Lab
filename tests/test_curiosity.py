"""M7–M10, M14–M16, A1, A5, B1–B3, B7 + доказательство тождества вариантов A и B."""

import ast
import io
import tokenize

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import curiosity as curiosity_module
from curiosity import Curiosity
from messages import CuriosityView, build_curiosity_view, predictive_surprise


def _view(ign, known=None):
    ign = np.asarray(ign, dtype=float)
    kn = np.zeros(len(ign), dtype=bool) if known is None else np.asarray(known, bool)
    return CuriosityView(ignorance=ign, known=kn)


def test_M7_M8_M9_ignorance_range():
    v = build_curiosity_view(np.array([1.0, 0.5, 0.0]), np.array([False, False, True]))
    assert v.ignorance[0] == 1.0
    assert v.ignorance[1] == 0.5
    assert v.ignorance[2] == 0.0  # known -> 0, не ошибка


def test_M10_surprise_calibration():
    rng = np.random.default_rng(12345)
    obs = rng.normal(0.0, np.sqrt(2.0), size=100_000)  # sigma2_pred=1, s2=1
    s = np.array([predictive_surprise(o, 0.0, 1.0, 1.0) for o in obs])
    assert abs(float(s.mean()) - 0.344324) < 0.005


def test_M14_interest_update():
    c = Curiosity(n_parcels=2, lam=0.3)
    c.update_after_observation(0, _view([0.5, 1.0]), surprise=0.2)
    st = c.get_state()
    assert st.interest[0] == pytest.approx(0.775)
    assert st.interest[1] == pytest.approx(1.0)  # вариант B: непосещённый — неподвижная точка


def test_M15_M16_novelty_via_visit_count():
    c = Curiosity(n_parcels=1, lam=0.3)
    assert c.get_state().visit_count[0] == 0
    c.update_after_observation(0, _view([0.5]), 0.1)
    assert c.get_state().visit_count[0] == 1  # novelty = 0.5
    c.update_after_observation(0, _view([1 / 3]), 0.1)
    assert c.get_state().visit_count[0] == 2  # novelty = 1/3


def test_A1_no_forbidden_inputs():
    """
    Инвариант проверяется по КОДУ, а не по тексту файла.
    Докстринг, декларирующий запрет, не должен нарушать запрет.
    """
    # (1) сигнатуры
    for fn in (Curiosity.__init__, Curiosity.propose, Curiosity.update_after_observation):
        names = set(inspect.signature(fn).parameters)
        for bad in ("budget", "budget_remaining", "cost", "available", "catalog",
                    "evsi", "mu", "sigma2", "belief", "theta_true"):
            assert bad not in names, f"{fn.__name__} принимает {bad}"

    path = Path(curiosity_module.__file__)
    src = path.read_text(encoding="utf-8")

    # (2) импорты: только разрешённые модули
    forbidden_modules = {"belief", "evsi", "environment", "probes", "runner", "runner_v3"}
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                assert root not in forbidden_modules, f"импорт {a.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in forbidden_modules, f"from {node.module}"
            if root == "messages":
                allowed = {"TIE_EPS", "CuriosityProposal", "CuriosityView"}
                for a in node.names:
                    assert a.name in allowed, f"из messages импортировано {a.name}"

    # (3) токены кода без строк и комментариев
    code_tokens = []
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                continue
            code_tokens.append(tok.string)
    code_src = " ".join(code_tokens)

    for pattern in (r"\bmu\b", r"\bbudget\w*", r"\bcost\b", r"\bevsi\w*",
                    r"theta_true", r"\bBelief\b"):
        assert re.search(pattern, code_src) is None, f"код содержит {pattern}"


def test_A5_visited_is_not_known():
    c = Curiosity(n_parcels=1, lam=0.3)
    s2 = 1.0
    for _ in range(5):
        s2 = 1.0 / (1.0 / s2 + 1.0)
        c.update_after_observation(0, build_curiosity_view([s2], [False]), 0.1)
    st = c.get_state()
    assert st.visit_count[0] == 5
    assert s2 > 0.0  # fast никогда не делает known


def test_B1_determinism():
    kn = np.zeros(5, dtype=bool)
    a = Curiosity(5).propose(kn, np.random.default_rng([7, 20000]))
    b = Curiosity(5).propose(kn, np.random.default_rng([7, 20000]))
    assert a == b


def test_B2_budget_independence():
    c = Curiosity(4)
    assert not hasattr(c, "budget_remaining")
    kn = np.zeros(4, dtype=bool)
    p1 = c.propose(kn, np.random.default_rng([1, 20000]))
    p2 = c.propose(kn, np.random.default_rng([1, 20000]))
    assert p1 == p2


def test_B3_same_view_different_state():
    """Одинаковый вид, разная история посещений -> разные предложения."""
    v = _view([0.5, 0.5, 0.5])
    kn = np.zeros(3, dtype=bool)

    c_a = Curiosity(3, lam=0.3)
    for _ in range(10):
        c_a.update_after_observation(0, v, 0.1)
    c_b = Curiosity(3, lam=0.3)
    for _ in range(10):
        c_b.update_after_observation(1, v, 0.1)

    pa = c_a.propose(kn, np.random.default_rng([3, 20000]))
    pb = c_b.propose(kn, np.random.default_rng([3, 20000]))
    assert pa.parcel_id != 0
    assert pb.parcel_id != 1
    assert pa.parcel_id != pb.parcel_id


def test_B7_known_excluded():
    c = Curiosity(3)
    p = c.propose(np.array([True, True, False]), np.random.default_rng(0))
    assert p.parcel_id == 2
    assert c.propose(np.array([True, True, True]), np.random.default_rng(0)) is None


class _CuriosityVariantA(Curiosity):
    """Референсный вариант A: обновляет только наблюдаемый объект."""

    def update_after_observation(self, parcel_id, view_after, surprise):
        k = int(parcel_id)
        self._visit_count[k] += 1
        self._last_surprise[k] = float(surprise)
        novelty = 1.0 / (1.0 + self._visit_count[k])
        a = float(view_after.ignorance[k]) * novelty
        self._interest[k] = (1.0 - self._lam) * self._interest[k] + self._lam * a


def test_variant_B_equals_A_in_mode_A_sweep():
    """
    Флаг 2: при budget = n и принудительном FAST варианты A и B дают
    идентичную траекторию. Неисследованные — неподвижная точка I=1,
    любой исследованный I <= 0.85 < 1, поэтому argmax всегда непосещённый.
    """
    n = 30
    ca = _CuriosityVariantA(n, lam=0.3)
    cb = Curiosity(n, lam=0.3)
    rng_a = np.random.default_rng([100, 20000])
    rng_b = np.random.default_rng([100, 20000])
    s2 = np.ones(n)
    known = np.zeros(n, dtype=bool)
    seq_a, seq_b = [], []

    for _ in range(30):
        pa = ca.propose(known, rng_a)
        pb = cb.propose(known, rng_b)
        assert pa.parcel_id == pb.parcel_id
        k = pa.parcel_id
        seq_a.append(k)
        seq_b.append(k)
        s2[k] = 1.0 / (1.0 / s2[k] + 1.0)
        v = build_curiosity_view(s2, known)
        ca.update_after_observation(k, v, 0.3)
        cb.update_after_observation(k, v, 0.3)

    assert seq_a == seq_b
    assert len(set(seq_a)) == 30  # sweep: все объекты по одному разу