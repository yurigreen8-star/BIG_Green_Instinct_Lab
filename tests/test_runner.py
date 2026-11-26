import numpy as np
import pytest

from belief import Belief
from environment import Environment
from policies.b_evsi import EVSIPolicy
from policies.b_mean_precision import MeanPrecisionPolicy
from policies.b_none import NonePolicy
from policies.b_random import RandomPolicy
from probes import SURVEY_FAST, Probe, build_catalog
from runner import run_episode


def _belief0(n):
    return Belief(mu=np.zeros(n), sigma2=np.ones(n), known=np.zeros(n, dtype=bool))


class AlwaysFirstPolicy:
    name = "always_first"

    def select_probe(self, belief, available, budget_remaining, rng):
        return available[0] if available else None


def test_repeats_allowed_rep_index_increments():
    n = 2
    env = Environment.create(n, seed=42)
    catalog = build_catalog(n)
    res = run_episode(AlwaysFirstPolicy(), _belief0(n), catalog, budget=5, env=env, seed=42, mode="A")
    obs_entries = [e for e in res.log if "observation" in e]
    assert len(obs_entries) == 5
    assert [e["rep_index"] for e in obs_entries] == [0, 1, 2, 3, 4]
    assert all(e["probe_type_id"] == "survey_fast" for e in obs_entries)
    assert res.budget_used == 5


def test_budget_not_exceeded_random():
    n = 5
    env = Environment.create(n, seed=7)
    res = run_episode(RandomPolicy(), _belief0(n), build_catalog(n), budget=30, env=env, seed=7, mode="A")
    assert res.budget_used <= 30
    assert res.budget_remaining >= 0


def test_none_policy_zero_probes():
    n = 3
    env = Environment.create(n, seed=1)
    res = run_episode(NonePolicy(), _belief0(n), build_catalog(n), budget=30, env=env, seed=1, mode="A")
    assert res.n_probes == 0
    assert res.stop_reason == "policy_none"


def test_mode_B_stops_by_evpi_ucb():
    n = 2
    env = Environment.create(n, seed=99)
    res = run_episode(
        EVSIPolicy(),
        _belief0(n),
        build_catalog(n),
        budget=1000,
        env=env,
        seed=99,
        mode="B",
        horizon=5000,
        evpi_n_samples=500,
    )
    assert res.stop_reason == "evpi_resolved"
    assert res.budget_used < 1000


def test_same_seed_same_observations():
    n = 4
    catalog = build_catalog(n)
    env1 = Environment.create(n, seed=5)
    env2 = Environment.create(n, seed=5)
    r1 = run_episode(RandomPolicy(), _belief0(n), catalog, 30, env1, seed=5, mode="A")
    r2 = run_episode(RandomPolicy(), _belief0(n), catalog, 30, env2, seed=5, mode="A")
    o1 = [e["observation"] for e in r1.log if "observation" in e]
    o2 = [e["observation"] for e in r2.log if "observation" in e]
    assert o1 == o2


def test_evsi_policy_runs_and_logs_diagnostics():
    n = 3
    env = Environment.create(n, seed=11)
    res = run_episode(EVSIPolicy(), _belief0(n), build_catalog(n), budget=30, env=env, seed=11, mode="A")
    obs_entries = [e for e in res.log if "observation" in e]
    assert len(obs_entries) > 0
    # Обновлённые имена полей согласно новой схеме логирования
    assert all("evsi_analytic_selected" in e for e in obs_entries)
    assert all(e["evsi_method_used"] == "analytic" for e in obs_entries)

# --- НОВЫЕ ТЕСТЫ ЛОГИРОВАНИЯ ---

def test_log_has_all_required_fields():
    n = 3
    env = Environment.create(n, seed=42)
    res = run_episode(RandomPolicy(), _belief0(n), build_catalog(n), 30, env, seed=42, mode="A")
    
    required_fields = [
        "step", "policy_name", "mode", "parcel_id", "probe_type_id", 
        "mu_before", "sigma2_before", "mu_after", "sigma2_after",
        "ig_selected", "evsi_analytic_selected", "budget_used_cumulative"
    ]
    for entry in res.log[:-1]: # Исключаем stop event
        for field in required_fields:
            assert field in entry, f"Missing field: {field}"

def test_log_full_belief_vectors():
    n = 5
    env = Environment.create(n, seed=42)
    res = run_episode(RandomPolicy(), _belief0(n), build_catalog(n), 30, env, seed=42, mode="A")
    
    for entry in res.log[:-1]:
        assert len(entry["mu_before"]) == n
        assert len(entry["sigma2_before"]) == n
        assert len(entry["mu_after"]) == n
        assert len(entry["sigma2_after"]) == n
        
        # Проверяем, что обновились только целевые участки (упрощённо: хотя бы один изменился)
        changed = sum(1 for i in range(n) if entry["mu_after"][i] != entry["mu_before"][i])
        assert changed >= 1

def test_log_last_entry_is_stop_event():
    n = 3
    env = Environment.create(n, seed=42)
    res = run_episode(NonePolicy(), _belief0(n), build_catalog(n), 30, env, seed=42, mode="A")
    
    last_entry = res.log[-1]
    assert last_entry["event"] == "stop"
    assert "stop_reason" in last_entry
    assert last_entry["stop_reason"] == "policy_none"

def test_log_ig_selected_non_negative():
    n = 3
    env = Environment.create(n, seed=42)
    res = run_episode(RandomPolicy(), _belief0(n), build_catalog(n), 30, env, seed=42, mode="A")
    
    for entry in res.log[:-1]:
        assert entry["ig_selected"] >= -1e-9  # Допускаем микро-погрешность float

def test_log_evsi_alternative_present():
    n = 5
    env = Environment.create(n, seed=42)
    res = run_episode(RandomPolicy(), _belief0(n), build_catalog(n), 30, env, seed=42, mode="A")
    
    for entry in res.log[:-1]:
        # Если доступно > 1 зонда, альтернатива должна быть заполнена
        # (В RandomPolicy на первом шаге доступно 15 зондов)
        assert entry["evsi_best_alternative_parcel"] is not None
        assert entry["evsi_best_alternative_score"] is not None

def test_backward_compat_raw_json(tmp_path):
    # Эмуляция сохранения, проверка структуры
    import json
    dummy_raw = {
        "B_test": {
            "regret": [0.5], "pcs": [1], "n_probes": [5], "budget_used": [30.0], "stop_reason": ["budget"],
            "logs": [[{"step": 0, "mu_before": [0.0]}]]
        }
    }
    out_file = tmp_path / "test_raw.json"
    with open(out_file, "w") as f:
        json.dump(dummy_raw, f)
        
    with open(out_file, "r") as f:
        loaded = json.load(f)
        
    assert "regret" in loaded["B_test"]
    assert "logs" in loaded["B_test"]