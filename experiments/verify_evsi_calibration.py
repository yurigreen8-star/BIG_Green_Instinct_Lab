"""
experiments/verify_evsi_calibration.py
Независимая математическая проверка калибровки EVSI и расчёта ΔV.
"""

import sys
from pathlib import Path
import json
import math
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from probes import SURVEY_FAST, SURVEY_STANDARD, SURVEY_PRECISE

PROBE_NOISE = {
    "survey_fast": 1.0,
    "survey_standard": 0.25,
    "survey_precise": 0.04,
}

def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def compute_evsi_independent(mu_before, sigma2_before, parcel_id, noise_var):
    """Независимый расчёт EVSI по формуле из спецификации."""
    mu_k = float(mu_before[parcel_id])
    sigma2_k = float(sigma2_before[parcel_id])
    
    if sigma2_k <= 0.0:
        return 0.0
        
    mu_others = np.delete(mu_before, parcel_id)
    m = float(np.max(mu_others)) if len(mu_others) > 0 else -np.inf
    
    if noise_var == 0.0:
        tau2 = sigma2_k
    else:
        tau2 = (sigma2_k * sigma2_k) / (sigma2_k + noise_var)
        
    if tau2 <= 0.0:
        return 0.0
        
    tau = math.sqrt(tau2)
    d = abs(mu_k - m) / tau
    
    evsi = tau * (phi(d) - d * Phi(-d))
    return max(0.0, float(evsi))

def bootstrap_ci(data, n_boot=2000, alpha=0.05):
    """Кластерный бутстрэп по эпизодам."""
    if len(data) == 0:
        return (0.0, 0.0)
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))

def main():
    # Пытаемся загрузить h2_results_raw.json, fallback на exp_100_scenarios_raw.json
    raw_file = Path(__file__).parent.parent / "results" / "h2_results_raw.json"
    if not raw_file.exists():
        raw_file = Path(__file__).parent.parent / "results" / "exp_100_scenarios_raw.json"
        
    if not raw_file.exists():
        print("❌ Файл с логами не найден.")
        return

    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("🔍 Независимая верификация EVSI и ΔV...\n")
    print("=" * 110)
    print(f"{'Политика':<20} | {'Mean EVSI logged':<16} | {'Mean EVSI recomputed':<18} | {'Mean ΔV':<10} | {'Mean Residual (R)':<18} | {'95% CI'}")
    print("-" * 110)

    results_summary = {}

    for policy_name, policy_data in raw_data.items():
        logs = policy_data.get("logs", [])
        if not logs:
            continue

        episode_residuals = []
        total_evsi_logged = 0.0
        total_evsi_recomp = 0.0
        total_delta_v = 0.0
        n_steps = 0
        n_episodes = len(logs)

        for ep_idx, episode_log in enumerate(logs):
            ep_evsi_logged = 0.0
            ep_evsi_recomp = 0.0
            ep_delta_v = 0.0

            for entry in episode_log:
                if entry.get("event") == "stop":
                    continue
                
                # Извлекаем данные
                parcel_id = entry["parcel_id"]
                probe_type_id = entry["probe_type_id"]
                noise_var = PROBE_NOISE.get(probe_type_id, 1.0)
                
                mu_before = np.array(entry["mu_before"])
                mu_after = np.array(entry["mu_after"])
                
                # 1. EVSI из лога (используем именно analytic_selected, а не best_alternative)
                evsi_logged = entry.get("evsi_analytic_selected", 0.0)
                if evsi_logged is None:
                    evsi_logged = 0.0
                    
                # 2. Независимый пересчёт EVSI
                evsi_recomp = compute_evsi_independent(mu_before, entry["sigma2_before"], parcel_id, noise_var)
                
                # 3. Фактическое ΔV (строго по глобальному максимуму вектора mu)
                delta_v = float(np.max(mu_after) - np.max(mu_before))
                
                ep_evsi_logged += evsi_logged
                ep_evsi_recomp += evsi_recomp
                ep_delta_v += delta_v
                
                total_evsi_logged += evsi_logged
                total_evsi_recomp += evsi_recomp
                total_delta_v += delta_v
                n_steps += 1

            # Остаток на уровне эпизода: R = S_ΔV - S_EVSI
            # (Используем recomputed EVSI для честной проверки математики)
            residual = ep_delta_v - ep_evsi_recomp
            episode_residuals.append(residual)

        mean_evsi_logged = total_evsi_logged / n_steps if n_steps > 0 else 0.0
        mean_evsi_recomp = total_evsi_recomp / n_steps if n_steps > 0 else 0.0
        mean_delta_v = total_delta_v / n_steps if n_steps > 0 else 0.0
        
        mean_r = np.mean(episode_residuals)
        ci_lower, ci_upper = bootstrap_ci(episode_residuals)

        results_summary[policy_name] = {
            "mean_evsi_logged": mean_evsi_logged,
            "mean_evsi_recomp": mean_evsi_recomp,
            "mean_delta_v": mean_delta_v,
            "mean_residual": mean_r,
            "ci_95": (ci_lower, ci_upper),
            "n_episodes": n_episodes,
            "n_steps": n_steps
        }

        print(f"{policy_name:<20} | {mean_evsi_logged:<16.6f} | {mean_evsi_recomp:<18.6f} | {mean_delta_v:<10.6f} | {mean_r:<18.6f} | [{ci_lower:.6f}, {ci_upper:.6f}]")

    print("=" * 110)

    # Сохранение результатов
    out_file = Path(__file__).parent.parent / "results" / "evsi_verification.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n💾 Результаты сохранены в {out_file}")

if __name__ == "__main__":
    main()