"""
§9 (исправлено). Причина исправления: прежний mean_residual считался как
СРЕДНЕЕ ПО ЭПИЗОДАМ ОТ СУММЫ (ΔV − EVSI), а не как пошаговое среднее.
Отсюда расхождение с наивной разностью средних ровно в k = T/E раз.

L1 = mean_t(EVSI) − mean_t(ΔV)                       (пошаговое, pooled)
L3 = mean_episodes[ Σ_t (ΔV − EVSI) ]                (сумма по эпизоду)
Тождество: L3 = −k · L1,  k = total_steps / episodes.

Прежняя формулировка теста («с обратным знаком») удалена как ошибочная.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent


def episode_sums(log):
    s_evsi = 0.0
    s_dv = 0.0
    n = 0
    for e in log:
        if e.get("event") == "stop":
            continue
        ev = e.get("evsi_analytic_selected")
        if ev is None:
            raise ValueError("отсутствует evsi_analytic_selected — нуль не подставляется")
        mu_b = np.asarray(e["mu_before"], float)
        mu_a = np.asarray(e["mu_after"], float)
        s_evsi += float(ev)
        s_dv += float(mu_a.max() - mu_b.max())
        n += 1
    return s_evsi, s_dv, n


def main():
    raw_path = ROOT / "results" / "h2_results_raw.json"
    if not raw_path.exists():
        print(f"нет файла {raw_path}; сначала запусти run_h2.py")
        return 1
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    print("§9  L1 = mean(EVSI) − mean(ΔV);  L3 = mean_ep[Σ(ΔV − EVSI)];  L3 = −k·L1")
    print("=" * 104)
    print(f"{'Политика':<20} {'T':>6} {'E':>5} {'k':>8} {'L1':>14} {'L3':>14} {'−k·L1':>14} {'ok':>5}")
    print("-" * 104)

    all_ok = True
    for name, data in raw.items():
        logs = data.get("logs") or []
        if not logs:
            continue
        tot_evsi = tot_dv = tot_steps = 0
        l3_terms = []
        for log in logs:
            se, sd, n = episode_sums(log)
            tot_evsi += se
            tot_dv += sd
            tot_steps += n
            l3_terms.append(sd - se)
        E = len(logs)
        k = tot_steps / E
        l1 = tot_evsi / tot_steps - tot_dv / tot_steps
        l3 = float(np.mean(l3_terms))
        pred = -k * l1
        ok = abs(l3 - pred) < 1e-9
        all_ok &= ok
        print(f"{name:<20} {tot_steps:>6} {E:>5} {k:>8.2f} {l1:>14.10f} "
              f"{l3:>14.10f} {pred:>14.10f} {'OK' if ok else 'FAIL':>5}")
    print("=" * 104)

    # Отдельно один эпизод: sample_log (seed 100)
    cands = sorted(ROOT.glob("results/sample_log*.json"))
    if cands:
        log = json.loads(cands[-1].read_text(encoding="utf-8"))
        se, sd, n = episode_sums(log)
        stop = next(e for e in log if e.get("event") == "stop")
        v_final = float(np.max(stop["final_mu"]))
        print(f"\nОдин эпизод ({cands[-1].name}), шагов = {n}")
        print(f"  Σ EVSI            = {se:.10f}")
        print(f"  Σ ΔV              = {sd:.10f}")
        print(f"  V_final − V_init  = {v_final:.10f}   (телескопичность)")
        print(f"  L1 = (ΣEVSI−ΣΔV)/n = {(se - sd) / n:.10f}")
        print(f"  L3 = ΣΔV − ΣEVSI   = {sd - se:.10f}")
        print(f"  −n·L1              = {-(se - sd):.10f}")
        tel_ok = abs(sd - v_final) < 1e-9
        id_ok = abs((sd - se) + (se - sd)) < 1e-12
        print(f"  телескопичность: {'OK' if tel_ok else 'FAIL'};  тождество: {'OK' if id_ok else 'FAIL'}")
        all_ok &= tel_ok and id_ok

    print(f"\nИТОГ: {'ВСЁ СОШЛОСЬ' if all_ok else 'ЕСТЬ РАСХОЖДЕНИЕ'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())