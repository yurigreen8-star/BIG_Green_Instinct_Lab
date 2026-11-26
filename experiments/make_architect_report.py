"""
experiments/make_architect_report.py
Компактный отчёт для архитектора из двух summary (budget 30 и 45).
Ничего не хардкодит: все числа читаются из JSON.
Выход: results/architect_report.txt (plain text, ASCII-совместимый).
"""

import json
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"

RUNS = [("budget=30", "v3_main_summary.json"),
        ("budget=45", "v3_b45_summary.json")]

GROUPS = ["CURIOSITY_SOLO", "RATIO_SOLO", "COMBINED", "COMBINED_IGNORE",
          "B_none", "B_random", "B_IG", "B_mean_prec", "B_EVSI",
          "B_EVSI_cost", "B_random_without_replacement"]

SWEEP_TRIPLE = ["CURIOSITY_SOLO", "B_IG", "B_random_without_replacement"]


def load(name):
    p = RES / name
    if not p.exists():
        sys.exit(f"НЕТ ФАЙЛА {p}. Сначала прогони run_v3_main.py с --tag.")
    return json.loads(p.read_text(encoding="utf-8"))


def power_estimate(p_raw, n):
    """Сколько эпизодов нужно, чтобы этот эффект стал значим / набрал 80% мощности."""
    if p_raw <= 0.0 or p_raw >= 1.0:
        return None, None
    z = stats.norm.isf(p_raw / 2.0)
    if z <= 0:
        return None, None
    n05 = n * (1.9600 / z) ** 2
    n80 = n * (2.8016 / z) ** 2
    return int(round(n05)), int(round(n80))


def main():
    L = []
    W = L.append
    W("BIG_Green Instinct Lab — v3.0 E0, отчёт для архитектора")
    W("Статус обоих прогонов: EXPLORATORY (сиды 0-99 использовались в v0.1)")
    W("")

    data = {}
    for label, fname in RUNS:
        d = load(fname)
        data[label] = d
        s = d["summary"]
        W("=" * 100)
        W(f"{label}   n_parcels={d['n_parcels']}  lambda={d['lambda_curiosity']}  "
          f"seeds={d['seed_range']}")
        W("-" * 100)
        W(f"{'Группа':<30}{'regret':>9}{'PCS%':>7}{'probes':>8}{'uniq':>7}"
          f"{'repeat':>8}{'Gini':>7}{'ctr':>7}{'L1':>12}")
        for g in GROUPS:
            if g not in s:
                continue
            r = s[g]
            l1 = r["stepwise_residual_mean"]
            l1s = "        n/a" if l1 is None else f"{l1:12.6f}"
            W(f"{g:<30}{r['mean_regret']:9.4f}{r['pcs_pct']:7.1f}"
              f"{r['mean_probes']:8.2f}{r['mean_unique']:7.2f}"
              f"{r['mean_repeats']:8.2f}{r['mean_gini']:7.3f}"
              f"{r['mean_counter_executed']:7.2f}{l1s}")
        W("")

        # Тождество трёх sweep-политик
        regs = {g: s[g]["mean_regret"] for g in SWEEP_TRIPLE if g in s}
        vals = list(regs.values())
        identical = max(vals) - min(vals) < 1e-12
        W(f"Тройное тождество CURIOSITY_SOLO / B_IG / B_random_without_replacement: "
          f"{'ВЫПОЛНЕНО (редукция доказана)' if identical else 'НАРУШЕНО'}")
        for g, v in regs.items():
            W(f"    {g:<32} {v:.16f}")
        W("")

        # Декомпозиция повторов
        c = s["COMBINED"]
        rep = c["mean_repeats"]
        ctr = c["mean_counter_executed"]
        W(f"COMBINED: повторов всего {rep:.2f} = Рацио {ctr:.2f} + "
          f"Любопытство {rep - ctr:.2f}")
        W(f"COMBINED: proposed={c['mean_counter_proposed']:.2f}, "
          f"executed={ctr:.2f} (отклонено верификацией: "
          f"{c['mean_counter_proposed'] - ctr:.2f})")
        W("")

        # Парные тесты
        W("Regret: парный двусторонний Wilcoxon, COMBINED vs контроли (Holm)")
        for t in d["regret_paired_tests"]:
            favorable = t["delta"] < 0.0
            n05, n80 = power_estimate(t["p_raw"], 100) if favorable else (None, None)
            if not t["significant"] and favorable and n05:
                extra = f"  [для p<0.05 нужно ~{n05} эпизодов, для 80% мощности ~{n80}]"
            elif not t["significant"] and not favorable:
                extra = "  [эффект в неблагоприятную сторону — мощность не считается]"
            else:
                extra = ""
            W(f"    vs {t['control']:<30} delta={t['delta']:+.4f}  "
              f"p_raw={t['p_raw']:.4e}  p_holm={t['p_holm']:.4e}  "
              f"{'ДА' if t['significant'] else 'НЕТ'}{extra}")

        # Контроль бюджета
        b = s.get("B_random_without_replacement")
        if b and abs(b["mean_budget"] - d["budget"]) > 1e-9:
            W(f"ВНИМАНИЕ: B_random_without_replacement тратит {b['mean_budget']:.1f} "
              f"из {d['budget']:.1f} — сравнение НЕ iso-budget.")
            W("")

        W("Факторный контраст (только аддитивные метрики, Y_0 = B_none):")
        for k, v in d["factorial_contrast_additive_only"].items():
            lo, hi = v["ci95"]
            degen = "  [ВЫРОЖДЕН: нулевая ширина CI]" if abs(hi - lo) < 1e-12 else ""
            W(f"    {k:<14} I_CR={v['I_CR']:+10.4f}  CI95 [{lo:.4f}, {hi:.4f}]{degen}")
        W("")

    W("=" * 100)
    W("ОГРАНИЧЕНИЯ И ОТКРЫТЫЕ ВОПРОСЫ")
    W("-" * 100)
    W("1. Оба прогона exploratory: сиды 0-99 уже использовались в v0.1.")
    W("   Подтверждающий прогон требует свежих сидов (300+).")
    W("2. Расписания (фиксированное и случайное чередование C/R) не реализованы.")
    W("   Отличие COMBINED от простого графика не проверено — H-D не закрыта.")
    W("3. Нет контроля 'sweep + случайные повторы'. При budget>n Любопытство")
    W("   стало причинно активным, но его порядок повторов может быть случайной")
    W("   перестановкой в маскировке. Нужен B_sweep_then_random_revisit.")
    W("4. B_random_without_replacement при budget>n не добирает бюджет,")
    W("   поэтому не является iso-budget контролем.")
    W("5. I_CR для budget_used вырожден (все тратят бюджет полностью).")
    W("6. I_CR не применяется к regret: механизмы замещающие, не аддитивные.")
    W("7. mean_diff тождественно stepwise_residual_mean при одной агрегации.")
    W("8. Калибровка EVSI: |L1| мало у всех групп; IG совпадает с изменением")
    W("   энтропии до машинной точности. Тождество L3 = -k*L1 проверено assert'ом.")
    W("")
    W("ГЛАВНЫЙ ВОПРОС К АРХИТЕКТОРУ:")
    W("COMBINED численно лучше B_EVSI_cost при budget=45, но не значимо.")
    W("Запускать подтверждающий прогон на ~1000 свежих сидах, или сначала")
    W("добавить недостающие контроли (расписания + случайные повторы)?")

    out = RES / "architect_report.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nСохранено: {out}  ({len(L)} строк)")


if __name__ == "__main__":
    main()