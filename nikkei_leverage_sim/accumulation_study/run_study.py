"""Build the accumulation×exit grid, run it on real 1570.T, write CSV + report.

Usage (from ``nikkei_leverage_sim/``)::

    python -m accumulation_study.run_study \
        --prices outputs_real/daily.csv --out accumulation_study/outputs

Outputs:
* ``rows.csv``       — every plan with full metrics (sorted by Calmar desc)
* ``REPORT_ACCUMULATION.md`` (written to the package dir) — headline tables,
  sub-window Calmar, and the honesty caveats.

The grid is *pre-registered* here (coarse, interpretable parameters) so the
study is not a nested over-fit; see the report's caveats section.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .engine import PlanMetrics, evaluate, month_start_indices, simulate
from .policies import (
    BuyTheDip,
    FixedDCA,
    GlideExit,
    HoldToEnd,
    LumpSum,
    MAExit,
    MomentumAllIn,
    MomentumExit,
    ScaledDCA,
    TakeProfit,
    TrailingStop,
    TrendFilterDCA,
    ValueAveraging,
    VolSpikeExit,
)
from .signals import build_signals

CAPITAL = 10_000_000.0
TRADING_DAYS_PER_MONTH = 21


@dataclass
class Plan:
    label: str
    accumulation: Callable
    exit: Callable
    cadence: str          # "monthly" | "daily"
    repeated: bool
    family: str           # accumulation family for grouping
    note: str = ""


def _dca_installments(cadence: str, horizon_months: int) -> int:
    return horizon_months if cadence == "monthly" else horizon_months * TRADING_DAYS_PER_MONTH


def _window_installments(cadence: str, n_days: int, n_months: int) -> int:
    return n_months if cadence == "monthly" else n_days


def build_grid(n_days: int, n_months: int) -> List[Plan]:
    """Pre-registered grid. ~200 plans: 12 accumulation × exit variants (monthly)
    plus a small daily cadence-sensitivity set and the dual-momentum control."""
    plans: List[Plan] = []

    def accel(cad: str):
        # Accumulation factories keyed by family; each returns a *fresh* object.
        full = _window_installments(cad, n_days, n_months)
        return {
            "lump": (lambda: LumpSum(), "lump"),
            "dca24": (lambda: FixedDCA(_dca_installments(cad, 24)), "dca"),
            "dca48": (lambda: FixedDCA(_dca_installments(cad, 48)), "dca"),
            "dca96": (lambda: FixedDCA(_dca_installments(cad, 96)), "dca"),
            "dipA": (lambda: BuyTheDip([-0.10, -0.20, -0.30], [1 / 3, 1 / 3, 1 / 3],
                                      _dca_installments(cad, 36)), "dip"),
            "dipB": (lambda: BuyTheDip([-0.15, -0.30, -0.45], [1 / 3, 1 / 3, 1 / 3],
                                      _dca_installments(cad, 36)), "dip"),
            "va48_0": (lambda: ValueAveraging(48, 0.0, 12 if cad == "monthly"
                                              else 252), "value_avg"),
            "va96_6": (lambda: ValueAveraging(96, 0.06, 12 if cad == "monthly"
                                              else 252), "value_avg"),
            "trend200": (lambda: TrendFilterDCA(200, full, True), "trend"),
            "trend100": (lambda: TrendFilterDCA(100, full, True), "trend"),
            "scaled_rsi": (lambda: ScaledDCA("rsi", full), "scaled"),
            "scaled_invvol": (lambda: ScaledDCA("invvol", full), "scaled"),
        }

    # Exit variants: (label, factory, [semantics]) — semantics restricted per agent.
    def exit_variants(n_days: int):
        glide_steps = 12  # last ~12 monthly de-risking steps
        glide_start = max(0, n_days - 12 * TRADING_DAYS_PER_MONTH)
        return [
            ("hold", lambda: HoldToEnd(), [False]),
            ("tp50", lambda: TakeProfit(0.50), [False, True]),
            ("tp100", lambda: TakeProfit(1.00), [False, True]),
            ("trail15", lambda: TrailingStop(0.15), [False, True]),
            ("trail25", lambda: TrailingStop(0.25), [False, True]),
            ("trail35", lambda: TrailingStop(0.35), [False, True]),
            ("maexit200", lambda: MAExit(200, 0.0), [False, True]),
            ("maexit200b", lambda: MAExit(200, 0.03), [False, True]),
            ("glide12", lambda: GlideExit(glide_start, glide_steps), [False]),
            ("volx45", lambda: VolSpikeExit(0.45), [True]),
        ]

    # --- Main grid: all accumulation × all exit variants, MONTHLY cadence ----
    acc_m = accel("monthly")
    for acc_label, (acc_factory, family) in acc_m.items():
        for ex_label, ex_factory, sems in exit_variants(n_days):
            for repeated in sems:
                tag = "rot" if repeated else "1shot"
                plans.append(Plan(
                    label=f"{acc_label}|{ex_label}|monthly|{tag}",
                    accumulation=acc_factory, exit=ex_factory,
                    cadence="monthly", repeated=repeated, family=family,
                ))

    # --- Cadence sensitivity: a few methods × key exits, DAILY ---------------
    acc_d = accel("daily")
    daily_acc = ["dca48", "trend200", "scaled_invvol"]
    daily_exits = [
        ("hold", lambda: HoldToEnd(), False),
        ("trail25", lambda: TrailingStop(0.25), True),
        ("maexit200", lambda: MAExit(200, 0.0), True),
    ]
    for acc_label in daily_acc:
        acc_factory, family = acc_d[acc_label]
        for ex_label, ex_factory, repeated in daily_exits:
            tag = "rot" if repeated else "1shot"
            plans.append(Plan(
                label=f"{acc_label}|{ex_label}|daily|{tag}",
                accumulation=acc_factory, exit=ex_factory,
                cadence="daily", repeated=repeated, family=family,
                note="cadence-sensitivity",
            ))

    # --- Dual (absolute) momentum on/off control (its own exit, rotating) ----
    plans.append(Plan(
        label="mom252|momexit|monthly|rot",
        accumulation=lambda: MomentumAllIn(252),
        exit=lambda: MomentumExit(252),
        cadence="monthly", repeated=True, family="momentum",
        note="dual-momentum all-in-one",
    ))
    return plans


def run_plan(plan: Plan, closes, dates, signals, n_days: int) -> PlanMetrics:
    buy_idx = month_start_indices(dates) if plan.cadence == "monthly" else None
    res = simulate(
        closes, CAPITAL, plan.accumulation(), plan.exit(),
        buy_day_indices=buy_idx, repeated=plan.repeated, signals=signals,
    )
    return evaluate(res, CAPITAL, n_days)


def load_prices(path: str) -> Tuple[List[str], List[float]]:
    dates: List[str] = []
    closes: List[float] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            dates.append(r["date"])
            closes.append(float(r["target_close"]))
    return dates, closes


def _subwindow_calmar(plan: Plan, closes, dates, splits) -> List[float]:
    out = []
    for lo, hi in splits:
        sub_c = closes[lo:hi]
        sub_d = dates[lo:hi]
        sig = build_signals(sub_c)
        buy_idx = month_start_indices(sub_d) if plan.cadence == "monthly" else None
        res = simulate(sub_c, CAPITAL, plan.accumulation(), plan.exit(),
                       buy_day_indices=buy_idx, repeated=plan.repeated, signals=sig)
        out.append(evaluate(res, CAPITAL, len(sub_c)).calmar)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="outputs_real/daily.csv")
    ap.add_argument("--out", default="accumulation_study/outputs")
    args = ap.parse_args(argv)

    dates, closes = load_prices(args.prices)
    n_days = len(closes)
    n_months = len(month_start_indices(dates))
    signals = build_signals(closes)
    plans = build_grid(n_days, n_months)

    rows = []
    metrics_by_label = {}
    for p in plans:
        m = run_plan(p, closes, dates, signals, n_days)
        metrics_by_label[p.label] = m
        row = {"label": p.label, "family": p.family, "cadence": p.cadence,
               "repeated": p.repeated, "note": p.note}
        row.update(m.to_row())
        rows.append(row)

    rows.sort(key=lambda r: r["calmar"], reverse=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out/'rows.csv'} ({len(rows)} plans), "
          f"window {dates[0]}..{dates[-1]} {n_days}d {n_months}mo")

    # Sub-window Calmar for the top plans + controls (regime robustness check).
    splits = [(0, n_days // 3), (n_days // 3, 2 * n_days // 3), (2 * n_days // 3, n_days)]
    interesting = [r["label"] for r in rows[:8]]
    for ctrl in ("lump|hold|monthly|1shot", "dca48|hold|monthly|1shot"):
        if ctrl not in interesting:
            interesting.append(ctrl)
    plan_by_label = {p.label: p for p in plans}
    subwin = {lbl: _subwindow_calmar(plan_by_label[lbl], closes, dates, splits)
              for lbl in interesting}

    summary = {
        "window": {"start": dates[0], "end": dates[-1], "n_days": n_days,
                   "n_months": n_months, "capital": CAPITAL},
        "n_plans": len(rows),
        "splits": [{"start": dates[lo], "end": dates[hi - 1]} for lo, hi in splits],
        "subwindow_calmar": subwin,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
