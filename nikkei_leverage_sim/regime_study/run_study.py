"""Replay the accumulation×exit grid on each historical regime.

Reuses ``accumulation_study`` verbatim (same grid, engine, metrics) but feeds it
the **synthetic 1570.T reconstructed from real N225** for each regime window.
The point is a like-for-like comparison: the only thing that changes between the
bull baseline and the lost decades is the *price path*.

Usage (from ``nikkei_leverage_sim/``)::

    python -m regime_study.run_study \
        --n225 data/benchmark_N225_long.csv \
        --real data/target_1570_T.csv \
        --out regime_study/outputs
"""
from __future__ import annotations

import sys
from pathlib import Path

# Self-contained import bootstrap (src + package root), mirroring the test
# conftest so the module runs whether or not the core package is pip-installed.
_PKG = Path(__file__).resolve().parents[1]  # nikkei_leverage_sim/
for _p in (str(_PKG / "src"), str(_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from typing import Dict, List  # noqa: E402

from accumulation_study.engine import month_start_indices  # noqa: E402
from accumulation_study.run_study import build_grid, run_plan  # noqa: E402
from accumulation_study.signals import build_signals  # noqa: E402

from .build_target import calibrate_base_drag, read_ohlc, synth_close_path  # noqa: E402
from .regimes import REGIMES, Regime, slice_window  # noqa: E402
from nikkei_leverage_sim.data import read_ohlc_csv  # noqa: E402


def _exit_of(label: str) -> str:
    return label.split("|")[1]


def _family_of(label: str, rows_by_label: Dict[str, dict]) -> str:
    return rows_by_label[label]["family"]


def run_regime(regime: Regime, full_close, n225) -> Dict:
    """Run the full grid on one regime; return rows + a compact headline dict.

    ``full_close`` is the synthetic close built **once over the entire N225
    series**, then sliced here — so every regime's first bar is anchored to a
    genuine prior close (no same-day-close edge artifact).
    """
    seg_close = full_close.loc[regime.start:regime.end]
    seg = slice_window(n225, regime)
    dates = [d.strftime("%Y-%m-%d") for d in seg_close.index]
    closes = seg_close.to_numpy().tolist()
    n_days = len(closes)
    n_months = len(month_start_indices(dates))
    signals = build_signals(closes)

    plans = build_grid(n_days, n_months)
    rows: List[dict] = []
    for p in plans:
        m = run_plan(p, closes, dates, signals, n_days)
        row = {"label": p.label, "family": p.family, "cadence": p.cadence,
               "repeated": p.repeated, "exit": _exit_of(p.label)}
        row.update(m.to_row())
        rows.append(row)
    rows.sort(key=lambda r: r["calmar"], reverse=True)
    by_label = {r["label"]: r for r in rows}

    monthly = [r for r in rows if r["cadence"] == "monthly"]
    holds = [r for r in monthly if r["exit"] == "hold"]
    exits = [r for r in monthly if r["exit"] != "hold"]
    best_hold = max(holds, key=lambda r: r["calmar"]) if holds else None
    best_exit = max(exits, key=lambda r: r["calmar"]) if exits else None

    # Best plan per accumulation family (hold-only, the "how to buy" question).
    fam_best: Dict[str, dict] = {}
    for r in holds:
        fam = r["family"]
        if fam not in fam_best or r["calmar"] > fam_best[fam]["calmar"]:
            fam_best[fam] = r

    lump_hold = by_label.get("lump|hold|monthly|1shot")
    dca_hold = by_label.get("dca48|hold|monthly|1shot")

    def _slim(r):
        if not r:
            return None
        return {k: r[k] for k in ("label", "total_return", "max_drawdown_pct",
                                  "calmar", "sortino", "avg_invested_pct", "n_sells")}

    headline = {
        "key": regime.key, "label": regime.label, "kind": regime.kind,
        "start": dates[0], "end": dates[-1], "n_days": n_days, "n_months": n_months,
        "n225_total_return": float(seg["Adj Close"].iloc[-1] / seg["Adj Close"].iloc[0] - 1.0),
        "top5": [_slim(r) for r in rows[:5]],
        "best_hold": _slim(best_hold),
        "best_exit": _slim(best_exit),
        "hold_beats_exit": (best_hold["calmar"] >= best_exit["calmar"])
        if (best_hold and best_exit) else None,
        "family_best_hold": {fam: _slim(r) for fam, r in sorted(
            fam_best.items(), key=lambda kv: kv[1]["calmar"], reverse=True)},
        "lump_hold": _slim(lump_hold),
        "dca_hold": _slim(dca_hold),
    }
    return {"rows": rows, "headline": headline}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n225", default="data/benchmark_N225_long.csv")
    ap.add_argument("--real", default="data/target_1570_T.csv")
    ap.add_argument("--out", default="regime_study/outputs")
    args = ap.parse_args(argv)

    n225 = read_ohlc(args.n225)
    real = read_ohlc_csv(args.real)[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    calib = calibrate_base_drag(n225, real)
    base_drag, calib_rate = calib["base_drag"], calib["calib_rate"]

    # Build the synthetic close ONCE over the whole series, then slice per regime
    # (so each regime's first bar has a real prior close — see run_regime).
    full_close = synth_close_path(n225, base_drag=base_drag, calib_rate=calib_rate)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    headlines = []
    for regime in REGIMES:
        res = run_regime(regime, full_close, n225)
        rdir = out / regime.key
        rdir.mkdir(parents=True, exist_ok=True)
        rows = res["rows"]
        with open(rdir / "rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        headlines.append(res["headline"])
        h = res["headline"]
        print(f"[{regime.key:16s}] {h['start']}..{h['end']} {h['n_days']}d "
              f"N225 {h['n225_total_return']*100:+.0f}%  "
              f"lump·hold {h['lump_hold']['total_return']*100:+.0f}% "
              f"(DD {h['lump_hold']['max_drawdown_pct']*100:.0f}%)  "
              f"hold>exit={h['hold_beats_exit']}")

    summary = {
        "calibration": calib,
        "regimes": headlines,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
