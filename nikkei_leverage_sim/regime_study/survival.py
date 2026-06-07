"""Margin / ruin survival of the *account-leveraged* strategy per regime.

The accumulation grid (:mod:`run_study`) is cash-only on ¥10M usable capital — it
captures the ETF's *internal* 2x but no account margin.  This module asks the
other question the prior reports raised: with **account leverage on top** (own
funds small versus a ¥10M gross position), does the strategy survive the lost
decade, or does the margin-call model force liquidation into ruin?

It reuses the **core engine unchanged**, with the project's validated stance:
* fixed default strategy (``optimization.enabled = False``) — the same fair,
  un-tuned ruleset used by ``REPORT_VALIDATION`` (no per-path data-snooping);
* ``force_liquidation = True`` so the corrected maintenance-ratio model
  (``REPORT_MARGIN_CALL.md`` §0.1) can actually fire;
* the synthetic 1570.T target driven by real N225, real N225 as the benchmark.

Capital levels mirror the margin-call report: ¥100M (structurally insensitive),
¥5M (=2x the ¥10M cap), ¥3.3M (≈3x, near the legal minimum).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
for _p in (str(_PKG / "src"), str(_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataclasses import dataclass  # noqa: E402
from typing import Dict, List  # noqa: E402

import numpy as np  # noqa: E402

from nikkei_leverage_sim.backtest import prepare_market_data, run_backtest  # noqa: E402
from nikkei_leverage_sim.config import load_config  # noqa: E402
from nikkei_leverage_sim.data import join_target_benchmark  # noqa: E402
from nikkei_leverage_sim.metrics import build_summary  # noqa: E402

from . import financing  # noqa: E402
from .regimes import Regime, slice_window  # noqa: E402

CAPITAL_LEVELS = [
    (100_000_000.0, "¥100M (本番前提)"),
    (5_000_000.0, "¥5M (≈2倍レバ)"),
    (3_300_000.0, "¥3.3M (≈3倍レバ)"),
]


@dataclass
class SurvivalRow:
    capital: float
    capital_label: str
    account_margin_rate: float    # regime-aware investor margin-loan rate (annual)
    final_equity: float
    own_funds_return: float       # final_equity/capital - 1
    worst_day_equity: float
    min_maintenance_ratio: float
    margin_warning_count: int
    margin_call_count: int
    forced_liquidation_count: int
    max_drawdown_pct: float
    ruined: bool                  # forced liquidation fired or equity went non-positive

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("final_equity", "worst_day_equity"):
            d[k] = round(d[k], 0)
        for k in ("own_funds_return", "max_drawdown_pct"):
            d[k] = round(d[k], 4)
        d["account_margin_rate"] = round(d["account_margin_rate"], 4)
        d["min_maintenance_ratio"] = round(d["min_maintenance_ratio"], 3)
        return d


def run_survival_regime(
    regime: Regime,
    full_target,
    n225,
    *,
    config_path: str = "examples/config_fast.yaml",
) -> List[dict]:
    """Run the fixed-strategy core backtest at each capital level for one regime.

    ``full_target`` is the synthetic OHLC built **once over the whole series**,
    sliced here so each regime's first bar has a real prior close.
    """
    target_df = full_target.loc[regime.start:regime.end]
    bench_df = slice_window(n225, regime)[
        ["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    joined = join_target_benchmark(target_df, bench_df)

    # Make the *account* margin interest regime-aware (consistent with the ETF's
    # internal financing): mean call rate + spread, ≈0.028 in the bull baseline.
    acct_rate = financing.account_margin_rate(regime.start, regime.end)

    rows: List[dict] = []
    for capital, label in CAPITAL_LEVELS:
        cfg = load_config(config_path)
        cfg.optimization.enabled = False          # fixed default strategy (fair)
        cfg.force_liquidation = True              # let the margin-call model fire
        cfg.initial_equity = capital
        cfg.annual_margin_interest_rate = acct_rate
        md = prepare_market_data(joined, cfg)
        result = run_backtest(md, cfg)
        # In ruin (debt) cases the core summary's CAGR raises a benign
        # invalid-power warning on negative final equity; we don't use that field
        # (own-funds return is computed directly), so silence it here.
        with np.errstate(invalid="ignore"):
            s = build_summary(result, cfg)
        risk = s["risk"]
        worst_day = float(risk["worst_day_equity"])
        forced = int(s.get("forced_liquidation_count", 0) or 0)
        mr = s["min_maintenance_ratio"]
        ruined = (forced > 0) or (worst_day <= 0.0) or (float(s["final_equity"]) <= 0.0)
        rows.append(SurvivalRow(
            capital=capital, capital_label=label,
            account_margin_rate=acct_rate,
            final_equity=float(s["final_equity"]),
            own_funds_return=float(s["final_equity"]) / capital - 1.0,
            worst_day_equity=worst_day,
            min_maintenance_ratio=float(mr) if mr is not None else float("nan"),
            margin_warning_count=int(s["margin_warning_count"]),
            margin_call_count=int(s["margin_call_count"]),
            forced_liquidation_count=forced,
            max_drawdown_pct=float(risk["max_drawdown_pct"]),
            ruined=bool(ruined),
        ).to_dict())
    return rows


def main(argv=None) -> None:
    import argparse
    import json

    from .build_target import build_synthetic_target, calibrate_base_drag, read_ohlc
    from .regimes import REGIMES
    from nikkei_leverage_sim.data import read_ohlc_csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--n225", default="data/benchmark_N225_long.csv")
    ap.add_argument("--real", default="data/target_1570_T.csv")
    ap.add_argument("--out", default="regime_study/outputs")
    ap.add_argument("--config", default="examples/config_fast.yaml")
    args = ap.parse_args(argv)

    n225 = read_ohlc(args.n225)
    real = read_ohlc_csv(args.real)[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    calib = calibrate_base_drag(n225, real)
    base_drag, calib_rate = calib["base_drag"], calib["calib_rate"]

    # Build the synthetic OHLC ONCE over the whole series, then slice per regime.
    full_target = build_synthetic_target(n225, base_drag=base_drag, calib_rate=calib_rate)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    survival: Dict[str, list] = {}
    for regime in REGIMES:
        rows = run_survival_regime(regime, full_target, n225,
                                   config_path=args.config)
        survival[regime.key] = rows
        for r in rows:
            print(f"[{regime.key:16s}] {r['capital_label']:18s} "
                  f"acctRate {r['account_margin_rate']*100:4.1f}%  "
                  f"own-funds {r['own_funds_return']*100:+8.1f}%  "
                  f"minMR {r['min_maintenance_ratio']:.3f}  "
                  f"calls {r['margin_call_count']:3d}  ruined={r['ruined']}")

    path = out / "survival.json"
    path.write_text(json.dumps(survival, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
