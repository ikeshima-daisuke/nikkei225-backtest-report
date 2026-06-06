"""Write summary JSON, CSV logs and PNG charts for a backtest run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt  # noqa: E402

from .backtest import BacktestResult
from .benchmark import build_benchmarks
from .metrics import build_summary, daily_returns, max_drawdown_abs
from .report import render_report_md

_DAILY_COLUMNS = [
    "date",
    "target_open",
    "target_close",
    "benchmark_close",
    "buy_amount",
    "buy_shares",
    "sell_value",
    "realized_pnl_before_tax",
    "realized_pnl_after_tax",
    "tax",
    "interest",
    "gross_exposure",
    "cash",
    "equity",
    "unrealized_pnl",
    "margin_ratio",
    "maintenance_ratio",
    "selected_params_id",
    "events",
]

_TRADE_COLUMNS = [
    "date",
    "side",
    "shares",
    "price",
    "value",
    "lot_id",
    "realized_pnl_before_tax",
    "realized_pnl_after_tax",
    "reason",
    "event_type",
]


def _daily_df(result: BacktestResult) -> pd.DataFrame:
    df = pd.DataFrame(result.daily_rows)
    if df.empty:
        return pd.DataFrame(columns=_DAILY_COLUMNS)
    return df[_DAILY_COLUMNS + ["took_profit"]]


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Running peak-to-current draw-down (negative or zero) of the equity."""
    peak = np.maximum.accumulate(equity)
    return equity - peak


def write_outputs(result: BacktestResult, out_dir: str | Path) -> Dict[str, Any]:
    """Write all output files to ``out_dir`` and return the summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = build_summary(result, result.config)
    (out / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    daily = _daily_df(result)
    daily[_DAILY_COLUMNS].to_csv(out / "daily.csv", index=False)

    trades = pd.DataFrame(result.trades)
    if trades.empty:
        trades = pd.DataFrame(columns=_TRADE_COLUMNS)
    else:
        for col in _TRADE_COLUMNS:
            if col not in trades.columns:
                trades[col] = np.nan
    trades[_TRADE_COLUMNS].to_csv(out / "trades.csv", index=False)

    _write_optimization_csv(result.optimization_rows, out / "optimization.csv")

    _write_charts(result, daily, out)

    # --- Passive benchmarks + structured Markdown report (Week 1, item F) ---
    n_days = len(daily)
    benchmarks = build_benchmarks(
        daily["target_close"].tolist() if not daily.empty else [],
        daily["benchmark_close"].tolist() if not daily.empty else [],
        result.config.initial_equity,
        n_days,
        dca_cap=result.config.max_gross_exposure,
    )
    period_start = period_end = ""
    if not daily.empty:
        dates = pd.to_datetime(daily["date"])
        period_start = dates.iloc[0].strftime("%Y-%m-%d")
        period_end = dates.iloc[-1].strftime("%Y-%m-%d")
    report_md = render_report_md(
        summary,
        benchmarks,
        period_start=period_start,
        period_end=period_end,
        n_sessions=n_days,
        dca_cap=result.config.max_gross_exposure,
    )
    (out / "report.md").write_text(report_md, encoding="utf-8")
    (out / "benchmarks.json").write_text(
        json.dumps([b.to_dict() for b in benchmarks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _write_optimization_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        pd.DataFrame(
            columns=[
                "date",
                "selected_params",
                "training_score",
                "training_net_profit",
                "training_max_drawdown",
                "training_max_unrealized_loss",
                "training_margin_call_count",
            ]
        ).to_csv(path, index=False)
        return
    flat = []
    for r in rows:
        flat.append(
            {
                "date": pd.Timestamp(r["date"]),
                "selected_params": json.dumps(r["selected_params"], ensure_ascii=False),
                "training_score": r.get("training_score"),
                "training_net_profit": r.get("training_net_profit"),
                "training_max_drawdown": r.get("training_max_drawdown"),
                "training_max_unrealized_loss": r.get("training_max_unrealized_loss"),
                "training_margin_call_count": r.get("training_margin_call_count"),
            }
        )
    pd.DataFrame(flat).to_csv(path, index=False)


def _write_charts(result: BacktestResult, daily: pd.DataFrame, out: Path) -> None:
    """Render the four standard charts.  Never raises on empty data."""
    if daily.empty:
        return
    dates = pd.to_datetime(daily["date"])
    equity = daily["equity"].to_numpy(dtype=float)
    exposure = daily["gross_exposure"].to_numpy(dtype=float)

    # Equity curve.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, equity, color="#1f77b4")
    ax.set_title("Equity curve")
    ax.set_ylabel("Equity (JPY)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=110)
    plt.close(fig)

    # Gross exposure curve.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, exposure, color="#ff7f0e")
    ax.axhline(
        result.config.max_gross_exposure,
        color="red",
        linestyle="--",
        label="max_gross_exposure",
    )
    ax.set_title("Gross exposure")
    ax.set_ylabel("Exposure (JPY)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "exposure_curve.png", dpi=110)
    plt.close(fig)

    # Equity draw-down curve.
    dd = _drawdown_series(equity)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(dates, dd, 0, color="#d62728", alpha=0.5)
    ax.set_title(f"Equity draw-down (max {max_drawdown_abs(equity):,.0f} JPY)")
    ax.set_ylabel("Draw-down (JPY)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "drawdown_curve.png", dpi=110)
    plt.close(fig)

    # Realized profit by day (after tax).
    realized = daily["realized_pnl_after_tax"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(dates, realized, color="#2ca02c", width=1.0)
    ax.set_title("Realized profit by day (after tax)")
    ax.set_ylabel("Realized P&L (JPY)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "realized_profit_by_day.png", dpi=110)
    plt.close(fig)

    # Underwater curve (draw-down as a percentage of the running peak).
    with np.errstate(divide="ignore", invalid="ignore"):
        peak = np.maximum.accumulate(equity)
        dd_pct = np.where(peak > 0, (equity - peak) / peak * 100.0, 0.0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(dates, dd_pct, 0, color="#9467bd", alpha=0.5)
    ax.set_title(f"Underwater curve (max {dd_pct.min():.1f}%)")
    ax.set_ylabel("Draw-down (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "underwater_curve.png", dpi=110)
    plt.close(fig)

    # Daily-return distribution (where the tail risk lives).
    rets = daily_returns(equity) * 100.0
    if rets.size:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(rets, bins=60, color="#1f77b4", alpha=0.8)
        var95 = float(np.quantile(rets, 0.05))
        ax.axvline(var95, color="red", linestyle="--", label=f"VaR95 {var95:.2f}%")
        ax.set_title("Daily equity-return distribution")
        ax.set_xlabel("Daily return (%)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "return_distribution.png", dpi=110)
        plt.close(fig)


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj
