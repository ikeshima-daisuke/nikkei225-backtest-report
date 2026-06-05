"""Stress testing (roadmap item D1 — "measure how bad it can get").

Three independent tools, all dependency-light (numpy / pandas) and reproducible:

1. :func:`regime_metrics` — slice a completed run's daily series by a historical
   crisis window and report how the *already-accumulated* book behaved through it
   (draw-down, peak unrealized loss, minimum maintenance ratio, forced closes).
   No look-ahead: it only reads a run that already happened.

2. :func:`sensitivity_grid` — replay a fixed strategy *parameter sequence* (a
   captured walk-forward sequence, or fixed params) under a grid of cost
   assumptions (slippage, margin interest).  Only the searched parameters are
   held fixed; take-profit timing still responds to the cost regime (net P&L
   includes interest), so each cell measures the strategy's outcome under those
   costs (cost effect + cost-response), not a pure cost derivative.

3. :func:`block_bootstrap_ruin` — Monte-Carlo ruin / forced-liquidation
   probability.  Paired (target, benchmark) daily bars are resampled in blocks
   (preserving autocorrelation, volatility clustering and the 2x-leverage
   co-movement), the strategy is run on each synthetic path with default params,
   and the distribution of outcomes (final equity, max draw-down, forced
   liquidations, ruin) is aggregated.

Everything operates on the existing engine (:func:`simulate`) so the
no-look-ahead and no-loss-cut invariants are inherited unchanged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .backtest import MarketData, prepare_market_data, simulate
from .config import Config, StrategyParams
from .data import join_target_benchmark
from .metrics import max_drawdown_pct

# Historical crisis windows available in the 1570.T sample (data starts 2014, so
# 2008/2011 are out of range — surfaced rather than silently faked).
HISTORICAL_REGIMES: List[Tuple[str, str, str]] = [
    ("2015 中国ショック / 円高 (Aug)", "2015-08-01", "2015-09-30"),
    ("2016 円急騰 / Brexit (H1)", "2016-01-01", "2016-06-30"),
    ("2018Q4 世界同時安", "2018-10-01", "2018-12-31"),
    ("2020 コロナショック", "2020-02-01", "2020-04-30"),
    ("2024 8月 円キャリー巻き戻し", "2024-07-15", "2024-08-31"),
]


# --------------------------------------------------------------------------- #
# 1. Historical-regime slicing
# --------------------------------------------------------------------------- #
@dataclass
class RegimeResult:
    name: str
    start: str
    end: str
    sessions: int
    benchmark_return: float
    target_return: float
    equity_start: float
    equity_end: float
    equity_return: float
    max_drawdown_pct: float
    peak_unrealized_loss: float
    min_maintenance_ratio: Optional[float]
    margin_call_days: int
    forced_liquidations: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _event_count(events: pd.Series, token: str) -> int:
    return int(events.fillna("").astype(str).str.contains(token).sum())


def regime_metrics(daily: pd.DataFrame, name: str, start: str, end: str) -> Optional[RegimeResult]:
    """Per-window stress metrics from a completed run's ``daily`` DataFrame.

    Returns ``None`` if the window does not overlap the data (so out-of-range
    regimes are skipped rather than silently producing zeros).
    """
    dates = pd.to_datetime(daily["date"])
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    mask = (dates >= lo) & (dates <= hi)
    win = daily.loc[mask]
    if win.empty:
        return None

    equity = win["equity"].to_numpy(dtype=float)
    upnl = win["unrealized_pnl"].to_numpy(dtype=float)
    bench = win["benchmark_close"].to_numpy(dtype=float)
    tgt = win["target_close"].to_numpy(dtype=float)

    maint = pd.to_numeric(win["maintenance_ratio"], errors="coerce").to_numpy(dtype=float)
    finite_maint = maint[np.isfinite(maint)]
    min_maint = float(finite_maint.min()) if finite_maint.size else None

    peak_uloss = float(-np.min(upnl)) if upnl.size and np.min(upnl) < 0 else 0.0

    return RegimeResult(
        name=name,
        start=start,
        end=end,
        sessions=int(win.shape[0]),
        benchmark_return=float(bench[-1] / bench[0] - 1.0) if bench[0] else 0.0,
        target_return=float(tgt[-1] / tgt[0] - 1.0) if tgt[0] else 0.0,
        equity_start=float(equity[0]),
        equity_end=float(equity[-1]),
        equity_return=float(equity[-1] / equity[0] - 1.0) if equity[0] else 0.0,
        max_drawdown_pct=max_drawdown_pct(equity),
        peak_unrealized_loss=peak_uloss,
        min_maintenance_ratio=min_maint,
        margin_call_days=_event_count(win["events"], "margin_call"),
        forced_liquidations=_event_count(win["events"], "forced_liquidation"),
    )


def all_regime_metrics(daily: pd.DataFrame) -> List[RegimeResult]:
    out: List[RegimeResult] = []
    for name, start, end in HISTORICAL_REGIMES:
        r = regime_metrics(daily, name, start, end)
        if r is not None:
            out.append(r)
    return out


# --------------------------------------------------------------------------- #
# 2. Cost sensitivity (decisions fixed, cost environment varied)
# --------------------------------------------------------------------------- #
ParamsProvider = Union[StrategyParams, Callable[[int], StrategyParams]]


@dataclass
class SensitivityCell:
    slippage_bps: float
    annual_margin_interest_rate: float
    net_realized_after_tax: float
    final_equity: float
    max_drawdown_equity: float
    max_unrealized_loss: float
    min_maintenance_ratio: Optional[float]
    forced_liquidations: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def sensitivity_grid(
    md: MarketData,
    base_cfg: Config,
    params_provider: ParamsProvider,
    slippage_bps_list: Sequence[float],
    interest_rate_list: Sequence[float],
) -> List[SensitivityCell]:
    """Replay ``params_provider`` under a grid of cost assumptions.

    The strategy *parameter sequence* is fixed (``params_provider`` — a fixed
    :class:`StrategyParams` or a captured ``f(i)->StrategyParams`` walk-forward
    sequence) and there is no re-optimization.  Only ``slippage_bps`` and
    ``annual_margin_interest_rate`` vary per cell.

    Note this is NOT a pure cost-isolation: take-profit decisions inside
    :func:`simulate` test ``net_pnl_before_tax`` which includes accrued interest,
    so a higher interest rate legitimately shifts *when* lots are sold.  Each cell
    therefore measures the strategy's realized outcome under that cost regime
    (cost effect + the strategy's own cost-response), holding only the searched
    parameters fixed — which is the intended question ("what does this strategy
    earn under different costs?"), not a marginal cost derivative.
    """
    from dataclasses import replace as _replace

    cells: List[SensitivityCell] = []
    for slip in slippage_bps_list:
        for rate in interest_rate_list:
            cfg = _replace(
                base_cfg, slippage_bps=float(slip), annual_margin_interest_rate=float(rate)
            )
            res = simulate(md, 0, md.n, params_provider, cfg, record=False)
            pf = res.portfolio
            min_maint = (
                pf.min_maintenance_ratio_seen
                if np.isfinite(pf.min_maintenance_ratio_seen)
                else None
            )
            cells.append(
                SensitivityCell(
                    slippage_bps=float(slip),
                    annual_margin_interest_rate=float(rate),
                    net_realized_after_tax=pf.realized_after_tax,
                    final_equity=res.final_equity,
                    max_drawdown_equity=res.max_drawdown_equity,
                    max_unrealized_loss=pf.max_unrealized_loss,
                    min_maintenance_ratio=min_maint,
                    forced_liquidations=pf.forced_liquidation_count,
                )
            )
    return cells


# --------------------------------------------------------------------------- #
# 3. Block-bootstrap Monte-Carlo ruin / forced-liquidation probability
# --------------------------------------------------------------------------- #
@dataclass
class BootstrapSummary:
    n_paths: int
    block_size: int
    horizon: int
    seed: int
    initial_equity: float
    force_liquidation: bool
    ruin_probability: float
    forced_liquidation_probability: float
    median_final_equity: float
    p05_final_equity: float
    p95_final_equity: float
    median_max_drawdown_pct: float
    p95_max_drawdown_pct: float
    worst_final_equity: float
    paths: List[Dict[str, float]] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d.pop("paths", None)
        return d


def _bar_ratios(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Stationary, resamplable descriptors of each daily bar."""
    close = df["Close"].to_numpy(dtype=float)
    logret = np.zeros(len(close), dtype=float)
    logret[1:] = np.log(close[1:] / close[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        open_r = np.where(close > 0, df["Open"].to_numpy(float) / close, 1.0)
        high_r = np.where(close > 0, df["High"].to_numpy(float) / close, 1.0)
        low_r = np.where(close > 0, df["Low"].to_numpy(float) / close, 1.0)
    return {"logret": logret, "open_r": open_r, "high_r": high_r, "low_r": low_r}


def _block_index_path(rng: np.random.Generator, n: int, block: int, horizon: int) -> np.ndarray:
    """A moving-block-bootstrap index sequence of length ``horizon`` over ``[1, n)``."""
    idx: List[int] = []
    # Valid block starts use rows 1..n-1 (row 0 has no close-to-close return).
    max_start = max(1, n - block)
    while len(idx) < horizon:
        start = int(rng.integers(1, max_start + 1))
        idx.extend(range(start, min(start + block, n)))
    return np.asarray(idx[:horizon], dtype=int)


def _synthetic_frame(
    base_index: pd.DatetimeIndex,
    ratios: Dict[str, np.ndarray],
    path_idx: np.ndarray,
    start_price: float,
) -> pd.DataFrame:
    """Rebuild an OHLC frame from a resampled index path."""
    logret = ratios["logret"][path_idx]
    close = start_price * np.exp(np.cumsum(logret))
    open_ = close * ratios["open_r"][path_idx]
    high = close * ratios["high_r"][path_idx]
    low = close * ratios["low_r"][path_idx]
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    n = len(close)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=base_index[:n],
    )


def block_bootstrap_ruin(
    joined: pd.DataFrame,
    cfg: Config,
    *,
    n_paths: int = 300,
    block_size: int = 20,
    horizon: Optional[int] = None,
    seed: int = 42,
) -> BootstrapSummary:
    """Monte-Carlo ruin / forced-liquidation probability via moving-block bootstrap.

    Paired target+benchmark daily bars are resampled in blocks (same indices for
    both series, preserving their co-movement), the strategy is run with default
    params on each synthetic path, and outcome statistics are aggregated.

    Ruin is defined as the *end-of-day* equity mark touching or falling below
    zero at any point (insolvency).  This is a LOWER BOUND on true insolvency
    probability: an intraday collapse (via the ``Low``) that recovers by the
    close is not counted, since the engine marks only at the close.  The
    forced-liquidation probability is the better early-warning metric.
    ``cfg.force_liquidation`` is honoured, so with the model on a maintenance
    breach forces a real loss.
    """
    target = joined[[c for c in joined.columns if c.startswith("target_")]].rename(
        columns=lambda c: c[len("target_"):].title().replace("Adj_Close", "Adj Close")
    )
    benchmark = joined[[c for c in joined.columns if c.startswith("benchmark_")]].rename(
        columns=lambda c: c[len("benchmark_"):].title().replace("Adj_Close", "Adj Close")
    )
    n = len(joined)
    horizon = int(horizon or n)
    rng = np.random.default_rng(seed)

    t_ratios = _bar_ratios(target)
    b_ratios = _bar_ratios(benchmark)
    t_start = float(target["Close"].iloc[0])
    b_start = float(benchmark["Close"].iloc[0])
    base_index = pd.DatetimeIndex(pd.bdate_range("2000-01-03", periods=horizon))

    default_params = cfg.strategy.default_params
    provider = lambda _i: default_params  # noqa: E731  no walk-forward on synthetic paths

    paths: List[Dict[str, float]] = []
    for _ in range(n_paths):
        path_idx = _block_index_path(rng, n, block_size, horizon)
        t_df = _synthetic_frame(base_index, t_ratios, path_idx, t_start)
        b_df = _synthetic_frame(base_index, b_ratios, path_idx, b_start)
        md = prepare_market_data(join_target_benchmark(t_df, b_df), cfg)
        res = simulate(md, 0, md.n, provider, cfg, record=False)
        eq = np.asarray(res.equity_curve, dtype=float)
        ruined = bool(eq.size and np.min(eq) <= 0.0)
        paths.append(
            {
                "final_equity": float(res.final_equity),
                "max_drawdown_pct": max_drawdown_pct(eq),
                "min_equity": float(np.min(eq)) if eq.size else cfg.initial_equity,
                "forced_liquidations": int(res.portfolio.forced_liquidation_count),
                "ruined": float(ruined),
            }
        )

    finals = np.array([p["final_equity"] for p in paths], dtype=float)
    dds = np.array([p["max_drawdown_pct"] for p in paths], dtype=float)
    forced = np.array([p["forced_liquidations"] for p in paths], dtype=float)
    ruined_arr = np.array([p["ruined"] for p in paths], dtype=float)

    return BootstrapSummary(
        n_paths=n_paths,
        block_size=block_size,
        horizon=horizon,
        seed=seed,
        initial_equity=cfg.initial_equity,
        force_liquidation=cfg.force_liquidation,
        ruin_probability=float(ruined_arr.mean()) if ruined_arr.size else 0.0,
        forced_liquidation_probability=float((forced > 0).mean()) if forced.size else 0.0,
        median_final_equity=float(np.median(finals)) if finals.size else 0.0,
        p05_final_equity=float(np.percentile(finals, 5)) if finals.size else 0.0,
        p95_final_equity=float(np.percentile(finals, 95)) if finals.size else 0.0,
        median_max_drawdown_pct=float(np.median(dds)) if dds.size else 0.0,
        p95_max_drawdown_pct=float(np.percentile(dds, 95)) if dds.size else 0.0,
        worst_final_equity=float(finals.min()) if finals.size else 0.0,
        paths=paths,
    )


# --------------------------------------------------------------------------- #
# Output writing (machine artifacts for the curated REPORT_STRESS.md)
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    return obj


def write_stress_outputs(
    out_dir: str | Path,
    regimes: Sequence[RegimeResult],
    sensitivity: Sequence[SensitivityCell],
    bootstrap: BootstrapSummary,
    *,
    regimes_thin: Optional[Sequence[RegimeResult]] = None,
    meta: Optional[Dict[str, Any]] = None,
    charts: bool = True,
) -> Dict[str, Any]:
    """Write stress.json + CSVs (+ optional charts) and return the summary dict.

    ``regimes`` are at the production own-funds level; the optional
    ``regimes_thin`` are the same windows at a thin (e.g. 2x-leverage) own-funds
    level with forced liquidation on, so per-crisis maintenance ratios and
    forced-close days are meaningful.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "meta": meta or {},
        "regimes": [r.to_dict() for r in regimes],
        "regimes_thin": [r.to_dict() for r in (regimes_thin or [])],
        "sensitivity": [c.to_dict() for c in sensitivity],
        "bootstrap": bootstrap.to_dict(),
    }
    (out / "stress.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame([r.to_dict() for r in regimes]).to_csv(out / "regimes.csv", index=False)
    if regimes_thin:
        pd.DataFrame([r.to_dict() for r in regimes_thin]).to_csv(
            out / "regimes_thin.csv", index=False
        )
    pd.DataFrame([c.to_dict() for c in sensitivity]).to_csv(out / "sensitivity.csv", index=False)
    pd.DataFrame(bootstrap.paths).to_csv(out / "bootstrap_paths.csv", index=False)

    if charts:
        _write_stress_charts(out, regimes, bootstrap)
    return summary


def _write_stress_charts(out: Path, regimes, bootstrap) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    if regimes:
        # ASCII labels (the start month) so the PNG renders without a CJK font.
        names = [r.start[:7] for r in regimes]
        dd = [r.max_drawdown_pct * 100.0 for r in regimes]
        uloss = [r.peak_unrealized_loss / 1e6 for r in regimes]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(names))
        ax.bar(x - 0.2, dd, width=0.4, color="#d62728", label="max DD (%)")
        ax2 = ax.twinx()
        ax2.bar(x + 0.2, uloss, width=0.4, color="#1f77b4", label="peak unrealized loss (M JPY)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("max draw-down (%)")
        ax2.set_ylabel("peak unrealized loss (¥M)")
        ax.set_title("Per-regime stress")
        fig.tight_layout()
        fig.savefig(out / "regime_stress.png", dpi=110)
        plt.close(fig)

    finals = np.array([p["final_equity"] for p in bootstrap.paths], dtype=float) / 1e6
    if finals.size:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(finals, bins=40, color="#2ca02c", alpha=0.8)
        ax.axvline(bootstrap.initial_equity / 1e6, color="black", linestyle=":", label="initial")
        ax.axvline(
            bootstrap.p05_final_equity / 1e6, color="red", linestyle="--", label="p05"
        )
        ax.set_title(
            f"Bootstrap final-equity distribution "
            f"(ruin {bootstrap.ruin_probability:.1%}, forced {bootstrap.forced_liquidation_probability:.1%})"
        )
        ax.set_xlabel("final equity (¥M)")
        ax.set_ylabel("paths")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "bootstrap_final_equity.png", dpi=110)
        plt.close(fig)
