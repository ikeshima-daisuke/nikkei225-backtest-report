"""Objective function and summary-metric helpers.

These functions are deliberately decoupled from the engine: they operate on a
duck-typed ``SimResult`` (any object exposing the documented attributes) plus
the :class:`~nikkei_leverage_sim.config.Config`, so the optimizer and the
reporting layer can both reuse them.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

from .config import Config


def max_drawdown_abs(equity_curve: Sequence[float]) -> float:
    """Maximum peak-to-trough drop of the equity curve, in yen (>= 0)."""
    peak = -math.inf
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return max_dd


def max_consecutive_without_tp(tp_flags: Sequence[bool]) -> int:
    """Longest run of consecutive trading days with no take-profit."""
    best = 0
    cur = 0
    for flag in tp_flags:
        if flag:
            cur = 0
        else:
            cur += 1
            if cur > best:
                best = cur
    return best


def no_tp_streak_penalty_days(tp_flags: Sequence[bool], grace: int) -> int:
    """Total excess no-take-profit days beyond ``grace``, summed over streaks."""
    total = 0
    cur = 0
    for flag in list(tp_flags) + [True]:  # sentinel closes the final streak
        if flag:
            if cur > grace:
                total += cur - grace
            cur = 0
        else:
            cur += 1
    return total


def objective_score(result: "SimResultLike", cfg: Config) -> float:
    """Composite optimization objective (higher is better).

    ``score = realized_after_tax + ending_unrealized_pnl``
    ``      - w_dd * max_drawdown_equity - w_ul * max_unrealized_loss``
    ``      - margin_call_penalty * margin_call_count``
    ``      - exposure_penalty * exposure_limit_hit_count``
    ``      - no_tp_penalty * no_take_profit_streak_penalty``
    """
    o = cfg.objective
    return (
        result.realized_after_tax
        + result.ending_unrealized_pnl
        - o.weight_max_drawdown_equity * result.max_drawdown_equity
        - o.weight_max_unrealized_loss * result.max_unrealized_loss
        - o.margin_call_penalty * result.margin_call_count
        - o.exposure_limit_hit_penalty * result.exposure_limit_hit_count
        - o.no_take_profit_streak_penalty * result.no_tp_streak_penalty
    )


def sharpe_like(daily_equity: Sequence[float]) -> float:
    """A rough (non-annualized-rigorous) Sharpe-like ratio of equity returns."""
    if len(daily_equity) < 3:
        return 0.0
    rets: List[float] = []
    for prev, cur in zip(daily_equity[:-1], daily_equity[1:]):
        if prev != 0:
            rets.append((cur - prev) / prev)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(252.0)


def annualized_return(equity_curve: Sequence[float], n_trading_days: int) -> float:
    """Compound annual growth rate of the equity curve."""
    if not equity_curve or n_trading_days <= 0:
        return 0.0
    start = equity_curve[0]
    end = equity_curve[-1]
    if start <= 0:
        return 0.0
    years = n_trading_days / 252.0
    if years <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


# ---------------------------------------------------------------------------
# Tail-risk / draw-down metrics (Week 1 — "measure how bad it can get").
#
# All operate on the *daily equity curve*, which already reflects positions,
# cash, accrued interest and unrealized P&L — satisfying the methodology rule
# that risk must be computed on a position/cash/valuation basis, not on a naive
# price-return series.  They are dependency-light (numpy + scipy.stats only) and
# never raise on short/degenerate input (return a neutral 0.0).
# ---------------------------------------------------------------------------


def daily_returns(equity_curve: Sequence[float]) -> np.ndarray:
    """Simple daily returns of the equity curve.

    Steps where the prior equity is non-positive (or the result is non-finite)
    are dropped, mirroring :func:`sharpe_like`.
    """
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size < 2:
        return np.empty(0, dtype=float)
    prev = arr[:-1]
    cur = arr[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, (cur - prev) / prev, np.nan)
    return rets[np.isfinite(rets)]


def sortino_ratio(
    equity_curve: Sequence[float], target: float = 0.0, periods: int = 252
) -> float:
    """Annualized Sortino ratio (excess return over downside deviation).

    Downside deviation is the RMS of *below-target* daily returns, averaged over
    the count of below-target days **only** — the conservative convention.  It
    deliberately does not dilute the denominator with flat/up days, so a strategy
    with sparse losses (like this no-stop-loss, realize-only-gains one) is not
    flattered by a large number of near-zero days.
    """
    rets = daily_returns(equity_curve)
    if rets.size < 2:
        return 0.0
    downside = rets[rets < target] - target
    if downside.size == 0:
        return 0.0
    dd = math.sqrt(float(np.mean(downside ** 2)))
    if dd == 0.0:
        return 0.0
    return (float(np.mean(rets)) - target) / dd * math.sqrt(periods)


def max_drawdown_pct(equity_curve: Sequence[float]) -> float:
    """Worst peak-to-trough drop as a positive fraction (0.12 == 12%)."""
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - arr) / peak, 0.0)
    return float(np.max(dd)) if dd.size else 0.0


def calmar_ratio(annual_ret: float, max_dd_fraction: float) -> float:
    """CAGR divided by max draw-down fraction (0 when there is no draw-down)."""
    if max_dd_fraction <= 0.0:
        return 0.0
    return annual_ret / max_dd_fraction


def ulcer_index(equity_curve: Sequence[float]) -> float:
    """Ulcer Index: RMS of the percentage draw-down series (in percent units)."""
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(peak > 0, (arr - peak) / peak * 100.0, 0.0)
    return float(math.sqrt(float(np.mean(dd_pct ** 2))))


def value_at_risk(equity_curve: Sequence[float], confidence: float = 0.95) -> float:
    """Historical 1-day VaR as a non-negative loss fraction.

    Returns the magnitude of the ``(1 - confidence)`` quantile of daily returns
    (e.g. confidence 0.95 -> 5th percentile), clamped at 0: a positive value is a
    loss of that fraction; 0 means even the tail day was not a loss.
    """
    rets = daily_returns(equity_curve)
    if rets.size < 2:
        return 0.0
    q = float(np.quantile(rets, 1.0 - confidence))
    return max(-q, 0.0)


def conditional_var(equity_curve: Sequence[float], confidence: float = 0.95) -> float:
    """Historical CVaR / expected shortfall as a non-negative loss fraction.

    The mean of all daily returns at or below the VaR threshold (the average of
    the worst ``1 - confidence`` tail), reported as a non-negative loss fraction
    (clamped at 0 when the tail is not a loss).
    """
    rets = daily_returns(equity_curve)
    if rets.size < 2:
        return 0.0
    q = np.quantile(rets, 1.0 - confidence)
    tail = rets[rets <= q]
    if tail.size == 0:
        return max(float(-q), 0.0)
    return max(float(-tail.mean()), 0.0)


def return_skew(equity_curve: Sequence[float]) -> float:
    """Sample skewness of daily returns (0 for fewer than 3 returns)."""
    rets = daily_returns(equity_curve)
    if rets.size < 3:
        return 0.0
    return float(stats.skew(rets, bias=False))


def return_kurtosis(equity_curve: Sequence[float]) -> float:
    """Sample excess kurtosis (Fisher) of daily returns (0 for fewer than 4)."""
    rets = daily_returns(equity_curve)
    if rets.size < 4:
        return 0.0
    return float(stats.kurtosis(rets, fisher=True, bias=False))


def drawdown_percentiles(
    equity_curve: Sequence[float], percentiles: Sequence[int] = (50, 90, 95, 99)
) -> Dict[int, float]:
    """Percentiles of the daily draw-down depth (positive fractions)."""
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size == 0:
        return {int(p): 0.0 for p in percentiles}
    peak = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - arr) / peak, 0.0)
    return {int(p): float(np.percentile(dd, p)) for p in percentiles}


def worst_day_equity(equity_curve: Sequence[float]) -> float:
    """Lowest equity ever reached (the literal worst day on the books)."""
    arr = np.asarray(equity_curve, dtype=float)
    return float(np.min(arr)) if arr.size else 0.0


def worst_daily_return(equity_curve: Sequence[float]) -> float:
    """Most negative single-day equity return (<= 0; 0 if never negative)."""
    rets = daily_returns(equity_curve)
    if rets.size == 0:
        return 0.0
    return float(np.min(rets))


def build_risk_metrics(equity_curve: Sequence[float], n_trading_days: int) -> Dict[str, float]:
    """Assemble the tail-risk / draw-down block used by the report and JSON."""
    ann = annualized_return(equity_curve, n_trading_days)
    mdd_pct = max_drawdown_pct(equity_curve)
    ddp = drawdown_percentiles(equity_curve)
    return {
        "sortino_ratio": sortino_ratio(equity_curve),
        "calmar_ratio": calmar_ratio(ann, mdd_pct),
        "ulcer_index": ulcer_index(equity_curve),
        "var_95_daily": value_at_risk(equity_curve, 0.95),
        "cvar_95_daily": conditional_var(equity_curve, 0.95),
        "return_skew": return_skew(equity_curve),
        "return_kurtosis_excess": return_kurtosis(equity_curve),
        "max_drawdown_pct": mdd_pct,
        "drawdown_pct_p50": ddp[50],
        "drawdown_pct_p90": ddp[90],
        "drawdown_pct_p95": ddp[95],
        "drawdown_pct_p99": ddp[99],
        "worst_day_equity": worst_day_equity(equity_curve),
        "worst_daily_return": worst_daily_return(equity_curve),
    }


class SimResultLike:  # pragma: no cover - typing aid only
    """Protocol-ish placeholder documenting required attributes.

    Any object with these attributes works with :func:`objective_score`.
    """

    realized_after_tax: float
    ending_unrealized_pnl: float
    max_drawdown_equity: float
    max_unrealized_loss: float
    margin_call_count: int
    exposure_limit_hit_count: int
    no_tp_streak_penalty: float


def build_summary(result, cfg: Config) -> Dict[str, object]:
    """Build the full summary dictionary from a recorded backtest result."""
    pf = result.portfolio
    daily = result.daily_rows
    n_days = len(daily)

    closed = pf.closed_lot_profits
    win_rate = (pf.closed_lot_wins / pf.closed_lot_count) if pf.closed_lot_count else 0.0
    avg_profit_per_lot = (sum(closed) / len(closed)) if closed else 0.0

    # Take-profit-day based figures.
    tp_day_profits = [r["realized_pnl_after_tax"] for r in daily if r["took_profit"]]
    days_with_profit = len(tp_day_profits)
    avg_per_tp_day = (sum(tp_day_profits) / days_with_profit) if days_with_profit else 0.0
    median_per_tp_day = _median(tp_day_profits)

    # Calendar span for per-calendar-day figure.
    if n_days >= 2:
        calendar_days = (daily[-1]["date"] - daily[0]["date"]).days or n_days
    else:
        calendar_days = max(n_days, 1)

    total_realized_after = pf.realized_after_tax
    ending_unrealized = pf.unrealized_pnl()

    avg_exposure = (pf.exposure_sum / pf.exposure_obs) if pf.exposure_obs else 0.0

    # Tail-risk / draw-down block (Week 1).  margin_call_rate joins it because it
    # is a survival metric, not a return metric.
    risk = build_risk_metrics(result.equity_curve, n_days)
    risk["margin_call_rate"] = (pf.margin_call_count / n_days) if n_days else 0.0

    profit_factor = (
        pf.closed_lot_gross_profit / pf.closed_lot_gross_loss
        if pf.closed_lot_gross_loss > 0
        else math.inf
        if pf.closed_lot_gross_profit > 0
        else 0.0
    )

    return {
        "initial_equity": cfg.initial_equity,
        "final_equity": result.final_equity,
        "net_realized_profit_before_tax": pf.realized_before_tax,
        "net_realized_profit_after_tax": total_realized_after,
        "ending_unrealized_pnl": ending_unrealized,
        "total_interest_paid": pf.total_interest_paid,
        "total_commission_paid": pf.total_commission_paid,
        "total_tax_paid": pf.total_tax_paid,
        "total_trades": pf.buy_trade_count + pf.sell_trade_count,
        "buy_trade_count": pf.buy_trade_count,
        "sell_trade_count": pf.sell_trade_count,
        "win_rate_of_closed_lots": win_rate,
        "average_profit_per_closed_lot": avg_profit_per_lot,
        "average_profit_per_calendar_day": total_realized_after / calendar_days,
        "average_profit_per_trading_day": total_realized_after / n_days if n_days else 0.0,
        "average_profit_per_take_profit_day": avg_per_tp_day,
        "median_profit_per_take_profit_day": median_per_tp_day,
        "days_with_realized_profit": days_with_profit,
        "max_gross_exposure": pf.max_gross_exposure_seen,
        "average_gross_exposure": avg_exposure,
        "exposure_limit_hit_count": pf.exposure_limit_hit_count,
        "max_unrealized_loss": pf.max_unrealized_loss,
        "max_drawdown_equity": result.max_drawdown_equity,
        "max_consecutive_days_without_take_profit": result.max_consecutive_no_tp,
        "max_lot_holding_days": pf.max_lot_holding_days,
        "margin_warning_count": pf.margin_warning_count,
        "margin_call_count": pf.margin_call_count,
        "forced_liquidation_count": pf.forced_liquidation_count,
        "min_maintenance_ratio": (
            pf.min_maintenance_ratio_seen
            if math.isfinite(pf.min_maintenance_ratio_seen)
            else None
        ),
        "annualized_return": annualized_return(result.equity_curve, n_days),
        "sharpe_like_ratio": sharpe_like(result.equity_curve),
        "profit_factor": profit_factor,
        # --- tail-risk / draw-down block (Week 1) ---
        "risk": risk,
        # --- data-quality audit trail (repaired vendor price glitches) ---
        "data_quality": {
            "price_glitch_repairs": getattr(result, "data_repairs", []) or [],
        },
        # --- run configuration echoed for reproducibility / audit ---
        "force_liquidation": cfg.force_liquidation,
        "maintenance_margin_ratio": cfg.maintenance_margin_ratio,
        "warning_margin_ratio": cfg.warning_margin_ratio,
        "random_seed": cfg.optimization.random_seed,
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0
