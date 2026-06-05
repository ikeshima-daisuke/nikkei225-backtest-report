"""Objective function and summary-metric helpers.

These functions are deliberately decoupled from the engine: they operate on a
duck-typed ``SimResult`` (any object exposing the documented attributes) plus
the :class:`~nikkei_leverage_sim.config.Config`, so the optimizer and the
reporting layer can both reuse them.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

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
