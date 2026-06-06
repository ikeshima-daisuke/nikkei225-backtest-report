"""Lookahead-safe accumulate-then-exit simulator and its metrics.

The engine is deliberately tiny and policy-agnostic: a *plan* is an
``AccumulationPolicy`` (how many yen to buy at today's close) paired with an
``ExitPolicy`` (what fraction of the holding to sell today, and whether that
ends the plan).  The same loop covers every method in :mod:`policies`.

Invariants (match the project's core):
* **No lookahead.** A policy receives :class:`Ctx` exposing ``closes`` and the
  index ``t``; it must read only ``closes[: t + 1]``.  Trades fill at
  ``closes[t]``.
* **Cash only.** No margin, no added leverage; uninvested cash earns 0% and
  cannot go negative (a buy is clamped to available cash).
* **Equity basis.** The reported curve is ``shares * close + cash`` starting at
  the full ``capital`` — so returns/draw-downs are on the *usable capital*
  (¥10M), the honest denominator from ``REPORT_REAL.md §10``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from nikkei_leverage_sim.metrics import (
    annualized_return,
    calmar_ratio,
    conditional_var,
    max_drawdown_abs,
    max_drawdown_pct,
    sharpe_like,
    sortino_ratio,
    ulcer_index,
)


@dataclass
class Ctx:
    """Everything a policy may look at on day ``t`` (lookahead-safe).

    A policy must index ``closes`` only up to ``t`` (inclusive).  Convenience
    fields summarise the position so simple rules need no re-derivation.
    """

    t: int
    n: int
    price: float
    closes: np.ndarray
    capital: float
    cash: float
    shares: float
    deployed: float          # cumulative yen ever bought (net of nothing)
    avg_cost: float          # mean entry price of the held shares (0 if flat)
    installment: float       # capital / number_of_scheduled_buy_days
    schedule_days_left: int  # scheduled buy days remaining in this phase
    equity: float            # mark-to-market equity *before* today's exit
    equity_peak: float       # running peak of equity so far
    phase_start_t: int       # index where the current accumulation phase began
    signals: "Optional[object]" = None  # precomputed lookahead-safe indicators


AccumulationPolicy = Callable[[Ctx], float]          # -> yen to buy today
ExitPolicy = Callable[[Ctx], "tuple[float, bool]"]   # -> (sell_fraction, terminate)


def month_start_indices(dates: Sequence[str]) -> List[int]:
    """Indices of the first trading day of each calendar month (``YYYY-MM``)."""
    seen = set()
    out: List[int] = []
    for i, d in enumerate(dates):
        ym = d[:7]
        if ym not in seen:
            seen.add(ym)
            out.append(i)
    return out


@dataclass
class SimResult:
    equity: List[float]
    shares_end: float
    cash_end: float
    deployed_total: float
    n_sells: int
    exited_early: bool
    n_buys: int = 0
    avg_invested_pct: float = 0.0  # mean(position_value / equity) over sessions


def simulate(
    closes: Sequence[float],
    capital: float,
    accumulation: AccumulationPolicy,
    exit_policy: ExitPolicy,
    *,
    buy_day_indices: Optional[Sequence[int]] = None,
    repeated: bool = False,
    signals: Optional[object] = None,
    cost_bps: float = 0.0,
) -> SimResult:
    """Run one accumulate-then-exit plan over ``closes``.

    ``buy_day_indices`` restricts *accumulation* to those sessions (e.g. month
    starts); ``None`` means every session.  Exit is checked **every** session.
    With ``repeated=False`` the first sell ends accumulation and the plan holds
    cash to the end (single round-trip).  With ``repeated=True`` a sell instead
    resets the phase anchors and accumulation resumes (rotating).
    """
    arr = np.asarray(closes, dtype=float)
    n = arr.size
    if n == 0:
        return SimResult([float(capital)], 0.0, float(capital), 0.0, 0, False)

    buy_set = set(range(n) if buy_day_indices is None else buy_day_indices)
    n_buy_days = max(len(buy_set), 1)
    installment = float(capital) / n_buy_days

    cash = float(capital)
    shares = 0.0
    cost_basis = 0.0       # total yen cost of currently held shares
    deployed = 0.0
    equity_curve = np.empty(n, dtype=float)
    peak = float(capital)
    phase_start = 0
    n_sells = 0
    n_buys = 0
    invested_frac_sum = 0.0
    exited = False
    cost_rate = max(0.0, float(cost_bps) / 10_000.0)

    # Precompute, per index, how many scheduled buy days remain in the window
    # (used by pacing policies); recomputed against phase_start lazily below.
    sorted_buy = sorted(buy_set)

    for t in range(n):
        price = arr[t]
        if not np.isfinite(price) or price <= 0:
            # Carry valuation forward on a bad tick; never trade on it.
            equity_curve[t] = shares * (arr[t - 1] if t else 0.0) + cash
            peak = max(peak, equity_curve[t])
            continue

        # --- 1) accumulation buy (only while in the accumulating phase) ------
        if (t in buy_set) and not exited:
            days_left = sum(1 for b in sorted_buy if b >= t and b >= phase_start)
            avg_cost = (cost_basis / shares) if shares > 0 else 0.0
            ctx = Ctx(
                t=t, n=n, price=price, closes=arr, capital=float(capital),
                cash=cash, shares=shares, deployed=deployed, avg_cost=avg_cost,
                installment=installment, schedule_days_left=days_left,
                equity=shares * price + cash, equity_peak=peak,
                phase_start_t=phase_start, signals=signals,
            )
            buy = accumulation(ctx)
            buy = float(max(0.0, min(buy, cash)))
            if buy > 0:
                shares += (buy * (1.0 - cost_rate)) / price  # fee eats into shares
                cash -= buy
                cost_basis += buy
                deployed += buy
                n_buys += 1

        # --- 2) exit decision (checked every session) -----------------------
        equity_pre = shares * price + cash
        avg_cost = (cost_basis / shares) if shares > 0 else 0.0
        ctx = Ctx(
            t=t, n=n, price=price, closes=arr, capital=float(capital),
            cash=cash, shares=shares, deployed=deployed, avg_cost=avg_cost,
            installment=installment, schedule_days_left=0,
            equity=equity_pre, equity_peak=peak, phase_start_t=phase_start,
            signals=signals,
        )
        frac, terminate = exit_policy(ctx)
        if shares > 0 and frac and frac > 0:
            frac = min(1.0, float(frac))
            sell_sh = shares * frac
            cash += sell_sh * price * (1.0 - cost_rate)  # fee off the proceeds
            cost_basis -= avg_cost * sell_sh
            shares -= sell_sh
            n_sells += 1
            if (not repeated) or terminate:
                exited = True
            else:
                phase_start = t + 1  # reset trailing anchors for the next leg

        equity_curve[t] = shares * price + cash
        peak = max(peak, equity_curve[t])
        if equity_curve[t] > 0:
            invested_frac_sum += (shares * price) / equity_curve[t]

    return SimResult(
        equity=equity_curve.tolist(),
        shares_end=shares,
        cash_end=cash,
        deployed_total=deployed,
        n_sells=n_sells,
        exited_early=exited,
        n_buys=n_buys,
        avg_invested_pct=invested_frac_sum / n if n else 0.0,
    )


@dataclass
class PlanMetrics:
    final_equity: float
    total_return: float
    cagr: float
    max_drawdown_pct: float
    max_drawdown_abs: float
    calmar: float
    sortino: float
    sharpe_like: float
    ulcer_index: float
    cvar_95_daily: float
    time_in_drawdown_pct: float
    avg_invested_pct: float
    deployed_total: float
    n_buys: int
    n_sells: int
    exited_early: bool

    def to_row(self) -> dict:
        return {
            "final_equity": round(self.final_equity, 2),
            "total_return": round(self.total_return, 6),
            "cagr": round(self.cagr, 6),
            "max_drawdown_pct": round(self.max_drawdown_pct, 6),
            "max_drawdown_abs": round(self.max_drawdown_abs, 2),
            "calmar": round(self.calmar, 4),
            "sortino": round(self.sortino, 4),
            "sharpe_like": round(self.sharpe_like, 4),
            "ulcer_index": round(self.ulcer_index, 4),
            "cvar_95_daily": round(self.cvar_95_daily, 6),
            "time_in_drawdown_pct": round(self.time_in_drawdown_pct, 4),
            "avg_invested_pct": round(self.avg_invested_pct, 4),
            "deployed_total": round(self.deployed_total, 2),
            "n_buys": self.n_buys,
            "n_sells": self.n_sells,
            "exited_early": self.exited_early,
        }


def _time_in_drawdown(equity: Sequence[float]) -> float:
    """Fraction of sessions spent below the running peak (underwater)."""
    arr = np.asarray(equity, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    return float(np.mean(arr < peak - 1e-9))


def evaluate(result: SimResult, capital: float, n_trading_days: int) -> PlanMetrics:
    """Reduce a :class:`SimResult` to comparable, Calmar-centred metrics."""
    eq = result.equity
    final = eq[-1] if eq else float(capital)
    total_return = final / capital - 1.0 if capital else 0.0
    cagr = annualized_return(eq, n_trading_days)
    mdd_pct = max_drawdown_pct(eq)
    return PlanMetrics(
        final_equity=final,
        total_return=total_return,
        cagr=cagr,
        max_drawdown_pct=mdd_pct,
        max_drawdown_abs=max_drawdown_abs(eq),
        calmar=calmar_ratio(cagr, mdd_pct),
        sortino=sortino_ratio(eq),
        sharpe_like=sharpe_like(eq),
        ulcer_index=ulcer_index(eq),
        cvar_95_daily=conditional_var(eq),
        time_in_drawdown_pct=_time_in_drawdown(eq),
        avg_invested_pct=result.avg_invested_pct,
        deployed_total=result.deployed_total,
        n_buys=result.n_buys,
        n_sells=result.n_sells,
        exited_early=result.exited_early,
    )
