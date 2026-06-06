"""Passive benchmark baselines for context (Week 1 — "vs just holding").

The active strategy is leveraged DCA with take-profit; a single number in
isolation cannot say whether the risk taken was worth it.  These baselines give
an apples-to-apples *capital* comparison: deploy the **same initial equity** and
either buy-and-hold the traded ETF (1570.T), buy-and-hold the index (N225),
**dollar-cost-average a fixed amount into either** (定額積立), or sit in cash.

Two flavours of "just buy it" are provided on purpose:

* **Lump-sum buy-and-hold** commits the whole initial equity on day one.  It is
  the *more aggressive* deployment of capital — the active strategy caps gross
  exposure far below the initial equity and accumulates gradually, so lump-sum
  is not a like-for-like risk twin.
* **定額積立 (fixed-amount DCA)** deploys the *same total capital*, but spread
  evenly across every session instead of all at once.  This is the like-for-like
  answer to the lump-sum caveat: like the active strategy it *accumulates* into
  the position over time rather than committing everything up front, so it
  isolates "what does the cleverness of the strategy buy you over robotically
  averaging in and just holding?".

All baselines are reference points, not perfect risk twins (the strategy still
caps exposure and takes profit); read them alongside the strategy's own exposure
and draw-down, which the report surfaces directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import (
    annualized_return,
    conditional_var,
    max_drawdown_pct,
    sharpe_like,
    sortino_ratio,
    ulcer_index,
)


@dataclass
class BenchmarkResult:
    """Summary metrics for one passive baseline over the backtest window."""

    name: str
    final_equity: float
    total_return: float
    annualized_return: float
    max_drawdown_pct: float
    sharpe_like: float
    sortino_ratio: float
    ulcer_index: float
    cvar_95_daily: float
    equity_curve: List[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_like": self.sharpe_like,
            "sortino_ratio": self.sortino_ratio,
            "ulcer_index": self.ulcer_index,
            "cvar_95_daily": self.cvar_95_daily,
        }


def buy_and_hold_curve(prices: Sequence[float], initial_equity: float) -> List[float]:
    """Lump-sum buy-and-hold equity curve for ``initial_equity``.

    Entry is the first finite, positive price; before that the capital sits in
    cash.  Non-finite later prices are forward-filled so a stray gap does not
    inject a spurious zero into the curve.
    """
    arr = np.asarray(prices, dtype=float)
    n = arr.size
    if n == 0:
        return [float(initial_equity)]
    finite = np.isfinite(arr) & (arr > 0)
    if not finite.any():
        return [float(initial_equity)] * n
    idx0 = int(np.argmax(finite))
    entry = float(arr[idx0])
    curve = np.full(n, float(initial_equity), dtype=float)
    last = entry
    for i in range(idx0, n):
        if finite[i]:
            last = float(arr[i])
        curve[i] = initial_equity * (last / entry)
    return curve.tolist()


def dca_curve(
    prices: Sequence[float],
    initial_equity: float,
    deploy_cap: Optional[float] = None,
) -> List[float]:
    """Fixed-amount dollar-cost-averaging (定額積立) equity curve.

    Capital is deployed in equal installments — one per finite-price session —
    instead of all on the first day.  The *total* amount put into the asset is
    ``initial_equity`` by default, or ``deploy_cap`` when given (clamped to
    ``initial_equity``): each finite session invests ``deployed_total /
    n_finite`` at that session's close; everything not (yet) invested waits in
    zero-interest cash.  Reported equity is the value of the shares accumulated
    so far (marked at the last finite price) plus that cash.

    ``deploy_cap`` lets only part of the account be averaged into the market
    while the rest stays in cash (a cash-allocation DCA).  **Caveat:** parking
    idle cash on the equity base *dilutes* the percentage return and draw-down,
    so a head-to-head against a strategy that risks only ``deploy_cap`` should be
    measured on the deployed capital (``initial_equity == deploy_cap``), not on a
    larger account — otherwise both metrics look smaller than they really are.
    Note too that the cap bounds *cash deployed*, not *position value*: in a
    strong uptrend the accumulated shares can mark far above ``deploy_cap``.

    The curve starts at ``initial_equity`` (nothing committed beyond the first
    installment, the rest still cash) and ends with the deployed capital fully
    invested.  Non-finite prices are skipped for *buying* and forward-filled for
    *valuation*, so a stray gap neither buys at a bogus price nor injects a
    spurious zero — mirroring :func:`buy_and_hold_curve`.
    """
    arr = np.asarray(prices, dtype=float)
    n = arr.size
    if n == 0:
        return [float(initial_equity)]
    finite = np.isfinite(arr) & (arr > 0)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return [float(initial_equity)] * n
    deployed_total = float(initial_equity)
    if deploy_cap is not None:
        deployed_total = min(float(deploy_cap), deployed_total)
    installment = deployed_total / n_finite
    shares = 0.0
    cash = float(initial_equity)
    last = float(arr[int(np.argmax(finite))])
    curve = np.empty(n, dtype=float)
    for i in range(n):
        if finite[i]:
            last = float(arr[i])
            shares += installment / last
            cash -= installment
        curve[i] = shares * last + cash
    return curve.tolist()


def _result_from_curve(
    name: str, equity_curve: Sequence[float], n_trading_days: int
) -> BenchmarkResult:
    curve = list(equity_curve)
    start = curve[0] if curve else float("nan")
    end = curve[-1] if curve else float("nan")
    total = (end / start - 1.0) if (curve and start) else 0.0
    return BenchmarkResult(
        name=name,
        final_equity=end if curve else 0.0,
        total_return=total,
        annualized_return=annualized_return(curve, n_trading_days),
        max_drawdown_pct=max_drawdown_pct(curve),
        sharpe_like=sharpe_like(curve),
        sortino_ratio=sortino_ratio(curve),
        ulcer_index=ulcer_index(curve),
        cvar_95_daily=conditional_var(curve),
        equity_curve=curve,
    )


def build_benchmarks(
    target_close: Sequence[float],
    benchmark_close: Sequence[float],
    initial_equity: float,
    n_trading_days: int,
    dca_cap: Optional[float] = None,
) -> List[BenchmarkResult]:
    """Build the standard passive baselines over the backtest window.

    Returns, in order: lump-sum buy-and-hold and fixed-amount DCA (定額積立) of
    1570.T, the same pair for N225, and a flat cash line.  Each asset is paired
    so the report can contrast "commit it all up front" against "average in".

    Lump-sum always deploys the full ``initial_equity``.  ``dca_cap`` caps how
    much the DCA baselines feed into the market (parking the rest in cash on the
    same equity base); pass the strategy's gross-exposure cap to make DCA a
    same-risk-budget twin of the active strategy.  When ``None``, DCA also
    deploys the full ``initial_equity``.
    """
    n = max(len(target_close), len(benchmark_close), 1)
    cash_curve = [float(initial_equity)] * n
    return [
        _result_from_curve(
            "1570.T Buy & Hold", buy_and_hold_curve(target_close, initial_equity), n_trading_days
        ),
        _result_from_curve(
            "1570.T 定額積立(DCA)",
            dca_curve(target_close, initial_equity, dca_cap),
            n_trading_days,
        ),
        _result_from_curve(
            "N225 Buy & Hold", buy_and_hold_curve(benchmark_close, initial_equity), n_trading_days
        ),
        _result_from_curve(
            "N225 定額積立(DCA)",
            dca_curve(benchmark_close, initial_equity, dca_cap),
            n_trading_days,
        ),
        _result_from_curve("Cash (no position)", cash_curve, n_trading_days),
    ]
