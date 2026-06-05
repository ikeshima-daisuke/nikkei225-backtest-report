"""Passive benchmark baselines for context (Week 1 — "vs just holding").

The active strategy is leveraged DCA with take-profit; a single number in
isolation cannot say whether the risk taken was worth it.  These baselines give
an apples-to-apples *capital* comparison: deploy the **same initial equity** and
either buy-and-hold the traded ETF (1570.T), buy-and-hold the index (N225), or
sit in cash.

Caveat (surfaced in the report): the strategy caps gross exposure far below the
initial equity and dollar-cost-averages, so a lump-sum buy-and-hold of the full
initial equity is a *more aggressive* deployment of capital.  The baselines are
reference points, not like-for-like risk twins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

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
) -> List[BenchmarkResult]:
    """Build the standard passive baselines over the backtest window.

    Returns buy-and-hold of 1570.T, buy-and-hold of N225, and a flat cash line,
    in that order.
    """
    n = max(len(target_close), len(benchmark_close), 1)
    cash_curve = [float(initial_equity)] * n
    return [
        _result_from_curve(
            "1570.T Buy & Hold", buy_and_hold_curve(target_close, initial_equity), n_trading_days
        ),
        _result_from_curve(
            "N225 Buy & Hold", buy_and_hold_curve(benchmark_close, initial_equity), n_trading_days
        ),
        _result_from_curve("Cash (no position)", cash_curve, n_trading_days),
    ]
