"""Unit tests for the passive benchmark baselines."""
from __future__ import annotations

import math

from nikkei_leverage_sim.benchmark import build_benchmarks, buy_and_hold_curve


def test_buy_and_hold_curve_scales_with_price():
    curve = buy_and_hold_curve([1000.0, 1100.0, 900.0, 2000.0], 1_000_000.0)
    assert curve[0] == 1_000_000.0
    assert math.isclose(curve[1], 1_100_000.0)
    assert math.isclose(curve[2], 900_000.0)
    assert math.isclose(curve[3], 2_000_000.0)


def test_buy_and_hold_holds_cash_before_first_finite_price():
    # Leading NaN -> capital sits in cash until the first real price.
    curve = buy_and_hold_curve([float("nan"), 1000.0, 1100.0], 1_000_000.0)
    assert curve[0] == 1_000_000.0  # cash
    assert math.isclose(curve[1], 1_000_000.0)  # entry
    assert math.isclose(curve[2], 1_100_000.0)


def test_buy_and_hold_empty_and_all_nan():
    assert buy_and_hold_curve([], 5.0) == [5.0]
    assert buy_and_hold_curve([float("nan"), float("nan")], 5.0) == [5.0, 5.0]


def test_build_benchmarks_order_and_metrics():
    target = [1000.0, 1100.0, 900.0, 1200.0, 2000.0]
    benchmark = [10000.0, 10500.0, 10200.0, 11000.0, 15000.0]
    results = build_benchmarks(target, benchmark, 1_000_000.0, 252)

    assert [r.name for r in results] == [
        "1570.T Buy & Hold",
        "N225 Buy & Hold",
        "Cash (no position)",
    ]
    etf, idx, cash = results
    # 1570.T doubled.
    assert math.isclose(etf.final_equity, 2_000_000.0)
    assert math.isclose(etf.total_return, 1.0)
    # N225 +50%.
    assert math.isclose(idx.total_return, 0.5)
    # Cash is flat: no return, no draw-down, neutral ratios.
    assert cash.total_return == 0.0
    assert cash.max_drawdown_pct == 0.0
    assert cash.sharpe_like == 0.0
    # The ETF had a deeper draw-down than the index in this series.
    assert etf.max_drawdown_pct > idx.max_drawdown_pct > 0.0


def test_benchmark_to_dict_is_json_friendly():
    results = build_benchmarks([1.0, 2.0], [1.0, 2.0], 100.0, 2)
    d = results[0].to_dict()
    assert "equity_curve" not in d  # curve is excluded from the compact dict
    assert set(d) >= {"name", "final_equity", "total_return", "max_drawdown_pct"}
