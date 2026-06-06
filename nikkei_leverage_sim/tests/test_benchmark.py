"""Unit tests for the passive benchmark baselines."""
from __future__ import annotations

import math

from nikkei_leverage_sim.benchmark import (
    build_benchmarks,
    buy_and_hold_curve,
    dca_curve,
)


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


def test_dca_curve_starts_at_equity_and_ends_fully_invested():
    # Steady +10%/step ramp: every installment compounds to the final price.
    prices = [100.0, 110.0, 121.0, 133.1]
    curve = dca_curve(prices, 1_000_000.0)
    assert curve[0] == 1_000_000.0  # first installment + the rest still cash
    # Fully invested at the end: shares (4 installments of 250k) marked at 133.1.
    inst = 1_000_000.0 / 4
    shares = sum(inst / p for p in prices)
    assert math.isclose(curve[-1], shares * prices[-1])


def test_dca_flat_price_preserves_capital():
    # No price movement -> averaging in neither gains nor loses; equity == equity.
    curve = dca_curve([500.0] * 6, 600_000.0)
    assert all(math.isclose(v, 600_000.0) for v in curve)


def test_dca_lags_lump_sum_in_a_rising_market():
    # In a monotonically rising market, committing up front (lump) beats
    # averaging in (DCA): DCA leaves capital idle in cash early on.
    prices = [100.0, 150.0, 200.0, 400.0]
    dca = dca_curve(prices, 1_000_000.0)
    lump = buy_and_hold_curve(prices, 1_000_000.0)
    assert dca[-1] < lump[-1]
    assert dca[-1] > 1_000_000.0  # still profits from the uptrend


def test_dca_skips_nonfinite_and_forward_fills_valuation():
    # A NaN session buys nothing and does not corrupt the curve; the next finite
    # price resumes investing the remaining installments.
    curve = dca_curve([100.0, float("nan"), 100.0], 900_000.0)
    assert all(v > 0 for v in curve)
    assert curve[1] == curve[0]  # NaN day: no buy, valuation forward-filled
    # Two finite sessions -> two installments of 450k, all at price 100.
    assert math.isclose(curve[-1], 900_000.0)


def test_dca_empty_and_all_nan():
    assert dca_curve([], 5.0) == [5.0]
    assert dca_curve([float("nan"), float("nan")], 5.0) == [5.0, 5.0]


def test_dca_deploy_cap_parks_remainder_in_cash():
    # Cap deployment at 100k of a 1M account: only 100k is fed into the market,
    # the other 900k sits in cash for the whole window.
    prices = [100.0, 200.0]  # doubles
    curve = dca_curve(prices, 1_000_000.0, deploy_cap=100_000.0)
    assert curve[0] == 1_000_000.0
    # Installment 50k/session at price 100 -> 1500 shares total; 900k stays cash.
    # End: 1000 sh from day0 (50k/100) + 250 sh from day1 (50k/200) = 1250 sh.
    shares = 50_000.0 / 100.0 + 50_000.0 / 200.0
    assert math.isclose(curve[-1], shares * 200.0 + 900_000.0)
    # The capped DCA's *total* return is diluted by the idle cash buffer.
    full = dca_curve(prices, 1_000_000.0)
    assert curve[-1] < full[-1]


def test_dca_cap_above_equity_is_a_noop():
    prices = [100.0, 150.0, 90.0]
    assert dca_curve(prices, 1_000.0, deploy_cap=10_000.0) == dca_curve(prices, 1_000.0)


def test_build_benchmarks_order_and_metrics():
    target = [1000.0, 1100.0, 900.0, 1200.0, 2000.0]
    benchmark = [10000.0, 10500.0, 10200.0, 11000.0, 15000.0]
    results = build_benchmarks(target, benchmark, 1_000_000.0, 252)

    assert [r.name for r in results] == [
        "1570.T Buy & Hold",
        "1570.T 定額積立(DCA)",
        "N225 Buy & Hold",
        "N225 定額積立(DCA)",
        "Cash (no position)",
    ]
    etf, etf_dca, idx, idx_dca, cash = results
    # 1570.T doubled.
    assert math.isclose(etf.final_equity, 2_000_000.0)
    assert math.isclose(etf.total_return, 1.0)
    # N225 +50%.
    assert math.isclose(idx.total_return, 0.5)
    # DCA into the same rising assets profits, but lags the lump-sum total
    # return (capital is deployed gradually rather than all up front).
    assert 0.0 < etf_dca.total_return < etf.total_return
    assert 0.0 < idx_dca.total_return < idx.total_return
    # Averaging in also rides a shallower draw-down than the lump-sum twin.
    assert etf_dca.max_drawdown_pct <= etf.max_drawdown_pct
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
