"""Engine invariants: capital basis, no-lookahead, and exit semantics."""
from __future__ import annotations

import math

from accumulation_study.engine import evaluate, month_start_indices, simulate
from accumulation_study.policies import HoldToEnd, LumpSum, TakeProfit, TrailingStop


def _dca(ctx):
    return ctx.installment


def test_equity_starts_at_capital_and_flat_price_preserves_it():
    res = simulate([500.0] * 10, 1_000_000.0, _dca, HoldToEnd())
    assert res.equity[0] == 1_000_000.0
    assert all(math.isclose(v, 1_000_000.0, rel_tol=1e-9) for v in res.equity)


def test_dca_rising_market_matches_closed_form():
    prices = [100.0, 110.0, 121.0, 133.1]
    res = simulate(prices, 1_000_000.0, _dca, HoldToEnd())
    inst = 1_000_000.0 / 4
    shares = sum(inst / p for p in prices)
    assert math.isclose(res.equity[-1], shares * prices[-1], rel_tol=1e-9)


def test_lump_buys_everything_on_day_one():
    prices = [100.0, 200.0]
    res = simulate(prices, 1_000_000.0, LumpSum(), HoldToEnd())
    assert math.isclose(res.equity[-1], 2_000_000.0, rel_tol=1e-9)  # fully invested, doubled
    assert res.n_buys == 1


def test_no_lookahead_future_prices_cannot_change_the_past():
    # Two series identical through index k, divergent after; equity[:k+1] must match.
    base = [100.0, 102.0, 99.0, 105.0, 110.0, 108.0]
    k = 3
    alt = base[: k + 1] + [10.0, 9999.0]  # wild future values
    a = simulate(base, 1_000_000.0, _dca, TrailingStop(0.10))
    b = simulate(alt, 1_000_000.0, _dca, TrailingStop(0.10))
    for i in range(k + 1):
        assert math.isclose(a.equity[i], b.equity[i], rel_tol=1e-12), i


def test_single_round_trip_sells_once_then_holds_cash():
    # Doubling triggers +100% take-profit; after selling, no further buys/sells.
    prices = [100.0, 100.0, 100.0, 250.0, 300.0, 50.0]
    res = simulate(prices, 1_000_000.0, _dca, TakeProfit(1.0), repeated=False)
    assert res.n_sells == 1
    assert res.exited_early is True
    # Final equity is the banked cash, immune to the later crash to 50.
    assert res.equity[-1] == res.equity[-2]


def test_repeated_reenters_after_selling():
    prices = [100.0, 250.0, 100.0, 250.0, 100.0]
    one = simulate(prices, 1_000_000.0, _dca, TakeProfit(1.0), repeated=False)
    rot = simulate(prices, 1_000_000.0, _dca, TakeProfit(1.0), repeated=True)
    assert rot.n_sells >= one.n_sells
    assert rot.exited_early is False


def test_month_start_indices_picks_first_session_per_month():
    dates = ["2020-01-06", "2020-01-07", "2020-02-03", "2020-02-04", "2020-03-02"]
    assert month_start_indices(dates) == [0, 2, 4]


def test_cost_bps_reduces_final_equity():
    prices = [100.0, 100.0, 100.0, 200.0, 50.0]
    free = simulate(prices, 1_000_000.0, LumpSum(), TakeProfit(0.5))
    paid = simulate(prices, 1_000_000.0, LumpSum(), TakeProfit(0.5), cost_bps=50.0)
    # A round trip (buy + sell) pays two fees, so the banked cash is smaller.
    assert paid.equity[-1] < free.equity[-1]
    # Zero cost is a no-op versus the default.
    same = simulate(prices, 1_000_000.0, LumpSum(), TakeProfit(0.5), cost_bps=0.0)
    assert same.equity[-1] == free.equity[-1]


def test_evaluate_reports_calmar_and_invested_fraction():
    res = simulate([100.0, 120.0, 90.0, 150.0], 1_000_000.0, LumpSum(), HoldToEnd())
    m = evaluate(res, 1_000_000.0, 4)
    assert m.calmar != 0.0
    assert 0.99 <= m.avg_invested_pct <= 1.0  # lump is ~fully invested throughout
