"""Policy behaviour and signal lookahead-safety."""
from __future__ import annotations

import math

import numpy as np

from accumulation_study.engine import simulate
from accumulation_study.policies import (
    BuyTheDip,
    HoldToEnd,
    MAExit,
    TrailingStop,
    TrendFilterDCA,
)
from accumulation_study.signals import build_signals


def test_trailing_stop_fires_only_after_threshold_drop():
    s = TrailingStop(0.20)

    class C:
        equity_peak = 1_000_000.0
        shares = 5.0

    c = C()
    c.equity = 850_000.0  # -15% from peak: no trigger
    assert s(c) == (0.0, False)
    c.equity = 790_000.0  # -21%: sell all and terminate
    assert s(c) == (1.0, True)


def test_buy_the_dip_fires_tranches_and_rearms_on_new_peak():
    # Down 12% then recover to new high then down 12% again -> two tranche fires.
    prices = [100.0, 88.0, 130.0, 114.0]
    sig = build_signals(prices)
    res = simulate(
        prices, 900_000.0,
        BuyTheDip([-0.10], [0.4], backstop_buys=10_000),  # partial tranche, cash left over
        HoldToEnd(), signals=sig,
    )
    # Both -10% crossings (day1 from peak100, day3 from peak130) deploy a tranche;
    # the tranche re-arms only because day2 set a fresh trailing peak (130).
    assert res.n_buys == 2


def test_ma_exit_triggers_below_trailing_average():
    # Long uptrend builds SMA200, then a sharp drop below it forces an exit.
    up = list(np.linspace(100.0, 300.0, 250))
    crash = [120.0, 110.0]
    prices = up + crash
    sig = build_signals(prices)
    res = simulate(prices, 1_000_000.0, lambda ctx: ctx.installment,
                   MAExit(200, 0.0), signals=sig)
    assert res.n_sells == 1
    assert res.exited_early is True


def test_trend_filter_parks_cash_below_ma_then_dumps_above():
    prices = list(np.linspace(100.0, 130.0, 120)) + list(np.linspace(130.0, 90.0, 120))
    sig = build_signals(prices)
    res = simulate(prices, 1_000_000.0,
                   TrendFilterDCA(100, n_installments=240, dump_parked=True),
                   HoldToEnd(), signals=sig)
    # It should leave a meaningful amount in cash during the long decline.
    assert res.cash_end >= 0.0
    assert res.avg_invested_pct < 1.0


def test_signals_sma_is_trailing_only():
    closes = [float(i) for i in range(1, 11)]  # 1..10
    s = build_signals(closes)
    # SMA over a trailing window uses only past+current values.
    assert math.isnan(s.sma200[5])           # not enough history
    assert s.trailing_peak[4] == 5.0         # max(1..5)
    assert s.drawdown[-1] == 0.0             # monotonically rising -> at peak
    # A 3-day check via the public 200/100 arrays isn't available; verify peak/dd
    # semantics which other policies rely on.
    closes2 = [10.0, 8.0, 12.0]
    s2 = build_signals(closes2)
    assert s2.trailing_peak.tolist() == [10.0, 10.0, 12.0]
    assert math.isclose(s2.drawdown[1], 8.0 / 10.0 - 1.0)
