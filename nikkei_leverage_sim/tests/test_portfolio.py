"""Portfolio unit tests."""
from __future__ import annotations

import pandas as pd

from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.portfolio import Portfolio


def _cfg(**kw) -> Config:
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_buy_adds_lot():
    pf = Portfolio(_cfg(commission_bps=0.0))
    lot = pf.buy(0, pd.Timestamp("2020-01-01"), 10, 1000.0)
    assert lot is not None
    assert len(pf.lots) == 1
    assert pf.lots[0].shares == 10
    assert pf.lots[0].entry_value == 10_000.0
    assert pf.buy_trade_count == 1


def test_sell_removes_only_target_lot():
    pf = Portfolio(_cfg())
    a = pf.buy(0, pd.Timestamp("2020-01-01"), 10, 1000.0)
    b = pf.buy(0, pd.Timestamp("2020-01-02"), 5, 1000.0)
    pf.mark_to_market(1100.0, 1)  # both in profit
    rec = pf.sell_lot(a, 2, pd.Timestamp("2020-01-03"), 1100.0)
    assert rec is not None
    assert len(pf.lots) == 1
    assert pf.lots[0].lot_id == b.lot_id


def test_losing_lot_is_not_sold():
    """sell_lot must refuse to realize a loss (no stop-loss allowed)."""
    pf = Portfolio(_cfg())
    lot = pf.buy(0, pd.Timestamp("2020-01-01"), 10, 1000.0)
    pf.mark_to_market(900.0, 1)  # now underwater
    rec = pf.sell_lot(lot, 2, pd.Timestamp("2020-01-03"), 900.0)
    assert rec is None
    assert len(pf.lots) == 1  # lot stays open


def test_daily_interest_accrues():
    cfg = _cfg(annual_margin_interest_rate=0.0365)  # ~0.0001/day
    pf = Portfolio(cfg)
    pf.buy(0, pd.Timestamp("2020-01-01"), 10, 1000.0)  # value 10_000
    pf.mark_to_market(1000.0, 1)
    expected_daily = 10_000.0 * 0.0365 / 365.0
    assert abs(pf.lots[0].accumulated_interest - expected_daily) < 1e-9
    pf.mark_to_market(1000.0, 2)
    assert abs(pf.lots[0].accumulated_interest - 2 * expected_daily) < 1e-9
    assert abs(pf.total_interest_paid - 2 * expected_daily) < 1e-9


def test_tax_only_on_positive_realized_pnl():
    cfg = _cfg(tax_rate=0.2, annual_margin_interest_rate=0.0, commission_bps=0.0)
    pf = Portfolio(cfg)
    lot = pf.buy(0, pd.Timestamp("2020-01-01"), 10, 1000.0)
    pf.mark_to_market(1200.0, 1)
    rec = pf.sell_lot(lot, 2, pd.Timestamp("2020-01-03"), 1200.0)
    # gross = 2000, tax = 400, net_after = 1600
    assert rec["realized_pnl_before_tax"] == 2000.0
    assert rec["tax"] == 400.0
    assert rec["realized_pnl_after_tax"] == 1600.0
    assert pf.total_tax_paid == 400.0


def test_no_tax_charged_when_forced_loss():
    cfg = _cfg(tax_rate=0.2, annual_margin_interest_rate=0.0)
    pf = Portfolio(cfg)
    lot = pf.buy(0, pd.Timestamp("2020-01-01"), 10, 1000.0)
    pf.mark_to_market(800.0, 1)
    rec = pf.sell_lot(lot, 2, pd.Timestamp("2020-01-03"), 800.0, force=True)
    assert rec is not None
    assert rec["realized_pnl_before_tax"] < 0
    assert rec["tax"] == 0.0  # no tax on a loss


def test_margin_ratio_infinite_when_flat():
    pf = Portfolio(_cfg())
    import math

    assert math.isinf(pf.margin_ratio())
