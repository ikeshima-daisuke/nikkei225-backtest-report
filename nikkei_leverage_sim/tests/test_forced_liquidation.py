"""Tests for the forced-liquidation (追証 / 強制ロスカット) model.

These cover three layers:

1. The pure :meth:`Portfolio.maintenance_ratio` / :meth:`Portfolio.force_liquidation_check`
   predicate (the maintenance-margin breach detector).
2. The interaction with :meth:`Portfolio.sell_lot` — a breach is the *only* path
   allowed to realize a loss (``force=True``), while the normal strategy path is
   still forbidden from realizing losses (no regression of the existing guard).
3. An end-to-end backtest on synthetic crash data: with the model enabled the
   book is force-closed (``forced_liquidation_count > 0``); with it disabled the
   breaches are still *counted* but no liquidation happens.
"""
from __future__ import annotations

import math

import pandas as pd

from nikkei_leverage_sim.backtest import prepare_market_data, run_backtest
from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.data import join_target_benchmark, make_synthetic_data
from nikkei_leverage_sim.portfolio import Portfolio

TS = pd.Timestamp("2020-01-01")
DEFAULT_MAINTENANCE = Config().maintenance_margin_ratio  # 0.30


def _cfg(**kw) -> Config:
    cfg = Config()
    # Zero out frictions so the maintenance-ratio arithmetic is exact in tests.
    cfg.annual_margin_interest_rate = 0.0
    cfg.commission_bps = 0.0
    cfg.tax_rate = 0.0
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
# 1. maintenance_ratio / force_liquidation_check predicate
# --------------------------------------------------------------------------- #
def test_maintenance_ratio_infinite_when_flat():
    pf = Portfolio(_cfg(initial_equity=1_000_000))
    assert math.isinf(pf.maintenance_ratio())
    assert pf.force_liquidation_check() is False  # nothing to liquidate


def test_force_liquidation_check_fires_only_below_threshold():
    # Own funds 350k against a 1,000,000 cost-basis position (~2.86x leverage).
    cfg = _cfg(initial_equity=350_000, maintenance_margin_ratio=0.30)
    pf = Portfolio(cfg)
    pf.buy(0, TS, 100, 10_000.0)  # entry_value = 1,000,000, gross = 1,000,000

    # No move: ratio = (350k + 0) / 1,000,000 = 0.35 >= 0.30 -> no breach.
    pf.mark_to_market(10_000.0, 1)
    assert abs(pf.maintenance_ratio() - 0.35) < 1e-9
    assert pf.force_liquidation_check() is False
    assert pf.margin_call_count == 0

    # 10% drop: upnl = -100k, gross = 900k -> ratio = 250k / 900k ≈ 0.278 < 0.30.
    pf.mark_to_market(9_000.0, 2)
    ratio = pf.maintenance_ratio()
    assert ratio < cfg.maintenance_margin_ratio
    assert pf.force_liquidation_check() is True
    assert pf.margin_call_count == 1
    # The running minimum maintenance ratio is recorded for reporting.
    assert pf.min_maintenance_ratio_seen <= ratio


def test_over_collateralized_book_never_breaches():
    """The production ¥100M-own-funds / ≤¥10M-position regime is inert."""
    cfg = _cfg(initial_equity=100_000_000, maintenance_margin_ratio=0.30)
    pf = Portfolio(cfg)
    pf.buy(0, TS, 1000, 10_000.0)  # entry_value = 10,000,000
    pf.mark_to_market(5_000.0, 1)  # a brutal 50% crash
    # ratio = (100M + (5M - 10M)) / 5M = 95M / 5M = 19 -> nowhere near 0.30.
    assert pf.maintenance_ratio() > 1.0
    assert pf.force_liquidation_check() is False


# --------------------------------------------------------------------------- #
# 2. sell_lot: forced path bypasses the loss guard; strategy path does not
# --------------------------------------------------------------------------- #
def test_loss_guard_intact_but_force_bypasses_it():
    pf = Portfolio(_cfg(initial_equity=350_000, tax_rate=0.20315))
    lot = pf.buy(0, TS, 100, 10_000.0)
    pf.mark_to_market(9_000.0, 1)  # underwater

    # Normal strategy sell still refuses to realize the loss (no regression).
    assert pf.sell_lot(lot, 2, TS, 9_000.0) is None
    assert len(pf.lots) == 1

    # Forced liquidation is the sole exception: it closes the losing lot.
    rec = pf.sell_lot(lot, 2, TS, 9_000.0, force=True)
    assert rec is not None
    assert rec["realized_pnl_before_tax"] < 0
    assert rec["tax"] == 0.0  # no tax on a realized loss
    assert rec["event_type"] == "forced_liquidation"
    assert rec["reason"] == "forced_liquidation"
    assert len(pf.lots) == 0


# --------------------------------------------------------------------------- #
# 3. End-to-end: enabled vs disabled on the same synthetic crash data
# --------------------------------------------------------------------------- #
def _market_data():
    target, benchmark = make_synthetic_data(n_days=500, seed=123)
    joined = join_target_benchmark(target, benchmark)
    return prepare_market_data(joined, Config())


def _run(md, *, force: bool, initial_equity: float) -> "BacktestResult":
    cfg = Config()
    cfg.force_liquidation = force
    cfg.initial_equity = initial_equity
    cfg.optimization.enabled = False  # default params throughout -> fast & deterministic
    return run_backtest(md, cfg)


def test_forced_liquidation_triggers_end_to_end():
    md = _market_data()
    # Thin own-funds base so the synthetic draw-down regimes breach maintenance.
    res_on = _run(md, force=True, initial_equity=300_000)
    pf = res_on.portfolio

    assert pf.forced_liquidation_count > 0, "expected at least one forced liquidation"
    assert pf.margin_call_count > 0
    assert math.isfinite(pf.min_maintenance_ratio_seen)
    assert pf.min_maintenance_ratio_seen < DEFAULT_MAINTENANCE

    forced_trades = [t for t in res_on.trades if t.get("event_type") == "forced_liquidation"]
    assert forced_trades, "forced liquidation must appear in the trade log"
    # A forced liquidation is allowed to realize a loss.
    assert any(t["realized_pnl_before_tax"] < 0 for t in forced_trades)


def test_disabled_model_counts_breaches_but_does_not_liquidate():
    md = _market_data()
    res_off = _run(md, force=False, initial_equity=300_000)
    pf = res_off.portfolio

    # The breach is still observed and reported (so a no-liquidation run shows
    # how many days it *would* have been margin-called)...
    assert pf.margin_call_count > 0
    # ...but nothing is force-closed, and no loss is ever realized.
    assert pf.forced_liquidation_count == 0
    assert all(t.get("event_type") != "forced_liquidation" for t in res_off.trades)
    assert all(t["realized_pnl_before_tax"] >= 0 for t in res_off.trades if t["side"] == "SELL")
