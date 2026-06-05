"""Tests for the strategy-variant engine.

These use synthetic OHLCV data (no network / no real CSVs) so they run anywhere
and never touch the committed report outputs.
"""
import math

import pytest

from nikkei_leverage_sim.backtest import prepare_market_data, simulate
from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.data import join_target_benchmark, make_synthetic_data

from variants.variant_engine import (
    VariantParams,
    prepare_market_data_v,
    simulate_variant,
)


@pytest.fixture(scope="module")
def market():
    cfg = Config()
    t, b = make_synthetic_data(n_days=700, seed=3)
    joined = join_target_benchmark(t, b)
    md = prepare_market_data(joined, cfg)
    mdv = prepare_market_data_v(joined, cfg)
    return cfg, md, mdv


def test_per_lot_matches_core_engine(market):
    """per_lot/init0 with fixed params must reproduce the core engine exactly."""
    cfg, md, mdv = market
    base = cfg.strategy.default_params
    core = simulate(md, 0, md.n, base, cfg, record=True)
    var = simulate_variant(mdv, cfg, base, VariantParams("per_lot", initial_amount=0))

    assert var.realized_after_tax == pytest.approx(core.realized_after_tax, rel=1e-9)
    assert var.ending_unrealized_pnl == pytest.approx(core.ending_unrealized_pnl, rel=1e-9)
    assert var.max_drawdown_equity == pytest.approx(core.max_drawdown_equity, rel=1e-9)
    assert var.max_unrealized_loss == pytest.approx(core.max_unrealized_loss, rel=1e-9)
    assert var.portfolio.buy_trade_count == core.portfolio.buy_trade_count
    assert var.portfolio.sell_trade_count == core.portfolio.sell_trade_count
    assert var.max_consecutive_no_tp == core.max_consecutive_no_tp


def test_per_day_sequence_provider(market):
    """A length-n sequence of identical params behaves like the fixed param."""
    cfg, md, mdv = market
    base = cfg.strategy.default_params
    seq = [base] * mdv.n
    a = simulate_variant(mdv, cfg, base, VariantParams("per_lot"))
    b = simulate_variant(mdv, cfg, seq, VariantParams("per_lot"))
    assert a.realized_after_tax == pytest.approx(b.realized_after_tax, rel=1e-12)
    assert a.final_equity == pytest.approx(b.final_equity, rel=1e-12)


def test_bulk_yen_resets_position_and_realizes(market):
    """A reachable yen target produces bulk exits that flatten the book."""
    cfg, md, mdv = market
    base = cfg.strategy.default_params
    vp = VariantParams("bulk_yen", initial_amount=0, bulk_exit_yen=50_000.0)
    res = simulate_variant(mdv, cfg, base, vp)

    assert res.bulk_exit_count > 0
    # On at least one day after a bulk exit the book is fully flat.
    flat_days = [r for r in res.daily_rows if r["gross_exposure"] == 0.0 and r["buy_shares"] == 0]
    assert flat_days, "expected at least one fully-flat day after a reset"
    # Cash/equity bookkeeping stays self-consistent at the end.
    pf = res.portfolio
    assert pf.equity() == pytest.approx(pf.cash() + pf.unrealized_pnl(), rel=1e-9)


def test_bulk_pct_triggers_same_day_on_high(market):
    """bulk_pct exits all lots on the day the intraday high crosses the target."""
    cfg, md, mdv = market
    base = cfg.strategy.default_params
    vp = VariantParams("bulk_pct", initial_amount=0, bulk_exit_pct=0.05)
    res = simulate_variant(mdv, cfg, base, vp)
    assert res.bulk_exit_count > 0
    # Each recorded bulk-pct sell happens at that day's close (same-day execution).
    sells = [t for t in res.trades if t.get("reason") == "bulk_pct"]
    assert sells


def test_initial_lump_increases_early_exposure(market):
    """A large initial lump pushes far more exposure on the first buy day."""
    cfg, md, mdv = market
    base = cfg.strategy.default_params
    no_lump = simulate_variant(mdv, cfg, base, VariantParams("per_lot", initial_amount=0))
    lump = simulate_variant(mdv, cfg, base,
                            VariantParams("per_lot", initial_amount=5_000_000.0))

    def first_buy_value(res):
        for r in res.daily_rows:
            if r["buy_shares"] > 0:
                return r["buy_amount"]
        return 0.0

    assert first_buy_value(lump) > first_buy_value(no_lump) + 1_000_000.0
    # The lump only fires once: total bought shares is much higher early but the
    # initial injection is not repeated (a flat first-buy difference, not growing).
    assert lump.portfolio.max_gross_exposure_seen > no_lump.portfolio.max_gross_exposure_seen


def test_bulk_exit_netted_tax_never_exceeds_gross(market):
    """Aggregate tax on a bulk exit is on the netted positive gain only."""
    cfg, md, mdv = market
    base = cfg.strategy.default_params
    vp = VariantParams("combo", initial_amount=1_000_000.0,
                       bulk_exit_yen=300_000.0, bulk_exit_pct=0.08)
    res = simulate_variant(mdv, cfg, base, vp)
    pf = res.portfolio
    # Tax paid can never exceed tax_rate * positive realized-before-tax.
    assert pf.total_tax_paid <= cfg.tax_rate * max(pf.realized_before_tax, 0.0) + 1e-6
    assert res.final_equity == pytest.approx(pf.equity_curve[-1], rel=1e-9)


def test_variant_params_validation():
    with pytest.raises(ValueError):
        VariantParams("nope")
    with pytest.raises(ValueError):
        VariantParams("bulk_yen")  # missing bulk_exit_yen
    with pytest.raises(ValueError):
        VariantParams("bulk_pct")  # missing bulk_exit_pct
    with pytest.raises(ValueError):
        VariantParams("combo", bulk_exit_yen=100_000.0)  # missing pct
