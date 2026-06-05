"""Tests for the execution-realism model (D2)."""
from __future__ import annotations

import math

import pytest

from nikkei_leverage_sim.backtest import (
    _exec_fill_prices,
    prepare_market_data,
    run_backtest,
    simulate,
)
from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.data import join_target_benchmark, make_synthetic_data
from nikkei_leverage_sim import execution as E


# --- fill-price helper ----------------------------------------------------- #
def test_fill_prices_next_open_uses_open_only():
    buy, sell = _exec_fill_prices("next_open", 100.0, 110.0, 90.0, 105.0, 0.01)
    assert buy == pytest.approx(101.0) and sell == pytest.approx(99.0)


def test_fill_prices_vwap_uses_ohlc_mean():
    mid = (100.0 + 110.0 + 90.0 + 106.0) / 4.0
    buy, sell = _exec_fill_prices("vwap", 100.0, 110.0, 90.0, 106.0, 0.0)
    assert buy == pytest.approx(mid) and sell == pytest.approx(mid)


def test_fill_prices_adverse_buys_high_sells_low():
    buy, sell = _exec_fill_prices("adverse", 100.0, 110.0, 90.0, 105.0, 0.01)
    assert buy == pytest.approx(110.0 * 1.01)
    assert sell == pytest.approx(90.0 * 0.99)


# --- config-driven engine behavior ---------------------------------------- #
def _md(n=400, seed=7, cfg=None):
    t, b = make_synthetic_data(n_days=n, seed=seed)
    joined = join_target_benchmark(t, b)
    return prepare_market_data(joined, cfg or Config())


def _run(fill_model="next_open", delay=0, participation=1.0, n=400):
    cfg = Config()
    cfg.optimization.enabled = False
    cfg.execution.fill_model = fill_model
    cfg.execution.execution_delay_days = delay
    cfg.execution.volume_participation = participation
    md = _md(n=n, cfg=cfg)
    return simulate(md, 0, md.n, cfg.strategy.default_params, cfg, record=False)


def test_unknown_fill_model_raises():
    cfg = Config()
    cfg.execution.fill_model = "nope"
    md = _md(cfg=cfg)
    with pytest.raises(ValueError):
        simulate(md, 0, md.n, cfg.strategy.default_params, cfg)


def test_adverse_model_never_beats_next_open():
    base = _run("next_open")
    adverse = _run("adverse")
    # Worse fills cannot improve realized P&L.
    assert adverse.portfolio.realized_after_tax <= base.portfolio.realized_after_tax + 1e-6
    # And it genuinely changes the outcome.
    assert adverse.portfolio.realized_after_tax != base.portfolio.realized_after_tax


def test_volume_participation_caps_buys():
    base = _run(participation=1.0)
    capped = _run(participation=1e-6)  # cap ~1-5 shares vs ~13 desired
    # A severe volume cap deploys far less capital (the direct effect of partial
    # fills). buy_trade_count is NOT asserted: partial fills feed back into the
    # exposure/unrealized signals and can change how often a buy fires.
    base_exp = base.portfolio.exposure_sum / max(base.portfolio.exposure_obs, 1)
    capped_exp = capped.portfolio.exposure_sum / max(capped.portfolio.exposure_obs, 1)
    assert capped_exp < base_exp * 0.5


def test_execution_delay_changes_results_and_is_default_off():
    base = _run(delay=0)
    delayed = _run(delay=3)
    assert delayed.portfolio.realized_after_tax != base.portfolio.realized_after_tax
    # Default config has zero delay (baseline reproduced elsewhere).
    assert Config().execution.execution_delay_days == 0


def test_negative_delay_raises():
    cfg = Config()
    cfg.execution.execution_delay_days = -1
    md = _md(cfg=cfg)
    with pytest.raises(ValueError):
        simulate(md, 0, md.n, cfg.strategy.default_params, cfg)


def test_delayed_forced_liquidation_resolves_and_closes_book():
    # Thin own funds on a synthetic crash with latency: a delayed forced
    # liquidation must still fire (no infinite postponement) and fully close the
    # book (not just a stale lot snapshot), with no fresh buys during the window.
    t, b = make_synthetic_data(n_days=500, seed=123)
    joined = join_target_benchmark(t, b)
    cfg = Config()
    cfg.optimization.enabled = False
    cfg.force_liquidation = True
    cfg.initial_equity = 300_000
    cfg.execution.execution_delay_days = 3
    md = prepare_market_data(joined, cfg)
    delay = cfg.execution.execution_delay_days
    res = simulate(md, 0, md.n, cfg.strategy.default_params, cfg, record=True)
    assert res.portfolio.forced_liquidation_count > 0  # liquidation did happen
    import pandas as pd

    daily = pd.DataFrame(res.daily_rows).reset_index(drop=True)
    forced_rows = daily.index[daily["events"].str.contains("forced_liquidation", na=False)]
    assert len(forced_rows) > 0
    fidx = int(forced_rows[0])  # the breach/decision day
    # The book is non-empty when the breach is detected ...
    assert daily["gross_exposure"].iloc[fidx] > 0.0
    # ... and is fully closed within the latency window (execution at fidx+1+delay),
    # confirming the whole book is liquidated (not just a stale snapshot) and the
    # forced close is not postponed forever.
    window = daily["gross_exposure"].iloc[fidx : fidx + delay + 3]
    assert (window <= 1e-6).any()


def test_default_config_is_unchanged_baseline():
    # The default (next_open, full fill, no delay) must equal a plain run.
    cfg = Config()
    cfg.optimization.enabled = False
    md = _md(cfg=cfg)
    a = simulate(md, 0, md.n, cfg.strategy.default_params, cfg, record=False)
    b = _run("next_open", 0, 1.0)
    assert a.portfolio.realized_after_tax == b.portfolio.realized_after_tax


# --- comparison module ----------------------------------------------------- #
def test_compare_execution_one_cell_per_scenario_and_baseline_matches():
    cfg = Config()
    cfg.optimization.enabled = False
    md = _md(cfg=cfg)
    cells = E.compare_execution(md, cfg, cfg.strategy.default_params)
    assert len(cells) == len(E.EXECUTION_SCENARIOS)
    assert cells[0].name.startswith("baseline")
    direct = simulate(md, 0, md.n, cfg.strategy.default_params, cfg, record=False)
    assert cells[0].net_realized_after_tax == pytest.approx(direct.portfolio.realized_after_tax)


def test_cfg_with_execution_normalizes_dirty_base():
    # Even if the input config already sets non-default D2 fields, the baseline
    # scenario (empty overrides) must be a clean next_open / full / zero-delay run.
    cfg = Config()
    cfg.execution.fill_model = "adverse"
    cfg.execution.execution_delay_days = 5
    cfg.execution.volume_participation = 0.01
    baseline = E._cfg_with_execution(cfg, {})
    assert baseline.execution.fill_model == "next_open"
    assert baseline.execution.execution_delay_days == 0
    assert baseline.execution.volume_participation == 1.0
    # An override still wins, others stay at default.
    vwap = E._cfg_with_execution(cfg, {"fill_model": "vwap"})
    assert vwap.execution.fill_model == "vwap"
    assert vwap.execution.execution_delay_days == 0


def test_write_execution_outputs(tmp_path):
    cfg = Config()
    cfg.optimization.enabled = False
    md = _md(cfg=cfg)
    cells = E.compare_execution(md, cfg, cfg.strategy.default_params)
    summary = E.write_execution_outputs(tmp_path, cells, meta={"seed": 1})
    assert (tmp_path / "execution.json").exists()
    assert (tmp_path / "execution.csv").exists()
    assert len(summary["scenarios"]) == len(E.EXECUTION_SCENARIOS)
