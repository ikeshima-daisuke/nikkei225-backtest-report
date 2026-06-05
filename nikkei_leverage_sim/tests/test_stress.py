"""Tests for the stress-testing module (D1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nikkei_leverage_sim.backtest import prepare_market_data, run_backtest
from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.data import join_target_benchmark, make_synthetic_data
from nikkei_leverage_sim import stress


def _full_run(n_days=400, seed=7):
    target, benchmark = make_synthetic_data(n_days=n_days, seed=seed)
    joined = join_target_benchmark(target, benchmark)
    cfg = Config()
    cfg.optimization.enabled = False  # default params -> fast
    md = prepare_market_data(joined, cfg)
    return run_backtest(md, cfg), md, cfg, joined


# --- 1. regime slicing ----------------------------------------------------- #
def test_regime_metrics_slices_window():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-03", "2020-02-04", "2020-02-05", "2020-05-01"]),
            "equity": [100.0, 90.0, 95.0, 200.0],
            "unrealized_pnl": [0.0, -10.0, -5.0, 0.0],
            "benchmark_close": [100.0, 90.0, 95.0, 150.0],
            "target_close": [100.0, 80.0, 90.0, 150.0],
            "maintenance_ratio": [1.0, 0.4, 0.5, np.inf],
            "events": ["", "margin_call", "forced_liquidation", ""],
        }
    )
    r = stress.regime_metrics(daily, "win", "2020-02-01", "2020-02-28")
    assert r is not None
    assert r.sessions == 3  # the May row is excluded
    assert r.peak_unrealized_loss == 10.0
    assert abs(r.min_maintenance_ratio - 0.4) < 1e-9
    assert r.margin_call_days == 1
    assert r.forced_liquidations == 1
    # peak-to-trough within the window: 100 -> 90 == 10%.
    assert abs(r.max_drawdown_pct - 0.10) < 1e-9


def test_regime_metrics_out_of_range_is_none():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-03"]),
            "equity": [100.0],
            "unrealized_pnl": [0.0],
            "benchmark_close": [100.0],
            "target_close": [100.0],
            "maintenance_ratio": [np.inf],
            "events": [""],
        }
    )
    assert stress.regime_metrics(daily, "x", "1990-01-01", "1990-12-31") is None


def test_all_regime_metrics_runs_on_real_shaped_daily():
    result, _md, _cfg, _joined = _full_run()
    daily = pd.DataFrame(result.daily_rows)
    regimes = stress.all_regime_metrics(daily)
    # Synthetic dates are 2019-.. so most named regimes are out of range; the call
    # must not raise and must return a (possibly empty) list of RegimeResult.
    assert isinstance(regimes, list)
    for r in regimes:
        assert r.sessions > 0


# --- 2. cost sensitivity --------------------------------------------------- #
def test_sensitivity_grid_shape_and_monotonicity():
    _result, md, cfg, _joined = _full_run()
    cells = stress.sensitivity_grid(
        md, cfg, cfg.strategy.default_params,
        slippage_bps_list=[2.0, 20.0], interest_rate_list=[0.028, 0.06],
    )
    assert len(cells) == 4
    by_key = {(c.slippage_bps, c.annual_margin_interest_rate): c for c in cells}
    # Higher slippage never improves net realized P&L.
    assert by_key[(20.0, 0.028)].net_realized_after_tax <= by_key[(2.0, 0.028)].net_realized_after_tax
    # Higher interest never improves it either.
    assert by_key[(2.0, 0.06)].net_realized_after_tax <= by_key[(2.0, 0.028)].net_realized_after_tax


# --- 3. bootstrap ruin ----------------------------------------------------- #
def test_block_index_path_length_and_range():
    rng = np.random.default_rng(0)
    idx = stress._block_index_path(rng, n=100, block=20, horizon=250)
    assert len(idx) == 250
    assert idx.min() >= 1 and idx.max() < 100


def test_synthetic_frame_ohlc_ordering():
    target, _benchmark = make_synthetic_data(n_days=120, seed=3)
    ratios = stress._bar_ratios(target.rename(columns=str))
    rng = np.random.default_rng(0)
    idx = stress._block_index_path(rng, n=len(target), block=10, horizon=60)
    frame = stress._synthetic_frame(
        pd.DatetimeIndex(pd.bdate_range("2000-01-03", periods=60)), ratios, idx, 1000.0
    )
    assert (frame["High"] >= frame[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (frame["Low"] <= frame[["Open", "Close"]].min(axis=1) + 1e-9).all()
    assert (frame["Close"] > 0).all()


def test_bootstrap_is_deterministic_and_well_formed():
    target, benchmark = make_synthetic_data(n_days=300, seed=7)
    joined = join_target_benchmark(target, benchmark)
    cfg = Config()
    cfg.initial_equity = 5_000_000.0
    cfg.force_liquidation = True
    a = stress.block_bootstrap_ruin(joined, cfg, n_paths=10, block_size=20, seed=11)
    b = stress.block_bootstrap_ruin(joined, cfg, n_paths=10, block_size=20, seed=11)
    assert a.n_paths == 10 and len(a.paths) == 10
    assert 0.0 <= a.ruin_probability <= 1.0
    assert 0.0 <= a.forced_liquidation_probability <= 1.0
    assert a.median_final_equity == b.median_final_equity  # deterministic given seed
    assert a.p05_final_equity <= a.median_final_equity <= a.p95_final_equity


def test_write_stress_outputs(tmp_path):
    result, md, cfg, joined = _full_run(n_days=300)
    daily = pd.DataFrame(result.daily_rows)
    regimes = stress.all_regime_metrics(daily)
    cells = stress.sensitivity_grid(
        md, cfg, cfg.strategy.default_params, [2.0], [0.028]
    )
    bs = stress.block_bootstrap_ruin(joined, cfg, n_paths=5, block_size=20, seed=1)
    summary = stress.write_stress_outputs(tmp_path, regimes, cells, bs, meta={"seed": 1})
    for fname in ("stress.json", "regimes.csv", "sensitivity.csv", "bootstrap_paths.csv"):
        assert (tmp_path / fname).exists()
    assert "bootstrap" in summary and "sensitivity" in summary
