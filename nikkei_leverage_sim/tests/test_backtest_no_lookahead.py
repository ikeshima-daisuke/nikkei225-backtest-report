"""Engine tests: look-ahead avoidance, open execution and exposure cap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nikkei_leverage_sim.config import Config, StrategyParams
from nikkei_leverage_sim.backtest import prepare_market_data, simulate
from nikkei_leverage_sim.data import join_target_benchmark, make_synthetic_data


def _flat_strategy(base_buy_amount: float = 1_000_000.0) -> StrategyParams:
    """A deterministic 'always buy the same amount' strategy (all weights 0)."""
    return StrategyParams(
        base_buy_amount=base_buy_amount,
        max_daily_buy_amount=2_000_000.0,
        score_threshold=0.0,
        score_scale=2.0,
        w_drawdown=0.0,
        w_rsi=0.0,
        w_ma_gap_25=0.0,
        w_ret_5=0.0,
        w_trend=0.0,
        w_vol=0.0,
        w_exposure=0.0,
        w_unrealized_loss=0.0,
        fixed_profit_yen=10**9,       # never take profit by yen
        base_take_profit_pct=10.0,    # never take profit by pct
        min_take_profit_pct=10.0,
    )


def _cfg(opt_enabled: bool = False) -> Config:
    cfg = Config()
    cfg.optimization.enabled = opt_enabled
    return cfg


def _run(joined, cfg, params):
    md = prepare_market_data(joined, cfg)
    return md, simulate(md, 0, md.n, params, cfg, record=True)


def test_execution_uses_next_open_price():
    """Recorded buy price must equal open*(1+slippage) of the execution day."""
    cfg = _cfg()
    target, benchmark = make_synthetic_data(n_days=400, seed=5)
    joined = join_target_benchmark(target, benchmark)
    md, res = _run(joined, cfg, _flat_strategy())

    slip = cfg.slippage_bps / 10_000.0
    buys = [t for t in res.trades if t["side"] == "BUY"]
    assert buys, "expected at least one buy"
    for t in buys:
        # Find the row index for this trade date.
        idx = int(np.where(md.dates == np.datetime64(t["date"]))[0][0])
        expected_price = md.target_open[idx] * (1.0 + slip)
        assert abs(t["price"] - expected_price) < 1e-6


def test_perturbing_execution_day_close_does_not_change_that_days_open_trade():
    """Changing day-i CLOSE must not change what is traded at day-i OPEN.

    The orders executed at the open of day ``i`` were decided at the close of
    day ``i-1``.  Therefore perturbing the close of day ``i`` may only change
    decisions for day ``i+1`` onwards — never day ``i``'s own open execution.
    This directly verifies 'no look-ahead'.
    """
    cfg = _cfg()
    target, benchmark = make_synthetic_data(n_days=420, seed=11)
    joined = join_target_benchmark(target, benchmark)
    params = _flat_strategy(base_buy_amount=400_000.0)

    md_base, base = _run(joined, cfg, params)
    base_daily = pd.DataFrame(base.daily_rows).set_index("date")

    # Pick a tradable day with an actual buy, in the middle of the series.
    buy_dates = [t["date"] for t in base.trades if t["side"] == "BUY"]
    pivot = buy_dates[len(buy_dates) // 2]

    # Perturb only the CLOSE (and adj close) of the pivot day, massively.
    perturbed = joined.copy()
    perturbed.loc[pivot, "target_close"] *= 3.0
    perturbed.loc[pivot, "target_adj_close"] *= 3.0
    perturbed.loc[pivot, "benchmark_close"] *= 3.0
    perturbed.loc[pivot, "benchmark_adj_close"] *= 3.0

    _, pert = _run(perturbed, cfg, params)
    pert_daily = pd.DataFrame(pert.daily_rows).set_index("date")

    # The open execution on the pivot day must be byte-identical.
    for col in ("target_open", "buy_amount", "buy_shares", "sell_value"):
        assert base_daily.loc[pivot, col] == pert_daily.loc[pivot, col], col


def test_first_day_has_no_trade():
    """Nothing can be executed on the very first row (no prior signal)."""
    cfg = _cfg()
    target, benchmark = make_synthetic_data(n_days=300, seed=3)
    joined = join_target_benchmark(target, benchmark)
    _, res = _run(joined, cfg, _flat_strategy())
    first = res.daily_rows[0]
    assert first["buy_shares"] == 0
    assert first["sell_value"] == 0.0


def test_gross_exposure_never_exceeds_cap_on_buy():
    """With flat prices the close-marked exposure must respect the 10M cap."""
    cfg = _cfg()
    cfg.max_gross_exposure = 10_000_000.0

    n = 260
    idx = pd.bdate_range("2020-01-01", periods=n)
    price = np.full(n, 1000.0)
    frame = {
        "Open": price, "High": price, "Low": price,
        "Close": price, "Adj Close": price, "Volume": np.ones(n),
    }
    target_df = pd.DataFrame(frame, index=pd.Index(idx, name="Date"))
    benchmark_df = target_df.copy()
    joined = join_target_benchmark(target_df, benchmark_df)

    params = _flat_strategy(base_buy_amount=2_000_000.0)  # ~1M/day
    md, res = _run(joined, cfg, params)

    gross = pd.DataFrame(res.daily_rows)["gross_exposure"].to_numpy()
    assert gross.max() <= cfg.max_gross_exposure + 1e-6
    # Exposure should actually approach the cap and then hit the limit.
    assert gross.max() > 0.9 * cfg.max_gross_exposure
    assert res.exposure_limit_hit_count > 0
