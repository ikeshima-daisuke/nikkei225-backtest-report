"""Indicator unit tests (offline / synthetic only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nikkei_leverage_sim.indicators import compute_indicators, rsi, rolling_volatility


def test_rsi_constant_uptrend_is_high():
    """A strictly rising series should produce RSI near 100."""
    s = pd.Series(np.arange(1, 60, dtype=float))
    r = rsi(s, 14)
    assert r.iloc[-1] > 99.0


def test_rsi_bounds():
    rng = np.random.default_rng(0)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    r = rsi(s, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_moving_average_gap():
    # Build a benchmark whose last value is exactly 10% above its 25-day MA.
    df = _benchmark_frame(np.linspace(100, 130, 300))
    out = compute_indicators(df)
    b = out["benchmark_adj_close"]
    expected = (b / b.rolling(25).mean() - 1.0)
    np.testing.assert_allclose(
        out["ma_gap_25"].dropna().to_numpy(),
        expected.dropna().to_numpy(),
        rtol=1e-9,
    )


def test_drawdown_252_matches_definition():
    rng = np.random.default_rng(1)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    out = compute_indicators(_benchmark_frame(prices))
    b = out["benchmark_adj_close"]
    roll_high = b.rolling(252, min_periods=1).max()
    expected = b / roll_high - 1.0
    np.testing.assert_allclose(out["drawdown_252"].to_numpy(), expected.to_numpy(), rtol=1e-12)
    # Draw-down is always <= 0.
    assert (out["drawdown_252"] <= 1e-9).all()


def test_vol_20_matches_std_of_returns():
    rng = np.random.default_rng(2)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 300)))
    out = compute_indicators(_benchmark_frame(prices))
    ret = out["benchmark_adj_close"].pct_change()
    expected = rolling_volatility(ret, 20)
    np.testing.assert_allclose(
        out["vol_20"].dropna().to_numpy(), expected.dropna().to_numpy(), rtol=1e-9
    )


def _benchmark_frame(prices: np.ndarray) -> pd.DataFrame:
    """Build a joined-style frame carrying both benchmark and target columns."""
    n = len(prices)
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame(
        {
            "target_open": prices,
            "target_high": prices * 1.01,
            "target_low": prices * 0.99,
            "target_close": prices,
            "target_adj_close": prices,
            "target_volume": np.ones(n),
            "benchmark_open": prices,
            "benchmark_high": prices * 1.01,
            "benchmark_low": prices * 0.99,
            "benchmark_close": prices,
            "benchmark_adj_close": prices,
            "benchmark_volume": np.ones(n),
        },
        index=pd.Index(idx, name="Date"),
    )
    return df
