"""Technical indicator calculations.

All indicators are computed *causally*: the value at row ``t`` only depends on
data up to and including row ``t``.  This is essential for avoiding look-ahead
bias in the backtest (we decide trades on day ``t`` using these values, then
execute on day ``t+1``).

Benchmark indicators are computed from the benchmark Adjusted Close (per the
spec), while a few target-ETF indicators are computed from the target series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI.

    Uses an exponential moving average with ``alpha = 1/period`` to approximate
    Wilder smoothing.  Returns values in ``[0, 100]``.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When average loss is zero (only gains) RSI is 100.
    out = out.where(avg_loss != 0.0, 100.0)
    # Leading value has no delta; keep as NaN until enough data.
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of daily returns (sample std)."""
    return returns.rolling(window).std(ddof=1)


def compute_indicators(joined: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators on a joined target+benchmark frame.

    Parameters
    ----------
    joined:
        DataFrame indexed by ``Date`` containing at least the columns
        ``benchmark_adj_close``, ``benchmark_high``, ``benchmark_low``,
        ``benchmark_close`` and ``target_close``.

    Returns
    -------
    DataFrame
        A copy of ``joined`` with indicator columns appended.
    """
    out = joined.copy()
    b = out["benchmark_adj_close"]

    # --- Benchmark returns ---
    out["ret_1"] = b.pct_change(1)
    out["ret_5"] = b.pct_change(5)
    out["ret_20"] = b.pct_change(20)

    # --- 252-day rolling high & drawdown ---
    out["rolling_high_252"] = b.rolling(252, min_periods=1).max()
    out["drawdown_252"] = b / out["rolling_high_252"] - 1.0

    # --- Moving averages and gaps ---
    out["ma_25"] = b.rolling(25).mean()
    out["ma_75"] = b.rolling(75).mean()
    out["ma_200"] = b.rolling(200).mean()
    out["ma_gap_25"] = b / out["ma_25"] - 1.0
    out["ma_gap_75"] = b / out["ma_75"] - 1.0
    out["ma_gap_200"] = b / out["ma_200"] - 1.0

    # --- Volatility ---
    out["vol_20"] = rolling_volatility(out["ret_1"], 20)
    out["vol_60"] = rolling_volatility(out["ret_1"], 60)

    # --- RSI / ATR on the benchmark ---
    out["RSI_14"] = rsi(b, 14)
    if {"benchmark_high", "benchmark_low", "benchmark_close"}.issubset(out.columns):
        out["ATR_14"] = atr(
            out["benchmark_high"], out["benchmark_low"], out["benchmark_close"], 14
        )
    else:  # pragma: no cover - benchmark always carries OHLC in practice
        out["ATR_14"] = np.nan

    # --- Target ETF indicators ---
    t = out["target_close"]
    out["target_ret_1"] = t.pct_change(1)
    out["target_vol_20"] = rolling_volatility(out["target_ret_1"], 20)
    if {"target_high", "target_low", "target_close"}.issubset(out.columns):
        out["target_atr_14"] = atr(
            out["target_high"], out["target_low"], out["target_close"], 14
        )
    else:  # pragma: no cover
        out["target_atr_14"] = np.nan

    return out
