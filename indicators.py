"""Vectorised technical indicators for the backtest.

Kept self-contained so the backtester does not depend on notify.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns to a OHLC DataFrame indexed by date.

    Input must contain columns: Open, High, Low, Close.
    """
    out = df.copy()
    close = out["Close"]

    # --- RSI(14) Wilder ---
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Use Wilder smoothing (ewm with alpha = 1/period, adjust=False) after
    # the first SMA seed. For a backtest of this size this is close enough.
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100.0 - 100.0 / (1.0 + rs)

    # --- Simple moving averages ---
    for n in (20, 25, 50, 75, 200):
        out[f"SMA{n}"] = close.rolling(n).mean()

    # --- Deviation from moving averages (%) ---
    for n in (25, 75, 200):
        out[f"DEV{n}"] = (close - out[f"SMA{n}"]) / out[f"SMA{n}"] * 100.0

    # --- Bollinger Band position (20, 2σ) ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    width = (bb_upper - bb_lower).replace(0, np.nan)
    out["BB_POS"] = (close - bb_lower) / width * 100.0

    # --- 52-week range (rolling 252 trading days) ---
    high_1y = close.rolling(252, min_periods=20).max()
    low_1y = close.rolling(252, min_periods=20).min()
    rng = (high_1y - low_1y).replace(0, np.nan)
    out["POS_52W"] = (close - low_1y) / rng * 100.0
    out["DRAWDOWN"] = (close - high_1y) / high_1y * 100.0  # negative when below high

    # --- Consecutive trading days where drawdown <= -10% ---
    below_10 = (out["DRAWDOWN"] <= -10.0).values
    streak = np.zeros(len(out), dtype=np.int32)
    run = 0
    for i, flag in enumerate(below_10):
        run = run + 1 if flag else 0
        streak[i] = run
    out["STREAK_DD10"] = streak

    # --- Returns ---
    for n in (1, 5, 20, 60):
        out[f"RET{n}"] = close.pct_change(n) * 100.0

    # --- Golden / death cross status ---
    out["GOLDEN"] = out["SMA50"] > out["SMA200"]

    return out


def define_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return mapping of signal-name -> boolean Series aligned to df.index.

    Boolean True means the condition is satisfied on that bar's close.
    Missing values (insufficient warm-up) are treated as False.
    """
    s: dict[str, pd.Series] = {}

    s["RSI<30"] = df["RSI14"] < 30
    s["RSI<35"] = df["RSI14"] < 35
    s["RSI<40"] = df["RSI14"] < 40

    s["BB<20"] = df["BB_POS"] < 20
    s["BB<10"] = df["BB_POS"] < 10

    s["52W<30"] = df["POS_52W"] < 30
    s["52W<20"] = df["POS_52W"] < 20

    s["DD<=-10"] = df["DRAWDOWN"] <= -10
    s["DD<=-15"] = df["DRAWDOWN"] <= -15
    s["DD<=-20"] = df["DRAWDOWN"] <= -20

    s["STREAK>=5"] = df["STREAK_DD10"] >= 5
    s["STREAK>=10"] = df["STREAK_DD10"] >= 10
    s["STREAK>=20"] = df["STREAK_DD10"] >= 20

    s["RET20<-5"] = df["RET20"] < -5
    s["RET5<-3"] = df["RET5"] < -3

    s["vsMA25<-5"] = df["DEV25"] < -5
    s["vsMA25<-7"] = df["DEV25"] < -7
    s["vsMA75<-5"] = df["DEV75"] < -5
    s["vsMA75<-10"] = df["DEV75"] < -10

    s["GOLDEN"] = df["GOLDEN"]

    return {k: v.fillna(False).astype(bool) for k, v in s.items()}
