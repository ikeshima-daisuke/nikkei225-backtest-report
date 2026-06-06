"""Pre-computed, lookahead-safe technical signals for the close series.

Every array is indexed by session ``t`` and uses **only** closes at or before
``t`` (trailing windows).  Computing them once per price series — instead of
re-slicing inside each policy on every day — keeps the grid fast while making
the no-lookahead guarantee structural: a policy can read ``signals.sma200[t]``
freely because that value, by construction, never peeks past ``t``.

Where a window has insufficient history (e.g. SMA-200 before day 200) the value
is ``NaN``; policies treat ``NaN`` as "no signal yet" and fall back to a stated
default (documented at each policy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Signals:
    closes: np.ndarray
    trailing_peak: np.ndarray   # max(close[0..t])
    drawdown: np.ndarray        # close[t]/trailing_peak[t] - 1  (<= 0)
    sma100: np.ndarray
    sma200: np.ndarray
    rsi14: np.ndarray           # Wilder RSI(14), 0..100, NaN until seeded
    vol20_ann: np.ndarray       # annualized 20d realized vol of daily returns
    mom252: np.ndarray          # close[t]/close[t-252] - 1, NaN for t<252


def _sma(closes: np.ndarray, n: int) -> np.ndarray:
    out = np.full(closes.size, np.nan)
    if closes.size >= n:
        c = np.cumsum(closes, dtype=float)
        c[n:] = c[n:] - c[:-n]
        out[n - 1 :] = c[n - 1 :] / n
    return out


def _rsi_wilder(closes: np.ndarray, n: int = 14) -> np.ndarray:
    out = np.full(closes.size, np.nan)
    if closes.size <= n:
        return out
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = gain[:n].mean()
    avg_loss = loss[:n].mean()
    for i in range(n, closes.size):
        if i > n:
            g = gain[i - 1]
            l = loss[i - 1]
            avg_gain = (avg_gain * (n - 1) + g) / n
            avg_loss = (avg_loss * (n - 1) + l) / n
        rs = np.inf if avg_loss == 0 else avg_gain / avg_loss
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _vol20_annualized(closes: np.ndarray, n: int = 20) -> np.ndarray:
    out = np.full(closes.size, np.nan)
    if closes.size < n + 1:
        return out
    rets = np.empty(closes.size)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0
    for t in range(n, closes.size):
        window = rets[t - n + 1 : t + 1]
        out[t] = float(np.std(window, ddof=1)) * np.sqrt(252.0)
    return out


def build_signals(closes) -> Signals:
    arr = np.asarray(closes, dtype=float)
    peak = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, arr / peak - 1.0, 0.0)
    mom = np.full(arr.size, np.nan)
    if arr.size > 252:
        mom[252:] = arr[252:] / arr[:-252] - 1.0
    return Signals(
        closes=arr,
        trailing_peak=peak,
        drawdown=dd,
        sma100=_sma(arr, 100),
        sma200=_sma(arr, 200),
        rsi14=_rsi_wilder(arr, 14),
        vol20_ann=_vol20_annualized(arr, 20),
        mom252=mom,
    )
