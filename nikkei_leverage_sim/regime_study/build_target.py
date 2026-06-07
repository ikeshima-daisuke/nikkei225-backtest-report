"""Reconstruct a synthetic 1570.T (2x daily-rebalanced ETF) from real N225.

1570.T listed only in 2012, so the lost decades cannot be tested on it directly.
But a daily-rebalanced 2x ETF is a *deterministic* function of the underlying
index's daily returns minus financing/expense drag — and the volatility decay
that hurts it in sideways/bear markets emerges automatically from compounding
``leverage × daily simple return``.  So we rebuild it from real ``^N225``.

Faithfulness is not assumed — it is **calibrated and checked against the real
1570.T over 2014–2026** (:func:`calibrate_base_drag`, and the package tests):
the construction reproduces beta≈2.0, daily-return correlation ≈0.99, and the
realized cumulative path to the basis point.  Only then is the same constructor
applied to 1989–2013, where there is no ground truth.

Construction (simple returns, per session ``t``)::

    close_ret = leverage * (n225_close[t]/n225_close[t-1] - 1) - daily_fee[t]
    target_close[t] = target_close[t-1] * (1 + close_ret)

with O/H/L scaled by the *leveraged* move from the prior close (the daily fee is
charged once, at the close), then clamped so OHLC stays internally valid.  The
per-session ``daily_fee`` comes from :mod:`regime_study.financing` and varies with
the prevailing JPY call rate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import financing

DEFAULT_LEVERAGE = 2.0
DEFAULT_BASE_PRICE = 10_000.0
# Window used to calibrate ``base_drag`` against the real 1570.T (low-rate era).
CALIB_START = "2014-01-06"
CALIB_END = "2026-06-05"


def read_ohlc(path: str | Path) -> pd.DataFrame:
    """Read an OHLCV CSV (``Date`` index); used for the long real N225 series."""
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df


def _simple_returns(close: np.ndarray) -> np.ndarray:
    out = np.empty(close.size)
    out[0] = 0.0
    out[1:] = close[1:] / close[:-1] - 1.0
    return out


def _require_finite(n225: pd.DataFrame, cols) -> None:
    """Fail fast on non-finite price inputs (one NaN would poison every later bar
    via ``cumprod``); points at the first bad date/column."""
    for c in cols:
        vals = n225[c].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            bad = n225.index[~np.isfinite(vals)][0]
            raise ValueError(
                f"build_target: non-finite {c!r} at {bad.date()}; "
                "clean or repair the series before constructing the synthetic target")


def synth_close_path(
    n225: pd.DataFrame,
    *,
    leverage: float = DEFAULT_LEVERAGE,
    base_drag: float,
    calib_rate: float,
    base_price: float = DEFAULT_BASE_PRICE,
) -> pd.Series:
    """Just the synthetic close series (the accumulation engine needs only this)."""
    _require_finite(n225, ["Adj Close"])
    nc = n225["Adj Close"].to_numpy(dtype=float)
    idx_ret = _simple_returns(nc)
    dates = n225.index
    fees = np.array(
        [
            financing.daily_fee(
                d.strftime("%Y-%m-%d"),
                leverage=leverage, base_drag=base_drag, calib_rate=calib_rate,
            )
            for d in dates
        ]
    )
    fees[0] = 0.0
    close_ret = leverage * idx_ret - fees
    close = base_price * np.cumprod(1.0 + close_ret)
    return pd.Series(close, index=dates, name="target_close")


def build_synthetic_target(
    n225: pd.DataFrame,
    *,
    leverage: float = DEFAULT_LEVERAGE,
    base_drag: float,
    calib_rate: float,
    base_price: float = DEFAULT_BASE_PRICE,
) -> pd.DataFrame:
    """Full synthetic 1570.T OHLCV frame (for the core margin/ruin engine)."""
    _require_finite(n225, ["Open", "High", "Low", "Adj Close"])
    dates = n225.index
    nc = n225["Adj Close"].to_numpy(dtype=float)
    no = n225["Open"].to_numpy(dtype=float)
    nh = n225["High"].to_numpy(dtype=float)
    nl = n225["Low"].to_numpy(dtype=float)
    nv = n225["Volume"].to_numpy(dtype=float)

    close = synth_close_path(
        n225, leverage=leverage, base_drag=base_drag,
        calib_rate=calib_rate, base_price=base_price,
    ).to_numpy()

    n = close.size
    open_ = np.empty(n)
    high = np.empty(n)
    low = np.empty(n)
    open_[0] = close[0]
    high[0] = close[0] * (1.0 + leverage * max(0.0, nh[0] / nc[0] - 1.0))
    low[0] = close[0] * (1.0 + leverage * min(0.0, nl[0] / nc[0] - 1.0))
    for t in range(1, n):
        prev = close[t - 1]
        open_[t] = prev * (1.0 + leverage * (no[t] / nc[t - 1] - 1.0))
        h_lvl = prev * (1.0 + leverage * (nh[t] / nc[t - 1] - 1.0))
        l_lvl = prev * (1.0 + leverage * (nl[t] / nc[t - 1] - 1.0))
        # Keep OHLC internally valid around the realized open/close.
        high[t] = max(open_[t], close[t], h_lvl)
        low[t] = max(1e-9, min(open_[t], close[t], l_lvl))

    out = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": nv,
        },
        index=dates,
    )
    out.index.name = "Date"
    return out


def calibrate_base_drag(
    n225: pd.DataFrame,
    real_target: pd.DataFrame,
    *,
    leverage: float = DEFAULT_LEVERAGE,
    calib_start: str = CALIB_START,
    calib_end: str = CALIB_END,
) -> dict:
    """Solve ``base_drag`` so the synthetic cumulative matches real 1570.T.

    Returns a dict with ``base_drag``, ``calib_rate`` and validation diagnostics
    (beta, daily-return correlation, annualized tracking error, cumulative ratio)
    over the overlap window.
    """
    from scipy.optimize import brentq  # local import (scipy is a core dep)

    real = real_target.loc[calib_start:calib_end]
    nb = n225.loc[real.index.min():real.index.max()]
    both = pd.DataFrame(
        {"n": nb["Adj Close"], "r": real["Adj Close"]}
    ).dropna()
    calib_rate = financing.mean_call_rate(calib_start, calib_end)

    nr = both["n"].pct_change().dropna()
    rr = both["r"].pct_change().dropna()
    common = nr.index.intersection(rr.index)
    nr, rr = nr.loc[common], rr.loc[common]
    target_cum = float((1.0 + rr).prod())

    # daily fee for these dates given a trial base_drag
    rates = np.array([financing.call_rate(d.strftime("%Y-%m-%d")) for d in nr.index])
    fee_ex_base = (leverage - 1.0) * (rates - calib_rate) / financing.TRADING_DAYS

    def cum_minus_target(base_drag: float) -> float:
        fee = base_drag / financing.TRADING_DAYS + fee_ex_base
        return float(np.prod(1.0 + leverage * nr.to_numpy() - fee)) - target_cum

    lo_b, hi_b = -0.05, 0.20
    f_lo, f_hi = cum_minus_target(lo_b), cum_minus_target(hi_b)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"calibrate_base_drag: bracket [{lo_b}, {hi_b}] does not straddle a root "
            f"(f(lo)={f_lo:.4g}, f(hi)={f_hi:.4g}, target_cum={target_cum:.4g}); "
            "the synthetic-vs-real cumulative gap is outside the expected range — "
            "check the overlap window and inputs.")
    base_drag = float(brentq(cum_minus_target, lo_b, hi_b))

    # diagnostics
    fee = base_drag / financing.TRADING_DAYS + fee_ex_base
    syn_ret = leverage * nr.to_numpy() - fee
    beta = float(np.polyfit(nr.to_numpy(), rr.to_numpy(), 1)[0])
    corr = float(np.corrcoef(rr.to_numpy(), leverage * nr.to_numpy())[0, 1])
    te = float(np.std(syn_ret - rr.to_numpy(), ddof=1) * np.sqrt(financing.TRADING_DAYS))
    return {
        "base_drag": base_drag,
        "base_drag_pct": round(base_drag * 100.0, 3),
        "calib_rate": calib_rate,
        "leverage": leverage,
        "calib_start": calib_start,
        "calib_end": calib_end,
        "n_days": int(common.size),
        "beta": round(beta, 4),
        "corr_daily_vs_2x": round(corr, 5),
        "tracking_error_ann": round(te, 4),
        "cum_real": round(target_cum, 4),
        "cum_synth": round(float(np.prod(1.0 + syn_ret)), 4),
    }


def write_daily_csv(close: pd.Series, path: str | Path) -> None:
    """Write a ``date,target_close`` CSV (the accumulation engine's input)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"date": close.index.strftime("%Y-%m-%d"),
                       "target_close": close.to_numpy()})
    df.to_csv(path, index=False)


def write_ohlc_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a full OHLCV CSV (the core margin/ruin engine's target input)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(path, index=False)
