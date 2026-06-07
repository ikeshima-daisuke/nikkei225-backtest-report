"""Synthetic 1570.T construction: correctness, decay, no-lookahead, fidelity."""
import numpy as np
import pandas as pd
import pytest

from regime_study import financing
from regime_study.build_target import (
    build_synthetic_target,
    calibrate_base_drag,
    synth_close_path,
)


def _n225(closes, *, start="2004-01-05"):
    """Tiny N225-like OHLC frame from a close path (one calendar year)."""
    closes = np.asarray(closes, dtype=float)
    dates = pd.bdate_range(start=start, periods=closes.size)
    opens = np.empty_like(closes)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    high = np.maximum(opens, closes) * 1.01
    low = np.minimum(opens, closes) * 0.99
    df = pd.DataFrame(
        {"Open": opens, "High": high, "Low": low, "Close": closes,
         "Adj Close": closes, "Volume": np.ones_like(closes) * 1e6},
        index=pd.Index(dates, name="Date"),
    )
    return df


def _zero_fee_kwargs(df):
    # Pin calib_rate to the (single-year) call rate so financing cancels and
    # base_drag=0 => zero total fee => pure leverage relationship.
    yr_rate = financing.call_rate(df.index[0].strftime("%Y-%m-%d"))
    return dict(base_drag=0.0, calib_rate=yr_rate)


def test_exact_2x_relationship_with_zero_fee():
    df = _n225([100.0, 105.0, 100.0])  # +5%, then -4.7619%
    close = synth_close_path(df, **_zero_fee_kwargs(df))
    # day1: 2*(+5%) = +10% ; day2: 2*(-4.7619%) = -9.5238%
    assert close.iloc[0] == pytest.approx(10_000.0)
    assert close.iloc[1] / close.iloc[0] - 1.0 == pytest.approx(0.10, abs=1e-9)
    r2 = 100.0 / 105.0 - 1.0
    assert close.iloc[2] / close.iloc[1] - 1.0 == pytest.approx(2 * r2, abs=1e-9)


def test_volatility_decay_round_trip_loses_even_without_fee():
    # Index returns to its start (+10% then -9.0909%) but 2x compounding decays.
    df = _n225([100.0, 110.0, 100.0])
    close = synth_close_path(df, **_zero_fee_kwargs(df))
    assert df["Adj Close"].iloc[-1] == pytest.approx(df["Adj Close"].iloc[0])
    assert close.iloc[-1] < close.iloc[0]  # the leverage decay, with zero costs


def test_fee_drags_a_flat_market_down():
    df = _n225([100.0] * 30)  # perfectly flat index
    flat = synth_close_path(df, base_drag=0.02, calib_rate=0.0)
    assert flat.iloc[-1] < flat.iloc[0]  # pure cost bleed when nothing moves


def test_high_rate_year_decays_faster_than_zirp():
    closes = [100.0] * 60
    df90 = _n225(closes, start="1990-01-03")
    df04 = _n225(closes, start="2004-01-05")
    kw = dict(base_drag=0.008, calib_rate=0.0)
    c90 = synth_close_path(df90, **kw)
    c04 = synth_close_path(df04, **kw)
    assert c90.iloc[-1] < c04.iloc[-1]  # 1990 financing >> ZIRP financing


def test_ohlc_internally_valid():
    rng = np.random.default_rng(0)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 50))
    df = _n225(closes)
    out = build_synthetic_target(df, base_drag=0.01, calib_rate=0.0)
    assert (out["High"] >= out["Low"]).all()
    assert (out["High"] >= out[["Open", "Close"]].max(axis=1) - 1e-6).all()
    assert (out["Low"] <= out[["Open", "Close"]].min(axis=1) + 1e-6).all()
    assert (out["Close"] > 0).all() and (out["Low"] > 0).all()
    # Close path matches the standalone close builder exactly.
    close = synth_close_path(df, base_drag=0.01, calib_rate=0.0)
    assert np.allclose(out["Close"].to_numpy(), close.to_numpy())


def test_no_lookahead_future_change_does_not_move_past():
    rng = np.random.default_rng(1)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 40))
    df = _n225(closes)
    base = synth_close_path(df, base_drag=0.01, calib_rate=0.0).to_numpy()
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("Adj Close")] *= 1.5  # perturb only the last day
    perturbed = synth_close_path(df2, base_drag=0.01, calib_rate=0.0).to_numpy()
    assert np.allclose(base[:-1], perturbed[:-1])  # earlier closes are untouched


def test_determinism():
    df = _n225(100.0 * np.cumprod(1.0 + np.linspace(-0.01, 0.01, 25)))
    a = synth_close_path(df, base_drag=0.015, calib_rate=0.0)
    b = synth_close_path(df, base_drag=0.015, calib_rate=0.0)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_non_finite_input_fails_fast():
    df = _n225([100.0, 101.0, 102.0, 103.0])
    df.iloc[2, df.columns.get_loc("Adj Close")] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        synth_close_path(df, base_drag=0.01, calib_rate=0.0)


# --- Fidelity vs the REAL 1570.T (skipped when the local, gitignored data is absent) ---
def _load_real_pair():
    from pathlib import Path

    from nikkei_leverage_sim.data import read_ohlc_csv
    from regime_study.build_target import read_ohlc

    root = Path(__file__).resolve().parents[2]
    n225_p = root / "data" / "benchmark_N225_long.csv"
    real_p = root / "data" / "target_1570_T.csv"
    if not (n225_p.exists() and real_p.exists()):
        pytest.skip("local market data not present (gitignored); fidelity check skipped")
    n225 = read_ohlc(n225_p)
    real = read_ohlc_csv(real_p)[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    return n225, real


def test_calibration_reproduces_real_1570():
    n225, real = _load_real_pair()
    d = calibrate_base_drag(n225, real)
    assert 1.90 <= d["beta"] <= 2.02            # daily-2x relationship
    assert d["corr_daily_vs_2x"] >= 0.98        # tracks 2x the index tightly
    assert abs(d["cum_synth"] / d["cum_real"] - 1.0) < 0.01  # cumulative matches
