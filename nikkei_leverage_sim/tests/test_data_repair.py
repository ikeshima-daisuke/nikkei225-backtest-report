"""Tests for the conservative price-glitch repair (data-quality layer)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nikkei_leverage_sim.data import (
    join_target_benchmark,
    load_market_data,
    read_ohlc_csv,
    repair_price_glitches,
)


def _ohlc(closes):
    idx = pd.Index(pd.bdate_range("2021-01-01", periods=len(closes)), name="Date")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": c,
            "High": c * 1.01,
            "Low": c * 0.99,
            "Close": c,
            "Adj Close": c,
            "Volume": np.full(len(c), 1_000_000.0),
        },
        index=idx,
    )


def test_repairs_two_day_plateau_glitch():
    # The real 1570.T 2021-04 signature: ~16k, glitch to ~8k for 2 days, back to 16k.
    df = _ohlc([16300, 16125, 16250, 8040, 8070, 15900, 16080])
    repaired, reps = repair_price_glitches(df)
    assert len(reps) == 2
    fixed = repaired["Close"].to_numpy()
    # Glitch bars are pulled back to the ~16k regime (between the brackets).
    assert 15000 < fixed[3] < 16500
    assert 15000 < fixed[4] < 16500
    # The whole OHLC bar was rescaled, not just Close.
    assert repaired["Open"].iloc[3] > 15000
    assert repaired["High"].iloc[3] > repaired["Low"].iloc[3]
    # Non-glitch bars are untouched.
    assert fixed[0] == 16300 and fixed[-1] == 16080


def test_repairs_single_day_spike():
    df = _ohlc([1000, 1010, 5000, 1005, 1015])  # one fake 5x spike
    repaired, reps = repair_price_glitches(df)
    assert len(reps) == 1
    assert reps[0]["original_close"] == 5000.0
    assert 1000 < repaired["Close"].iloc[2] < 1020


def test_repair_value_is_causal_forward_fill():
    # The repaired value must be the last KNOWN-GOOD close (the bar before the
    # run), never a future-aware interpolation toward the reversion bar.
    df = _ohlc([16300, 16125, 16250, 8040, 8070, 15900, 16080])
    repaired, reps = repair_price_glitches(df)
    assert [r["repaired_close"] for r in reps] == [16250.0, 16250.0]
    assert repaired["Close"].iloc[3] == pytest.approx(16250.0)
    assert repaired["Close"].iloc[4] == pytest.approx(16250.0)


def test_tightened_reversion_leaves_partial_rebound_crash_alone():
    # >40% drop that rebounds only to 0.85x of pre (within the OLD 25% tol but
    # outside the new 10% tol) is a real move, not a glitch -> not repaired.
    _repaired, reps = repair_price_glitches(_ohlc([1000, 540, 850, 860]))
    assert reps == []
    # A genuine snap-back to within 10% IS treated as a glitch.
    _r2, reps2 = repair_price_glitches(_ohlc([1000, 540, 960, 970]))
    assert len(reps2) == 1


def test_two_separate_glitches_both_repaired():
    # In-place repair must let the second glitch be detected with good brackets.
    df = _ohlc([1000, 1010, 5000, 1005, 1015, 5100, 1020, 1010])
    _repaired, reps = repair_price_glitches(df)
    assert len(reps) == 2
    assert [r["date"][-5:] for r in reps]  # both recorded


def test_leaves_real_crash_alone():
    # A sustained 50% decline that does NOT revert -> a real crash, not a glitch.
    df = _ohlc([1000, 950, 600, 500, 480, 520, 510])
    _repaired, reps = repair_price_glitches(df)
    assert reps == []


def test_leaves_clean_data_alone():
    df = _ohlc([1000, 1010, 1005, 1020, 1015, 1030])
    repaired, reps = repair_price_glitches(df)
    assert reps == []
    assert np.allclose(repaired["Close"], df["Close"])


def test_read_ohlc_csv_repairs_and_records(tmp_path):
    df = _ohlc([16300, 16125, 16250, 8040, 8070, 15900, 16080])
    path = tmp_path / "target.csv"
    df.reset_index().to_csv(path, index=False)

    with pytest.warns(UserWarning, match="glitch"):
        repaired = read_ohlc_csv(path)
    assert len(repaired.attrs["data_repairs"]) == 2

    # Opt-out reproduces the raw (glitched) series.
    raw = read_ohlc_csv(path, repair_glitches=False)
    assert raw.attrs["data_repairs"] == []
    assert raw["Close"].iloc[3] == 8040.0


def test_load_market_data_threads_repairs_labelled(tmp_path):
    tgt = _ohlc([16300, 16125, 16250, 8040, 8070, 15900, 16080])
    bch = _ohlc([1000, 1010, 1005, 1020, 1015, 1030, 1025])  # clean
    tp = tmp_path / "t.csv"
    bp = tmp_path / "b.csv"
    tgt.reset_index().to_csv(tp, index=False)
    bch.reset_index().to_csv(bp, index=False)

    with pytest.warns(UserWarning):
        joined = load_market_data(tp, bp)
    reps = joined.attrs["data_repairs"]
    assert len(reps) == 2
    assert all(r["series"] == "target" for r in reps)


def test_join_sets_empty_repairs_for_clean_inputs():
    tgt = _ohlc([1000, 1010, 1005, 1020])
    bch = _ohlc([2000, 2010, 2005, 2020])
    joined = join_target_benchmark(tgt, bch)
    assert joined.attrs["data_repairs"] == []


def test_repair_audit_trail_reaches_summary(tmp_path):
    """End-to-end: a repaired glitch is recorded in summary.json's data_quality."""
    from nikkei_leverage_sim.backtest import prepare_market_data, run_backtest
    from nikkei_leverage_sim.config import Config
    from nikkei_leverage_sim.metrics import build_summary

    n = 300
    rng = np.random.default_rng(0)
    base = 1000.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    base[200] = base[199] * 0.4  # inject a 2-day glitch that reverts
    base[201] = base[199] * 0.4
    base[202] = base[199] * 1.0
    tp = tmp_path / "t.csv"
    bp = tmp_path / "b.csv"
    _ohlc(base).reset_index().to_csv(tp, index=False)
    _ohlc(1000.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))).reset_index().to_csv(
        bp, index=False
    )

    with pytest.warns(UserWarning):
        joined = load_market_data(tp, bp)  # repair happens at CSV-read time
    assert joined.attrs["data_repairs"]

    cfg = Config()
    cfg.optimization.enabled = False
    md = prepare_market_data(joined, cfg)
    assert md.data_repairs  # threaded into MarketData

    result = run_backtest(md, cfg)
    summary = build_summary(result, cfg)
    repairs = summary["data_quality"]["price_glitch_repairs"]
    assert len(repairs) >= 1
    assert "date" in repairs[0] and "series" in repairs[0]
