"""Regime definitions and window slicing."""
import numpy as np
import pandas as pd

from regime_study.regimes import REGIME_BY_KEY, REGIMES, slice_window


def test_keys_unique_and_one_baseline():
    keys = [r.key for r in REGIMES]
    assert len(keys) == len(set(keys))
    baselines = [r for r in REGIMES if r.kind == "baseline"]
    assert len(baselines) == 1 and baselines[0].key == "bull_real"
    assert all(r.kind in ("adverse", "baseline") for r in REGIMES)


def test_lookup_consistent():
    for r in REGIMES:
        assert REGIME_BY_KEY[r.key] is r


def test_slice_window_bounds():
    dates = pd.bdate_range("1989-01-02", "2026-06-05")
    df = pd.DataFrame({"Adj Close": np.arange(len(dates), dtype=float)},
                      index=pd.Index(dates, name="Date"))
    r = REGIME_BY_KEY["lost_decade_1"]
    seg = slice_window(df, r)
    assert str(seg.index.min().date()) >= r.start
    assert str(seg.index.max().date()) <= r.end
    assert len(seg) > 0
