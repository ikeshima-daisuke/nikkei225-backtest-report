"""Named historical regimes for the replay.

Each :class:`Regime` is a date window into the real ``^N225`` series.  The lost-
decade windows are where a daily-rebalanced 2x ETF is expected to break (sideways/
bear + the early-1990s high-rate financing); the bull window is the real-era
baseline that the prior reports were built on (the synthetic build reproduces the
real 1570.T there, so it is an apples-to-apples control).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class Regime:
    key: str
    start: str
    end: str
    label: str
    kind: str  # "adverse" | "baseline"


REGIMES: List[Regime] = [
    Regime("bubble_burst", "1990-01-01", "1995-12-31",
           "バブル崩壊 1990-1995（最高金利期）", "adverse"),
    Regime("lost_decade_1", "1990-01-01", "2003-04-30",
           "失われた10年→2003大底", "adverse"),
    Regime("deflation_grind", "2000-01-01", "2012-12-31",
           "デフレ横ばい 2000-2012", "adverse"),
    Regime("gfc", "2007-06-01", "2009-03-31",
           "リーマン・ショック 2007-2009", "adverse"),
    Regime("two_lost_decades", "1990-01-01", "2013-12-31",
           "二つの失われた10年 1990-2013", "adverse"),
    Regime("bull_real", "2014-01-06", "2026-06-05",
           "強気ベースライン 2014-2026（実era）", "baseline"),
]

REGIME_BY_KEY = {r.key: r for r in REGIMES}


def slice_window(df: pd.DataFrame, regime: Regime) -> pd.DataFrame:
    """Return the rows of ``df`` (Date-indexed) within ``regime``'s window."""
    return df.loc[regime.start:regime.end]
