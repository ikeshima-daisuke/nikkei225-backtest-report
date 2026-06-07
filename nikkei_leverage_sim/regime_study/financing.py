"""Time-varying JPY financing cost for a daily-rebalanced 2x ETF.

A 2x ETF holds ``leverage``× the index funded by 1× equity plus ``(leverage-1)``×
borrowing.  The borrowed leg accrues roughly the short-term money-market rate
(plus a small spread).  In the ZIRP/NIRP era (≈2011–2024) that cost is ~0, which
is why a flat drag calibrated on 2014–2026 understates the 1990s: Japan's
overnight call rate was **4–7.4% in 1990–92**, so the leverage carry cost alone
was several percent per year on top of the expense ratio and decay.

Ignoring this would *flatter* the lost-decade replay (understate the head-wind),
so the regime study models it explicitly.

Rate schedule
-------------
``CALL_RATE_ANNUAL`` holds approximate **annual averages of the Bank of Japan
uncollateralized overnight call rate** (the policy rate), in decimal.  These are
public, well-documented figures; annual granularity is sufficient for a multi-
year drag and keeps the model auditable.  Pre-1989 and post-table years fall
back to the nearest endpoint.

Fee model (simple-return, per session)
--------------------------------------
``annual_fee(date) = base_drag + (leverage - 1) * (call_rate(date) - calib_rate)``

``base_drag`` is the all-in drag (expense ratio + spread + tracking) calibrated
against the *real* 1570.T over a low-rate window (:mod:`build_target`); subtracting
``calib_rate`` (the mean call rate over that same window, ≈0) avoids double-
counting the financing already baked into ``base_drag``.  The per-session fee is
``annual_fee / TRADING_DAYS``.
"""
from __future__ import annotations

from typing import Dict

TRADING_DAYS = 245.0  # ~Japanese trading days per year (matches the data ~245-249)

# Approximate annual-average BoJ uncollateralized overnight call rate (decimal).
# Sources: BoJ time-series (policy/call rate).  Rounded to the basis point.
CALL_RATE_ANNUAL: Dict[int, float] = {
    1989: 0.0487,
    1990: 0.0724,
    1991: 0.0738,
    1992: 0.0458,
    1993: 0.0306,
    1994: 0.0220,
    1995: 0.0121,
    1996: 0.0047,
    1997: 0.0048,
    1998: 0.0037,
    1999: 0.0006,   # ZIRP begins Feb 1999
    2000: 0.0011,   # ZIRP lifted Aug 2000
    2001: 0.0006,   # QE Mar 2001
    2002: 0.0002,
    2003: 0.0002,
    2004: 0.0002,
    2005: 0.0002,
    2006: 0.0021,   # ZIRP ends Jul 2006
    2007: 0.0048,
    2008: 0.0047,   # cut to 0.1% Dec 2008
    2009: 0.0010,
    2010: 0.0010,
    2011: 0.0008,
    2012: 0.0008,
    2013: 0.0007,
    2014: 0.0007,
    2015: 0.0006,
    2016: -0.0004,  # NIRP Jan 2016
    2017: -0.0004,
    2018: -0.0006,
    2019: -0.0006,
    2020: -0.0003,
    2021: -0.0002,
    2022: -0.0002,
    2023: -0.0004,
    2024: 0.0005,   # NIRP exit Mar 2024
    2025: 0.0025,
    2026: 0.0050,
}

_MIN_YEAR = min(CALL_RATE_ANNUAL)
_MAX_YEAR = max(CALL_RATE_ANNUAL)


def call_rate(date: str) -> float:
    """Annual call rate for a ``YYYY-MM-DD`` (or ``YYYY...``) date string."""
    year = int(str(date)[:4])
    if year < _MIN_YEAR:
        year = _MIN_YEAR
    elif year > _MAX_YEAR:
        year = _MAX_YEAR
    return CALL_RATE_ANNUAL[year]


def mean_call_rate(start: str, end: str) -> float:
    """Mean annual call rate over the inclusive ``[start, end]`` year span."""
    y0, y1 = int(str(start)[:4]), int(str(end)[:4])
    years = [y for y in range(min(y0, y1), max(y0, y1) + 1)]
    rates = [call_rate(f"{y}-06-30") for y in years]
    return sum(rates) / len(rates) if rates else 0.0


# Retail margin-loan spread over the overnight call rate.  Calibrated so that in
# the ZIRP/NIRP bull era (2014-2026, mean call rate ≈0) the investor's account
# margin rate reproduces the core engine's default ``annual_margin_interest_rate``
# (0.028) — so the bull baseline is unchanged and only adverse regimes get the
# (historically real) higher carry.
CORE_DEFAULT_MARGIN_RATE = 0.028
ACCOUNT_MARGIN_SPREAD = CORE_DEFAULT_MARGIN_RATE - mean_call_rate("2014-01-06", "2026-06-05")


def account_margin_rate(start: str, end: str, *, spread: float = ACCOUNT_MARGIN_SPREAD) -> float:
    """Regime-appropriate investor margin-loan rate = mean call rate + spread.

    Makes the *account* financing in the survival test as regime-aware as the
    *ETF-internal* financing in :func:`daily_fee`; reduces to ≈0.028 (the core
    default) in the bull baseline so that run is unchanged.
    """
    return mean_call_rate(start, end) + spread


def annual_fee(date: str, *, leverage: float, base_drag: float, calib_rate: float) -> float:
    """All-in annual drag for ``date`` (decimal); see module docstring."""
    return base_drag + (leverage - 1.0) * (call_rate(date) - calib_rate)


def daily_fee(date: str, *, leverage: float, base_drag: float, calib_rate: float) -> float:
    """Per-session simple-return drag for ``date`` (decimal)."""
    return annual_fee(
        date, leverage=leverage, base_drag=base_drag, calib_rate=calib_rate
    ) / TRADING_DAYS
