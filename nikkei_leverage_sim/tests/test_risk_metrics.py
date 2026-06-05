"""Unit tests for the Week 1 tail-risk / draw-down metrics.

Inputs are chosen so the expected values are hand-verifiable (or reduced to a
direct numpy reference for the quantile-based ones), and every metric is checked
to degrade gracefully on short / degenerate input.
"""
from __future__ import annotations

import math

import numpy as np

from nikkei_leverage_sim import metrics as m


def test_daily_returns_basic_and_zero_denominator_dropped():
    rets = m.daily_returns([100.0, 110.0, 99.0])
    assert np.allclose(rets, [0.10, -0.10])
    # A non-positive prior equity drops that step (mirrors sharpe_like).
    rets2 = m.daily_returns([0.0, 100.0, 110.0])
    assert np.allclose(rets2, [0.10])


def test_max_drawdown_pct():
    # peaks: 100,120,120,130,130 -> worst (130-80)/130.
    curve = [100.0, 120.0, 90.0, 130.0, 80.0]
    assert math.isclose(m.max_drawdown_pct(curve), (130.0 - 80.0) / 130.0)


def test_max_drawdown_pct_monotonic_is_zero():
    assert m.max_drawdown_pct([1.0, 2.0, 3.0]) == 0.0


def test_calmar_ratio():
    assert math.isclose(m.calmar_ratio(0.10, 0.50), 0.20)
    # No draw-down -> undefined -> 0 (not infinity).
    assert m.calmar_ratio(0.10, 0.0) == 0.0


def test_ulcer_index_is_zero_on_monotonic_and_positive_otherwise():
    assert m.ulcer_index([1.0, 2.0, 3.0]) == 0.0
    assert m.ulcer_index([100.0, 90.0, 95.0, 80.0]) > 0.0


def test_var_and_cvar_sign_and_wiring():
    # Build a curve whose daily returns are exactly ``r`` (within float error).
    r = np.array([-0.06, -0.05, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05])
    equity = (100.0 * np.cumprod(1.0 + r)).tolist()
    equity = [100.0] + equity  # so daily_returns(equity) == r
    rec = m.daily_returns(equity)
    assert np.allclose(rec, r, atol=1e-9)

    q = np.quantile(r, 0.05)
    assert math.isclose(m.value_at_risk(equity, 0.95), -q, abs_tol=1e-9)
    tail = r[r <= q]
    assert math.isclose(m.conditional_var(equity, 0.95), -tail.mean(), abs_tol=1e-9)
    # CVaR is at least as severe as VaR.
    assert m.conditional_var(equity) >= m.value_at_risk(equity) - 1e-12


def test_sortino_positive_for_upward_drift():
    curve = [100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0]
    assert m.sortino_ratio(curve) > 0.0
    # No downside at all -> defined as 0 (avoid div-by-zero blow-up).
    assert m.sortino_ratio([100.0, 101.0, 102.0, 103.0]) == 0.0


def test_sortino_downside_denominator_excludes_flat_days():
    # One -4% day then four flat days. Downside deviation must average over the
    # single below-target day only (RMS = 0.04), NOT dilute over all 5 days.
    curve = [100.0, 96.0, 96.0, 96.0, 96.0, 96.0]
    rets = m.daily_returns(curve)  # [-0.04, 0, 0, 0, 0]
    expected = (float(np.mean(rets)) - 0.0) / 0.04 * math.sqrt(252)
    assert math.isclose(m.sortino_ratio(curve), expected, rel_tol=1e-9)


def test_skew_and_kurtosis_guards_and_values():
    # Too short -> neutral 0.0.
    assert m.return_skew([100.0, 101.0]) == 0.0
    assert m.return_kurtosis([100.0, 101.0, 102.0]) == 0.0
    # Right-skewed returns -> positive skew.
    r = np.array([-0.01, -0.01, -0.01, -0.01, 0.10])
    equity = [100.0] + (100.0 * np.cumprod(1.0 + r)).tolist()
    assert m.return_skew(equity) > 0.0
    assert math.isfinite(m.return_kurtosis(equity))


def test_drawdown_percentiles_monotonic_all_zero():
    ddp = m.drawdown_percentiles([1.0, 2.0, 3.0, 4.0])
    assert set(ddp) == {50, 90, 95, 99}
    assert all(v == 0.0 for v in ddp.values())


def test_drawdown_percentiles_increasing_with_level():
    curve = [100.0, 120.0, 90.0, 130.0, 80.0, 110.0]
    ddp = m.drawdown_percentiles(curve)
    assert ddp[50] <= ddp[90] <= ddp[95] <= ddp[99]
    assert ddp[99] > 0.0


def test_worst_day_helpers():
    curve = [100.0, 120.0, 90.0, 130.0, 80.0]
    assert m.worst_day_equity(curve) == 80.0
    assert m.worst_daily_return(curve) < 0.0


def test_build_risk_metrics_keys_and_degenerate():
    keys = {
        "sortino_ratio",
        "calmar_ratio",
        "ulcer_index",
        "var_95_daily",
        "cvar_95_daily",
        "return_skew",
        "return_kurtosis_excess",
        "max_drawdown_pct",
        "drawdown_pct_p50",
        "drawdown_pct_p90",
        "drawdown_pct_p95",
        "drawdown_pct_p99",
        "worst_day_equity",
        "worst_daily_return",
    }
    block = m.build_risk_metrics([100.0, 110.0, 105.0, 120.0], 4)
    assert set(block) == keys
    # Empty curve must not raise and must be all-neutral / finite.
    empty = m.build_risk_metrics([], 0)
    assert set(empty) == keys
    assert all(v == 0.0 for v in empty.values())
