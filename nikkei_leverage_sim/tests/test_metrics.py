"""Metrics unit tests."""
from __future__ import annotations

import math

from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.metrics import (
    max_consecutive_without_tp,
    max_drawdown_abs,
    no_tp_streak_penalty_days,
    objective_score,
)


def test_max_drawdown_abs():
    curve = [100.0, 120.0, 90.0, 130.0, 80.0]
    # Peak 130 -> trough 80 == 50; peak 120 -> 90 == 30.  Worst is 50.
    assert max_drawdown_abs(curve) == 50.0


def test_max_drawdown_zero_when_monotonic():
    assert max_drawdown_abs([1.0, 2.0, 3.0]) == 0.0


def test_max_consecutive_without_tp():
    # True == took profit (resets the streak).
    flags = [True, False, False, False, True, False, False]
    assert max_consecutive_without_tp(flags) == 3
    assert max_consecutive_without_tp([False] * 7) == 7
    assert max_consecutive_without_tp([True, True]) == 0


def test_no_tp_streak_penalty_days():
    # grace 2: a run of 5 with no TP contributes 5-2 == 3 excess days.
    flags = [False, False, False, False, False, True]
    assert no_tp_streak_penalty_days(flags, grace=2) == 3
    # Two separate runs both above grace.
    flags2 = [False, False, False, True, False, False, False, False]
    assert no_tp_streak_penalty_days(flags2, grace=2) == (3 - 2) + (4 - 2)
    # Runs at or below grace contribute nothing.
    assert no_tp_streak_penalty_days([False, False, True], grace=2) == 0


class _Result:
    realized_after_tax = 100_000.0
    ending_unrealized_pnl = 20_000.0
    max_drawdown_equity = 40_000.0
    max_unrealized_loss = 10_000.0
    margin_call_count = 1
    exposure_limit_hit_count = 2
    no_tp_streak_penalty = 3.0


def test_objective_score_combines_terms():
    cfg = Config()
    r = _Result()
    expected = (
        100_000.0
        + 20_000.0
        - 0.5 * 40_000.0
        - 0.5 * 10_000.0
        - 1_000_000.0 * 1
        - 100_000.0 * 2
        - 50_000.0 * 3.0
    )
    assert math.isclose(objective_score(r, cfg), expected)
