"""Unit tests for the statistical-validation helpers (pure, fast, seeded)."""
from __future__ import annotations

import math

import numpy as np

from accumulation_study.validate import (
    _ci,
    _log_returns,
    _moving_block,
    _path_from_logrets,
)


def test_log_returns_and_path_roundtrip():
    closes = np.array([100.0, 110.0, 99.0, 120.0])
    lr = _log_returns(closes)
    rebuilt = _path_from_logrets(float(closes[0]), lr)
    assert np.allclose(rebuilt, closes)


def test_moving_block_preserves_length_and_uses_real_values():
    lr = np.arange(1.0, 51.0)  # distinct values
    rng = np.random.default_rng(0)
    out = _moving_block(lr, block=5, rng=rng)
    assert out.size == lr.size
    assert set(out.tolist()).issubset(set(lr.tolist()))


def test_moving_block_is_seed_reproducible():
    lr = np.arange(1.0, 51.0)
    a = _moving_block(lr, 5, np.random.default_rng(7))
    b = _moving_block(lr, 5, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_ci_brackets_the_median():
    samples = np.linspace(0.0, 1.0, 1001)
    lo, hi = _ci(samples, conf=0.90)
    assert math.isclose(lo, 0.05, abs_tol=0.01)
    assert math.isclose(hi, 0.95, abs_tol=0.01)
