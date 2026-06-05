"""Shared pytest fixtures.  No network access anywhere — synthetic data only."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the ``src`` layout importable without an editable install.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nikkei_leverage_sim.config import Config  # noqa: E402
from nikkei_leverage_sim.data import join_target_benchmark, make_synthetic_data  # noqa: E402
from nikkei_leverage_sim.indicators import compute_indicators  # noqa: E402


@pytest.fixture
def base_config() -> Config:
    return Config()


@pytest.fixture
def synthetic_joined():
    target, benchmark = make_synthetic_data(n_days=500, seed=123)
    return join_target_benchmark(target, benchmark)


@pytest.fixture
def synthetic_indicators(synthetic_joined):
    return compute_indicators(synthetic_joined)
