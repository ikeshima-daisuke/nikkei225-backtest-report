"""Nikkei leverage ETF margin-long accumulation & take-profit backtester."""
from __future__ import annotations

from .config import Config, StrategyParams, load_config
from .backtest import (
    BacktestResult,
    MarketData,
    SimResult,
    prepare_market_data,
    run_backtest,
    simulate,
)

__all__ = [
    "Config",
    "StrategyParams",
    "load_config",
    "MarketData",
    "SimResult",
    "BacktestResult",
    "prepare_market_data",
    "run_backtest",
    "simulate",
]

__version__ = "0.1.0"
