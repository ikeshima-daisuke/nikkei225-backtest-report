"""Strategy-variant backtests for the Nikkei leverage accumulation simulator.

This sub-package is **fully self-contained** and imports the validated core
engine pieces (``Portfolio``, ``strategy`` scoring, ``metrics``) without
modifying any core module.  It adds three exit-rule variants plus an
initial-lump-sum entry option on top of the same daily DCA buy engine, so the
*only* thing that changes between runs is the entry/exit rule -- a clean
controlled comparison.

See :mod:`nikkei_leverage_sim.variants.variant_engine` for the engine and
:mod:`nikkei_leverage_sim.variants.grid` for the grid-search driver.
"""
from __future__ import annotations

from .variant_engine import (
    VariantParams,
    VariantResult,
    prepare_market_data_v,
    simulate_variant,
)

__all__ = [
    "VariantParams",
    "VariantResult",
    "prepare_market_data_v",
    "simulate_variant",
]
