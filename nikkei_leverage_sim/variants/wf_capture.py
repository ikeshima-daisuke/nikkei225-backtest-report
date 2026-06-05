"""Capture the fast walk-forward's per-day buy-engine parameters once.

The published "fast" backtest re-selects :class:`StrategyParams` every few days
via :class:`~nikkei_leverage_sim.optimizer.WalkForwardOptimizer` (a causal,
trailing-window search).  Running that walk-forward is the only expensive part
of a run (~2.5 min).  We run it **once**, record the exact parameter object used
for the decision at the close of each day, and then *replay* that fixed per-day
sequence through the (cheap) variant engine for every grid combo.

This makes the variant grid both realistic (every combo buys exactly like the
real fast strategy) and fast (each combo is a single millisecond-scale pass),
while keeping the comparison controlled: the buy engine is identical across all
variants, so only the entry-lump / exit-rule differs.
"""
from __future__ import annotations

from typing import List

from nikkei_leverage_sim.backtest import MarketData, prepare_market_data, simulate
from nikkei_leverage_sim.config import Config, StrategyParams


def capture_walkforward_params(md: MarketData, cfg: Config) -> List[StrategyParams]:
    """Run the fast walk-forward once and return per-day ``StrategyParams``.

    ``seq[i]`` is the parameter set the optimizer selected for the decision made
    at the close of day ``i`` (the params the core engine actually used that
    day).  Days on which no decision is made keep the last selection, so the
    sequence is always fully populated and length ``md.n``.
    """
    from nikkei_leverage_sim.optimizer import WalkForwardOptimizer

    opt = WalkForwardOptimizer(md, cfg)
    seq: List[StrategyParams] = [cfg.strategy.default_params] * md.n
    last = cfg.strategy.default_params

    def provider(i: int) -> StrategyParams:
        nonlocal last
        p = opt.params_at_close(i)
        seq[i] = p
        last = p
        return p

    # record=False keeps it light; we only need the optimizer's selections.
    simulate(md, 0, md.n, provider, cfg, record=False)
    # Forward-fill any untouched (non-decision / pre-warmup) days with the
    # selection that was current at that point.
    cur = cfg.strategy.default_params
    for i in range(md.n):
        if md.valid[i]:
            cur = seq[i]
        else:
            seq[i] = cur
    return seq
