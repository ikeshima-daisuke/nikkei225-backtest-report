"""Walk-forward parameter optimization.

The optimizer never looks into the future: at the close of day ``i`` it scores a
fixed set of candidate parameter sets on the trailing training window
``[i-lookback+1, i]`` (data we already have), picks the best by the composite
objective, and applies it to the decision that executes on day ``i+1``.

Two search methods are supported:

* ``random`` – ``n_trials`` uniform samples from the parameter bounds, seeded for
  reproducibility (the default).
* ``grid`` – a coarse deterministic lattice over the most influential params.

The candidate set is generated **once** (deterministically from the seed) and
reused on every rebalance day.  This keeps the walk-forward tractable and makes
"same seed -> same selection" trivially true.
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .backtest import MarketData, simulate
from .config import Config, StrategyParams
from .metrics import objective_score

# Searchable parameter bounds (inclusive).
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "base_buy_amount": (50_000.0, 1_000_000.0),
    "max_daily_buy_amount": (100_000.0, 2_000_000.0),
    "score_threshold": (-2.0, 2.0),
    "score_scale": (0.5, 5.0),
    "w_drawdown": (0.0, 5.0),
    "w_rsi": (0.0, 5.0),
    "w_ma_gap_25": (0.0, 5.0),
    "w_ret_5": (0.0, 5.0),
    "w_trend": (-3.0, 3.0),
    "w_vol": (0.0, 5.0),
    "w_exposure": (0.0, 10.0),
    "w_unrealized_loss": (0.0, 10.0),
    "fixed_profit_yen": (3_000.0, 50_000.0),
    "base_take_profit_pct": (0.002, 0.03),
    "min_take_profit_pct": (0.001, 0.01),
    "exposure_tp_sensitivity": (0.0, 0.9),
    "vol_tp_multiplier": (0.0, 5.0),
}

# Params used for the coarse grid (others stay at the default value).
_GRID_LEVELS: Dict[str, List[float]] = {
    "base_buy_amount": [100_000.0, 300_000.0, 600_000.0],
    "score_threshold": [-1.0, 0.0, 1.0],
    "w_drawdown": [1.0, 3.0],
    "base_take_profit_pct": [0.004, 0.008, 0.015],
    "fixed_profit_yen": [5_000.0, 15_000.0],
}


def make_random_candidates(
    cfg: Config, n_trials: int, seed: int
) -> List[StrategyParams]:
    """Generate ``n_trials`` random candidates (plus the default at index 0)."""
    rng = np.random.default_rng(seed)
    candidates: List[StrategyParams] = [cfg.strategy.default_params.replace()]
    for _ in range(max(0, n_trials - 1)):
        kwargs = {
            name: float(rng.uniform(lo, hi)) for name, (lo, hi) in PARAM_BOUNDS.items()
        }
        # Keep min_take_profit_pct <= base_take_profit_pct for sane targets.
        if kwargs["min_take_profit_pct"] > kwargs["base_take_profit_pct"]:
            kwargs["min_take_profit_pct"] = kwargs["base_take_profit_pct"]
        candidates.append(StrategyParams(**kwargs))
    return candidates


def make_grid_candidates(cfg: Config, n_trials: int) -> List[StrategyParams]:
    """Generate a coarse grid of candidates, truncated to ``n_trials``."""
    base = cfg.strategy.default_params
    names = list(_GRID_LEVELS)
    candidates: List[StrategyParams] = [base.replace()]
    for combo in itertools.product(*(_GRID_LEVELS[n] for n in names)):
        kwargs = dict(zip(names, combo))
        candidates.append(base.replace(**kwargs))
        if len(candidates) >= n_trials:
            break
    return candidates


def make_candidates(cfg: Config) -> List[StrategyParams]:
    """Build the candidate set according to the configured search method."""
    o = cfg.optimization
    if o.method == "grid":
        return make_grid_candidates(cfg, o.n_trials)
    return make_random_candidates(cfg, o.n_trials, o.random_seed)


def select_best_params(
    md: MarketData,
    lo: int,
    hi: int,
    candidates: List[StrategyParams],
    cfg: Config,
) -> Tuple[StrategyParams, Dict[str, float]]:
    """Score every candidate on rows ``[lo, hi)`` and return the best one.

    Returns ``(best_params, training_metrics)``.
    """
    best_score = -np.inf
    best: Optional[StrategyParams] = None
    best_metrics: Dict[str, float] = {}
    for cand in candidates:
        r = simulate(md, lo, hi, cand, cfg, record=False)
        score = objective_score(r, cfg)
        if score > best_score:
            best_score = score
            best = cand
            best_metrics = {
                "training_score": score,
                "training_net_profit": r.realized_after_tax + r.ending_unrealized_pnl,
                "training_max_drawdown": r.max_drawdown_equity,
                "training_max_unrealized_loss": r.max_unrealized_loss,
                "training_margin_call_count": r.margin_call_count,
            }
    assert best is not None  # candidates always non-empty (default at idx 0)
    return best, best_metrics


class WalkForwardOptimizer:
    """Provides per-day parameters for the live backtest via walk-forward search."""

    def __init__(self, md: MarketData, cfg: Config) -> None:
        self.md = md
        self.cfg = cfg
        self.optimization_rows: List[Dict[str, object]] = []
        self._default = cfg.strategy.default_params
        self._current = cfg.strategy.default_params
        self._last_rebalance = -(10**9)
        self._candidates = make_candidates(cfg) if cfg.optimization.enabled else []
        # Cumulative count of tradable days for fast train-window sizing.
        self._cumvalid = np.cumsum(md.valid.astype(np.int64))

    def _valid_count(self, lo: int, hi_inclusive: int) -> int:
        upper = self._cumvalid[hi_inclusive]
        lower = self._cumvalid[lo - 1] if lo > 0 else 0
        return int(upper - lower)

    def params_at_close(self, i: int) -> StrategyParams:
        """Return parameters for the decision made at the close of day ``i``.

        (That decision is executed on day ``i+1``.)
        """
        o = self.cfg.optimization
        if not o.enabled:
            return self._default

        lo = max(0, i - o.lookback_days + 1)
        if self._valid_count(lo, i) < o.min_train_days:
            return self._current  # not enough history yet -> keep current/default

        # Rebalance cadence.
        rebalance = (i - self._last_rebalance) >= max(1, o.apply_days)
        if o.rebalance_frequency == "weekly":
            rebalance = rebalance and (i - self._last_rebalance) >= 5
        if not rebalance:
            return self._current

        best, metrics = select_best_params(self.md, lo, i + 1, self._candidates, self.cfg)
        self._current = best
        self._last_rebalance = i

        exec_index = i + 1 if i + 1 < self.md.n else i
        self.optimization_rows.append(
            {
                "date": self.md.dates[exec_index],
                "selected_params": best.to_dict(),
                **metrics,
            }
        )
        return self._current
