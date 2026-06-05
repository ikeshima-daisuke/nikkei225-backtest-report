"""Optimizer tests: reproducibility, no future leakage, best-candidate choice."""
from __future__ import annotations

import numpy as np

from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.backtest import prepare_market_data, run_backtest, simulate
from nikkei_leverage_sim.data import join_target_benchmark, make_synthetic_data
from nikkei_leverage_sim.metrics import objective_score
from nikkei_leverage_sim.optimizer import (
    make_random_candidates,
    select_best_params,
)


def _fast_cfg() -> Config:
    cfg = Config()
    cfg.optimization.enabled = True
    cfg.optimization.method = "random"
    cfg.optimization.random_seed = 42
    cfg.optimization.n_trials = 8
    cfg.optimization.lookback_days = 60
    cfg.optimization.min_train_days = 50
    return cfg


def test_random_candidates_reproducible_with_seed():
    cfg = Config()
    a = make_random_candidates(cfg, 10, seed=42)
    b = make_random_candidates(cfg, 10, seed=42)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    c = make_random_candidates(cfg, 10, seed=43)
    assert [x.to_dict() for x in a] != [x.to_dict() for x in c]


def test_select_best_returns_highest_objective():
    cfg = _fast_cfg()
    target, benchmark = make_synthetic_data(n_days=200, seed=9)
    joined = join_target_benchmark(target, benchmark)
    md = prepare_market_data(joined, cfg)
    candidates = make_random_candidates(cfg, 12, seed=7)

    lo, hi = 120, 190
    best, _ = select_best_params(md, lo, hi, candidates, cfg)

    # Recompute the objective for every candidate independently.
    scored = [
        (objective_score(simulate(md, lo, hi, c, cfg), cfg), c) for c in candidates
    ]
    best_score = max(s for s, _ in scored)
    assert abs(objective_score(simulate(md, lo, hi, best, cfg), cfg) - best_score) < 1e-9


def test_optimizer_ignores_future_data_outside_window():
    """Perturbing rows AFTER the window must not change the selection."""
    cfg = _fast_cfg()
    target, benchmark = make_synthetic_data(n_days=220, seed=4)
    joined = join_target_benchmark(target, benchmark)
    candidates = make_random_candidates(cfg, 10, seed=1)
    lo, hi = 130, 200

    md1 = prepare_market_data(joined, cfg)
    best1, _ = select_best_params(md1, lo, hi, candidates, cfg)

    # Mutate only rows at/after ``hi`` (strictly future relative to the window).
    perturbed = joined.copy()
    future_idx = perturbed.index[hi:]
    for col in ("target_open", "target_close", "target_adj_close",
                "benchmark_close", "benchmark_adj_close"):
        perturbed.loc[future_idx, col] *= 1.5

    md2 = prepare_market_data(perturbed, cfg)
    best2, _ = select_best_params(md2, lo, hi, candidates, cfg)

    assert best1.to_dict() == best2.to_dict()


def test_full_walkforward_run_is_reproducible():
    cfg = _fast_cfg()
    target, benchmark = make_synthetic_data(n_days=300, seed=2)
    joined = join_target_benchmark(target, benchmark)

    md_a = prepare_market_data(joined, cfg)
    res_a = run_backtest(md_a, cfg)
    md_b = prepare_market_data(joined, cfg)
    res_b = run_backtest(md_b, cfg)

    assert res_a.final_equity == res_b.final_equity
    assert [r["selected_params"] for r in res_a.optimization_rows] == [
        r["selected_params"] for r in res_b.optimization_rows
    ]
