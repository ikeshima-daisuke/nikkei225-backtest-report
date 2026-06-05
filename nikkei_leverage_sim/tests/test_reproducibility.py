"""Reproducibility / seed-determinism tests.

The only source of randomness in the engine is the walk-forward optimizer's
candidate generation (``np.random.default_rng(random_seed)``); the market data
is deterministic CSV/synthetic input.  These tests pin that down:

* the candidate set is a pure function of the seed (same seed -> identical, a
  different seed -> different);
* two full backtests with the same seed produce byte-identical results;
* the seed actually used is recorded in ``summary.json``.
"""
from __future__ import annotations

from nikkei_leverage_sim.backtest import prepare_market_data, run_backtest
from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.data import join_target_benchmark, make_synthetic_data
from nikkei_leverage_sim.metrics import build_summary
from nikkei_leverage_sim.optimizer import make_random_candidates


def _market_data():
    target, benchmark = make_synthetic_data(n_days=400, seed=7)
    return prepare_market_data(join_target_benchmark(target, benchmark), Config())


def _cfg(seed: int) -> Config:
    cfg = Config()
    cfg.optimization.enabled = True
    cfg.optimization.method = "random"
    cfg.optimization.random_seed = seed
    cfg.optimization.n_trials = 10
    cfg.optimization.lookback_days = 200
    cfg.optimization.min_train_days = 150
    cfg.optimization.apply_days = 5
    cfg.optimization.rebalance_frequency = "weekly"
    return cfg


def _fingerprint(result) -> tuple:
    pf = result.portfolio
    metrics = (
        round(result.final_equity, 6),
        round(pf.realized_after_tax, 6),
        round(pf.unrealized_pnl(), 6),
        round(pf.total_interest_paid, 6),
        round(pf.total_tax_paid, 6),
        pf.buy_trade_count,
        pf.sell_trade_count,
        pf.margin_call_count,
        pf.forced_liquidation_count,
    )
    trades = tuple(
        (str(t["date"]), t["side"], t["shares"], round(float(t["price"]), 6), t["lot_id"])
        for t in result.trades
    )
    return metrics, trades


def test_candidate_generation_is_deterministic_and_seed_sensitive():
    cfg = Config()
    same_a = [p.to_dict() for p in make_random_candidates(cfg, 25, 42)]
    same_b = [p.to_dict() for p in make_random_candidates(cfg, 25, 42)]
    diff = [p.to_dict() for p in make_random_candidates(cfg, 25, 43)]
    assert same_a == same_b          # same seed -> identical candidate set
    assert same_a != diff            # different seed -> different candidate set


def test_same_seed_reproduces_identical_backtest():
    md = _market_data()
    r1 = run_backtest(md, _cfg(42))
    r2 = run_backtest(md, _cfg(42))
    assert _fingerprint(r1) == _fingerprint(r2)


def test_seed_is_recorded_in_summary():
    md = _market_data()
    cfg = _cfg(2024)
    summary = build_summary(run_backtest(md, cfg), cfg)
    assert summary["random_seed"] == 2024
