"""Forced-liquidation (追証/強制ロスカット) effect verification — Week 0.5.

Quantifies how much the headline fast-version result (¥10.48M realized, ¥5.28M
peak unrealized loss, 0 margin calls) was *optimistic* because it assumed
¥100,000,000 of own funds backing a position capped at ¥10M cost basis — a
regime no broker would ever margin-call.

Method — gold-standard isolation by *replay*:

  1. Run the fast-version walk-forward optimizer once at ¥100M and capture the
     exact per-day selected-parameter sequence.  That sequence is deterministic
     given (data, seed) and the *same strategy* the headline result used.
  2. Replay that identical sequence (no re-optimization) at several own-funds /
     committed-margin levels, toggling only the margin-call model OFF vs ON.

Because position sizing never depends on own funds, every OFF replay is the
identical trade tape regardless of capital; realized P&L only diverges once the
ON model force-closes the book on a maintenance-margin breach.  That divergence
is exactly "the optimism".

Run:  PYTHONUTF8=1 ./.venv/Scripts/python.exe verify_margin_call.py
"""
from __future__ import annotations

from pathlib import Path

from nikkei_leverage_sim.backtest import prepare_market_data, simulate
from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.data import load_market_data
from nikkei_leverage_sim.optimizer import WalkForwardOptimizer

HERE = Path(__file__).resolve().parent
JOINED = load_market_data(
    HERE / "data" / "target_1570_T.csv", HERE / "data" / "benchmark_N225.csv"
)


def _fast_cfg(initial_equity: float, force: bool) -> Config:
    """The fast-version config (config_fast.yaml), with overrides."""
    cfg = Config()
    cfg.initial_equity = initial_equity
    cfg.force_liquidation = force
    cfg.optimization.enabled = True
    cfg.optimization.method = "random"
    cfg.optimization.random_seed = 42
    cfg.optimization.n_trials = 40
    cfg.optimization.lookback_days = 252
    cfg.optimization.min_train_days = 200
    cfg.optimization.apply_days = 5
    cfg.optimization.rebalance_frequency = "weekly"
    return cfg


MD = prepare_market_data(JOINED, Config())

# 1. Capture the fast-version (¥100M) walk-forward parameter sequence once.
print("Optimizing fast-version (¥100M) walk-forward parameter sequence ...", flush=True)
_opt = WalkForwardOptimizer(MD, _fast_cfg(100_000_000, force=False))
PARAM_SEQ = [_opt.params_at_close(i) for i in range(MD.n)]
_provider = lambda i: PARAM_SEQ[i]  # noqa: E731  fixed replay, no re-optimization


def replay(initial_equity: float, force: bool) -> dict:
    cfg = Config()
    cfg.initial_equity = initial_equity
    cfg.force_liquidation = force
    res = simulate(MD, 0, MD.n, _provider, cfg, record=True)
    pf = res.portfolio
    return {
        "realized": pf.realized_after_tax,
        "max_uloss": pf.max_unrealized_loss,
        "min_maint": pf.min_maintenance_ratio_seen,
        "mc_days": pf.margin_call_count,
        "forced": pf.forced_liquidation_count,
        "final_equity": res.final_equity,
        "max_dd": res.max_drawdown_equity,
        "buys": pf.buy_trade_count,
        "sells": pf.sell_trade_count,
    }


def yen(x: float) -> str:
    return f"¥{x / 1e6:8.3f}M"


def fmt_maint(v: float) -> str:
    return "inf" if v == float("inf") else f"{v:.3f}"


SCENARIOS = [
    ("¥100M own funds (production assumption)", 100_000_000),
    ("¥10M  own funds (1.0x max position)", 10_000_000),
    ("¥5M   own funds (2.0x max leverage)", 5_000_000),
    ("¥3.3M own funds (3.0x, ~legal min margin)", 3_300_000),
]

print()
print("=" * 104)
print("Forced-liquidation effect — replay of the fast-version optimized strategy on real 1570.T")
print("data (2014-01-06 .. 2026-06-05, 3,032 sessions). Trade tape identical across OFF rows.")
print("=" * 104)
print(
    f"{'scenario':<44} {'model':<5} {'realized':>11} {'max_uloss':>11} "
    f"{'min_maint':>10} {'mc_days':>8} {'forced':>7}"
)
print("-" * 104)
rows = {}
for label, eq in SCENARIOS:
    for force in (False, True):
        r = replay(eq, force)
        rows[(eq, force)] = r
        print(
            f"{label:<44} {'ON' if force else 'OFF':<5} "
            f"{yen(r['realized']):>11} {yen(r['max_uloss']):>11} "
            f"{fmt_maint(r['min_maint']):>10} {r['mc_days']:>8} {r['forced']:>7}"
        )
    print("-" * 104)

print()
print("Δ from the optimistic (OFF) model, per capital level:")
for label, eq in SCENARIOS:
    off, on = rows[(eq, False)], rows[(eq, True)]
    d_real = on["realized"] - off["realized"]
    pct = (d_real / off["realized"] * 100.0) if off["realized"] else 0.0
    print(
        f"  {label:<44} realized {yen(off['realized'])} -> {yen(on['realized'])} "
        f"({d_real / 1e6:+.3f}M, {pct:+.1f}%)  forced={on['forced']}  mc_days={on['mc_days']}"
    )
