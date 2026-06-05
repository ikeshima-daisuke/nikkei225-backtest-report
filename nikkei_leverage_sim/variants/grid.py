"""Grid-search driver for the strategy variants.

Builds the full factorial of *entry-lump* x *exit-rule* combos, runs each one
through the variant engine (replaying a captured walk-forward buy sequence so
every combo buys exactly like the published fast strategy), and emits one flat
metrics row per combo.

Categories
----------
* ``per_lot``  : initial x {per-lot take-profit}                 (6 combos)
* ``bulk_yen`` : initial x bulk_exit_yen                          (6 x 7 = 42)
* ``bulk_pct`` : initial x bulk_exit_pct                          (6 x 8 = 48)
* ``combo``    : initial x bulk_exit_yen x bulk_exit_pct          (6 x 7 x 8 = 336)

Total: 432 combos (covers the ~340 the task asked for, plus the combined
yen-or-pct exploration).
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence

from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.metrics import build_summary

from .variant_engine import (
    MarketDataV,
    VariantParams,
    VariantResult,
    simulate_variant,
)

# --- Grid axes (yen amounts; pct as fractions) ------------------------------
INITIAL_AMOUNTS: List[float] = [0.0, 500_000.0, 1_000_000.0, 2_000_000.0,
                                5_000_000.0, 10_000_000.0]
BULK_YEN: List[float] = [100_000.0, 300_000.0, 500_000.0, 1_000_000.0,
                         2_000_000.0, 5_000_000.0, 10_000_000.0]
BULK_PCT: List[float] = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]


def build_grid() -> List[VariantParams]:
    """Return the full list of variant combos (deterministic order)."""
    combos: List[VariantParams] = []
    for init in INITIAL_AMOUNTS:
        combos.append(VariantParams("per_lot", initial_amount=init))
    for init, yen in itertools.product(INITIAL_AMOUNTS, BULK_YEN):
        combos.append(VariantParams("bulk_yen", initial_amount=init, bulk_exit_yen=yen))
    for init, pct in itertools.product(INITIAL_AMOUNTS, BULK_PCT):
        combos.append(VariantParams("bulk_pct", initial_amount=init, bulk_exit_pct=pct))
    for init, yen, pct in itertools.product(INITIAL_AMOUNTS, BULK_YEN, BULK_PCT):
        combos.append(VariantParams("combo", initial_amount=init,
                                    bulk_exit_yen=yen, bulk_exit_pct=pct))
    return combos


def _extra_metrics(res: VariantResult, n_days: int) -> Dict[str, Any]:
    """Metrics not in ``build_summary`` that matter for the 'satisfying
    take-profit' framing: time spent underwater and harvest cadence."""
    underwater = 0
    max_uw_streak = 0
    cur = 0
    for r in res.daily_rows:
        if r["unrealized_pnl"] < 0:
            underwater += 1
            cur += 1
            if cur > max_uw_streak:
                max_uw_streak = cur
        else:
            cur = 0
    harvest_days = sum(1 for r in res.daily_rows if r["took_profit"])
    return {
        "days_underwater": underwater,
        "pct_days_underwater": underwater / n_days if n_days else 0.0,
        "max_consecutive_underwater_days": max_uw_streak,
        "harvest_days": harvest_days,
        "bulk_exit_count": res.bulk_exit_count,
    }


def run_combo(
    mdv: MarketDataV,
    cfg: Config,
    params_provider,
    vp: VariantParams,
) -> Dict[str, Any]:
    """Run one combo and return a flat metrics row."""
    res = simulate_variant(mdv, cfg, params_provider, vp, record=True)
    summary = build_summary(res, cfg)
    n_days = len(res.daily_rows)
    extra = _extra_metrics(res, n_days)

    total_net = res.realized_after_tax + res.ending_unrealized_pnl
    row: Dict[str, Any] = {
        "label": vp.label(),
        "category": vp.variant,
        "initial_amount": vp.initial_amount,
        "bulk_exit_yen": vp.bulk_exit_yen if vp.bulk_exit_yen else "",
        "bulk_exit_pct": vp.bulk_exit_pct if vp.bulk_exit_pct else "",
        "n_trading_days": n_days,
        "net_profit_after_tax": total_net,
        "realized_after_tax": res.realized_after_tax,
        "ending_unrealized_pnl": res.ending_unrealized_pnl,
        "final_equity": summary["final_equity"],
        "annualized_return": summary["annualized_return"],
        "sharpe_like": summary["sharpe_like_ratio"],
        "max_drawdown_equity": res.max_drawdown_equity,
        "max_unrealized_loss": res.max_unrealized_loss,
        "total_interest_paid": summary["total_interest_paid"],
        "total_tax_paid": summary["total_tax_paid"],
        "buy_trade_count": summary["buy_trade_count"],
        "sell_trade_count": summary["sell_trade_count"],
        "harvest_days": extra["harvest_days"],
        "bulk_exit_count": extra["bulk_exit_count"],
        "avg_profit_per_harvest_day": summary["average_profit_per_take_profit_day"],
        "median_profit_per_harvest_day": summary["median_profit_per_take_profit_day"],
        "win_rate_closed_lots": summary["win_rate_of_closed_lots"],
        "profit_factor": summary["profit_factor"],
        "max_consecutive_days_without_take_profit": summary[
            "max_consecutive_days_without_take_profit"
        ],
        "max_lot_holding_days": summary["max_lot_holding_days"],
        "days_underwater": extra["days_underwater"],
        "pct_days_underwater": extra["pct_days_underwater"],
        "max_consecutive_underwater_days": extra["max_consecutive_underwater_days"],
        "margin_call_count": summary["margin_call_count"],
        "exposure_limit_hit_count": summary["exposure_limit_hit_count"],
        "max_gross_exposure": summary["max_gross_exposure"],
        "average_gross_exposure": summary["average_gross_exposure"],
    }
    return row


# --- Worker-pool plumbing (module-level globals for spawn safety) -----------
_W: Dict[str, Any] = {}


def _init_worker(mdv: MarketDataV, cfg: Config, seq) -> None:
    _W["mdv"] = mdv
    _W["cfg"] = cfg
    _W["seq"] = seq


def _run_one(vp: VariantParams) -> Dict[str, Any]:
    return run_combo(_W["mdv"], _W["cfg"], _W["seq"], vp)


def run_grid(
    mdv: MarketDataV,
    cfg: Config,
    seq,
    combos: Optional[Sequence[VariantParams]] = None,
    processes: int = 1,
) -> List[Dict[str, Any]]:
    """Run all ``combos`` (default: the full grid) and return metric rows.

    ``processes > 1`` fans out across a process pool (each worker holds the
    shared market data + captured param sequence via an initializer).
    """
    combos = list(combos) if combos is not None else build_grid()
    if processes <= 1:
        return [run_combo(mdv, cfg, seq, vp) for vp in combos]

    from concurrent.futures import ProcessPoolExecutor

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=processes, initializer=_init_worker, initargs=(mdv, cfg, seq)
    ) as ex:
        for row in ex.map(_run_one, combos, chunksize=4):
            rows.append(row)
    return rows
