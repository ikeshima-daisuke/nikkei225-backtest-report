"""Strategy-variant simulation engine.

This engine deliberately mirrors the core
:func:`nikkei_leverage_sim.backtest.simulate` loop (same look-ahead policy:
decide at the *close* of day ``t``, execute at the *open* of ``t+1``) and reuses
the validated :class:`~nikkei_leverage_sim.portfolio.Portfolio` accounting and
:mod:`~nikkei_leverage_sim.strategy` scoring functions verbatim.  It adds:

* **Initial lump sum** (``initial_amount``) -- a one-off fixed buy injected into
  the very first buy decision (executed at the next open, subject to the gross
  exposure cap).  ``0`` reproduces the pure-DCA baseline; a value at/above the
  exposure cap approximates buy-and-hold.

* **Three exit rules** (``variant``):

  - ``per_lot``  -- the existing per-lot take-profit (identical to the core
    engine; used as the in-grid control).
  - ``bulk_yen`` -- when the *aggregate* net unrealized P&L reaches
    ``bulk_exit_yen``, sell **every** open lot at the next open, then reset and
    resume DCA the following day.
  - ``bulk_pct`` -- when the day's *high* reaches ``avg_entry * (1 + bulk_exit_pct)``
    (volume-weighted average entry of all open lots), sell **every** open lot at
    that day's *close*, then reset.
  - ``combo``    -- both bulk rules active at once (pct fires same-day at close;
    yen fires next-open).  Whichever triggers first in time wins.

Bulk exits sell *all* lots, including any individually underwater ones, so they
realize an aggregate result.  Tax is applied to the **netted** aggregate gain
(gains and losses offset, as in a Japanese tokutei-kouza same-day bulk close) --
this keeps the comparison against the per-lot strategy fair (the per-lot
strategy never realizes a loss, so it has no offset to give up).

Nothing here is imported by the core package; the core test-suite is untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from nikkei_leverage_sim.config import Config, StrategyParams
from nikkei_leverage_sim.indicators import compute_indicators
from nikkei_leverage_sim.metrics import max_drawdown_abs, max_consecutive_without_tp
from nikkei_leverage_sim.portfolio import Portfolio
from nikkei_leverage_sim import strategy as strat

# Indicator keys the buy-scoring needs each day (same set the core engine uses).
_IND_KEYS = ("drawdown_252", "RSI_14", "ma_gap_25", "ret_5", "ma_gap_200", "vol_20")

VALID_VARIANTS = ("per_lot", "bulk_yen", "bulk_pct", "combo")


@dataclass(slots=True)
class VariantParams:
    """Entry/exit-rule parameters layered on top of the DCA buy engine."""

    variant: str = "per_lot"          # one of VALID_VARIANTS
    initial_amount: float = 0.0       # one-off lump injected into the first buy
    bulk_exit_yen: Optional[float] = None    # aggregate yen profit -> sell all (next open)
    bulk_exit_pct: Optional[float] = None    # high >= avg_entry*(1+pct) -> sell all (close)

    def __post_init__(self) -> None:
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {self.variant!r}")
        if self.variant in ("bulk_yen", "combo") and not self.bulk_exit_yen:
            raise ValueError(f"{self.variant} requires bulk_exit_yen > 0")
        if self.variant in ("bulk_pct", "combo") and not self.bulk_exit_pct:
            raise ValueError(f"{self.variant} requires bulk_exit_pct > 0")

    def label(self) -> str:
        bits = [self.variant]
        if self.initial_amount:
            bits.append(f"init{int(self.initial_amount):d}")
        else:
            bits.append("init0")
        if self.bulk_exit_yen:
            bits.append(f"yen{int(self.bulk_exit_yen):d}")
        if self.bulk_exit_pct:
            bits.append(f"pct{self.bulk_exit_pct:g}")
        return "__".join(bits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "initial_amount": self.initial_amount,
            "bulk_exit_yen": self.bulk_exit_yen,
            "bulk_exit_pct": self.bulk_exit_pct,
        }


@dataclass(slots=True)
class MarketDataV:
    """Numpy arrays for the variant engine (adds ``target_high`` vs core)."""

    dates: np.ndarray
    target_open: np.ndarray
    target_close: np.ndarray
    target_high: np.ndarray
    benchmark_close: np.ndarray
    valuation: np.ndarray
    vol_20: np.ndarray
    ind_rows: List[Dict[str, float]]
    valid: np.ndarray
    n: int


def prepare_market_data_v(joined: pd.DataFrame, cfg: Config) -> MarketDataV:
    """Compute indicators and pack arrays, including the target intraday high."""
    df = compute_indicators(joined)

    if cfg.execution.valuation_price == "adj_close":
        valuation = df["target_adj_close"].to_numpy(dtype=float)
    else:
        valuation = df["target_close"].to_numpy(dtype=float)

    ind_arrays = {k: df[k].to_numpy(dtype=float) for k in _IND_KEYS}
    n = len(df)
    ind_rows: List[Dict[str, float]] = [
        {k: ind_arrays[k][i] for k in _IND_KEYS} for i in range(n)
    ]
    valid = np.ones(n, dtype=bool)
    for k in _IND_KEYS:
        valid &= ~np.isnan(ind_arrays[k])
    valid &= df["target_open"].to_numpy(dtype=float) > 0

    return MarketDataV(
        dates=df.index.to_numpy(),
        target_open=df["target_open"].to_numpy(dtype=float),
        target_close=df["target_close"].to_numpy(dtype=float),
        target_high=df["target_high"].to_numpy(dtype=float),
        benchmark_close=df["benchmark_close"].to_numpy(dtype=float),
        valuation=valuation,
        vol_20=ind_arrays["vol_20"],
        ind_rows=ind_rows,
        valid=valid,
        n=n,
    )


@dataclass
class VariantResult:
    """Outcome of a variant simulation (duck-types the core ``SimResult`` for
    reuse with :func:`nikkei_leverage_sim.metrics.build_summary`)."""

    realized_after_tax: float
    ending_unrealized_pnl: float
    max_drawdown_equity: float
    max_unrealized_loss: float
    margin_call_count: int
    exposure_limit_hit_count: int
    no_tp_streak_penalty: float
    max_consecutive_no_tp: int
    final_equity: float
    equity_curve: List[float]
    portfolio: Portfolio
    daily_rows: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    bulk_exit_count: int = 0


def _avg_entry(pf: Portfolio) -> float:
    """Volume-weighted average entry price across open lots (0 if flat)."""
    tot_shares = 0
    tot_value = 0.0
    for lot in pf.lots:
        tot_shares += lot.shares
        tot_value += lot.entry_value
    if tot_shares <= 0:
        return 0.0
    return tot_value / tot_shares


def _bulk_exit_all(
    pf: Portfolio,
    lot_by_id: Dict[int, Any],
    sell_price: float,
    index: int,
    date: Any,
    record: bool,
    trades: List[Dict[str, Any]],
    reason: str,
) -> Optional[Dict[str, float]]:
    """Close *every* open lot at ``sell_price`` with netted (aggregate) tax.

    Realizes an aggregate result -- individually underwater lots are sold too.
    Tax applies to the positive netted aggregate only (gains/losses offset).
    Updates portfolio cumulative figures so ``cash()``/``equity()`` stay
    consistent.  Returns a per-day rollup dict, or ``None`` when flat.
    """
    lots = list(pf.lots)
    if not lots:
        return None

    total_value = 0.0
    agg_before = 0.0
    n_wins = 0
    comm_bps = pf.cfg.commission_bps
    for lot in lots:
        sell_value = lot.shares * sell_price
        sell_comm = sell_value * comm_bps / 10_000.0
        net_before = lot.net_pnl_before_tax(sell_price, sell_comm)
        agg_before += net_before
        total_value += sell_value
        pf.total_commission_paid += sell_comm
        if net_before > 0:
            n_wins += 1
        # Per-lot pre-tax contributions feed the (secondary) profit-factor stats.
        pf.closed_lot_profits.append(net_before)
        if net_before > 0:
            pf.closed_lot_gross_profit += net_before
        else:
            pf.closed_lot_gross_loss += -net_before

    tax = max(agg_before, 0.0) * pf.cfg.tax_rate
    agg_after = agg_before - tax

    pf.realized_before_tax += agg_before
    pf.realized_after_tax += agg_after
    pf.total_tax_paid += tax
    pf.sell_trade_count += 1            # one bulk transaction
    pf.closed_lot_count += len(lots)
    pf.closed_lot_wins += n_wins

    pf.lots.clear()
    lot_by_id.clear()

    if record:
        trades.append(
            {
                "date": date,
                "side": "SELL_ALL",
                "shares": sum(l.shares for l in lots),
                "price": sell_price,
                "value": total_value,
                "lot_id": -1,
                "n_lots": len(lots),
                "realized_pnl_before_tax": agg_before,
                "realized_pnl_after_tax": agg_after,
                "tax": tax,
                "reason": reason,
            }
        )
    return {
        "value": total_value,
        "realized_before": agg_before,
        "realized_after": agg_after,
        "tax": tax,
        "n_lots": float(len(lots)),
    }


def _resolve_provider(params_provider, n: int):
    """Return a callable ``get_params(i) -> StrategyParams``.

    Accepts a fixed :class:`StrategyParams` (used every day), a pre-computed
    sequence of length ``n`` (one per day -- e.g. a captured walk-forward
    selection), or a callable ``f(i) -> StrategyParams``.
    """
    if isinstance(params_provider, StrategyParams):
        fixed = params_provider
        return lambda _i: fixed
    if callable(params_provider):
        return params_provider
    seq = list(params_provider)
    if len(seq) != n:
        raise ValueError(f"params sequence length {len(seq)} != n_rows {n}")
    return lambda i: seq[i]


def simulate_variant(
    md: MarketDataV,
    cfg: Config,
    params_provider,
    vparams: VariantParams,
    record: bool = True,
) -> VariantResult:
    """Run the variant engine over all rows.

    ``params_provider`` drives the daily DCA buy sizing (and, for the
    ``per_lot`` variant, the per-lot take-profit).  It may be a fixed
    :class:`StrategyParams`, a per-day sequence of length ``md.n`` (e.g. a
    captured walk-forward param selection), or a callable ``f(i)``.  ``vparams``
    selects the entry/exit rule.
    """
    get_params = _resolve_provider(params_provider, md.n)
    pf = Portfolio(cfg)
    lot_by_id: Dict[int, Any] = {}
    slip = cfg.slippage_bps / 10_000.0
    max_gross = cfg.max_gross_exposure
    variant = vparams.variant
    yen_target = vparams.bulk_exit_yen
    pct_target = vparams.bulk_exit_pct

    pending: Optional[Dict[str, Any]] = None
    initial_remaining = float(vparams.initial_amount)

    daily_rows: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    reset_flags: List[bool] = []
    bulk_exit_count = 0

    for i in range(md.n):
        date_i = md.dates[i]
        open_p = md.target_open[i]

        day_buy_amount = 0.0
        day_buy_shares = 0
        day_sell_value = 0.0
        day_realized_before = 0.0
        day_realized_after = 0.0
        day_tax = 0.0
        day_events: List[str] = []

        # --- 1. Execute the decision made at the previous close, at open_i ---
        if pending is not None:
            sell_price = open_p * (1.0 - slip)
            if pending.get("sell_all"):
                roll = _bulk_exit_all(
                    pf, lot_by_id, sell_price, i, date_i, record, trades,
                    reason="bulk_yen",
                )
                if roll is not None:
                    day_sell_value += roll["value"]
                    day_realized_before += roll["realized_before"]
                    day_realized_after += roll["realized_after"]
                    day_tax += roll["tax"]
                    bulk_exit_count += 1
            else:
                for lid in pending.get("sell_lot_ids", ()):  # per_lot exits
                    lot = lot_by_id.get(lid)
                    if lot is None:
                        continue
                    rec = pf.sell_lot(lot, i, date_i, sell_price)
                    if rec is not None:
                        day_sell_value += rec["value"]
                        day_realized_before += rec["realized_pnl_before_tax"]
                        day_realized_after += rec["realized_pnl_after_tax"]
                        day_tax += rec["tax"]
                        del lot_by_id[lid]
                        if record:
                            trades.append(rec)

            # Buy after sells (sells free up exposure capacity).
            buy_amount = pending.get("buy_amount", 0.0)
            if buy_amount > 0.0:
                buy_price = open_p * (1.0 + slip)
                capacity = max(0.0, max_gross - pf.gross_exposure())
                desired_shares = math.floor(buy_amount / buy_price)
                capped = min(buy_amount, capacity)
                shares = int(math.floor(capped / buy_price))
                if desired_shares > shares and desired_shares >= 1:
                    pf.exposure_limit_hit_count += 1
                    day_events.append("exposure_limit_hit")
                if shares >= 1:
                    lot = pf.buy(i, date_i, shares, buy_price)
                    if lot is not None:
                        lot_by_id[lot.lot_id] = lot
                        day_buy_shares = shares
                        day_buy_amount = shares * buy_price
                        if record:
                            trades.append(
                                {
                                    "date": date_i,
                                    "side": "BUY",
                                    "shares": shares,
                                    "price": buy_price,
                                    "value": day_buy_amount,
                                    "lot_id": lot.lot_id,
                                    "realized_pnl_before_tax": 0.0,
                                    "realized_pnl_after_tax": 0.0,
                                    "reason": "buy_signal",
                                }
                            )

        # --- 2. Mark to market at close_i (accrue interest, margin events) ---
        interest_before = pf.total_interest_paid
        mevents = pf.mark_to_market(md.valuation[i], i)
        interest_today = pf.total_interest_paid - interest_before
        day_events.extend(mevents)

        # --- 3. Same-day bulk-pct exit: high_i >= avg_entry*(1+pct) -> close_i ---
        if pct_target and variant in ("bulk_pct", "combo") and pf.lots:
            avg_entry = _avg_entry(pf)
            if avg_entry > 0.0 and md.target_high[i] >= avg_entry * (1.0 + pct_target):
                close_sell = md.valuation[i] * (1.0 - slip)
                roll = _bulk_exit_all(
                    pf, lot_by_id, close_sell, i, date_i, record, trades,
                    reason="bulk_pct",
                )
                if roll is not None:
                    day_sell_value += roll["value"]
                    day_realized_before += roll["realized_before"]
                    day_realized_after += roll["realized_after"]
                    day_tax += roll["tax"]
                    bulk_exit_count += 1
                    # Re-mark (now flat) so the recorded equity is post-exit.
                    pf.equity_curve[-1] = pf.equity()

        took_profit = day_sell_value > 0.0

        gross = pf.gross_exposure()
        equity = pf.equity()
        upnl = pf.unrealized_pnl()
        margin_ratio = pf.margin_ratio()

        # No-take-profit streak resets on a profit day or any flat day.
        reset_flags.append(took_profit or gross <= 0.0)

        if record:
            daily_rows.append(
                {
                    "date": pd.Timestamp(date_i),
                    "target_open": open_p,
                    "target_close": md.target_close[i],
                    "benchmark_close": md.benchmark_close[i],
                    "buy_amount": day_buy_amount,
                    "buy_shares": day_buy_shares,
                    "sell_value": day_sell_value,
                    "realized_pnl_before_tax": day_realized_before,
                    "realized_pnl_after_tax": day_realized_after,
                    "tax": day_tax,
                    "interest": interest_today,
                    "gross_exposure": gross,
                    "cash": pf.cash(),
                    "equity": equity,
                    "unrealized_pnl": upnl,
                    "margin_ratio": margin_ratio if math.isfinite(margin_ratio) else np.inf,
                    "events": ";".join(day_events),
                    "took_profit": took_profit,
                }
            )

        # --- 4. Decide for the next day using day-i close information ---
        if md.valid[i]:
            params_i = get_params(i)
            gross_now = pf.gross_exposure()
            exposure_ratio = gross_now / cfg.max_gross_exposure if cfg.max_gross_exposure else 0.0
            upnl_now = pf.unrealized_pnl()
            unrealized_loss_ratio = max(-upnl_now / cfg.max_gross_exposure, 0.0)

            buy_amount = strat.desired_buy_amount(
                md.ind_rows[i], params_i, exposure_ratio, unrealized_loss_ratio
            )
            # Inject the one-off initial lump into the first buy decision.
            if initial_remaining > 0.0:
                buy_amount += initial_remaining
                initial_remaining = 0.0

            decision: Dict[str, Any] = {"buy_amount": buy_amount}

            if variant == "per_lot":
                price = md.valuation[i]
                req_pct = strat.required_profit_pct(params_i, exposure_ratio, md.vol_20[i])
                sell_ids: List[int] = []
                for lot in pf.lots:
                    net = lot.net_pnl_before_tax(price)
                    pct = lot.profit_pct(price)
                    if strat.should_take_profit(net, pct, params_i, req_pct):
                        sell_ids.append(lot.lot_id)
                decision["sell_lot_ids"] = sell_ids
            elif variant in ("bulk_yen", "combo"):
                # bulk-yen check (next-open execution); pct handled same-day above.
                if yen_target and pf.lots and pf.unrealized_pnl() >= yen_target:
                    decision["sell_all"] = True
            # bulk_pct: nothing to schedule for next open.

            pending = decision
        else:
            pending = None

    equity_curve = pf.equity_curve
    grace = cfg.objective.no_take_profit_grace_days
    from nikkei_leverage_sim.metrics import no_tp_streak_penalty_days

    return VariantResult(
        realized_after_tax=pf.realized_after_tax,
        ending_unrealized_pnl=pf.unrealized_pnl(),
        max_drawdown_equity=max_drawdown_abs(equity_curve),
        max_unrealized_loss=pf.max_unrealized_loss,
        margin_call_count=pf.margin_call_count,
        exposure_limit_hit_count=pf.exposure_limit_hit_count,
        no_tp_streak_penalty=float(no_tp_streak_penalty_days(reset_flags, grace)),
        max_consecutive_no_tp=max_consecutive_without_tp(reset_flags),
        final_equity=equity_curve[-1] if equity_curve else cfg.initial_equity,
        equity_curve=equity_curve,
        portfolio=pf,
        daily_rows=daily_rows,
        trades=trades,
        bulk_exit_count=bulk_exit_count,
    )
