"""The simulation engine and walk-forward backtest driver.

Look-ahead policy (strictly enforced by the loop structure):

* On day ``t`` (using indicators computed from data up to the *close* of ``t``)
  we *decide* the buy amount and which lots to take profit on.
* On day ``t+1`` those orders are *executed at the open*.  Share counts are
  computed from ``open_{t+1}`` (a price we did not know on day ``t``).
* The position is then *valued at the close* of ``t+1``.

Because the decision for execution-day ``i`` is always produced from the signal
of day ``i-1`` and stored in ``pending``, the engine can never use day ``i``'s
close to trade at day ``i``'s open.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import Config, StrategyParams
from .indicators import compute_indicators
from .metrics import (
    max_consecutive_without_tp,
    max_drawdown_abs,
    no_tp_streak_penalty_days,
)
from .portfolio import Portfolio
from . import strategy as strat

# Indicator keys the strategy needs each day.
_IND_KEYS = ("drawdown_252", "RSI_14", "ma_gap_25", "ret_5", "ma_gap_200", "vol_20")

_FILL_MODELS = ("next_open", "vwap", "adverse")


def _exec_fill_prices(
    model: str, open_p: float, high_p: float, low_p: float, close_p: float, slip: float
) -> tuple[float, float]:
    """Return ``(buy_fill, sell_fill)`` for the execution day under ``model``.

    All models use only the *execution day's* own bar (a decision made at the
    prior close never sees future data, so the look-ahead invariant holds).

    * ``next_open`` — open*(1±slip); fills depend only on the open (default).
    * ``vwap``      — ((O+H+L+C)/4)*(1±slip); a VWAP proxy over the day.
    * ``adverse``   — buy at High*(1+slip), sell at Low*(1-slip); worst-case fill.
    """
    if model == "vwap":
        mid = (open_p + high_p + low_p + close_p) / 4.0
        return mid * (1.0 + slip), mid * (1.0 - slip)
    if model == "adverse":
        return high_p * (1.0 + slip), low_p * (1.0 - slip)
    return open_p * (1.0 + slip), open_p * (1.0 - slip)  # next_open


@dataclass(slots=True)
class MarketData:
    """Pre-extracted numpy arrays for fast iteration in the engine."""

    dates: np.ndarray
    target_open: np.ndarray
    target_close: np.ndarray
    benchmark_close: np.ndarray
    valuation: np.ndarray          # price used for daily marking
    vol_20: np.ndarray
    ind_rows: List[Dict[str, float]]
    valid: np.ndarray              # bool mask: indicators present -> tradable
    n: int
    target_high: np.ndarray = field(default_factory=lambda: np.empty(0))
    target_low: np.ndarray = field(default_factory=lambda: np.empty(0))
    target_volume: np.ndarray = field(default_factory=lambda: np.empty(0))
    data_repairs: List[Dict[str, Any]] = field(default_factory=list)


def prepare_market_data(joined: pd.DataFrame, cfg: Config) -> MarketData:
    """Compute indicators and pack everything into a :class:`MarketData`."""
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

    return MarketData(
        dates=df.index.to_numpy(),
        target_open=df["target_open"].to_numpy(dtype=float),
        target_close=df["target_close"].to_numpy(dtype=float),
        benchmark_close=df["benchmark_close"].to_numpy(dtype=float),
        valuation=valuation,
        vol_20=ind_arrays["vol_20"],
        ind_rows=ind_rows,
        valid=valid,
        n=n,
        target_high=df["target_high"].to_numpy(dtype=float),
        target_low=df["target_low"].to_numpy(dtype=float),
        target_volume=df["target_volume"].to_numpy(dtype=float),
        data_repairs=list(getattr(joined, "attrs", {}).get("data_repairs", []) or []),
    )


@dataclass(slots=True)
class Decision:
    """An order set decided at the close of day ``t``, executed at open ``t+1``."""

    buy_amount: float
    sell_lot_ids: List[int]
    params_id: int
    forced: bool = False


@dataclass
class SimResult:
    """Outcome of a simulation run (full or training window)."""

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
    tp_flags: List[bool] = field(default_factory=list)


ParamsProvider = Union[StrategyParams, Callable[[int], StrategyParams]]


def _decide(
    pf: Portfolio, md: MarketData, i: int, params: StrategyParams, params_id: int
) -> Decision:
    """Build the order set for execution on the next day, using day-``i`` info."""
    gross = pf.gross_exposure()
    exposure_ratio = gross / pf.cfg.max_gross_exposure if pf.cfg.max_gross_exposure else 0.0
    upnl = pf.unrealized_pnl()
    unrealized_loss_ratio = max(-upnl / pf.cfg.max_gross_exposure, 0.0)

    buy_amount = strat.desired_buy_amount(
        md.ind_rows[i], params, exposure_ratio, unrealized_loss_ratio
    )

    # Take-profit evaluation on the day-i close valuation.
    price = md.valuation[i]
    req_pct = strat.required_profit_pct(params, exposure_ratio, md.vol_20[i])
    sell_ids: List[int] = []
    for lot in pf.lots:
        net = lot.net_pnl_before_tax(price)
        pct = lot.profit_pct(price)
        if strat.should_take_profit(net, pct, params, req_pct):
            sell_ids.append(lot.lot_id)

    return Decision(buy_amount=buy_amount, sell_lot_ids=sell_ids, params_id=params_id)


def simulate(
    md: MarketData,
    lo: int,
    hi: int,
    params_provider: ParamsProvider,
    cfg: Config,
    record: bool = False,
) -> SimResult:
    """Run the engine over rows ``[lo, hi)`` and return a :class:`SimResult`.

    ``params_provider`` is either a fixed :class:`StrategyParams` (used for
    training-window simulations) or a callable ``f(i) -> StrategyParams`` that
    supplies the parameters for the decision made at the close of day ``i``.
    """
    if callable(params_provider):
        get_params = params_provider
    else:
        _fixed = params_provider
        get_params = lambda _i: _fixed  # noqa: E731

    pf = Portfolio(cfg)
    lot_by_id: Dict[int, Any] = {}
    slip = cfg.slippage_bps / 10_000.0
    max_gross = cfg.max_gross_exposure

    # Execution-realism settings (D2).  Defaults reproduce the original
    # next-open, full-fill, zero-delay behaviour bit-for-bit.
    fill_model = cfg.execution.fill_model
    if fill_model not in _FILL_MODELS:
        raise ValueError(f"Unknown fill_model {fill_model!r}; expected one of {_FILL_MODELS}")
    participation = cfg.execution.volume_participation
    exec_delay = int(cfg.execution.execution_delay_days)
    if exec_delay < 0:
        raise ValueError("execution_delay_days must be >= 0")
    has_hl = md.target_high.size == md.n and md.target_low.size == md.n
    has_vol = md.target_volume.size == md.n

    # Decisions scheduled by execution-day index (supports latency).  At delay 0
    # a decision made at close i is keyed to i+1 (next open), as before.
    pending_by_exec: Dict[int, Decision] = {}
    # Index at which an in-flight forced liquidation will execute (None = none).
    # While set, no new decisions are scheduled (the broker is liquidating), so a
    # delayed forced close cannot be postponed forever or accumulate fresh buys.
    forced_exec_idx: Optional[int] = None
    daily_rows: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    tp_flags: List[bool] = []
    reset_flags: List[bool] = []
    # Map distinct params objects to stable small ids for the daily log.
    pid_map: Dict[int, int] = {}

    for i in range(lo, hi):
        date_i = md.dates[i]
        open_p = md.target_open[i]

        day_buy_amount = 0.0
        day_buy_shares = 0
        day_sell_value = 0.0
        day_realized_before = 0.0
        day_realized_after = 0.0
        day_tax = 0.0
        day_events: List[str] = []
        exec_params_id = -1

        # --- 1. Execute the decision scheduled for this day, at open_i ---
        pending = pending_by_exec.pop(i, None)
        if pending is not None:
            exec_params_id = pending.params_id
            high_p = md.target_high[i] if has_hl else open_p
            low_p = md.target_low[i] if has_hl else open_p
            buy_price, sell_price = _exec_fill_prices(
                fill_model, open_p, high_p, low_p, md.target_close[i], slip
            )
            # A forced liquidation closes the ENTIRE book that exists at execution
            # time (not a stale decision-time snapshot), so lots opened during a
            # latency window are also liquidated.
            if pending.forced:
                forced_exec_idx = None
                sell_ids = list(lot_by_id.keys())
            else:
                sell_ids = pending.sell_lot_ids
            forced_closes = 0
            for lid in sell_ids:
                lot = lot_by_id.get(lid)
                if lot is None:
                    continue
                if pending.forced:
                    rec = pf.sell_lot(lot, i, date_i, sell_price, force=True)
                else:
                    rec = pf.sell_lot(lot, i, date_i, sell_price)
                if rec is not None:
                    day_sell_value += rec["value"]
                    day_realized_before += rec["realized_pnl_before_tax"]
                    day_realized_after += rec["realized_pnl_after_tax"]
                    day_tax += rec["tax"]
                    del lot_by_id[lid]
                    if pending.forced:
                        forced_closes += 1
                    if record:
                        trades.append(rec)
            # One forced-liquidation *event* per day the whole book is closed out
            # by a margin call (counted at execution, not at decision time).
            if pending.forced and forced_closes > 0:
                pf.forced_liquidation_count += 1

            # Buy after sells (sells free up exposure capacity).
            if pending.buy_amount > 0.0:
                capacity = max(0.0, max_gross - pf.gross_exposure())
                desired_shares = math.floor(pending.buy_amount / buy_price)
                capped = min(pending.buy_amount, capacity)
                shares = int(math.floor(capped / buy_price))
                if desired_shares > shares and desired_shares >= 1:
                    pf.exposure_limit_hit_count += 1
                    day_events.append("exposure_limit_hit")
                # Volume-participation partial fill (buy side): cap shares at a
                # fraction of the execution day's volume.
                if participation < 1.0 and has_vol:
                    vol_cap = int(math.floor(participation * md.target_volume[i]))
                    if shares > vol_cap:
                        shares = max(vol_cap, 0)
                        day_events.append("partial_fill")
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
                                    "event_type": "buy",
                                }
                            )

        # --- 2. Mark to market at close_i (accrue interest, margin events) ---
        interest_before = pf.total_interest_paid
        mevents = pf.mark_to_market(md.valuation[i], i)
        interest_today = pf.total_interest_paid - interest_before
        day_events.extend(mevents)

        # --- 2b. Maintenance-margin (追証) check after the close ---
        # Runs every day so the minimum maintenance ratio is tracked even when
        # forced liquidation is disabled; only forces a liquidation (next open)
        # when ``force_liquidation`` is enabled and the book is non-empty.
        margin_breach = pf.force_liquidation_check()
        # Only trigger a *new* forced liquidation when one is not already in
        # flight (so a delayed liquidation is scheduled once, not every day it
        # stays in breach during the latency window).
        force_now = (
            margin_breach
            and cfg.force_liquidation
            and bool(pf.lots)
            and forced_exec_idx is None
        )
        if margin_breach:
            day_events.append("margin_call")
            if force_now:
                day_events.append("forced_liquidation")

        took_profit = day_sell_value > 0.0
        tp_flags.append(took_profit)

        gross = pf.gross_exposure()
        equity = pf.equity()
        margin_ratio = pf.margin_ratio()
        maintenance_ratio = pf.maintenance_ratio()

        # A "no-take-profit streak" only accrues on days we actually hold a
        # position (a flat day has nothing to take profit on, so it resets the
        # streak rather than penalising it).  ``reset_flag`` True == reset.
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
                    "unrealized_pnl": pf.unrealized_pnl(),
                    "margin_ratio": margin_ratio if math.isfinite(margin_ratio) else np.inf,
                    "maintenance_ratio": (
                        maintenance_ratio if math.isfinite(maintenance_ratio) else np.inf
                    ),
                    "selected_params_id": exec_params_id,
                    "events": ";".join(day_events),
                    "took_profit": took_profit,
                }
            )

        # --- 3. Decide for a future day using day-i close information ---
        # Execution lands at open_{i+1+exec_delay} (next open plus any latency).
        exec_idx = i + 1 + exec_delay
        if forced_exec_idx is not None:
            # A forced liquidation is already in flight: the broker is taking the
            # book down.  Schedule nothing new (no fresh buys) and do not reschedule
            # the forced close (so a delayed liquidation cannot be postponed).
            pass
        elif force_now:
            # System event takes precedence: cancel all outstanding orders and
            # schedule a full-book forced close (the actual lot set is resolved at
            # execution time, above).
            pending_by_exec.clear()
            pending_by_exec[exec_idx] = Decision(
                buy_amount=0.0,
                sell_lot_ids=[],
                params_id=exec_params_id,
                forced=True,
            )
            forced_exec_idx = exec_idx
        elif md.valid[i]:
            params = get_params(i)
            pid = pid_map.get(id(params))
            if pid is None:
                pid = len(pid_map)
                pid_map[id(params)] = pid
            pending_by_exec[exec_idx] = _decide(pf, md, i, params, pid)

    equity_curve = pf.equity_curve
    grace = cfg.objective.no_take_profit_grace_days
    result = SimResult(
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
        tp_flags=tp_flags,
    )
    return result


@dataclass
class BacktestResult:
    """Full backtest output (records + optimization log)."""

    sim: SimResult
    optimization_rows: List[Dict[str, Any]]
    config: Config
    data_repairs: List[Dict[str, Any]] = field(default_factory=list)

    # Convenience pass-throughs used by the reporting layer.
    @property
    def daily_rows(self) -> List[Dict[str, Any]]:
        return self.sim.daily_rows

    @property
    def trades(self) -> List[Dict[str, Any]]:
        return self.sim.trades

    @property
    def portfolio(self) -> Portfolio:
        return self.sim.portfolio

    @property
    def equity_curve(self) -> List[float]:
        return self.sim.equity_curve

    @property
    def max_drawdown_equity(self) -> float:
        return self.sim.max_drawdown_equity

    @property
    def max_consecutive_no_tp(self) -> int:
        return self.sim.max_consecutive_no_tp

    @property
    def final_equity(self) -> float:
        return self.sim.final_equity


def run_backtest(md: MarketData, cfg: Config) -> BacktestResult:
    """Run the full walk-forward backtest over all available data.

    When optimization is enabled, parameters are re-selected (walk-forward) at
    the configured cadence using only trailing data, then applied to the next
    day's execution.  When disabled, the default parameters are used throughout.
    """
    # Local import avoids a module-level import cycle (optimizer imports us).
    from .optimizer import WalkForwardOptimizer

    opt = WalkForwardOptimizer(md, cfg)
    result = simulate(md, 0, md.n, opt.params_at_close, cfg, record=True)
    return BacktestResult(
        sim=result,
        optimization_rows=opt.optimization_rows,
        config=cfg,
        data_repairs=list(md.data_repairs),
    )
