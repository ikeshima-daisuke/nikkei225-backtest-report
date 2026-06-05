"""Lot-based margin portfolio.

The portfolio manages a list of :class:`Lot` objects (margin *long* positions).
Key rules enforced here:

* Buying creates a new lot; selling closes specific lots only.
* A loss is **never** realized by the *strategy* — the engine only ever asks to
  sell lots that are in profit, and :meth:`Portfolio.sell_lot` additionally
  refuses to realize a negative ``net_pnl_before_tax`` (a final safety guard).
  The **sole exception** is a broker *forced liquidation* (追証/強制ロスカット):
  when :meth:`Portfolio.force_liquidation_check` reports a maintenance-margin
  breach, the engine closes the whole book via ``sell_lot(..., force=True)``,
  which bypasses that guard.  This is a *system* event, not a stop-loss.
* Margin interest accrues daily on each open lot's market value.
* Tax is charged only on positive realized P&L of a closed lot.

Cash / equity convention (margin trading with large own funds):

* ``cash    = initial_equity + cumulative realized net P&L (after tax)``
* ``equity  = cash + unrealized P&L (net, before tax)``
* ``gross_exposure = sum of open lots' market value``
* ``margin_ratio = equity / gross_exposure`` (``inf`` when flat)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Config


@dataclass(slots=True)
class Lot:
    """A single margin-long position."""

    lot_id: int
    entry_index: int          # integer row index of the entry day
    entry_date: Any           # pandas Timestamp / date
    shares: int
    entry_price: float
    entry_value: float        # shares * entry_price
    accumulated_interest: float = 0.0
    commission_paid: float = 0.0   # entry-side commission
    market_value: float = 0.0      # updated on mark-to-market

    def net_pnl_before_tax(self, price: float, sell_commission: float = 0.0) -> float:
        """Net P&L of this lot if marked/sold at ``price`` (before tax)."""
        gross = self.shares * price - self.entry_value
        return gross - self.accumulated_interest - self.commission_paid - sell_commission

    def profit_pct(self, price: float) -> float:
        """Gross profit fraction relative to the entry value."""
        if self.entry_value == 0:
            return 0.0
        return (self.shares * price - self.entry_value) / self.entry_value


class Portfolio:
    """Mutable portfolio state plus running statistics."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.lots: List[Lot] = []
        self._next_lot_id = 0

        # Cumulative realized figures.
        self.realized_before_tax = 0.0
        self.realized_after_tax = 0.0
        self.total_interest_paid = 0.0
        self.total_commission_paid = 0.0
        self.total_tax_paid = 0.0

        # Trade / lot statistics.
        self.buy_trade_count = 0
        self.sell_trade_count = 0
        self.closed_lot_count = 0
        self.closed_lot_wins = 0
        self.closed_lot_profits: List[float] = []
        self.closed_lot_gross_profit = 0.0   # sum of positive realized (profit factor)
        self.closed_lot_gross_loss = 0.0     # sum of |negative realized|

        # Risk tracking.
        self.max_unrealized_loss = 0.0  # positive magnitude of worst unrealized P&L
        self.max_gross_exposure_seen = 0.0
        self.exposure_sum = 0.0
        self.exposure_obs = 0
        self.equity_curve: List[float] = []
        self.margin_warning_count = 0
        self.margin_call_count = 0          # days the maintenance ratio breached
        self.forced_liquidation_count = 0   # distinct forced-liquidation events
        self.min_maintenance_ratio_seen = math.inf
        self.exposure_limit_hit_count = 0
        self.max_lot_holding_days = 0

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    def gross_exposure(self) -> float:
        return sum(lot.market_value for lot in self.lots)

    def unrealized_pnl(self) -> float:
        """Net unrealized P&L over open lots (before tax)."""
        return sum(
            lot.market_value - lot.entry_value - lot.accumulated_interest - lot.commission_paid
            for lot in self.lots
        )

    def cash(self) -> float:
        return self.cfg.initial_equity + self.realized_after_tax

    def equity(self) -> float:
        return self.cash() + self.unrealized_pnl()

    def margin_ratio(self) -> float:
        ge = self.gross_exposure()
        if ge <= 0.0:
            return math.inf
        return self.equity() / ge

    def maintenance_ratio(self) -> float:
        """Broker-style 委託保証金維持率 used for the forced-liquidation (追証) check.

        ``maintenance_ratio = (own_funds + min(unrealized_pnl, 0)) / gross_position``

        where ``own_funds = initial_equity + realized_after_tax`` is the current
        deposited collateral (the initial margin **plus realized P&L, so realized
        losses from a prior forced close reduce it** — and net of the financing
        and tax already booked into realized P&L), and ``gross_position`` is the
        current mark-to-market value of all open lots.  Only *unrealized losses*
        count against collateral (paper gains are excluded, the conservative
        Japanese 維持率 convention); open-lot financing cost is already inside
        ``unrealized_pnl``.  Returns ``inf`` when the book is flat.

        Design notes:
        * Deliberate deviation from the literal Week-0.5 brief, which wrote the
          numerator as ``... + gross_position``.  That extra term makes the ratio
          ``>= 1`` for any positive own funds, so a 30%% threshold could never
          trigger; it is dropped (the standard 維持率 = collateral / position).
        * The collateral uses ``own_funds`` (= ``cash()``), not the old
          ``initial_equity - total_interest_paid - total_tax_paid`` buffer, which
          both *ignored realized losses* (overstating margin after a forced close)
          and *double-counted open-lot interest* (subtracted once here and again
          inside ``unrealized_pnl``).  Stricter than ``margin_ratio`` because it
          excludes unrealized gains from collateral.
        """
        ge = self.gross_exposure()
        if ge <= 0.0:
            return math.inf
        own_funds = self.cash()  # initial_equity + realized_after_tax
        return (own_funds + min(self.unrealized_pnl(), 0.0)) / ge

    def force_liquidation_check(self) -> bool:
        """Daily post-close maintenance check (追証 / forced loss-cut).

        Computes :meth:`maintenance_ratio` after the close has been marked, and
        returns ``True`` when it has fallen below ``cfg.maintenance_margin_ratio``
        — i.e. a real broker would issue a margin call and force-close the whole
        book at the next open.  Side effects (run exactly once per trading day by
        the engine):

        * tracks the running minimum maintenance ratio
          (:attr:`min_maintenance_ratio_seen`), reported for *every* run so the
          "lowest maintenance ratio" is visible even when forced liquidation is
          disabled;
        * increments :attr:`margin_call_count` on a breach, regardless of the
          ``force_liquidation`` flag (so a no-liquidation run still reports how
          many days it *would* have been margin-called).

        The actual selling and the per-event :attr:`forced_liquidation_count`
        are handled by the engine, which only acts on a breach when
        ``cfg.force_liquidation`` is enabled (the loss-cut is a *system* event,
        never a strategy stop-loss — see :meth:`sell_lot`).
        """
        ratio = self.maintenance_ratio()
        if math.isfinite(ratio):
            if ratio < self.min_maintenance_ratio_seen:
                self.min_maintenance_ratio_seen = ratio
            if ratio < self.cfg.maintenance_margin_ratio:
                self.margin_call_count += 1
                return True
        return False

    # ------------------------------------------------------------------ #
    # Daily update
    # ------------------------------------------------------------------ #
    def mark_to_market(self, price: float, index: int) -> List[str]:
        """Update market values, accrue interest and check margin events.

        Returns a list of event strings raised on this day.
        """
        rate = self.cfg.annual_margin_interest_rate / 365.0
        for lot in self.lots:
            lot.market_value = lot.shares * price
            interest = lot.market_value * rate
            lot.accumulated_interest += interest
            self.total_interest_paid += interest
            holding = index - lot.entry_index
            if holding > self.max_lot_holding_days:
                self.max_lot_holding_days = holding

        events: List[str] = []
        ge = self.gross_exposure()
        self.max_gross_exposure_seen = max(self.max_gross_exposure_seen, ge)
        self.exposure_sum += ge
        self.exposure_obs += 1

        upnl = self.unrealized_pnl()
        if upnl < 0 and -upnl > self.max_unrealized_loss:
            self.max_unrealized_loss = -upnl

        eq = self.cash() + upnl
        self.equity_curve.append(eq)

        # The maintenance-ratio breach (追証 / margin call) and the resulting
        # forced liquidation are handled by the engine via
        # ``force_liquidation_check`` *after* this mark-to-market, so a breach is
        # counted exactly once and the book is closed at the next open.  Here we
        # only flag the informational *warning* band (maintenance <= ratio < warn).
        if ge > 0.0:
            ratio = self.maintenance_ratio()
            if self.cfg.maintenance_margin_ratio <= ratio < self.cfg.warning_margin_ratio:
                self.margin_warning_count += 1
                events.append("margin_warning")
        return events

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    def buy(
        self, index: int, date: Any, shares: int, exec_price: float
    ) -> Optional[Lot]:
        """Open a new lot of ``shares`` at ``exec_price``."""
        if shares <= 0:
            return None
        entry_value = shares * exec_price
        commission = entry_value * self.cfg.commission_bps / 10_000.0
        lot = Lot(
            lot_id=self._next_lot_id,
            entry_index=index,
            entry_date=date,
            shares=shares,
            entry_price=exec_price,
            entry_value=entry_value,
            commission_paid=commission,
            market_value=entry_value,
        )
        self._next_lot_id += 1
        self.lots.append(lot)
        self.buy_trade_count += 1
        self.total_commission_paid += commission
        return lot

    def sell_lot(
        self, lot: Lot, index: int, date: Any, exec_price: float, force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Close ``lot`` at ``exec_price``.

        Refuses to realize a loss (returns ``None`` and leaves the lot open if
        ``net_pnl_before_tax < 0``) unless ``force`` is set, which is only used
        by the optional forced-liquidation path (a *system* event, never a
        strategy stop-loss).  Returns a trade record dict on success.
        """
        sell_value = lot.shares * exec_price
        sell_commission = sell_value * self.cfg.commission_bps / 10_000.0
        net_before_tax = lot.net_pnl_before_tax(exec_price, sell_commission)

        # Hard guard: never realize a loss (the strategy has no stop-loss).
        if net_before_tax < 0.0 and not force:
            return None

        tax = max(net_before_tax, 0.0) * self.cfg.tax_rate
        net_after_tax = net_before_tax - tax

        # Update cumulative figures.
        self.realized_before_tax += net_before_tax
        self.realized_after_tax += net_after_tax
        self.total_tax_paid += tax
        self.total_commission_paid += sell_commission

        self.sell_trade_count += 1
        self.closed_lot_count += 1
        self.closed_lot_profits.append(net_after_tax)
        if net_after_tax > 0:
            self.closed_lot_wins += 1
            self.closed_lot_gross_profit += net_after_tax
        else:
            self.closed_lot_gross_loss += -net_after_tax

        self.lots.remove(lot)
        return {
            "date": date,
            "side": "SELL",
            "shares": lot.shares,
            "price": exec_price,
            "value": sell_value,
            "lot_id": lot.lot_id,
            "realized_pnl_before_tax": net_before_tax,
            "realized_pnl_after_tax": net_after_tax,
            "tax": tax,
            "reason": "forced_liquidation" if force else "take_profit",
            "event_type": "forced_liquidation" if force else "take_profit",
        }
