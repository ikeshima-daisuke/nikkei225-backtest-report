"""Accumulation and exit policies for the comparison study.

Each policy is a small stateful object (``__call__(ctx)``); state that must
restart when a *rotating* plan re-enters (trailing anchors, tranche counters) is
keyed on ``ctx.phase_start_t`` and reset automatically via :class:`_PhaseAware`.
All signals come pre-computed and lookahead-safe from :mod:`signals`.

Accumulation ``__call__`` -> yen to BUY at today's close.
Exit ``__call__`` -> ``(sell_fraction, terminate)``.

The method families implement the catalog recommended for a 2x leveraged ETF
judged on Calmar (avoid large draw-downs while compounding the trend):
DCA / lump (controls), buy-the-dip ladder, value averaging, trend-filter DCA,
vol-/RSI-scaled DCA, dual (absolute) momentum; exits: hold, take-profit,
trailing stop, MA-cross, target-date glide, vol-spike.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .engine import Ctx


class _PhaseAware:
    """Mixin: call :meth:`_maybe_reset` first thing each day to honour phases."""

    def __init__(self) -> None:
        self._phase = -1

    def _maybe_reset(self, ctx: Ctx) -> None:
        if ctx.phase_start_t != self._phase:
            self._phase = ctx.phase_start_t
            self.reset()

    def reset(self) -> None:  # overridden by stateful policies
        pass


# ---------------------------------------------------------------------------
# Accumulation policies
# ---------------------------------------------------------------------------
class LumpSum(_PhaseAware):
    """Deploy all available cash on the first buy-day of the phase."""

    name = "lump"

    def reset(self) -> None:
        self._done = False

    def __call__(self, ctx: Ctx) -> float:
        self._maybe_reset(ctx)
        if self._done:
            return 0.0
        self._done = True
        return ctx.cash


class FixedDCA(_PhaseAware):
    """Constant yen per buy-day until ``n_installments`` have been made."""

    def __init__(self, n_installments: int):
        super().__init__()
        self.n_installments = max(1, int(n_installments))
        self.name = f"dca{self.n_installments}"

    def reset(self) -> None:
        self._made = 0

    def __call__(self, ctx: Ctx) -> float:
        self._maybe_reset(ctx)
        if self._made >= self.n_installments:
            return 0.0
        self._made += 1
        return ctx.capital / self.n_installments


class BuyTheDip(_PhaseAware):
    """Hold cash; release tranches as price falls from its trailing peak.

    A tranche fires once when the drawdown first crosses its threshold and
    re-arms only after a fresh trailing peak (no thrashing).  Any cash still
    idle after ``backstop_buys`` buy-days is dollar-cost-averaged out over the
    rest of the window, so a calm decade does not leave capital unused forever.
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        weights: Sequence[float],
        backstop_buys: int,
    ):
        super().__init__()
        self.thresholds = list(thresholds)
        self.weights = list(weights)
        self.backstop_buys = max(1, int(backstop_buys))
        thr = "_".join(str(int(-t * 100)) for t in thresholds)
        self.name = f"dip{thr}"

    def reset(self) -> None:
        self._armed = [True] * len(self.thresholds)
        self._peak_seen = -math.inf
        self._buys = 0
        self._spent = 0.0

    def __call__(self, ctx: Ctx) -> float:
        self._maybe_reset(ctx)
        self._buys += 1
        s = ctx.signals
        # Re-arm tranches on a new trailing peak.
        if s.trailing_peak[ctx.t] > self._peak_seen:
            self._peak_seen = float(s.trailing_peak[ctx.t])
            self._armed = [True] * len(self.thresholds)
        dd = float(s.drawdown[ctx.t])
        buy = 0.0
        for i, thr in enumerate(self.thresholds):
            if self._armed[i] and dd <= thr:
                self._armed[i] = False
                buy += ctx.capital * self.weights[i]
        # Stale-cash backstop: DCA the remainder once the window is half gone.
        if self._buys >= self.backstop_buys:
            remaining_buys = max(1, ctx.schedule_days_left)
            buy += ctx.cash / remaining_buys
        return min(buy, ctx.cash)


class ValueAveraging(_PhaseAware):
    """Buy-only value averaging: top up to a value path that grows each period.

    Target after ``k`` buys is ``(k / n_periods) * capital * (1+g)^(k/period)``
    capped at ``capital``; we buy the shortfall versus the marked position
    (never sell on overshoot, to respect the accumulate-only spirit).
    """

    def __init__(self, n_periods: int, annual_growth: float, periods_per_year: float):
        super().__init__()
        self.n_periods = max(1, int(n_periods))
        self.g = float(annual_growth)
        self.ppy = float(periods_per_year)
        self.name = f"va{self.n_periods}_{int(annual_growth*100)}"

    def reset(self) -> None:
        self._k = 0

    def __call__(self, ctx: Ctx) -> float:
        self._maybe_reset(ctx)
        self._k += 1
        frac = min(1.0, self._k / self.n_periods)
        growth = (1.0 + self.g) ** (self._k / self.ppy) if self.g else 1.0
        target = min(ctx.capital, frac * ctx.capital * growth)
        position_value = ctx.shares * ctx.price
        shortfall = target - position_value
        return max(0.0, min(shortfall, ctx.cash))


class TrendFilterDCA(_PhaseAware):
    """DCA, but skip (park) contributions while ``close < SMA_n``.

    Parked cash is deployed on the next session back above the average
    (``dump_parked=True``) or simply skipped.  Before the SMA is seeded we
    default to investing (treat unknown trend as "in").
    """

    def __init__(self, n_ma: int, n_installments: int, dump_parked: bool):
        super().__init__()
        self.n_ma = int(n_ma)
        self.n_installments = max(1, int(n_installments))
        self.dump_parked = bool(dump_parked)
        self.name = f"trend{self.n_ma}{'D' if dump_parked else 'S'}"

    def reset(self) -> None:
        self._made = 0
        self._parked = 0.0

    def __call__(self, ctx: Ctx) -> float:
        self._maybe_reset(ctx)
        if self._made >= self.n_installments:
            return 0.0
        self._made += 1
        amount = ctx.capital / self.n_installments
        ma = ctx.signals.sma100[ctx.t] if self.n_ma == 100 else ctx.signals.sma200[ctx.t]
        in_trend = (not math.isfinite(ma)) or (ctx.price >= ma)
        if in_trend:
            buy = amount + (self._parked if self.dump_parked else 0.0)
            self._parked = 0.0
            return min(buy, ctx.cash)
        # Below trend: park this installment for later.
        self._parked += amount
        return 0.0


class ScaledDCA(_PhaseAware):
    """DCA whose per-period amount is scaled by an oversold/low-vol multiplier.

    ``signal='rsi'``: buy more when RSI(14) is low (mean-reversion).
    ``signal='invvol'``: buy more when 20d realized vol is low (the decay-aware
    choice — high realized vol is when a 2x ETF bleeds most).  Multiplier is
    clipped to ``[0.5, 2.0]``; total deployment is still bounded by cash.
    """

    def __init__(self, signal: str, n_installments: int, target_vol: float = 0.25):
        super().__init__()
        self.signal = signal
        self.n_installments = max(1, int(n_installments))
        self.target_vol = float(target_vol)
        self.name = f"scaled_{signal}{self.n_installments}"

    def reset(self) -> None:
        self._made = 0

    def _mult(self, ctx: Ctx) -> float:
        s = ctx.signals
        if self.signal == "rsi":
            r = s.rsi14[ctx.t]
            if not math.isfinite(r):
                return 1.0
            # RSI 30 -> 2.0x, 50 -> 1.0x, 70 -> 0.5x (piecewise linear, clipped).
            if r <= 50:
                m = 1.0 + (50 - r) / 20.0      # 50->1.0, 30->2.0
            else:
                m = 1.0 - (r - 50) / 40.0       # 50->1.0, 70->0.5
            return min(2.0, max(0.5, m))
        v = s.vol20_ann[ctx.t]
        if not math.isfinite(v) or v <= 0:
            return 1.0
        return min(2.0, max(0.5, self.target_vol / v))

    def __call__(self, ctx: Ctx) -> float:
        self._maybe_reset(ctx)
        if self._made >= self.n_installments:
            return 0.0
        self._made += 1
        base = ctx.capital / self.n_installments
        return min(base * self._mult(ctx), ctx.cash)


class MomentumAllIn:
    """Absolute-momentum accumulation: go fully invested while 12m momentum>0.

    Pairs with :class:`MomentumExit` (rotating) to form the dual-momentum
    on/off switch.  Before the lookback is seeded we default to invested.
    """

    def __init__(self, lookback: int = 252):
        self.lookback = int(lookback)
        self.name = f"mom{lookback}"

    def __call__(self, ctx: Ctx) -> float:
        m = ctx.signals.mom252[ctx.t]
        invest = (not math.isfinite(m)) or (m >= 0.0)
        return ctx.cash if invest else 0.0


# ---------------------------------------------------------------------------
# Exit policies  ->  (sell_fraction, terminate)
# ---------------------------------------------------------------------------
class HoldToEnd:
    name = "hold"

    def __call__(self, ctx: Ctx) -> Tuple[float, bool]:
        return 0.0, False


class TakeProfit:
    """Sell everything when the held position's gain reaches ``x``."""

    def __init__(self, x: float):
        self.x = float(x)
        self.name = f"tp{int(x*100)}"

    def __call__(self, ctx: Ctx) -> Tuple[float, bool]:
        if ctx.shares > 0 and ctx.avg_cost > 0 and ctx.price / ctx.avg_cost - 1.0 >= self.x:
            return 1.0, True
        return 0.0, False


class TrailingStop:
    """Sell everything if equity falls ``y`` below its running peak."""

    def __init__(self, y: float):
        self.y = float(y)
        self.name = f"trail{int(y*100)}"

    def __call__(self, ctx: Ctx) -> Tuple[float, bool]:
        if ctx.equity_peak > 0 and ctx.equity / ctx.equity_peak - 1.0 <= -self.y:
            return 1.0, True
        return 0.0, False


class MAExit:
    """Sell everything when ``close < SMA_n * (1 - band)``."""

    def __init__(self, n_ma: int, band: float = 0.0):
        self.n_ma = int(n_ma)
        self.band = float(band)
        self.name = f"maexit{n_ma}" + (f"_{int(band*100)}" if band else "")

    def __call__(self, ctx: Ctx) -> Tuple[float, bool]:
        if ctx.shares <= 0:
            return 0.0, False
        ma = ctx.signals.sma100[ctx.t] if self.n_ma == 100 else ctx.signals.sma200[ctx.t]
        if math.isfinite(ma) and ctx.price < ma * (1.0 - self.band):
            return 1.0, True
        return 0.0, False


class GlideExit:
    """Calendar de-risking: sell ``1/k`` of peak shares over the last ``k`` buy-days.

    Purely time-based (lookahead-safe), but **endpoint-dependent** — it flatters
    a window that ends near a high, so it is reported with that caveat.
    """

    def __init__(self, start_index: int, k_steps: int):
        self.start_index = int(start_index)
        self.k_steps = max(1, int(k_steps))
        self.name = f"glide{self.k_steps}"
        self._base_shares = None
        self._steps = 0

    def __call__(self, ctx: Ctx) -> Tuple[float, bool]:
        if ctx.t < self.start_index or ctx.shares <= 0:
            return 0.0, False
        if self._base_shares is None:
            self._base_shares = ctx.shares
        self._steps += 1
        # Sell a constant slice of the original holding each step.
        target_sell_shares = self._base_shares / self.k_steps
        frac = min(1.0, target_sell_shares / ctx.shares)
        terminate = self._steps >= self.k_steps
        return frac, terminate


class VolSpikeExit:
    """Move to cash when 20d annualized realized vol exceeds ``vstar``."""

    def __init__(self, vstar: float):
        self.vstar = float(vstar)
        self.name = f"volx{int(vstar*100)}"

    def __call__(self, ctx: Ctx) -> Tuple[float, bool]:
        if ctx.shares <= 0:
            return 0.0, False
        v = ctx.signals.vol20_ann[ctx.t]
        if math.isfinite(v) and v > self.vstar:
            return 1.0, False  # rotating regime toggle, not terminal
        return 0.0, False


class MomentumExit:
    """Exit to cash while 12m absolute momentum is negative (rotating)."""

    def __init__(self, lookback: int = 252):
        self.lookback = int(lookback)
        self.name = f"momexit{lookback}"

    def __call__(self, ctx: Ctx) -> Tuple[float, bool]:
        if ctx.shares <= 0:
            return 0.0, False
        m = ctx.signals.mom252[ctx.t]
        if math.isfinite(m) and m < 0.0:
            return 1.0, False
        return 0.0, False
