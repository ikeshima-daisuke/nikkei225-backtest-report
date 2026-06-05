"""Buy-sizing and take-profit logic.

These are *pure* functions of the indicator values, portfolio state and the
:class:`~nikkei_leverage_sim.config.StrategyParams`.  Keeping them side-effect
free makes them easy to unit-test and reuse from both the live walk-forward
loop and the optimizer's training simulations.

All indicator inputs are plain floats (already computed causally on day ``t``);
the produced decision is executed on day ``t+1`` by the engine.
"""
from __future__ import annotations

import math
from typing import Mapping

from .config import StrategyParams

# Reference scales used to map raw indicators into roughly [-1, 1] / [0, 1].
_DRAWDOWN_SCALE = 0.30  # 30% draw-down -> normalized_drawdown == 1
_RSI_PIVOT = 50.0
_RSI_SCALE = 50.0
_MA_GAP_SCALE = 0.05  # 5% below MA -> 1.0
_RET5_SCALE = 0.05  # 5% 5-day drop -> 1.0
_TREND_SCALE = 0.10
_VOL_SCALE = 0.02


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalized_drawdown(drawdown_252: float) -> float:
    """Deeper (more negative) draw-down -> larger positive value in [0, ~1]."""
    return _clip(-drawdown_252 / _DRAWDOWN_SCALE, 0.0, 2.0)


def normalized_rsi_oversold(rsi_14: float) -> float:
    """Lower RSI (oversold) -> larger value; monotonically decreasing in RSI."""
    return _clip((_RSI_PIVOT - rsi_14) / _RSI_SCALE, 0.0, 1.0)


def normalized_ma_gap(ma_gap_25: float) -> float:
    """Price below the MA (negative gap) -> larger positive dip signal."""
    return _clip(-ma_gap_25 / _MA_GAP_SCALE, -1.0, 1.0)


def normalized_short_drop(ret_5: float) -> float:
    """A larger recent 5-day drop -> larger positive value."""
    return _clip(-ret_5 / _RET5_SCALE, -1.0, 1.0)


def trend_component(ma_gap_200: float) -> float:
    """Positive in an up-trend (price above the 200-day MA)."""
    return _clip(ma_gap_200 / _TREND_SCALE, -1.0, 1.0)


def normalized_vol(vol_20: float) -> float:
    """Higher volatility -> larger positive value in [0, ~1]."""
    return _clip(vol_20 / _VOL_SCALE, 0.0, 2.0)


def buy_score(
    ind: Mapping[str, float],
    params: StrategyParams,
    exposure_ratio: float,
    unrealized_loss_ratio: float,
) -> float:
    """Compute the raw buy score from indicators and portfolio state.

    Higher score -> larger desired purchase.  Dip signals (draw-down, oversold
    RSI, negative MA gap, recent drop) push the score up; high volatility, high
    existing exposure and existing unrealized losses push it down.
    """
    return (
        params.w_drawdown * normalized_drawdown(ind["drawdown_252"])
        + params.w_rsi * normalized_rsi_oversold(ind["RSI_14"])
        + params.w_ma_gap_25 * normalized_ma_gap(ind["ma_gap_25"])
        + params.w_ret_5 * normalized_short_drop(ind["ret_5"])
        + params.w_trend * trend_component(ind["ma_gap_200"])
        - params.w_vol * normalized_vol(ind["vol_20"])
        - params.w_exposure * exposure_ratio
        - params.w_unrealized_loss * unrealized_loss_ratio
    )


def desired_buy_amount(
    ind: Mapping[str, float],
    params: StrategyParams,
    exposure_ratio: float,
    unrealized_loss_ratio: float,
) -> float:
    """Map the buy score to a yen amount, clipped to ``[0, max_daily]``.

    ``buy_amount = base * sigmoid(scale * (score - threshold))`` then clipped.
    The hard exposure cap is applied later, at execution time.
    """
    score = buy_score(ind, params, exposure_ratio, unrealized_loss_ratio)
    raw = params.base_buy_amount * sigmoid(
        params.score_scale * (score - params.score_threshold)
    )
    return _clip(raw, 0.0, params.max_daily_buy_amount)


def required_profit_pct(
    params: StrategyParams, exposure_ratio: float, vol_20: float
) -> float:
    """Take-profit threshold (as a fraction) for a lot.

    Larger exposure -> shallower target (take profits sooner to relieve the
    book); higher volatility -> wider target.  Floored at ``min_take_profit_pct``.
    """
    raw = (
        params.base_take_profit_pct * (1.0 - exposure_ratio * params.exposure_tp_sensitivity)
        + vol_20 * params.vol_tp_multiplier
    )
    return max(params.min_take_profit_pct, raw)


def should_take_profit(
    net_pnl_before_tax: float,
    profit_pct: float,
    params: StrategyParams,
    req_profit_pct: float,
) -> bool:
    """Decide whether a single lot should be closed for profit.

    A lot is taken only when it is *in profit* (no loss is ever realized) and
    either the yen target or the percentage target is met.
    """
    if net_pnl_before_tax <= 0.0:
        return False
    return (net_pnl_before_tax >= params.fixed_profit_yen) or (
        profit_pct >= req_profit_pct
    )
