"""Strategy (buy sizing + take profit) unit tests."""
from __future__ import annotations

from nikkei_leverage_sim.config import StrategyParams
from nikkei_leverage_sim import strategy as strat


def _ind(**overrides) -> dict:
    base = {
        "drawdown_252": -0.05,
        "RSI_14": 50.0,
        "ma_gap_25": 0.0,
        "ret_5": 0.0,
        "ma_gap_200": 0.0,
        "vol_20": 0.01,
    }
    base.update(overrides)
    return base


def test_deeper_drawdown_increases_buy_score():
    p = StrategyParams()
    shallow = strat.buy_score(_ind(drawdown_252=-0.02), p, 0.0, 0.0)
    deep = strat.buy_score(_ind(drawdown_252=-0.25), p, 0.0, 0.0)
    assert deep > shallow


def test_higher_exposure_reduces_buy_amount():
    p = StrategyParams()
    low = strat.desired_buy_amount(_ind(drawdown_252=-0.2), p, 0.1, 0.0)
    high = strat.desired_buy_amount(_ind(drawdown_252=-0.2), p, 0.9, 0.0)
    assert high < low


def test_higher_unrealized_loss_reduces_buy_amount():
    p = StrategyParams()
    low = strat.desired_buy_amount(_ind(drawdown_252=-0.2), p, 0.2, 0.0)
    high = strat.desired_buy_amount(_ind(drawdown_252=-0.2), p, 0.2, 0.5)
    assert high < low


def test_buy_amount_never_exceeds_max_daily():
    p = StrategyParams(base_buy_amount=5_000_000, max_daily_buy_amount=1_000_000)
    # Very strong dip signal -> raw amount would blow past the cap.
    amt = strat.desired_buy_amount(
        _ind(drawdown_252=-0.5, RSI_14=10.0, ma_gap_25=-0.1, ret_5=-0.1), p, 0.0, 0.0
    )
    assert amt <= p.max_daily_buy_amount + 1e-9


def test_take_profit_fixed_yen():
    p = StrategyParams(fixed_profit_yen=5_000, base_take_profit_pct=999.0, min_take_profit_pct=999.0)
    # pct target impossible -> only the yen rule can fire.
    assert strat.should_take_profit(6_000.0, 0.001, p, 999.0) is True
    assert strat.should_take_profit(4_000.0, 0.001, p, 999.0) is False


def test_take_profit_fixed_pct():
    p = StrategyParams(fixed_profit_yen=10**9)  # yen rule effectively off
    req = 0.01
    assert strat.should_take_profit(100.0, 0.02, p, req) is True
    assert strat.should_take_profit(100.0, 0.005, p, req) is False


def test_losing_lot_never_takes_profit():
    p = StrategyParams(fixed_profit_yen=1.0)
    assert strat.should_take_profit(-1.0, -0.05, p, 0.0) is False


def test_required_profit_pct_floor_and_exposure_scaling():
    p = StrategyParams(
        base_take_profit_pct=0.01,
        min_take_profit_pct=0.003,
        exposure_tp_sensitivity=0.5,
        vol_tp_multiplier=0.0,
    )
    # Higher exposure -> shallower required pct.
    low_exp = strat.required_profit_pct(p, 0.0, 0.0)
    high_exp = strat.required_profit_pct(p, 1.0, 0.0)
    assert high_exp < low_exp
    # Floored at the minimum.
    p2 = StrategyParams(
        base_take_profit_pct=0.004, min_take_profit_pct=0.003,
        exposure_tp_sensitivity=0.9, vol_tp_multiplier=0.0,
    )
    assert strat.required_profit_pct(p2, 1.0, 0.0) == 0.003
