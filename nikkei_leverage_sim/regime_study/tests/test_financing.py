"""Time-varying JPY financing drag behaves as documented."""
from regime_study import financing


def test_call_rate_known_years():
    # Early-1990s peak vs ZIRP trough vs NIRP.
    assert financing.call_rate("1990-06-30") > 0.06          # ~7.2%
    assert financing.call_rate("1991-01-01") > 0.07
    assert abs(financing.call_rate("2004-06-30")) < 0.001    # ~0%
    assert financing.call_rate("2018-06-30") < 0.0           # negative (NIRP)


def test_call_rate_clamps_out_of_range():
    # Years before/after the table fall back to the nearest endpoint.
    assert financing.call_rate("1980-01-01") == financing.CALL_RATE_ANNUAL[min(financing.CALL_RATE_ANNUAL)]
    assert financing.call_rate("2100-01-01") == financing.CALL_RATE_ANNUAL[max(financing.CALL_RATE_ANNUAL)]


def test_high_rate_year_costs_more_than_zirp():
    kw = dict(leverage=2.0, base_drag=0.008, calib_rate=0.0)
    fee_1990 = financing.annual_fee("1990-06-30", **kw)
    fee_2004 = financing.annual_fee("2004-06-30", **kw)
    # 1990's ~7% call rate adds ~7% financing on the (leverage-1) borrowed leg.
    assert fee_1990 - fee_2004 > 0.05


def test_leverage_scales_financing():
    kw = dict(base_drag=0.0, calib_rate=0.0)
    fee_2x = financing.annual_fee("1990-06-30", leverage=2.0, **kw)
    fee_3x = financing.annual_fee("1990-06-30", leverage=3.0, **kw)
    # 3x borrows twice as much as 2x, so financing roughly doubles.
    assert fee_3x > 1.9 * fee_2x


def test_account_margin_rate_matches_core_default_in_bull_era():
    # Spread is calibrated so the ZIRP bull window reproduces the core 0.028.
    bull = financing.account_margin_rate("2014-01-06", "2026-06-05")
    assert abs(bull - financing.CORE_DEFAULT_MARGIN_RATE) < 5e-4


def test_account_margin_rate_higher_in_high_rate_regime():
    bull = financing.account_margin_rate("2014-01-06", "2026-06-05")
    bubble = financing.account_margin_rate("1990-01-01", "1995-12-31")
    assert bubble > bull + 0.03  # early-1990s call rates add several percent


def test_daily_fee_is_annual_over_trading_days():
    a = financing.annual_fee("1995-06-30", leverage=2.0, base_drag=0.01, calib_rate=0.0)
    d = financing.daily_fee("1995-06-30", leverage=2.0, base_drag=0.01, calib_rate=0.0)
    assert abs(d - a / financing.TRADING_DAYS) < 1e-12
