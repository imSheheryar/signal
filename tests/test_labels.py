import numpy as np
import pandas as pd
import pytest

from signal_earnings.labels import event_windows, realised_volatility, top_tercile_threshold


@pytest.fixture
def business_days():
    return pd.bdate_range("2020-01-01", "2020-12-31")


def test_event_windows_are_strictly_around_the_call(business_days):
    call = pd.Timestamp("2020-06-01")
    pre, post = event_windows(business_days, call)

    assert len(business_days[pre]) == 60
    assert len(business_days[post]) == 5
    assert business_days[pre][-1] < call
    assert business_days[post][0] > call
    assert business_days[pre][-1] < business_days[post][0]


def test_event_windows_refuse_incomplete_history(business_days):
    with pytest.raises(ValueError, match="pre-call"):
        event_windows(business_days, business_days[10])


def test_event_windows_refuse_incomplete_forward_data(business_days):
    with pytest.raises(ValueError, match="no trading day after|post-call"):
        event_windows(business_days, business_days[-1])


def test_constant_returns_have_zero_volatility():
    prices = pd.Series(100 * 1.01 ** np.arange(30), index=pd.bdate_range("2020-01-01", periods=30))
    assert realised_volatility(prices) == pytest.approx(0.0, abs=1e-12)


def test_market_adjustment_removes_common_movement():
    prices = pd.Series(100 * 1.01 ** np.arange(30), index=pd.bdate_range("2020-01-01", periods=30))
    # Passing the same series as its own benchmark leaves zero excess return.
    assert realised_volatility(prices, market=prices) == pytest.approx(0.0, abs=1e-12)


def test_volatility_is_annualised_and_positive():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 252)
    prices = pd.Series(100 * np.cumprod(1 + returns), index=pd.bdate_range("2020-01-01", periods=252))
    vol = realised_volatility(prices)
    assert 0.10 < vol < 0.25


def test_top_tercile_threshold_uses_only_supplied_values():
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert top_tercile_threshold(values) == pytest.approx(6.3333, abs=1e-3)


def test_top_tercile_threshold_rejects_empty_input():
    with pytest.raises(ValueError):
        top_tercile_threshold(pd.Series([np.nan, np.nan]))
