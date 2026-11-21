"""Volatility labels and event-window construction.

The central leakage risk in this project is using prices from the call day or
from after the label window. These helpers make the window boundaries explicit
and testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def event_windows(
    index: pd.Index,
    call_date: str | pd.Timestamp,
    pre_days: int = 60,
    post_days: int = 5,
) -> tuple[slice, slice]:
    """Return (pre, post) positional slices around a call date.

    The post window starts on the first trading day *strictly after* the call
    date. Calls frequently happen after the close, so including the call day
    would leak the market's reaction into the "pre" window.

    Raises ValueError when there is not enough history or forward data.
    """
    idx = pd.DatetimeIndex(index).normalize()
    call = pd.Timestamp(call_date).normalize()
    pre_end = int(np.searchsorted(idx, call, side="left"))
    start = int(np.searchsorted(idx, call, side="right"))
    if start >= len(idx):
        raise ValueError(f"no trading day after {call.date()}")
    if pre_end < pre_days:
        raise ValueError(f"only {pre_end} pre-call days for {call.date()}")
    if start + post_days > len(idx):
        raise ValueError(f"only {len(idx) - start} post-call days for {call.date()}")
    return slice(pre_end - pre_days, pre_end), slice(start, start + post_days)


def realised_volatility(
    prices: pd.Series,
    market: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised standard deviation of daily returns.

    When ``market`` is supplied, returns are market-adjusted (excess) returns.
    The two series are aligned on the price index before differencing, so a
    missing benchmark day cannot silently shift the window.
    """
    prices = pd.Series(prices).astype(float).sort_index()
    if len(prices) < 3:
        return float("nan")
    returns = prices.pct_change().dropna()
    if market is not None:
        market = pd.Series(market).astype(float).reindex(prices.index).ffill()
        market_returns = market.pct_change().reindex(returns.index)
        returns = (returns - market_returns).dropna()
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def top_tercile_threshold(values: pd.Series | np.ndarray, quantile: float = 2 / 3) -> float:
    """Learn a binary-label threshold, normally on the training period only."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("no finite values")
    return float(np.quantile(arr, quantile))
