"""Build market-adjusted volatility labels for every call with usable prices.

Windows are deliberately explicit: the post-call window starts on the first
trading day strictly after the call, and the pre-call window ends on the last
trading day before it. Both boundaries are covered by tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from signal_earnings.labels import (
    event_windows,
    realised_volatility,
    top_tercile_threshold,
)

OUT = Path("artifacts")
features = pd.read_parquet(OUT / "calls_features.parquet")
prices = pd.read_parquet(OUT / "prices_close.parquet")
prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
prices = prices.sort_index()
spy = prices["SPY"].dropna()

rows = []
for row in features.itertuples():
    ticker = row.ticker
    if ticker not in prices.columns:
        continue
    series = prices[ticker].dropna()
    if len(series) < 100:
        continue
    try:
        pre_slice, post_slice = event_windows(series.index, row.date)
    except ValueError:
        continue

    pre_prices = series.iloc[pre_slice]
    post_prices = series.iloc[post_slice]
    if pre_prices.isna().any() or post_prices.isna().any():
        continue

    pre_market = spy.reindex(pre_prices.index).ffill()
    post_market = spy.reindex(post_prices.index).ffill()
    if pre_market.isna().any() or post_market.isna().any():
        continue

    pre_vol = realised_volatility(pre_prices, pre_market)
    post_vol = realised_volatility(post_prices, post_market)
    if not np.isfinite(pre_vol) or not np.isfinite(post_vol) or pre_vol <= 1e-8:
        continue

    pre_returns = pre_prices.pct_change().dropna()
    post_returns = post_prices.pct_change().dropna()
    pre_excess = pre_returns - pre_market.pct_change().reindex(pre_returns.index)
    post_excess = post_returns - post_market.pct_change().reindex(post_returns.index)
    if len(pre_excess.dropna()) < 40 or len(post_excess.dropna()) < 4:
        continue

    rows.append(
        {
            "call_id": row.call_id,
            "ticker": ticker,
            "date": row.date,
            "pre_vol": pre_vol,
            "post_vol": post_vol,
            "vol_ratio": post_vol / pre_vol,
            "post_excess_ret": float(post_excess.sum()),
            "n_pre": int(pre_excess.notna().sum()),
            "n_post": int(post_excess.notna().sum()),
        }
    )

labels = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
train = labels.date < "2017-01-01"

# The operational label is the post-call volatility level. The ratio label is
# kept for the ablation that showed it was harder to predict.
post_threshold = top_tercile_threshold(labels.loc[train, "post_vol"])
ratio_threshold = top_tercile_threshold(labels.loc[train, "vol_ratio"])
labels["label"] = (labels.post_vol >= post_threshold).astype(int)
labels["ratio_label"] = (labels.vol_ratio >= ratio_threshold).astype(int)
labels["target_ratio"] = labels.vol_ratio
labels.to_parquet(OUT / "calls_labels.parquet", index=False)

print("calls with labels", len(labels), "of", len(features))
print("tickers", labels.ticker.nunique())
print(
    "post-vol threshold (train)",
    round(post_threshold, 4),
    "rate",
    round(labels.label.mean(), 3),
)
print("ratio threshold (train)", round(ratio_threshold, 3), "rate", round(labels.ratio_label.mean(), 3))
print(labels[["pre_vol", "post_vol", "vol_ratio", "post_excess_ret"]].describe().round(4).to_string())
print("by year", labels.groupby(labels.date.dt.year).label.agg(["count", "mean"]).round(3).to_string())
