"""Build the JSON payload for the static GitHub Pages dashboard.

The Streamlit app is the full interactive version. This static mirror embeds
the same held-out artifacts so recruiters can explore it without a server.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
A = ROOT / "artifacts"
SITE = ROOT / "site"

calls = pd.read_parquet(A / "demo_calls.parquet")
drivers = pd.read_parquet(A / "demo_drivers.parquet")
prosody = pd.read_parquet(A / "demo_prosody.parquet")
metrics = pd.read_csv(A / "metrics.csv")

calls["date"] = pd.to_datetime(calls["date"]).dt.strftime("%Y-%m-%d")
drivers_by_call = {
    call_id: [
        {
            "feature": row.feature,
            "value": None if pd.isna(row.value) else round(float(row.value), 4),
            "contribution": round(float(row.contribution), 4),
        }
        for row in group.sort_values("contribution", key=lambda s: s.abs(), ascending=False).head(6).itertuples()
    ]
    for call_id, group in drivers.groupby("call_id")
}

prosody_by_call: dict[str, dict[str, list[float]]] = {}
for call_id, group in prosody.groupby("call_id"):
    series = {}
    for metric in ("Mean pitch", "Mean intensity"):
        values = group[group.metric == metric].sort_values("segment").value.dropna().tolist()
        if len(values) > 140:
            step = math.ceil(len(values) / 140)
            values = values[::step]
        series[metric] = [round(float(v), 1) for v in values]
    prosody_by_call[call_id] = series

payload = {
    "metrics": [
        {
            "model": row.model,
            "val_auc": round(float(row.val_auc), 4),
            "auc": round(float(row.auc), 4),
            "auc_lo": round(float(row.auc_lo), 4),
            "auc_hi": round(float(row.auc_hi), 4),
            "ap": round(float(row.ap), 4),
            "brier": round(float(row.brier), 4),
            "logloss": round(float(row.logloss), 4),
        }
        for row in metrics.sort_values("val_auc", ascending=False).itertuples()
    ],
    "calls": [
        {
            "call_id": row.call_id,
            "ticker": row.ticker,
            "date": row.date,
            "risk": round(float(row.risk), 4),
            "risk_pct": round(float(row.risk_pct), 4),
            "outcome": row.outcome,
            "high_vol": bool(row.y),
            "pre_vol": round(float(row.pre_vol), 4),
            "post_vol": round(float(row.post_vol), 4),
            "pred_pre_vol": round(float(row.pred_pre_vol), 4),
            "pred_text": round(float(row.pred_text_tfidf), 4),
            "pred_voice": round(float(row.pred_acoustic), 4),
            "pred_late_pre_text": round(float(row.pred_late_pre_text), 4),
            "pred_late_pre_text_ac": round(float(row.pred_late_pre_text_ac), 4),
            "pred_late_all": round(float(row.pred_late_all), 4),
            "pred_fusion": round(float(row.pred_fusion), 4),
            "snippet": str(row.snippet)[:1600],
            "drivers": drivers_by_call.get(row.call_id, []),
            "prosody": prosody_by_call.get(row.call_id, {"Mean pitch": [], "Mean intensity": []}),
        }
        for row in calls.sort_values(["ticker", "date"]).itertuples()
    ],
}

SITE.mkdir(exist_ok=True)
(SITE / "data.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"calls={len(payload['calls'])} metrics={len(payload['metrics'])}")
print(f"data.json={(SITE / 'data.json').stat().st_size / 1e6:.2f} MB")
