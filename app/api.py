"""Read-only API over the held-out Signal artifacts.

The service deliberately exposes the evaluated test set rather than pretending
to score arbitrary new calls: a real deployment would need the full acoustic
and market feature pipeline, which is not available from a text string alone.
That constraint is stated in the OpenAPI description and in the README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

app = FastAPI(
    title="Signal API",
    version="0.1.0",
    description=(
        "Held-out earnings-call risk scores from the Signal project. "
        "Read-only: the API serves the evaluated 2017 H2–2018 H1 test set, "
        "not live inference on arbitrary transcripts."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


class CallSummary(BaseModel):
    call_id: str
    ticker: str
    date: str
    risk: float = Field(ge=0, le=1, description="Calibrated probability of a high-volatility call")
    risk_percentile: float = Field(ge=0, le=1)
    outcome: Literal["High volatility", "Normal"]
    pre_vol: float
    post_vol: float


class Driver(BaseModel):
    feature: str
    value: float | None
    contribution: float


class CallDetail(CallSummary):
    actual_high_vol: bool
    drivers: list[Driver]
    transcript_excerpt: str


class MetricsRow(BaseModel):
    model: str
    val_auc: float
    auc: float
    auc_lo: float
    auc_hi: float
    ap: float
    brier: float
    logloss: float


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calls = pd.read_parquet(ARTIFACTS / "demo_calls.parquet")
    drivers = pd.read_parquet(ARTIFACTS / "demo_drivers.parquet")
    metrics = pd.read_csv(ARTIFACTS / "metrics.csv")
    calls["date"] = pd.to_datetime(calls["date"]).dt.strftime("%Y-%m-%d")
    return calls, drivers, metrics


CALLS, DRIVERS, METRICS = _load()


def _summary(row: pd.Series) -> CallSummary:
    return CallSummary(
        call_id=row.call_id,
        ticker=row.ticker,
        date=row.date,
        risk=round(float(row.risk), 4),
        risk_percentile=round(float(row.risk_pct), 4),
        outcome=row.outcome,
        pre_vol=round(float(row.pre_vol), 4),
        post_vol=round(float(row.post_vol), 4),
    )


@app.get("/health", tags=["system"])
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "calls": len(CALLS),
        "tickers": int(CALLS.ticker.nunique()),
        "model": "late_pre_text_ac",
    }


@app.get("/calls", response_model=list[CallSummary], tags=["calls"])
def list_calls(
    ticker: str | None = Query(default=None, description="Exact ticker, e.g. PEP"),
    min_risk: float = Query(default=0.0, ge=0, le=1),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[CallSummary]:
    frame = CALLS
    if ticker:
        frame = frame[frame.ticker.str.upper() == ticker.upper()]
    frame = frame[frame.risk >= min_risk].sort_values("risk", ascending=False)
    return [_summary(row) for _, row in frame.iloc[offset : offset + limit].iterrows()]


@app.get("/calls/{call_id}", response_model=CallDetail, tags=["calls"])
def get_call(call_id: str) -> CallDetail:
    match = CALLS[CALLS.call_id == call_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"unknown call_id: {call_id}")
    row = match.iloc[0]
    drivers = DRIVERS[DRIVERS.call_id == call_id].copy()
    drivers["abs"] = drivers.contribution.abs()
    drivers = drivers.sort_values("abs", ascending=False).head(6)
    return CallDetail(
        **_summary(row).model_dump(),
        actual_high_vol=bool(row.y),
        drivers=[
            Driver(
                feature=d.feature,
                value=round(float(d.value), 4) if pd.notna(d.value) else None,
                contribution=round(float(d.contribution), 4),
            )
            for d in drivers.itertuples()
        ],
        transcript_excerpt=str(row.snippet)[:1200],
    )


@app.get("/metrics", response_model=list[MetricsRow], tags=["model"])
def metrics() -> list[MetricsRow]:
    ordered = METRICS.sort_values("val_auc", ascending=False)
    return [
        MetricsRow(
            model=row.model,
            val_auc=round(float(row.val_auc), 4),
            auc=round(float(row.auc), 4),
            auc_lo=round(float(row.auc_lo), 4),
            auc_hi=round(float(row.auc_hi), 4),
            ap=round(float(row.ap), 4),
            brier=round(float(row.brier), 4),
            logloss=round(float(row.logloss), 4),
        )
        for row in ordered.itertuples()
    ]
