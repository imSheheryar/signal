"""Evaluation helpers used by the modelling and API layers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_ci(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a binary ranking metric.

    Resamples that contain only one class are skipped because AUC and average
    precision are undefined there. Returns (nan, nan) if no valid resample.
    """
    y = np.asarray(y_true)
    p = np.asarray(scores, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and scores must have the same shape")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(metric(y[idx], p[idx]))
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def top_k_lift(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    k: float = 0.2,
) -> dict[str, float]:
    """Precision, recall and lift when reviewing the top ``k`` fraction.

    ``lift`` is precision divided by the base rate. This is the operational
    question an analyst actually asks: if I can only read 20% of the calls,
    how many of the volatile ones do I catch?
    """
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1]")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and scores must have the same shape")
    n = len(y)
    n_review = max(1, round(k * n))
    order = np.argsort(-p)[:n_review]
    hits = int(y[order].sum())
    positives = int(y.sum())
    base_rate = positives / n
    precision = hits / n_review
    recall = hits / positives if positives else float("nan")
    return {
        "n_reviewed": n_review,
        "hits": hits,
        "precision": precision,
        "recall": recall,
        "base_rate": base_rate,
        "lift": precision / base_rate if base_rate else float("nan"),
        "threshold": float(np.min(p[order])),
    }
