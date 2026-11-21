"""Lexical features for earnings-call transcripts.

The features are deliberately simple and interpretable. They are not meant to
compete with a transformer; they exist so the text ablation has a transparent
baseline that can be explained to a non-specialist.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z']+")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")

HEDGES = {
    "approximately", "around", "believe", "could", "estimate", "expect",
    "likely", "may", "might", "possibly", "potential", "probably", "roughly",
    "should", "uncertain", "unclear", "would",
}
NEGATIONS = {
    "decline", "declined", "decrease", "decreased", "difficult", "down",
    "headwind", "loss", "losses", "lower", "negative", "risk", "risks",
    "soft", "softness", "weak", "weaker", "weakness", "worse",
}
FORWARD = {
    "ahead", "coming", "expect", "forecast", "future", "guidance", "next",
    "outlook", "plan", "project", "remainder", "upcoming", "will", "year",
}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def lexical_features(text: str) -> dict[str, float]:
    """Return count, rate and diversity features for one transcript.

    Rates are per 100 words so calls of different lengths are comparable.
    Empty or whitespace-only text returns zeros rather than raising.
    """
    tokens = _tokens(text or "")
    n = len(tokens)
    if n == 0:
        return {
            "n_words": 0,
            "n_unique": 0,
            "ttr": 0.0,
            "n_numbers": 0,
            "hedge_rate": 0.0,
            "neg_rate": 0.0,
            "fwd_rate": 0.0,
            "modal_rate": 0.0,
        }

    counts = {
        "hedge": sum(t in HEDGES for t in tokens),
        "neg": sum(t in NEGATIONS for t in tokens),
        "fwd": sum(t in FORWARD for t in tokens),
        "modal": sum(t in MODALS for t in tokens),
    }
    return {
        "n_words": n,
        "n_unique": len(set(tokens)),
        "ttr": len(set(tokens)) / n,
        "n_numbers": len(_NUMBER.findall(text)),
        **{f"{k}_rate": 100.0 * v / n for k, v in counts.items()},
    }
