import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from signal_earnings.evaluation import bootstrap_ci, top_k_lift


def test_top_k_lift_reviews_the_highest_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    result = top_k_lift(y, scores, k=0.4)

    assert result["n_reviewed"] == 2
    assert result["hits"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["base_rate"] == 0.4
    assert result["lift"] == pytest.approx(2.5)


def test_top_k_lift_validates_fraction():
    with pytest.raises(ValueError):
        top_k_lift([0, 1], [0.1, 0.9], k=0)


def test_bootstrap_ci_brackets_perfect_ranking():
    y = np.array([0] * 20 + [1] * 20)
    scores = np.array([0.1] * 20 + [0.9] * 20)
    lo, hi = bootstrap_ci(y, scores, roc_auc_score, n_boot=100, seed=1)

    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)
