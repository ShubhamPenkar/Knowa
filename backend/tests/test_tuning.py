"""Focused tests for model-scoring and hyperparameter tuning in app.ml.tuning."""

import numpy as np
import pandas as pd
import pytest

from app.ml.tuning import (
    _binary_auc_scorer,
    _cv_scorer,
    _default_params,
    tune_tree_hyperparameters,
)


class DummyProbaEstimator:
    """Mock estimator returning configured predict_proba."""

    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def predict_proba(self, X):
        return self._proba


class DummyDecisionEstimator:
    """Mock estimator returning configured decision_function."""

    def __init__(self, scores):
        self._scores = np.asarray(scores)

    def decision_function(self, X):
        return self._scores


class DummyPredictEstimator:
    """Mock estimator returning configured discrete predictions."""

    def __init__(self, preds):
        self._preds = np.asarray(preds)

    def predict(self, X):
        return self._preds


# =============================================================================
# 1. Binary ROC-AUC Scorer Unit Tests
# =============================================================================


def test_binary_auc_scorer_valid_2d_proba():
    """Valid binary prediction with 2D predict_proba returns exact ROC-AUC."""
    # True labels: [0, 0, 1, 1]
    # Pos-class probas: [0.1, 0.4, 0.6, 0.9] -> perfect ranking
    proba = np.array([
        [0.9, 0.1],
        [0.6, 0.4],
        [0.4, 0.6],
        [0.1, 0.9],
    ])
    estimator = DummyProbaEstimator(proba)
    score = _binary_auc_scorer(estimator, None, [0, 0, 1, 1])
    assert score == 1.0


def test_binary_auc_scorer_valid_1d_proba():
    """Valid binary prediction with 1D predict_proba array."""
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    estimator = DummyProbaEstimator(proba)
    score = _binary_auc_scorer(estimator, None, [0, 0, 1, 1])
    assert score == 1.0


def test_binary_auc_scorer_valid_decision_function():
    """Estimator with decision_function is scored correctly."""
    estimator = DummyDecisionEstimator([-1.5, -0.2, 0.3, 2.1])
    score = _binary_auc_scorer(estimator, None, [0, 0, 1, 1])
    assert score == 1.0


def test_binary_auc_scorer_valid_predict_fallback():
    """Estimator with only predict fallback is scored."""
    estimator = DummyPredictEstimator([0, 0, 1, 1])
    score = _binary_auc_scorer(estimator, None, [0, 0, 1, 1])
    assert score == 1.0


def test_binary_auc_scorer_single_class_raises_not_silent():
    """Mathematically undefined ROC-AUC on single-class fold raises ValueError.

    Must NEVER silently substitute 0.0 or 0.5.
    """
    proba = np.array([[0.2, 0.8], [0.4, 0.6]])
    estimator = DummyProbaEstimator(proba)

    # All-positive fold
    with pytest.raises(ValueError, match="Only one class present in y_true"):
        _binary_auc_scorer(estimator, None, [1, 1])

    # All-negative fold
    with pytest.raises(ValueError, match="Only one class present in y_true"):
        _binary_auc_scorer(estimator, None, [0, 0])


def test_binary_auc_scorer_multiclass_y_raises():
    """Multiclass target in binary scorer raises ValueError."""
    proba = np.array([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
    estimator = DummyProbaEstimator(proba)

    with pytest.raises(ValueError, match="Binary ROC-AUC scoring requires exactly 2 classes"):
        _binary_auc_scorer(estimator, None, [0, 1, 2])


def test_binary_auc_scorer_multiclass_proba_shape_raises():
    """3-column predict_proba for binary scoring raises ValueError."""
    proba = np.array([
        [0.2, 0.5, 0.3],
        [0.1, 0.2, 0.7],
    ])
    estimator = DummyProbaEstimator(proba)

    with pytest.raises(ValueError, match="Expected binary classification probabilities with 2 columns"):
        _binary_auc_scorer(estimator, None, [0, 1])


# =============================================================================
# 2. Scorer Strategy Selection Tests
# =============================================================================


def test_cv_scorer_selection():
    """Prefers built-in scorers where supported; uses custom for xgboost."""
    # Regression -> built-in
    assert _cv_scorer("regression") == "neg_root_mean_squared_error"

    # RandomForest / LightGBM -> built-in roc_auc
    assert _cv_scorer("binary_classification", family="random_forest") == "roc_auc"
    assert _cv_scorer("binary_classification", family="lightgbm") == "roc_auc"

    # XGBoost -> _binary_auc_scorer (scikit-learn 1.4 MRO fix)
    assert _cv_scorer("binary_classification", family="xgboost") == _binary_auc_scorer

    # Unsupported problem types raise ValueError
    with pytest.raises(ValueError, match="Unsupported problem_type: multiclass"):
        _cv_scorer("multiclass")


# =============================================================================
# 3. Hyperparameter Tuning Integration Tests
# =============================================================================


def test_tune_tree_hyperparameters_unsupported_problem_type():
    """Unsupported problem_type raises ValueError explicitly."""
    X = pd.DataFrame(np.random.randn(20, 2))
    y = pd.Series([0, 1] * 10)

    with pytest.raises(ValueError, match="Unsupported problem_type: multiclass"):
        tune_tree_hyperparameters(X, y, problem_type="multiclass")


def test_tune_tree_hyperparameters_single_class_dataset_returns_defaults():
    """Dataset with only 1 class immediately returns defaults without failed search."""
    X = pd.DataFrame(np.random.randn(20, 2))
    y = pd.Series([1] * 20)

    defaults = _default_params("random_forest", "binary_classification")
    result = tune_tree_hyperparameters(
        X, y, family="random_forest", problem_type="binary_classification", n_trials=5
    )
    assert result == defaults
    assert "_optuna_best_score" not in result


def test_tune_tree_hyperparameters_insufficient_minority_samples_returns_defaults():
    """Dataset with only 1 minority sample cannot form CV folds and returns defaults."""
    X = pd.DataFrame(np.random.randn(20, 2))
    y = pd.Series([0] * 19 + [1])

    defaults = _default_params("random_forest", "binary_classification")
    result = tune_tree_hyperparameters(
        X, y, family="random_forest", problem_type="binary_classification", n_trials=5
    )
    assert result == defaults
    assert "_optuna_best_score" not in result


def test_tune_tree_hyperparameters_binary_classification_all_families():
    """Valid binary classification tuning works across random_forest, lightgbm, and xgboost."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(60, 4), columns=list("abcd"))
    y = pd.Series(rng.binomial(1, 0.5, 60))

    for family in ("random_forest", "lightgbm", "xgboost"):
        result = tune_tree_hyperparameters(
            X,
            y,
            family=family,
            problem_type="binary_classification",
            n_trials=2,
            random_state=42,
        )
        assert "_optuna_best_score" in result
        score = result["_optuna_best_score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert result.get("_optuna_trials", 0) > 0


def test_tune_tree_hyperparameters_regression():
    """Valid regression tuning works and returns negative RMSE score."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(60, 4), columns=list("abcd"))
    y = pd.Series(rng.randn(60))

    result = tune_tree_hyperparameters(
        X,
        y,
        family="random_forest",
        problem_type="regression",
        n_trials=2,
        random_state=42,
    )
    assert "_optuna_best_score" in result
    score = result["_optuna_best_score"]
    assert isinstance(score, float)
    # neg_root_mean_squared_error is negative and > -1e8
    assert -1e8 < score <= 0.0


def test_tune_tree_hyperparameters_failed_scoring_falls_back_to_defaults(monkeypatch):
    """When cross_val_score fails or returns NaN, trials are rejected and defaults returned.

    Never treats 0.0 or 0.5 as valid scores.
    """
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(30, 2), columns=["x1", "x2"])
    y = pd.Series(rng.binomial(1, 0.5, 30))

    # Simulate a scenario where scoring always fails with NaN
    import app.ml.tuning

    def mock_cross_val_score(*args, **kwargs):
        return np.array([np.nan, np.nan, np.nan])

    monkeypatch.setattr(app.ml.tuning, "cross_val_score", mock_cross_val_score)

    defaults = _default_params("random_forest", "binary_classification")
    result = tune_tree_hyperparameters(
        X,
        y,
        family="random_forest",
        problem_type="binary_classification",
        n_trials=3,
        random_state=42,
    )
    # Since all trials failed, Optuna's best_value is <= -1e8, fallback to defaults
    assert result == defaults
    assert "_optuna_best_score" not in result
