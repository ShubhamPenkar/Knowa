"""Per-project classification threshold tuning for imbalanced targets.

Global default remains 0.5 (Telco and other balanced-ish cases). Each project
can store a tuned threshold in ``feature_config["_project"]["decision_threshold"]``,
fit on the calibration fold at train time.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


DEFAULT_DECISION_THRESHOLD = 0.5


def get_decision_threshold(project: Any) -> float:
    """Return per-project decision threshold; 0.5 when unset."""
    fc = getattr(project, "feature_config", None) or {}
    if not isinstance(fc, dict):
        return DEFAULT_DECISION_THRESHOLD
    meta = fc.get("_project") or {}
    thr = meta.get("decision_threshold")
    if thr is None:
        return DEFAULT_DECISION_THRESHOLD
    try:
        return float(thr)
    except (TypeError, ValueError):
        return DEFAULT_DECISION_THRESHOLD


def set_decision_threshold_meta(
    feature_config: Optional[dict[str, Any]],
    threshold: float,
    *,
    metric: str,
    metric_value: float,
    tuned_on: str = "calibration",
) -> dict[str, Any]:
    """Merge threshold metadata into feature_config (creates dict if needed)."""
    fc = dict(feature_config or {})
    fc["_project"] = {
        **(fc.get("_project") or {}),
        "decision_threshold": round(float(threshold), 6),
        "decision_threshold_metric": metric,
        "decision_threshold_metric_value": round(float(metric_value), 6),
        "decision_threshold_tuned_on": tuned_on,
    }
    return fc


def tune_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    metric: Literal["balanced_accuracy", "f1", "accuracy"] = "balanced_accuracy",
    grid_step: float = 0.01,
) -> tuple[float, dict[str, Any]]:
    """
    Pick threshold maximizing balanced accuracy, F1, or accuracy on a labeled fold.

    For imbalanced targets where the negative class is the majority, accuracy-optimal
    thresholds often sit above 0.5; when several thresholds tie on accuracy, the
    highest threshold is chosen so held-out lift vs majority is less likely to flip.

    Uses a fine grid on [0.01, 0.99]. Returns (best_threshold, metadata).
    """
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_proba, dtype=float).ravel()
    if len(y) == 0 or len(p) == 0:
        return DEFAULT_DECISION_THRESHOLD, {
            "metric": metric,
            "metric_value": 0.0,
            "n_samples": 0,
            "note": "empty_fold",
        }
    if len(np.unique(y)) < 2:
        return DEFAULT_DECISION_THRESHOLD, {
            "metric": metric,
            "metric_value": 0.0,
            "n_samples": len(y),
            "note": "single_class_fold",
        }

    thresholds = np.round(np.arange(grid_step, 1.0 - grid_step / 2, grid_step), 4)
    uniq = np.unique(np.clip(np.round(p, 4), 0.0, 1.0))
    candidates = np.unique(np.concatenate([thresholds, uniq]))
    scored: list[tuple[float, float]] = []
    for thr in candidates:
        pred = (p >= thr).astype(int)
        if metric == "f1":
            score = float(f1_score(y, pred, zero_division=0))
        elif metric == "accuracy":
            score = float(accuracy_score(y, pred))
        else:
            score = float(balanced_accuracy_score(y, pred))
        scored.append((float(thr), score))

    best_score = max(s for _, s in scored)
    tol = 1e-9
    tied = [thr for thr, s in scored if s >= best_score - tol]
    if metric == "accuracy":
        best_thr = max(tied)
    else:
        best_thr = tied[0] if tied else DEFAULT_DECISION_THRESHOLD

    return best_thr, {
        "metric": metric,
        "metric_value": round(best_score, 6),
        "n_samples": int(len(y)),
        "pos_rate": round(float(y.mean()), 6),
    }


def threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Accuracy, lift vs majority, balanced accuracy, F1 at a given threshold."""
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_proba, dtype=float).ravel()
    n = len(y)
    if n == 0:
        return {}
    pred = (p >= threshold).astype(int)
    acc = float((pred == y).mean())
    n_pos = int(y.sum())
    n_neg = n - n_pos
    maj = 1 if n_pos >= n_neg else 0
    maj_acc = max(n_pos, n_neg) / n
    f1 = float(f1_score(y, pred, zero_division=0))
    ba = float(balanced_accuracy_score(y, pred))
    return {
        "threshold": float(threshold),
        "accuracy": round(acc, 4),
        "majority_baseline": round(maj_acc, 4),
        "lift_pp": round((acc - maj_acc) * 100, 2),
        "balanced_accuracy": round(ba, 4),
        "f1": round(f1, 4),
    }
