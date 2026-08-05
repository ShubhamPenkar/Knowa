"""Probability calibration after scoring (Phase 1.5).

Fits Platt (sigmoid) or isotonic regression on a held-out calibration set
so reported probabilities are better calibrated before conformal wrapping.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CalibrationMethod = Literal["isotonic", "sigmoid", "none"]


class ProbabilityCalibrator:
    """Maps raw scores/probabilities to calibrated probabilities."""

    def __init__(self, method: CalibrationMethod = "isotonic"):
        self.method: CalibrationMethod = method if method in ("isotonic", "sigmoid", "none") else "isotonic"
        self.model = None
        self.is_fitted = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ProbabilityCalibrator":
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if self.method == "none" or len(y_true) < 10 or len(np.unique(y_true)) < 2:
            self.is_fitted = False
            self.model = None
            return self

        if self.method == "isotonic":
            # Requires sorted unique-ish support; isotonic handles duplicates
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(y_pred, y_true)
            self.model = iso
        else:
            # Platt scaling: logistic on 1-d score
            lr = LogisticRegression(C=1e3, solver="lbfgs", max_iter=1000)
            lr.fit(y_pred.reshape(-1, 1), y_true.astype(int))
            self.model = lr

        self.is_fitted = True
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if not self.is_fitted or self.model is None or self.method == "none":
            return np.clip(y_pred, 0.0, 1.0)

        if self.method == "isotonic":
            out = self.model.predict(y_pred)
        else:
            out = self.model.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(out, dtype=float).ravel(), 0.0, 1.0)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "method": self.method,
                "model": self.model,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path: str) -> "ProbabilityCalibrator":
        if not os.path.exists(path):
            self.is_fitted = False
            return self
        data = joblib.load(path)
        self.method = data.get("method", "isotonic")
        self.model = data.get("model")
        self.is_fitted = bool(data.get("is_fitted", False))
        return self
