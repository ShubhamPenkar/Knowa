"""Conformal prediction and abstention utilities (Phase 1a).

Implements inductive (split) conformal intervals around point predictions.
Works for binary classification probabilities and continuous regression targets.

Abstention (`low_confidence`) uses base-model disagreement and interval width,
with classification thresholds chosen so residual conformal bands (which are
often ≈0.5–0.8 wide on binary labels) do not fire on every row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import joblib
import numpy as np


@dataclass
class UncertaintyResult:
    """Point estimate plus calibrated uncertainty and abstention flag."""

    prediction: float
    lower: float
    upper: float
    interval_width: float
    coverage_level: float
    disagreement: float
    low_confidence: bool
    abstention_reason: Optional[str] = None

    def as_interval_dict(self) -> dict[str, float]:
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "level": float(self.coverage_level),
            "width": float(self.interval_width),
        }


class ConformalCalibrator:
    """
    Split-conformal residual calibrator.

    For classification, predictions are probabilities in [0, 1].
    For regression, predictions are continuous values.

    Fitted on a held-out calibration set via absolute residual scores.
    """

    # Residual |y−p| on {0,1} labels routinely yields quantile ≈ 0.3–0.6;
    # flag only when the band is nearly uninformative on [0, 1].
    DEFAULT_CLASSIFICATION_WIDTH = 0.85
    DEFAULT_REGRESSION_WIDTH = 0.4
    DEFAULT_DISAGREEMENT = 0.25

    def __init__(
        self,
        alpha: float = 0.1,
        problem_type: str = "binary_classification",
        disagreement_threshold: float = DEFAULT_DISAGREEMENT,
        interval_width_threshold: Optional[float] = None,
    ):
        self.alpha = alpha
        self.problem_type = problem_type
        self.disagreement_threshold = disagreement_threshold
        if interval_width_threshold is None:
            interval_width_threshold = (
                self.DEFAULT_REGRESSION_WIDTH
                if problem_type == "regression"
                else self.DEFAULT_CLASSIFICATION_WIDTH
            )
        self.interval_width_threshold = interval_width_threshold
        self.quantile: Optional[float] = None
        self.is_fitted = False
        self.n_calibration: int = 0

    @property
    def coverage_level(self) -> float:
        return 1.0 - self.alpha

    def apply_policy(
        self,
        *,
        disagreement_threshold: Optional[float] = None,
        interval_width_threshold: Optional[float] = None,
    ) -> "ConformalCalibrator":
        """Update inference-time abstention policy without refitting quantiles."""
        if disagreement_threshold is not None:
            self.disagreement_threshold = float(disagreement_threshold)
        if interval_width_threshold is not None:
            self.interval_width_threshold = float(interval_width_threshold)
        return self

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalCalibrator":
        """Fit residual quantile on calibration predictions."""
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if len(y_true) == 0:
            raise ValueError("Calibration set is empty")

        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        # Finite-sample conformal quantile level
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.quantile = float(np.quantile(residuals, q_level))
        # Guard against zero-width from perfect fit / tiny sets
        if self.quantile <= 0:
            self.quantile = float(np.std(residuals) + 1e-6)
        self.n_calibration = n
        self.is_fitted = True
        return self

    def predict_interval(
        self,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (lower, upper, width) arrays for point predictions.

        If not fitted, falls back to a conservative fixed half-width.
        """
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if self.is_fitted and self.quantile is not None:
            q = self.quantile
        else:
            # Unfitted: half-width that will not auto-abstain under default policy
            q = 0.35 if self.problem_type != "regression" else 0.5 * self.interval_width_threshold

        lower = y_pred - q
        upper = y_pred + q
        if self.problem_type != "regression":
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)
        width = upper - lower
        return lower, upper, width

    def evaluate_uncertainty(
        self,
        y_pred: float,
        disagreement: float,
    ) -> UncertaintyResult:
        """Build uncertainty result for a single prediction."""
        lower, upper, width = self.predict_interval(np.array([y_pred]))
        lower_f = float(lower[0])
        upper_f = float(upper[0])
        width_f = float(width[0])

        reasons: list[str] = []
        if disagreement > self.disagreement_threshold:
            reasons.append(
                f"base model disagreement ({disagreement:.3f}) exceeds "
                f"threshold ({self.disagreement_threshold})"
            )

        if width_f > self.interval_width_threshold:
            # Classification residual bands are often 0.4–0.8 wide by design;
            # threshold defaults to ~0.85 so only near-uninformative bands abstain.
            reasons.append(
                f"conformal interval width ({width_f:.3f}) exceeds "
                f"threshold ({self.interval_width_threshold})"
            )

        low_confidence = len(reasons) > 0
        return UncertaintyResult(
            prediction=float(y_pred),
            lower=lower_f,
            upper=upper_f,
            interval_width=width_f,
            coverage_level=self.coverage_level,
            disagreement=float(disagreement),
            low_confidence=low_confidence,
            abstention_reason="; ".join(reasons) if reasons else None,
        )

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "alpha": self.alpha,
                "problem_type": self.problem_type,
                "disagreement_threshold": self.disagreement_threshold,
                "interval_width_threshold": self.interval_width_threshold,
                "quantile": self.quantile,
                "is_fitted": self.is_fitted,
                "n_calibration": self.n_calibration,
            },
            path,
        )

    def load(self, path: str) -> "ConformalCalibrator":
        data = joblib.load(path)
        self.alpha = data["alpha"]
        self.problem_type = data.get("problem_type", "binary_classification")
        self.disagreement_threshold = data.get(
            "disagreement_threshold", self.DEFAULT_DISAGREEMENT
        )
        # Prefer saved threshold, but bump legacy 0.4 classification defaults
        # that flagged nearly every residual-conformal band.
        saved_w = data.get("interval_width_threshold")
        if saved_w is None:
            self.interval_width_threshold = (
                self.DEFAULT_REGRESSION_WIDTH
                if self.problem_type == "regression"
                else self.DEFAULT_CLASSIFICATION_WIDTH
            )
        elif self.problem_type != "regression" and float(saved_w) <= 0.45:
            self.interval_width_threshold = self.DEFAULT_CLASSIFICATION_WIDTH
        else:
            self.interval_width_threshold = float(saved_w)
        self.quantile = data.get("quantile")
        self.is_fitted = data.get("is_fitted", False)
        self.n_calibration = data.get("n_calibration", 0)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "coverage_level": self.coverage_level,
            "quantile": self.quantile,
            "is_fitted": self.is_fitted,
            "n_calibration": self.n_calibration,
            "disagreement_threshold": self.disagreement_threshold,
            "interval_width_threshold": self.interval_width_threshold,
        }
