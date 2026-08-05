"""Foundation tabular model path (Phase 1b).

Prefers TabPFN when installed and data fits its size constraints.
Falls back to sklearn HistGradientBoosting so training always succeeds
without GPU/tabpfn. Both paths share conformal calibration + abstention.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from app.ml.calibration import ConformalCalibrator, UncertaintyResult
from app.ml.models.base_model import BaseModel
from app.ml.probability_calibration import ProbabilityCalibrator


def tabpfn_available() -> bool:
    try:
        from tabpfn import TabPFNClassifier  # noqa: F401
        return True
    except Exception:
        return False


class FoundationModel(BaseModel):
    """
    Small/medium tabular foundation path with conformal uncertainty.

    backend:
      - "tabpfn": Prior-data fitted network (when package + size allow)
      - "hist_gradient_boosting": robust sklearn fallback
    """

    def __init__(
        self,
        version: str = "1.0",
        problem_type: str = "binary_classification",
        conformal_alpha: float = 0.1,
        disagreement_threshold: float = 0.25,
        interval_width_threshold: float = 0.4,
        prefer_tabpfn: bool = True,
        tabpfn_max_rows: int = 10_000,
        tabpfn_max_features: int = 500,
        random_state: int = 42,
        probability_calibration: str = "isotonic",
    ):
        super().__init__("foundation", version)
        self.problem_type = problem_type
        self.prefer_tabpfn = prefer_tabpfn
        self.tabpfn_max_rows = tabpfn_max_rows
        self.tabpfn_max_features = tabpfn_max_features
        self.random_state = random_state
        self.backend: str = "hist_gradient_boosting"
        self.backend_detail: str = ""
        self.calibrator = ConformalCalibrator(
            alpha=conformal_alpha,
            problem_type=problem_type,
            disagreement_threshold=disagreement_threshold,
            interval_width_threshold=interval_width_threshold,
        )
        self.prob_calibrator = ProbabilityCalibrator(method=probability_calibration)  # type: ignore[arg-type]
        self.probability_calibration_method = probability_calibration

    def _is_regression(self) -> bool:
        return self.problem_type == "regression"

    def _to_numpy(self, X: pd.DataFrame) -> np.ndarray:
        arr = X.to_numpy(dtype=float, copy=False)
        if np.isnan(arr).any():
            # TabPFN / HGB: fill residual NaNs with 0 after upstream medians
            arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def _try_tabpfn(self, X: np.ndarray, y: np.ndarray) -> bool:
        if not self.prefer_tabpfn:
            return False
        n_rows, n_features = X.shape
        if n_rows > self.tabpfn_max_rows or n_features > self.tabpfn_max_features:
            self.backend_detail = (
                f"TabPFN skipped: size rows={n_rows} feats={n_features} "
                f"over limits ({self.tabpfn_max_rows}, {self.tabpfn_max_features})"
            )
            return False
        if not tabpfn_available():
            self.backend_detail = "TabPFN package not installed; using HistGradientBoosting"
            return False
        try:
            if self._is_regression():
                from tabpfn import TabPFNRegressor

                model = TabPFNRegressor(device="cpu", n_estimators=8)
                model.fit(X, y)
            else:
                from tabpfn import TabPFNClassifier

                model = TabPFNClassifier(device="cpu", n_estimators=8)
                model.fit(X, y)
            self.model = model
            self.backend = "tabpfn"
            self.backend_detail = "TabPFN fitted on CPU"
            return True
        except Exception as e:
            self.backend_detail = f"TabPFN fit failed ({e}); using HistGradientBoosting"
            print(f"FoundationModel: {self.backend_detail}")
            return False

    def _fit_hgb(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._is_regression():
            self.model = HistGradientBoostingRegressor(
                max_depth=6,
                max_iter=200,
                learning_rate=0.08,
                random_state=self.random_state,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_depth=6,
                max_iter=200,
                learning_rate=0.08,
                random_state=self.random_state,
            )
        self.model.fit(X, y)
        self.backend = "hist_gradient_boosting"
        if not self.backend_detail:
            self.backend_detail = "HistGradientBoosting foundation fallback"

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
        calibration_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> dict[str, float]:
        self.feature_names = list(X.columns)
        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=float).ravel()

        if not self._try_tabpfn(X_np, y_np):
            self._fit_hgb(X_np, y_np)

        self.is_trained = True

        # Prefer dedicated calib set → isotonic/Platt then conformal
        if calibration_data is not None:
            X_cal, y_cal = calibration_data
            raw = self._raw_predict(X_cal)
            cal_true = np.asarray(y_cal, dtype=float).ravel()
        elif validation_data is not None:
            X_cal, y_cal = validation_data
            raw = self._raw_predict(X_cal)
            cal_true = np.asarray(y_cal, dtype=float).ravel()
        else:
            raw = self._raw_predict(X)
            cal_true = y_np

        if not self._is_regression():
            self.prob_calibrator.fit(cal_true, raw)
            cal_pred = self.prob_calibrator.transform(raw)
        else:
            self.prob_calibrator = ProbabilityCalibrator(method="none")
            cal_pred = raw

        try:
            self.calibrator.fit(cal_true, cal_pred)
        except ValueError:
            self.calibrator.fit(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]) if self._is_regression() else np.array([0.05, 0.95]),
            )

        metrics: dict[str, Any] = {
            "foundation_backend": 1.0 if self.backend == "tabpfn" else 0.0,
            "conformal_quantile": float(self.calibrator.quantile or 0.0),
            "conformal_coverage_target": float(self.calibrator.coverage_level),
            "prob_calibration_fitted": float(self.prob_calibrator.is_fitted),
        }

        # Use held-out test (validation_data) as canonical reported metrics when present
        eval_X, eval_y = (validation_data if validation_data else (X, y))
        eval_y_np = np.asarray(eval_y, dtype=float).ravel()

        if self._is_regression():
            y_pred = self.predict(eval_X if isinstance(eval_X, pd.DataFrame) else X)
            # If eval_X is series-aligned dataframe
            if not isinstance(eval_X, pd.DataFrame):
                y_pred = self.predict(X)
                eval_y_np = y_np
            else:
                y_pred = self.predict(eval_X)
            core = {
                "ensemble_mae": mean_absolute_error(eval_y_np, y_pred),
                "ensemble_mse": mean_squared_error(eval_y_np, y_pred),
                "ensemble_rmse": float(np.sqrt(mean_squared_error(eval_y_np, y_pred))),
                "ensemble_r2_score": r2_score(eval_y_np, y_pred),
                "foundation_mae": mean_absolute_error(eval_y_np, y_pred),
                "foundation_mse": mean_squared_error(eval_y_np, y_pred),
                "foundation_rmse": float(np.sqrt(mean_squared_error(eval_y_np, y_pred))),
                "foundation_r2_score": r2_score(eval_y_np, y_pred),
            }
            self.training_metrics = {**core, **metrics}
        else:
            if isinstance(eval_X, pd.DataFrame):
                y_pred = self.predict(eval_X)
                y_proba = self.predict_proba(eval_X)
            else:
                y_pred = self.predict(X)
                y_proba = self.predict_proba(X)
                eval_y_np = y_np
            from sklearn.metrics import brier_score_loss

            core = {
                "ensemble_accuracy": accuracy_score(eval_y_np, y_pred),
                "ensemble_precision": precision_score(eval_y_np, y_pred, zero_division=0),
                "ensemble_recall": recall_score(eval_y_np, y_pred, zero_division=0),
                "ensemble_f1_score": f1_score(eval_y_np, y_pred, zero_division=0),
                "ensemble_auc_roc": roc_auc_score(eval_y_np, y_proba) if len(np.unique(eval_y_np)) > 1 else 0.5,
                "ensemble_brier": float(brier_score_loss(eval_y_np, y_proba)),
                "foundation_accuracy": accuracy_score(eval_y_np, y_pred),
                "foundation_precision": precision_score(eval_y_np, y_pred, zero_division=0),
                "foundation_recall": recall_score(eval_y_np, y_pred, zero_division=0),
                "foundation_f1_score": f1_score(eval_y_np, y_pred, zero_division=0),
                "foundation_auc_roc": roc_auc_score(eval_y_np, y_proba) if len(np.unique(eval_y_np)) > 1 else 0.5,
            }
            if validation_data:
                core["ensemble_test_auc_roc"] = core["ensemble_auc_roc"]
                core["ensemble_test_accuracy"] = core["ensemble_accuracy"]
            self.training_metrics = {**core, **metrics}

        return self.training_metrics

    def _raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated model output."""
        X_np = self._to_numpy(X[self.feature_names] if self.feature_names else X)
        if self._is_regression():
            return np.asarray(self.model.predict(X_np), dtype=float).ravel()
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X_np), dtype=float)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.ravel()
        return np.clip(np.asarray(self.model.predict(X_np), dtype=float).ravel(), 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if self._is_regression():
            return self.predict_proba(X)
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        raw = self._raw_predict(X)
        if self._is_regression():
            return raw
        if getattr(self, "prob_calibrator", None) is not None and self.prob_calibrator.is_fitted:
            return self.prob_calibrator.transform(raw)
        return np.clip(raw, 0.0, 1.0)

    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if self._is_regression():
            # Invert normalized interval width proxy when calibrator fitted
            point = self.predict_proba(X)
            results = self.predict_with_uncertainty(X)
            conf = []
            for r in results:
                w = r.interval_width
                conf.append(float(1.0 / (1.0 + w)))
            return np.asarray(conf, dtype=float)
        proba = self.predict_proba(X)
        return 2.0 * np.abs(proba - 0.5)

    def _point_disagreement(self, X: pd.DataFrame) -> np.ndarray:
        """
        Single-model disagreement proxy for foundation path.

        Uses proximity to decision boundary (classification) or relative
        conformal half-width scale (regression) so abstention still fires.
        """
        point = self.predict_proba(X)
        if self._is_regression():
            q = float(self.calibrator.quantile or 0.1) if self.calibrator.is_fitted else 0.1
            scale = np.maximum(np.abs(point), 1e-6)
            # Relative residual scale; kept mild so conformal width is primary signal
            return np.clip(q / scale, 0.0, 1.0) * 0.2
        # classification: soft boundary ambiguity (primary abstention via CI width)
        ambiguity = np.clip(1.0 - 2.0 * np.abs(point - 0.5), 0.0, 1.0)
        return ambiguity * 0.2

    def predict_with_uncertainty(self, X: pd.DataFrame) -> list[UncertaintyResult]:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        point = self.predict_proba(X)
        disagreement = self._point_disagreement(X)
        results: list[UncertaintyResult] = []
        for i in range(len(X)):
            results.append(
                self.calibrator.evaluate_uncertainty(
                    y_pred=float(point[i]),
                    disagreement=float(disagreement[i]),
                )
            )
        return results

    def get_primary_model(self) -> "FoundationModel":
        """Self — KernelSHAP uses predict_proba for non-tree names."""
        return self

    def get_feature_importance(self) -> dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        if hasattr(self.model, "feature_importances_"):
            imp = np.asarray(self.model.feature_importances_, dtype=float)
            return dict(zip(self.feature_names, imp.tolist()))
        # TabPFN / no native importance — equal priors; SHAP fills at explain time
        n = max(len(self.feature_names), 1)
        return {f: 1.0 / n for f in self.feature_names}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, "foundation_estimator.joblib"))
        self.calibrator.save(os.path.join(path, "calibrator.joblib"))
        if self.prob_calibrator is not None:
            self.prob_calibrator.save(os.path.join(path, "prob_calibrator.joblib"))
        metadata = {
            "version": self.version,
            "problem_type": self.problem_type,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "backend": self.backend,
            "backend_detail": self.backend_detail,
            "model_name": self.model_name,
            "probability_calibration_method": getattr(self, "probability_calibration_method", "isotonic"),
        }
        joblib.dump(metadata, os.path.join(path, "foundation_meta.joblib"))
        # Human-readable + loader hint
        with open(os.path.join(path, "route_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "strategy": "foundation_model",
                    "backend": self.backend,
                    "backend_detail": self.backend_detail,
                    "problem_type": self.problem_type,
                    "version": self.version,
                },
                f,
                indent=2,
            )

    def load(self, path: str) -> None:
        meta_path = os.path.join(path, "foundation_meta.joblib")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"foundation_meta.joblib not found under {path}")
        metadata = joblib.load(meta_path)
        self.version = metadata.get("version", self.version)
        self.problem_type = metadata.get("problem_type", self.problem_type)
        self.feature_names = metadata.get("feature_names", [])
        self.training_metrics = metadata.get("training_metrics", {})
        self.backend = metadata.get("backend", "hist_gradient_boosting")
        self.backend_detail = metadata.get("backend_detail", "")
        self.model = joblib.load(os.path.join(path, "foundation_estimator.joblib"))
        cal_path = os.path.join(path, "calibrator.joblib")
        self.calibrator = ConformalCalibrator(problem_type=self.problem_type)
        if os.path.exists(cal_path):
            self.calibrator.load(cal_path)
        method = metadata.get("probability_calibration_method", "isotonic")
        self.probability_calibration_method = method
        self.prob_calibrator = ProbabilityCalibrator(method=method)  # type: ignore[arg-type]
        pc = os.path.join(path, "prob_calibrator.joblib")
        if os.path.exists(pc):
            self.prob_calibrator.load(pc)
        self.is_trained = True
