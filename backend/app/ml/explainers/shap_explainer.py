"""SHAP explainer for tree and non-tree tabular models."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from app.ml.models.base_model import BaseModel

_TREE_MODEL_NAMES = {
    "xgboost",
    "lightgbm",
    "random_forest",
    "randomforest",
    "hist_gradient_boosting",
    "histgradientboosting",
}

_TREE_ESTIMATOR_TYPES = {
    "XGBClassifier",
    "XGBRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
}


class SHAPExplainer:
    """
    Local + global SHAP explanations.

    Prefers TreeExplainer for tree estimators; KernelExplainer for wrappers
    (foundation path, stacking ensemble) via predict_proba.
    """

    def __init__(
        self,
        model: BaseModel,
        background_data: Optional[pd.DataFrame] = None,
        *,
        max_background: int = 80,
    ):
        self.model = model
        self.background_data = background_data
        self.max_background = max_background
        self.explainer = None
        self.mode: str = "unknown"  # tree | kernel
        self.feature_names: list[str] = list(
            getattr(model, "feature_names", None)
            or (list(background_data.columns) if background_data is not None else [])
        )
        self._initialize_explainer()

    def _estimator(self):
        return getattr(self.model, "model", None)

    def _is_tree_candidate(self) -> bool:
        name = str(getattr(self.model, "model_name", "") or "").lower()
        backend = str(getattr(self.model, "backend", "") or "").lower()
        est = self._estimator()
        est_name = type(est).__name__ if est is not None else ""
        if name in _TREE_MODEL_NAMES or backend in _TREE_MODEL_NAMES:
            return est is not None
        if est_name in _TREE_ESTIMATOR_TYPES:
            return True
        if "hist" in backend and "gradient" in backend and est is not None:
            return True
        return False

    def _predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """KernelSHAP/LIME-style (n, 2) probabilities for class 0/1."""
        df = pd.DataFrame(np.asarray(X), columns=self.feature_names)
        p = np.asarray(self.model.predict_proba(df), dtype=float).ravel()
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])

    def _sample_background(self) -> pd.DataFrame:
        if self.background_data is None or len(self.background_data) == 0:
            raise ValueError("Background data required for Kernel SHAP")
        bg = self.background_data
        missing = [c for c in self.feature_names if c not in bg.columns]
        if missing:
            # align columns
            for c in missing:
                bg = bg.copy()
                bg[c] = 0
        bg = bg[self.feature_names]
        if len(bg) > self.max_background:
            return shap.sample(bg, self.max_background)
        return bg

    def _initialize_explainer(self) -> None:
        if not getattr(self.model, "is_trained", False):
            raise ValueError("Model must be trained before creating explainer")

        if not self.feature_names and self.background_data is not None:
            self.feature_names = list(self.background_data.columns)

        # 1) Tree path
        if self._is_tree_candidate():
            try:
                self.explainer = shap.TreeExplainer(self._estimator())
                self.mode = "tree"
                return
            except Exception:
                self.explainer = None

        # 2) Kernel on full model.predict_proba (ensemble meta, TabPFN, etc.)
        bg = self._sample_background()
        self.explainer = shap.KernelExplainer(self._predict_proba_matrix, bg.values)
        self.mode = "kernel"

    def _extract_positive_class_shap(self, shap_values: Any, n_rows: int = 1) -> np.ndarray:
        """Normalize SHAP outputs to shape (n_rows, n_features) for the positive class."""
        # List form: [class0, class1]
        if isinstance(shap_values, list):
            vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            arr = np.asarray(vals, dtype=float)
        else:
            arr = np.asarray(shap_values, dtype=float)

        # (n, f, classes) 3d
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
        # single instance (f,)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if n_rows == 1 and arr.shape[0] > 1:
            # sometimes returns full bg + x — take first
            arr = arr[:1]
        return arr

    def explain_instance(self, instance: pd.DataFrame) -> dict[str, Any]:
        if not self.feature_names:
            self.feature_names = list(instance.columns)

        inst = instance.copy()
        for c in self.feature_names:
            if c not in inst.columns:
                inst[c] = 0
        inst = inst[self.feature_names]

        if self.mode == "tree":
            raw = self.explainer.shap_values(inst)
            shap_row = self._extract_positive_class_shap(raw, n_rows=1)[0]
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                bv = np.asarray(base_value).ravel()
                base_value = float(bv[1] if len(bv) > 1 else bv[0])
            else:
                base_value = float(base_value)
        else:
            # KernelExplainer: nsamples kept modest for interactive latency
            raw = self.explainer.shap_values(inst.values, nsamples=100)
            shap_row = self._extract_positive_class_shap(raw, n_rows=1)[0]
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                bv = np.asarray(base_value).ravel()
                base_value = float(bv[1] if len(bv) > 1 else bv[0])
            else:
                base_value = float(base_value)

        feature_values = inst.iloc[0].values
        explanations = []
        for name, value, shap_val in zip(self.feature_names, feature_values, shap_row):
            sv = float(shap_val)
            explanations.append(
                {
                    "feature": name,
                    "value": float(value) if isinstance(value, (int, float, np.number)) else value,
                    "importance": abs(sv),
                    "shap_value": sv,
                    "direction": "positive" if sv > 0 else "negative",
                    "contribution": "increases_risk" if sv > 0 else "decreases_risk",
                }
            )
        explanations.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "base_value": float(base_value),
            "explanations": explanations,
            # Signed for consistency / direction; abs stored separately as importance
            "feature_importance": {e["feature"]: e["shap_value"] for e in explanations},
            "method": self.mode,
        }

    def explain_global(self, data: pd.DataFrame, max_samples: int = 300) -> dict[str, Any]:
        sample = data if len(data) <= max_samples else data.sample(n=max_samples, random_state=42)
        for c in self.feature_names:
            if c not in sample.columns:
                sample = sample.copy()
                sample[c] = 0
        sample = sample[self.feature_names]

        if self.mode == "tree":
            raw = self.explainer.shap_values(sample)
        else:
            raw = self.explainer.shap_values(sample.values, nsamples=50)

        arr = self._extract_positive_class_shap(raw, n_rows=len(sample))
        mean_abs = np.abs(arr).mean(axis=0)

        global_importance = {
            name: float(imp) for name, imp in zip(self.feature_names, mean_abs)
        }
        sorted_importance = dict(
            sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
        )
        return {
            "feature_importance": sorted_importance,
            "sample_size": len(sample),
            "method": self.mode,
            "shap_values_summary": {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            },
        }

    def get_top_factors(
        self, instance: pd.DataFrame, n: int = 5
    ) -> tuple[list[str], list[str]]:
        explanation = self.explain_instance(instance)
        explanations = explanation["explanations"]
        risk = [e["feature"] for e in explanations if e["direction"] == "positive"][:n]
        protect = [e["feature"] for e in explanations if e["direction"] == "negative"][:n]
        return risk, protect
