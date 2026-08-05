"""LIME explainer for local tabular explanations."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from app.ml.models.base_model import BaseModel


class LIMEExplainer:
    """Local linear explanations around a single prediction."""

    def __init__(
        self,
        model: BaseModel,
        training_data: pd.DataFrame,
        categorical_features: Optional[list[int]] = None,
        feature_names: Optional[list[str]] = None,
        class_names: Optional[list[str]] = None,
        *,
        max_background: int = 500,
    ):
        self.model = model
        if not getattr(self.model, "is_trained", False):
            raise ValueError("Model must be trained before creating explainer")

        self.feature_names = list(
            feature_names
            or getattr(model, "feature_names", None)
            or training_data.columns
        )

        bg = training_data.copy()
        for c in self.feature_names:
            if c not in bg.columns:
                bg[c] = 0
        bg = bg[self.feature_names]
        if len(bg) > max_background:
            bg = bg.sample(n=max_background, random_state=42)
        self.training_data = bg

        # After FeatureTransformer, features are numeric codes
        self.categorical_features = categorical_features or []
        self.class_names = class_names or ["No", "Yes"]

        self.explainer = LimeTabularExplainer(
            training_data=self.training_data.values.astype(float),
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(np.asarray(X), columns=self.feature_names)
        proba = np.asarray(self.model.predict_proba(df), dtype=float).ravel()
        proba = np.clip(proba, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - proba, proba])

    def explain_instance(
        self,
        instance: pd.DataFrame,
        num_features: int = 10,
        num_samples: int = 1200,
    ) -> dict[str, Any]:
        inst = instance.copy()
        for c in self.feature_names:
            if c not in inst.columns:
                inst[c] = 0
        inst = inst[self.feature_names]
        instance_array = inst.values[0].astype(float)

        explanation = self.explainer.explain_instance(
            instance_array,
            self._predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,),
        )

        lime_exp = explanation.as_list(label=1)
        feature_values = dict(zip(self.feature_names, instance_array))

        explanations = []
        for feature_expr, weight in lime_exp:
            feature_name = self._parse_feature_name(feature_expr)
            if feature_name in feature_values:
                value = feature_values[feature_name]
            else:
                value = feature_expr
            w = float(weight)
            explanations.append(
                {
                    "feature": feature_name,
                    "expression": feature_expr,
                    "value": float(value) if isinstance(value, (int, float, np.number)) else value,
                    "importance": abs(w),
                    "lime_weight": w,
                    "direction": "positive" if w > 0 else "negative",
                    "contribution": "increases_risk" if w > 0 else "decreases_risk",
                }
            )

        explanations.sort(key=lambda x: x["importance"], reverse=True)

        local_pred = explanation.local_pred[0] if hasattr(explanation, "local_pred") else None
        score = explanation.score if hasattr(explanation, "score") else None

        return {
            "explanations": explanations,
            # Signed for direction-aware consistency
            "feature_importance": {e["feature"]: e["lime_weight"] for e in explanations},
            "intercept": float(explanation.intercept[1]) if hasattr(explanation, "intercept") else 0.0,
            "local_prediction": float(local_pred) if local_pred is not None else None,
            "model_fidelity": float(score) if score is not None else None,
            "method": "lime",
        }

    def _parse_feature_name(self, expression: str) -> str:
        operators = [" <= ", " < ", " >= ", " > ", " = ", "="]
        for op in operators:
            if op in expression:
                return expression.split(op)[0].strip()
        return expression

    def get_top_factors(
        self, instance: pd.DataFrame, n: int = 5
    ) -> tuple[list[str], list[str]]:
        explanation = self.explain_instance(instance)
        explanations = explanation["explanations"]
        risk = [e["feature"] for e in explanations if e["direction"] == "positive"][:n]
        protect = [e["feature"] for e in explanations if e["direction"] == "negative"][:n]
        return risk, protect
