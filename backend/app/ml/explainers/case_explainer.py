"""Unified case-level explainability (Phase 2).

Runs SHAP + LIME on the *routed* scoring model (foundation or ensemble),
computes Explanation Consistency Score, and returns business-facing drivers.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from app.config import get_settings
from app.ml.explainers.consistency_scorer import ConsistencyScorer
from app.ml.explainers.lime_explainer import LIMEExplainer
from app.ml.explainers.shap_explainer import SHAPExplainer


def _humanize(name: str) -> str:
    return (
        str(name)
        .replace("__", " ")
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
    )


def _strength(impact: float) -> str:
    a = abs(impact)
    if a >= 0.12:
        return "strong"
    if a >= 0.04:
        return "moderate"
    return "mild"


def _driver_sentence(
    feature: str,
    value: Any,
    impact: float,
    outcome: str,
) -> dict[str, Any]:
    label = _humanize(feature)
    val_s = "—" if value is None else value
    up = impact > 0
    strength = _strength(impact)
    if up:
        text = (
            f"{label} (value: {val_s}) is a {strength} factor "
            f"raising the chance of {outcome}."
        )
        direction = "increases"
    else:
        text = (
            f"{label} (value: {val_s}) is a {strength} factor "
            f"lowering the chance of {outcome}."
        )
        direction = "decreases"
    return {
        "feature": feature,
        "label": label,
        "value": value,
        "impact": float(impact),
        "direction": direction,
        "strength": strength,
        "text": text,
    }


class CaseExplainer:
    """Orchestrate SHAP + LIME + consistency for one case."""

    def __init__(self, consistency_threshold: Optional[float] = None):
        settings = get_settings()
        thr = (
            consistency_threshold
            if consistency_threshold is not None
            else settings.explanation_consistency_threshold
        )
        self.scorer = ConsistencyScorer(consistency_threshold=thr)

    def explain(
        self,
        model,
        instance: pd.DataFrame,
        background: pd.DataFrame,
        *,
        raw_features: Optional[dict[str, Any]] = None,
        outcome_label: str = "the outcome",
        positive_label: str = "Yes",
        negative_label: str = "No",
        top_k: int = 5,
        run_lime: bool = True,
        run_shap: bool = True,
    ) -> dict[str, Any]:
        """
        Explain one transformed feature row.

        raw_features: original (pre-transform) values for display when available.
        """
        errors: list[str] = []
        shap_result: dict[str, Any] = {"explanations": [], "feature_importance": {}, "method": None}
        lime_result: dict[str, Any] = {"explanations": [], "feature_importance": {}, "method": None}

        # Prefer explaining the routed model end-to-end
        # (ensemble stack / foundation calibrated proba), not a loose base only
        explain_model = model

        if run_shap:
            try:
                shap_explainer = SHAPExplainer(explain_model, background_data=background)
                shap_result = shap_explainer.explain_instance(instance)
            except Exception as e:
                errors.append(f"SHAP failed: {e}")

        if run_lime:
            try:
                lime_explainer = LIMEExplainer(
                    explain_model,
                    background,
                    feature_names=list(getattr(explain_model, "feature_names", None) or instance.columns),
                    class_names=[str(negative_label), str(positive_label)],
                )
                lime_result = lime_explainer.explain_instance(instance)
            except Exception as e:
                errors.append(f"LIME failed: {e}")

        shap_imp = shap_result.get("feature_importance") or {}
        lime_imp = lime_result.get("feature_importance") or {}

        if shap_imp and lime_imp:
            consistency = self.scorer.calculate_consistency(shap_imp, lime_imp)
        elif shap_imp or lime_imp:
            consistency = {
                "consistency_score": None,
                "trust_level": "single_method",
                "metrics": {},
                "plain": (
                    "Only one explanation method succeeded — using it as the best available “why”, "
                    "without a cross-check."
                ),
                "should_flag": True,
                "methods_partial": True,
            }
        else:
            consistency = {
                "consistency_score": 0.0,
                "trust_level": "unavailable",
                "metrics": {},
                "plain": "Explanations could not be produced for this case.",
                "should_flag": True,
                "methods_partial": True,
            }

        # Prefer SHAP order when available; enrich values from raw features
        primary_exps = shap_result.get("explanations") or lime_result.get("explanations") or []
        drivers = []
        for exp in primary_exps[:top_k]:
            feat = exp["feature"]
            impact = float(exp.get("shap_value", exp.get("lime_weight", exp.get("impact", 0))))
            raw_val = None
            if raw_features is not None:
                # Prefer raw column if present; fall back to transformed
                raw_val = raw_features.get(feat, exp.get("value"))
            else:
                raw_val = exp.get("value")
            drivers.append(_driver_sentence(feat, raw_val, impact, outcome_label))

        top_factors = [
            {
                "feature": d["feature"],
                "value": d["value"],
                "impact": d["impact"],
                "direction": d["direction"],
                "strength": d["strength"],
                "text": d["text"],
            }
            for d in drivers
        ]

        methods = []
        if shap_imp:
            methods.append("shap")
        if lime_imp:
            methods.append("lime")

        shap_top = [
            {
                "feature": e["feature"],
                "value": e.get("value"),
                "impact": e.get("shap_value", e.get("importance", 0)),
                "direction": "increases" if e.get("direction") == "positive" else "decreases",
            }
            for e in (shap_result.get("explanations") or [])[:top_k]
        ]
        lime_top = [
            {
                "feature": e["feature"],
                "expression": e.get("expression"),
                "value": e.get("value"),
                "impact": e.get("lime_weight", e.get("importance", 0)),
                "direction": "increases" if e.get("direction") == "positive" else "decreases",
            }
            for e in (lime_result.get("explanations") or [])[:top_k]
        ]

        return {
            "methods_available": methods,
            "degraded": len(methods) < 2 or bool(errors),
            "errors": errors or None,
            "shap": {
                "top_features": shap_top,
                "base_value": shap_result.get("base_value"),
                "method": shap_result.get("method"),
            },
            "lime": {
                "top_features": lime_top,
                "model_fidelity": lime_result.get("model_fidelity"),
                "local_prediction": lime_result.get("local_prediction"),
            },
            "consistency": {
                "score": consistency.get("consistency_score"),
                "trust_level": consistency.get("trust_level"),
                "flag": bool(consistency.get("should_flag", False)),
                "plain": consistency.get("plain"),
                "metrics": consistency.get("metrics"),
                "top_k_features": consistency.get("top_k_features"),
            },
            "drivers": drivers,
            "all_factors": shap_result.get("explanations") or lime_result.get("explanations") or [],
            "top_factors": top_factors,
        }
