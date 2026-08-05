"""Impact calculator for estimating action effects on outcome probability."""

from __future__ import annotations

from typing import Any, Optional

from app.recommendations.action_catalog import Action, _norm


class ImpactCalculator:
    """
    Expected impact of an action on positive-outcome probability.

    Uses base impact potential × driver relevance × risk band.
    Optional historical effectiveness (Phase 6 feedback) can blend in.
    """

    def __init__(self, effectiveness_data: Optional[dict[str, dict]] = None):
        self.effectiveness_data = effectiveness_data or {}

    def calculate_impact(
        self,
        action: Action,
        features: dict[str, Any],
        probability: float,
        feature_importance: Optional[dict[str, float]] = None,
        *,
        soft_case: bool = False,
    ) -> dict[str, Any]:
        base_impact = float(action.impact_potential)
        p = float(probability)

        if action.code in self.effectiveness_data:
            hist = self.effectiveness_data[action.code]
            historical_factor = float(hist.get("success_rate", 0.5)) * 1.15
            base_impact = (base_impact + historical_factor) / 2

        relevance_multiplier = self._relevance_multiplier(
            action, feature_importance or {}
        )
        risk_multiplier = self._risk_multiplier(p)

        # Soft / low-trust cases: de-scale aggressive interventions slightly
        uncertainty_factor = 0.85 if soft_case and action.category == "save" else 1.0
        # Monitor action impact is intentionally tiny
        if action.code in ("monitor_only", "monitor_attrition"):
            impact_score = 0.08
            expected_reduction = 0.0
            new_p = p
        else:
            impact_score = min(
                1.0, base_impact * relevance_multiplier * risk_multiplier * uncertainty_factor
            )
            # Cap expected reduction so we never claim magic
            max_possible_reduction = min(p * 0.55, 0.35)
            expected_reduction = max_possible_reduction * impact_score
            new_p = max(0.02, p - expected_reduction)

        return {
            "impact_score": round(impact_score, 4),
            "expected_probability_reduction": round(expected_reduction, 4),
            "new_probability_estimate": round(new_p, 4),
            "probability_reduction_percent": round(
                (expected_reduction / p) * 100, 1
            )
            if p > 1e-6
            else 0.0,
            "components": {
                "base_impact": round(base_impact, 4),
                "relevance_multiplier": round(relevance_multiplier, 4),
                "risk_multiplier": round(risk_multiplier, 4),
                "uncertainty_factor": uncertainty_factor,
            },
        }

    def _relevance_multiplier(
        self, action: Action, feature_importance: dict[str, float]
    ) -> float:
        if not feature_importance or not action.target_features:
            return 1.0

        # Only count *risk-raising* magnitude for addressable levers
        positives = {
            k: v for k, v in feature_importance.items() if float(v) > 0
        }
        total = sum(abs(v) for v in positives.values()) or sum(
            abs(v) for v in feature_importance.values()
        ) or 1.0

        target_imp = 0.0
        for t in action.target_features:
            tn = _norm(t)
            for k, v in feature_importance.items():
                kn = _norm(k)
                if kn == tn or tn in kn or kn in tn:
                    target_imp += abs(float(v))
                    break

        ratio = min(1.0, target_imp / total)
        # 0.55 … 1.45
        return 0.55 + 0.9 * ratio

    def _risk_multiplier(self, probability: float) -> float:
        if probability >= 0.85:
            return 0.75
        if probability >= 0.65:
            return 0.95
        if probability >= 0.4:
            return 1.05
        if probability >= 0.2:
            return 0.95
        return 0.7  # already calm — little "save" room
