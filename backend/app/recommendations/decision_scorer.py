"""Decision scorer — hybrid rule filter + weighted impact/cost/relevance.

Final Decision Score = α·Impact + β·(1 − Cost) + γ·Relevance

Consumes Phase-2 drivers and Phase-3 action_context for relevance boosts.
"""

from __future__ import annotations

from typing import Any, Optional

from app.config import get_settings
from app.recommendations.action_catalog import (
    Action,
    action_from_custom,
    build_feature_importance,
    feature_lookup,
    get_applicable_actions,
    get_all_actions,
    _norm,
)
from app.recommendations.cost_calculator import CostCalculator
from app.recommendations.impact_calculator import ImpactCalculator


class DecisionScorer:
    def __init__(
        self,
        impact_weight: Optional[float] = None,
        cost_weight: Optional[float] = None,
        relevance_weight: Optional[float] = None,
        effectiveness_data: Optional[dict] = None,
    ):
        settings = get_settings()
        self.impact_weight = (
            impact_weight if impact_weight is not None else settings.impact_weight
        )
        self.cost_weight = (
            cost_weight if cost_weight is not None else settings.cost_weight
        )
        self.relevance_weight = (
            relevance_weight
            if relevance_weight is not None
            else settings.relevance_weight
        )
        # Normalize weights
        total = self.impact_weight + self.cost_weight + self.relevance_weight
        if total > 0:
            self.impact_weight /= total
            self.cost_weight /= total
            self.relevance_weight /= total

        self.impact_calculator = ImpactCalculator(effectiveness_data)
        self.cost_calculator = CostCalculator()

    def score_case(
        self,
        *,
        features: dict[str, Any],
        probability: float,
        top_factors: Optional[list[dict[str, Any]]] = None,
        action_context: Optional[dict[str, Any]] = None,
        custom_actions: Optional[list[Action]] = None,
        soft_case: bool = False,
        low_confidence: bool = False,
        consistency_trust: Optional[str] = None,
        outcome_label: str = "the outcome",
        top_n: int = 5,
        domain: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Full hybrid recommendation for one case.

        Returns ranked recommendations + decision summary.
        """
        p = float(probability)
        feature_importance = build_feature_importance(top_factors)
        customer_value = self._estimate_customer_value(features)

        catalog = get_applicable_actions(features, p, domain=domain)
        # Merge custom org actions
        seen = {a.code for a in catalog}
        for ca in custom_actions or []:
            if ca.code not in seen:
                catalog.append(ca)
                seen.add(ca.code)

        # If still empty, full soft set for this domain
        if not catalog:
            catalog = list(get_all_actions(domain))[:6]

        soft = soft_case or low_confidence or (
            consistency_trust in ("low", "unavailable")
        )

        scored: list[dict[str, Any]] = []
        for action in catalog:
            scored.append(
                self.score_action(
                    action,
                    features=features,
                    probability=p,
                    feature_importance=feature_importance,
                    customer_value=customer_value,
                    action_context=action_context,
                    soft_case=soft,
                    outcome_label=outcome_label,
                )
            )

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top = scored[:top_n]
        for i, rec in enumerate(top):
            rec["rank"] = i + 1

        summary = self.get_decision_summary(
            top, p, soft=soft, outcome_label=outcome_label, domain=domain
        )

        return {
            "recommendations": top,
            "decision_summary": summary,
            "scoring": {
                "impact_weight": round(self.impact_weight, 3),
                "cost_weight": round(self.cost_weight, 3),
                "relevance_weight": round(self.relevance_weight, 3),
                "soft_case": soft,
                "n_candidates": len(scored),
                "domain": domain or "telco",
            },
        }

    def score_action(
        self,
        action: Action,
        *,
        features: dict[str, Any],
        probability: float,
        feature_importance: Optional[dict[str, float]] = None,
        customer_value: Optional[float] = None,
        action_context: Optional[dict[str, Any]] = None,
        soft_case: bool = False,
        outcome_label: str = "the outcome",
    ) -> dict[str, Any]:
        impact_result = self.impact_calculator.calculate_impact(
            action,
            features,
            probability,
            feature_importance,
            soft_case=soft_case,
        )
        cost_result = self.cost_calculator.calculate_cost(
            action, features, customer_value
        )
        relevance_score = self._calculate_relevance(
            action, feature_importance or {}, action_context
        )
        impact_score = impact_result["impact_score"]
        cost_score = cost_result["cost_score"]

        final_score = (
            self.impact_weight * impact_score
            + self.cost_weight * (1.0 - cost_score)
            + self.relevance_weight * relevance_score
        )

        # Soft cases: lightly prefer low-cost moves in ranking
        if soft_case and cost_score > 0.45:
            final_score *= 0.92

        reasoning = self._generate_reasoning(
            action,
            impact_score=impact_score,
            cost_score=cost_score,
            relevance_score=relevance_score,
            features=features,
            probability=probability,
            action_context=action_context,
            reduction=impact_result["expected_probability_reduction"],
            outcome_label=outcome_label,
        )

        return {
            "action_code": action.code,
            "action_name": action.name,
            "name": action.name,
            "description": action.description,
            "category": action.category,
            "final_score": round(final_score, 4),
            "impact_score": round(impact_score, 4),
            "cost_score": round(cost_score, 4),
            "cost_label": cost_result.get("cost_label"),
            "relevance_score": round(relevance_score, 4),
            "expected_probability_reduction": impact_result[
                "expected_probability_reduction"
            ],
            "new_probability_estimate": impact_result["new_probability_estimate"],
            "probability_reduction_percent": impact_result.get(
                "probability_reduction_percent"
            ),
            "impact_is_illustrative": True,
            "impact_disclaimer": (
                "Illustrative estimate from action-catalog heuristics — "
                "not a re-simulated outcome for this case."
            ),
            "reasoning": reasoning,
            "implementation_time": action.implementation_time,
        }

    def score_all_applicable(
        self,
        features: dict[str, Any],
        churn_probability: float,
        feature_importance: Optional[dict[str, float]] = None,
        customer_value: Optional[float] = None,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Legacy API used by RecommendationService."""
        top_factors = [
            {"feature": k, "impact": v} for k, v in (feature_importance or {}).items()
        ]
        result = self.score_case(
            features=features,
            probability=churn_probability,
            top_factors=top_factors,
            top_n=top_n,
        )
        return result["recommendations"]

    def _calculate_relevance(
        self,
        action: Action,
        feature_importance: dict[str, float],
        action_context: Optional[dict[str, Any]],
    ) -> float:
        if not feature_importance and not action_context:
            return float(action.impact_potential) * 0.6

        total = sum(abs(v) for v in feature_importance.values()) or 1.0
        addressed = 0.0
        for t in action.target_features:
            tn = _norm(t)
            for k, v in feature_importance.items():
                kn = _norm(k)
                if kn == tn or tn in kn or kn in tn:
                    addressed += abs(float(v))
                    break
        relevance = min(1.0, addressed / total) if action.target_features else 0.35

        # Boost when action targets Phase-3 primary lever
        if action_context:
            primary = action_context.get("primary_lever") or {}
            pf = str(primary.get("feature") or "").lower()
            if pf and action.target_features:
                pt = _norm(pf)
                for t in action.target_features:
                    if _norm(t) in pt or pt in _norm(t):
                        relevance = min(1.0, relevance + 0.25)
                        break
            for af in action_context.get("addressable_factors") or []:
                afn = _norm(str(af.get("feature") or ""))
                for t in action.target_features:
                    if afn and (_norm(t) in afn or afn in _norm(t)):
                        relevance = min(1.0, relevance + 0.1)
                        break

        # Generic actions always somewhat relevant
        if not action.target_features:
            relevance = max(relevance, 0.4 if action.code == "check_in_light" else 0.25)
        if action.code in ("monitor_only", "monitor_attrition"):
            relevance = 0.5

        return float(min(1.0, relevance))

    def _estimate_customer_value(self, features: dict[str, Any]) -> float:
        for key in (
            "total_charges",
            "TotalCharges",
            "lifetime_value",
            "spend",
            "monthly_income",
            "MonthlyIncome",
        ):
            logical = "monthly_income" if "income" in key.lower() else "total_charges"
            v = feature_lookup(features, logical, None)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        # fallback: monthly * tenure proxy
        monthly = feature_lookup(features, "monthly_charges", 0) or 0
        tenure = feature_lookup(features, "tenure", 0) or 0
        try:
            return float(monthly) * float(tenure)
        except (TypeError, ValueError):
            return 0.0

    def _generate_reasoning(
        self,
        action: Action,
        *,
        impact_score: float,
        cost_score: float,
        relevance_score: float,
        features: dict[str, Any],
        probability: float,
        action_context: Optional[dict[str, Any]],
        reduction: float,
        outcome_label: str,
    ) -> str:
        parts = []
        if impact_score >= 0.6:
            parts.append("Strong expected effect on this case")
        elif impact_score >= 0.35:
            parts.append("Solid expected effect")
        else:
            parts.append("Lighter-touch move")

        if cost_score <= 0.25:
            parts.append("low cost to try")
        elif cost_score <= 0.55:
            parts.append("balanced cost vs. likely benefit")
        else:
            parts.append("heavier cost — use when stakes justify it")

        if relevance_score >= 0.55:
            targets = ", ".join(action.target_features[:2]) if action.target_features else "broader risk"
            parts.append(f"aligned with active drivers ({targets})")
        elif relevance_score >= 0.3:
            parts.append("partially matches current drivers")

        if reduction >= 0.05:
            parts.append(
                f"illustrative shift on {outcome_label}: about −{reduction:.0%} "
                f"(guide, not a guarantee)"
            )

        if action_context and action_context.get("primary_lever"):
            pl = action_context["primary_lever"]
            if pl.get("suggestion") and action.category in ("save", "service", "custom"):
                parts.append(f"insight focus: {pl.get('display_name') or pl.get('feature')}")

        return ". ".join(parts).capitalize() + "."

    def get_decision_summary(
        self,
        recommendations: list[dict[str, Any]],
        probability: float,
        *,
        soft: bool = False,
        outcome_label: str = "the outcome",
        domain: Optional[str] = None,
    ) -> dict[str, Any]:
        if not recommendations:
            return {
                "strategy": "Monitor",
                "description": "No applicable actions for this case.",
                "current_probability": round(probability, 4),
                "expected_new_probability": round(probability, 4),
            }

        top = recommendations[0]
        # Diminishing stack of top-3 expected reductions
        stacked = 0.0
        for i, r in enumerate(recommendations[:3]):
            stacked += float(r.get("expected_probability_reduction") or 0) * (0.7**i)
        new_p = max(0.02, probability - stacked)

        hr = domain == "hr_attrition"
        if probability >= 0.7:
            strategy = "Urgent stay intervention" if hr else "Urgent intervention"
        elif probability >= 0.5:
            strategy = "Active retention" if hr else "Active retention"
        elif probability >= 0.3:
            strategy = "Preventive engagement"
        else:
            strategy = "Relationship maintenance" if not hr else "Steady-state monitoring"

        if soft:
            strategy = f"{strategy} (soft — prefer lighter first steps)"

        return {
            "strategy": strategy,
            "description": f"Lead with “{top['action_name']}”. {top.get('reasoning', '')}",
            "current_probability": round(probability, 4),
            "expected_new_probability": round(new_p, 4),
            "expected_reduction": round(probability - new_p, 4),
            "impact_is_illustrative": True,
            "impact_disclaimer": (
                "Illustrative stack estimate from action-catalog heuristics — "
                "not a re-simulated outcome for this case."
            ),
            "recommended_actions_count": len(recommendations),
            "primary_action_code": top.get("action_code"),
            "outcome": outcome_label,
            "domain": domain or "telco",
        }
