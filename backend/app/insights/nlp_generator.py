"""Case-level business insight generation (Phase 3).

Turns Phase-2 drivers (SHAP/LIME) into:
  - executive summary
  - prioritized plain-language insights with next-step hints
  - theme rollup
  - context for recommendation scoring (Phase 4)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from app.insights.feature_mapping import (
    get_action_hint,
    get_feature_info,
    get_value_interpretation,
)
from app.insights.templates import (
    format_insight,
    get_overall_severity,
    get_severity_from_importance,
    get_template,
)


class InsightGenerator:
    """Convert local explanations into business-facing case intelligence."""

    def __init__(self, max_insights: int = 6):
        self.max_insights = max_insights

    def generate_case_insights(
        self,
        *,
        drivers: list[dict[str, Any]],
        probability: float,
        features: Optional[dict[str, Any]] = None,
        outcome_label: str = "the outcome",
        consistency: Optional[dict[str, Any]] = None,
        low_confidence: bool = False,
        soft_range: bool = False,
    ) -> dict[str, Any]:
        """
        Primary API for SaaS project predictions.

        drivers: CaseExplainer top_factors / drivers
          ({feature, impact|shap, direction, value, strength, text?})
        """
        features = features or {}
        outcome = self._clean_outcome(outcome_label)
        explanations = self._drivers_to_explanations(drivers, features)

        if not explanations:
            return self._empty(probability, outcome)

        max_imp = max(abs(e["importance"]) for e in explanations) or 1.0
        insights: list[dict[str, Any]] = []
        risk_names: list[str] = []
        protect_names: list[str] = []
        themes: dict[str, list[str]] = defaultdict(list)

        for exp in explanations[: self.max_insights]:
            feature = exp["feature"]
            impact = float(exp.get("signed_impact", 0.0))
            importance = abs(impact) if impact != 0 else float(exp["importance"])
            raises = exp["contribution"] == "increases_risk"
            value = exp.get("value", features.get(feature))
            info = get_feature_info(feature)
            category = info.get("category", "other")
            severity = get_severity_from_importance(importance, max_imp)
            value_interp = get_value_interpretation(feature, value)
            contribution = "increases_risk" if raises else "decreases_risk"
            template = get_template(category, contribution, severity)
            text = format_insight(
                template,
                feature,
                info,
                value,
                value_interp,
                outcome=outcome,
            )
            # Prefer CaseExplainer sentence when present (keeps Phase 2 voice)
            if exp.get("driver_text"):
                text = exp["driver_text"]

            suggestion = get_action_hint(feature, raises)
            if raises:
                if severity == "high":
                    display_severity = "critical"
                elif severity == "medium":
                    display_severity = "warning"
                else:
                    display_severity = "info"
                risk_names.append(info.get("display_name", feature))
            else:
                display_severity = "positive"
                protect_names.append(info.get("display_name", feature))

            themes[category].append(info.get("display_name", feature))

            insights.append(
                {
                    "text": text,
                    "reason": text,
                    "suggestion": suggestion,
                    "severity": display_severity,
                    "feature": feature,
                    "display_name": info.get("display_name", feature),
                    "category": category,
                    "importance": round(importance, 6),
                    "impact": round(impact, 6),
                    "contribution": contribution,
                    "direction": "increasing" if raises else "decreasing",
                    "strength": exp.get("strength")
                    or ("strong" if severity == "high" else "moderate" if severity == "medium" else "mild"),
                    "value": value,
                    "value_interpretation": value_interp,
                }
            )

        risk_level = self._get_risk_level(probability)
        summary = self._summary(
            probability=probability,
            risk_level=risk_level,
            outcome=outcome,
            risk_factors=risk_names[:3],
            protective_factors=protect_names[:3],
            consistency=consistency,
            low_confidence=low_confidence,
            soft_range=soft_range,
        )
        headline = self._headline(risk_level, probability, outcome)
        theme_rollup = [
            {
                "category": cat,
                "label": self._theme_label(cat),
                "features": feats,
                "count": len(feats),
            }
            for cat, feats in sorted(themes.items(), key=lambda kv: -len(kv[1]))
        ]
        action_context = self.generate_action_context(insights)

        trust_note = None
        if consistency and consistency.get("plain"):
            trust_note = consistency.get("plain")
        elif low_confidence or soft_range:
            trust_note = (
                "Treat these points as a prioritization brief — uncertainty is elevated for this case."
            )

        return {
            "insights": insights,
            "summary": summary,
            "headline": headline,
            "risk_level": risk_level,
            "risk_factors": risk_names,
            "protective_factors": protect_names,
            "theme_rollup": theme_rollup,
            "overall_severity": get_overall_severity([i["severity"] for i in insights]),
            "trust_note": trust_note,
            "action_context": action_context,
            "outcome": outcome,
            "probability": round(float(probability), 4),
        }

    # Back-compat for InsightService (legacy Prediction table)
    def generate_insights(
        self,
        explanations: list[dict[str, Any]],
        features: dict[str, Any],
        churn_probability: float,
        outcome_label: str = "churn",
    ) -> dict[str, Any]:
        drivers = []
        for e in explanations:
            contrib = e.get("contribution", "increases_risk")
            imp = float(e.get("importance", 0))
            signed = imp if contrib == "increases_risk" else -imp
            if "shap_value" in e:
                signed = float(e["shap_value"])
            drivers.append(
                {
                    "feature": e["feature"],
                    "impact": signed,
                    "value": e.get("value", features.get(e["feature"])),
                    "direction": "increases" if signed >= 0 else "decreases",
                }
            )
        return self.generate_case_insights(
            drivers=drivers,
            probability=churn_probability,
            features=features,
            outcome_label=outcome_label,
        )

    def generate_action_context(self, insights: list[dict[str, Any]]) -> dict[str, Any]:
        """Bridge to Phase 4 — which levers look addressable first."""
        addressable = []
        for insight in insights:
            if insight.get("contribution") != "increases_risk":
                continue
            addressable.append(
                {
                    "feature": insight["feature"],
                    "display_name": insight.get("display_name"),
                    "category": insight.get("category"),
                    "severity": insight.get("severity"),
                    "importance": insight.get("importance"),
                    "suggestion": insight.get("suggestion"),
                }
            )
        addressable.sort(key=lambda x: abs(float(x.get("importance") or 0)), reverse=True)
        return {
            "addressable_factors": addressable,
            "primary_lever": addressable[0] if addressable else None,
            "total_risk_factors": sum(
                1 for i in insights if i.get("contribution") == "increases_risk"
            ),
            "total_protective_factors": sum(
                1 for i in insights if i.get("contribution") == "decreases_risk"
            ),
        }

    def _drivers_to_explanations(
        self, drivers: list[dict[str, Any]], features: dict[str, Any]
    ) -> list[dict[str, Any]]:
        out = []
        for d in drivers:
            feature = d.get("feature")
            if not feature:
                continue
            impact = d.get("impact")
            if impact is None:
                impact = d.get("shap_value", d.get("lime_weight", 0))
            impact = float(impact or 0)
            direction = str(d.get("direction") or "")
            if direction in ("increases", "increasing", "positive"):
                raises = True
            elif direction in ("decreases", "decreasing", "negative"):
                raises = False
            else:
                raises = impact >= 0
            signed = abs(impact) if raises else -abs(impact)
            # prefer original signed impact if provided
            if impact != 0:
                signed = impact
                raises = impact >= 0
            out.append(
                {
                    "feature": feature,
                    "importance": abs(signed),
                    "signed_impact": signed,
                    "contribution": "increases_risk" if raises else "decreases_risk",
                    "value": d.get("value", features.get(feature)),
                    "strength": d.get("strength"),
                    "driver_text": d.get("text"),
                }
            )
        out.sort(key=lambda e: e["importance"], reverse=True)
        return out

    def _clean_outcome(self, label: str) -> str:
        s = str(label or "the outcome").strip().lower()
        if s in ("outcome", "the outcome", "target", ""):
            return "the outcome"
        # avoid "chance of churned" weirdness
        return s

    def _get_risk_level(self, p: float) -> str:
        if p >= 0.8:
            return "critical"
        if p >= 0.6:
            return "high"
        if p >= 0.4:
            return "medium"
        return "low"

    def _headline(self, risk_level: str, probability: float, outcome: str) -> str:
        pct = round(probability * 100)
        mapping = {
            "critical": f"Critical attention — {pct}% chance of {outcome}",
            "high": f"Elevated priority — {pct}% chance of {outcome}",
            "medium": f"Watch list — {pct}% chance of {outcome}",
            "low": f"Lower priority — {pct}% chance of {outcome}",
        }
        return mapping.get(risk_level, f"{pct}% chance of {outcome}")

    def _summary(
        self,
        *,
        probability: float,
        risk_level: str,
        outcome: str,
        risk_factors: list[str],
        protective_factors: list[str],
        consistency: Optional[dict[str, Any]],
        low_confidence: bool,
        soft_range: bool,
    ) -> str:
        pct = round(probability * 100)
        parts = []
        openings = {
            "critical": f"This case sits at critical priority ({pct}% chance of {outcome}).",
            "high": f"This case is higher priority ({pct}% chance of {outcome}).",
            "medium": f"This case is moderate priority ({pct}% chance of {outcome}).",
            "low": f"This case is lower priority ({pct}% chance of {outcome}).",
        }
        parts.append(openings.get(risk_level, f"{pct}% chance of {outcome}."))

        if risk_factors:
            if len(risk_factors) == 1:
                parts.append(f"Main pressure: {risk_factors[0]}.")
            else:
                parts.append(
                    "Main pressures: "
                    + ", ".join(risk_factors[:-1])
                    + f", and {risk_factors[-1]}."
                )
        if protective_factors:
            if len(protective_factors) == 1:
                parts.append(f"Working in your favour: {protective_factors[0]}.")
            else:
                parts.append(
                    "Working in your favour: "
                    + ", ".join(protective_factors[:-1])
                    + f", and {protective_factors[-1]}."
                )

        if low_confidence or soft_range:
            parts.append("Uncertainty is elevated — confirm before a costly intervention.")
        if consistency and consistency.get("trust_level") == "low":
            parts.append("Explanation methods disagree more than usual — verify the story.")
        elif consistency and consistency.get("trust_level") == "high":
            parts.append("Drivers are cross-checked by two explanation methods.")

        return " ".join(parts)

    def _theme_label(self, category: str) -> str:
        return {
            "engagement": "Engagement & recency",
            "financial": "Price & value",
            "contract": "Commitment & billing",
            "services": "Product mix",
            "support": "Support & satisfaction",
            "other": "Other signals",
        }.get(category, category.title())

    def _empty(self, probability: float, outcome: str) -> dict[str, Any]:
        return {
            "insights": [],
            "summary": f"No clear drivers available. Estimated chance of {outcome}: {round(probability*100)}%.",
            "headline": self._headline(self._get_risk_level(probability), probability, outcome),
            "risk_level": self._get_risk_level(probability),
            "risk_factors": [],
            "protective_factors": [],
            "theme_rollup": [],
            "overall_severity": "info",
            "trust_note": None,
            "action_context": {
                "addressable_factors": [],
                "primary_lever": None,
                "total_risk_factors": 0,
                "total_protective_factors": 0,
            },
            "outcome": outcome,
            "probability": round(float(probability), 4),
        }
