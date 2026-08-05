"""Rule templates for business insights (outcome-agnostic phrasing)."""

from __future__ import annotations

from typing import Any


def get_severity_from_importance(importance: float, max_importance: float) -> str:
    if max_importance <= 0:
        return "low"
    rel = importance / max_importance
    if rel >= 0.7:
        return "high"
    if rel >= 0.3:
        return "medium"
    return "low"


def get_overall_severity(display_severities: list[str]) -> str:
    order = {"critical": 4, "warning": 3, "info": 2, "positive": 1}
    if not display_severities:
        return "info"
    return max(display_severities, key=lambda s: order.get(s, 0))


# Category × direction × strength → plain sentence (uses {outcome}, {display_name}, …)
INSIGHT_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "engagement": {
        "increases_risk": {
            "high": "{display_name} ({value}) points to weak habit or recency — a strong driver of {outcome}.",
            "medium": "{display_name} ({value}) suggests lighter engagement than peers, adding to {outcome} risk.",
            "low": "{display_name} plays a milder role in elevating {outcome} chance.",
        },
        "decreases_risk": {
            "high": "{display_name} ({value}) shows solid ongoing engagement, pulling down {outcome} risk.",
            "medium": "{display_name} supports stability through healthier engagement.",
            "low": "{display_name} offers a mild protective signal on engagement.",
        },
    },
    "financial": {
        "increases_risk": {
            "high": "{display_name} ({value}) is pushing cost or value pressure — a major {outcome} driver here.",
            "medium": "{display_name} ({value}) is a meaningful financial pressure marker.",
            "low": "{display_name} has a smaller financial effect on {outcome}.",
        },
        "decreases_risk": {
            "high": "{display_name} ({value}) reads as a valuable relationship signal, reducing {outcome} risk.",
            "medium": "{display_name} suggests acceptable value perception.",
            "low": "{display_name} contributes a mild protective financial signal.",
        },
    },
    "contract": {
        "increases_risk": {
            "high": "{value_interpretation} That setup is a critical lever for {outcome}.",
            "medium": "Commitment / billing setup ({value}) is contributing to {outcome} risk.",
            "low": "Contract or payment configuration has a mild effect here.",
        },
        "decreases_risk": {
            "high": "{value_interpretation} That is a strong retention anchor against {outcome}.",
            "medium": "Commitment structure supports staying put.",
            "low": "Contract setup provides a mild safety net.",
        },
    },
    "services": {
        "increases_risk": {
            "high": "{display_name} ({value}) signals thinner product attachment — a clear {outcome} push.",
            "medium": "Service mix around {display_name} leaves room to deepen attachment.",
            "low": "{display_name} has only a light effect via product mix.",
        },
        "decreases_risk": {
            "high": "{display_name} ({value}) shows deeper product attachment, reducing {outcome} risk.",
            "medium": "Active service mix supports retention.",
            "low": "{display_name} adds a mild protective product signal.",
        },
    },
    "support": {
        "increases_risk": {
            "high": "{display_name} ({value}) is a loud dissatisfaction signal — treat as urgent for {outcome}.",
            "medium": "{display_name} hints at friction that can still be fixed.",
            "low": "{display_name} is a milder support-related concern.",
        },
        "decreases_risk": {
            "high": "{display_name} ({value}) reflects a healthy support/satisfaction posture.",
            "medium": "{display_name} looks reasonably healthy.",
            "low": "{display_name} is a mild positive support signal.",
        },
    },
    "other": {
        "increases_risk": {
            "high": "{display_name} ({value}) is among the strongest factors raising {outcome}.",
            "medium": "{display_name} ({value}) meaningfully raises {outcome} chance.",
            "low": "{display_name} has a mild upward effect on {outcome}.",
        },
        "decreases_risk": {
            "high": "{display_name} ({value}) is among the strongest factors lowering {outcome}.",
            "medium": "{display_name} ({value}) helps hold {outcome} down.",
            "low": "{display_name} provides a mild protective effect.",
        },
    },
}


def get_template(category: str, contribution: str, severity: str) -> str:
    cat = INSIGHT_TEMPLATES.get(category, INSIGHT_TEMPLATES["other"])
    side = cat.get(contribution, cat["increases_risk"])
    return side.get(severity, side["medium"])


def format_insight(
    template: str,
    feature_name: str,
    feature_info: dict[str, Any],
    value: Any,
    value_interpretation: str = "",
    outcome: str = "the outcome",
) -> str:
    fmt = {
        "feature_name": feature_name,
        "display_name": feature_info.get("display_name", feature_name),
        "business_concept": feature_info.get("business_concept", feature_name),
        "value": value if value is not None else "—",
        "unit": feature_info.get("unit") or "",
        "value_interpretation": value_interpretation or "",
        "outcome": outcome,
    }
    try:
        text = template.format(**fmt)
    except KeyError:
        text = f"{fmt['display_name']}: {fmt['value']}"
    # tidy double spaces / empty unit residue
    return " ".join(text.split())
