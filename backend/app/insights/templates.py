"""Rule templates for generating business insights."""

from typing import Any


# Insight templates by category and direction
INSIGHT_TEMPLATES = {
    # Engagement insights
    "engagement": {
        "increases_risk": {
            "high": "{display_name} of {value} {unit} indicates declining engagement, significantly increasing churn risk.",
            "medium": "{display_name} suggests reduced engagement, contributing to churn risk.",
            "low": "{display_name} shows some engagement concerns.",
        },
        "decreases_risk": {
            "high": "{display_name} of {value} {unit} demonstrates strong customer loyalty, substantially reducing churn risk.",
            "medium": "{display_name} indicates good engagement levels.",
            "low": "{display_name} provides some stability.",
        },
    },
    
    # Financial insights
    "financial": {
        "increases_risk": {
            "high": "High {display_name} of ${value} may be driving cost-sensitivity and churn consideration.",
            "medium": "{display_name} of ${value} is a notable factor in churn risk.",
            "low": "{display_name} has minor impact on churn likelihood.",
        },
        "decreases_risk": {
            "high": "{display_name} pattern indicates a valuable, stable customer relationship.",
            "medium": "{display_name} suggests reasonable value perception.",
            "low": "{display_name} provides marginal retention benefit.",
        },
    },
    
    # Contract insights
    "contract": {
        "increases_risk": {
            "high": "{value_interpretation} This is a critical churn indicator requiring immediate attention.",
            "medium": "Contract arrangement ({value}) contributes moderately to churn risk.",
            "low": "Contract type has minor influence on retention.",
        },
        "decreases_risk": {
            "high": "{value_interpretation} This provides strong retention protection.",
            "medium": "Contract arrangement supports customer retention.",
            "low": "Contract type offers some stability.",
        },
    },
    
    # Services insights
    "services": {
        "increases_risk": {
            "high": "Lack of {display_name} ({value}) indicates limited product engagement, strongly associated with churn.",
            "medium": "Service configuration suggests opportunity to deepen engagement.",
            "low": "Service selection has minor churn implications.",
        },
        "decreases_risk": {
            "high": "{display_name} subscription demonstrates deep product investment, significantly reducing churn risk.",
            "medium": "Active service usage indicates engaged customer.",
            "low": "Service configuration provides some retention benefit.",
        },
    },
    
    # Support insights
    "support": {
        "increases_risk": {
            "high": "{display_name} of {value} signals serious customer dissatisfaction requiring urgent intervention.",
            "medium": "{display_name} indicates customer frustration that needs addressing.",
            "low": "{display_name} suggests minor concerns.",
        },
        "decreases_risk": {
            "high": "Excellent {display_name} of {value} indicates high customer satisfaction.",
            "medium": "{display_name} shows customer is reasonably satisfied.",
            "low": "{display_name} provides some positive signal.",
        },
    },
    
    # Default fallback
    "other": {
        "increases_risk": {
            "high": "{display_name} is a significant factor increasing churn risk.",
            "medium": "{display_name} contributes to churn likelihood.",
            "low": "{display_name} has minor churn impact.",
        },
        "decreases_risk": {
            "high": "{display_name} significantly reduces churn risk.",
            "medium": "{display_name} supports customer retention.",
            "low": "{display_name} provides slight retention benefit.",
        },
    },
}


def get_severity_from_importance(importance: float, max_importance: float) -> str:
    """
    Determine severity level from importance score.
    
    Args:
        importance: Absolute importance value
        max_importance: Maximum importance in the explanation
        
    Returns:
        Severity: 'critical', 'warning', 'info', or 'positive'
    """
    if max_importance == 0:
        return "info"
    
    relative_importance = importance / max_importance
    
    if relative_importance >= 0.7:
        return "high"
    elif relative_importance >= 0.3:
        return "medium"
    else:
        return "low"


def get_template(
    category: str,
    contribution: str,
    severity: str
) -> str:
    """Get appropriate template for insight generation."""
    category_templates = INSIGHT_TEMPLATES.get(category, INSIGHT_TEMPLATES["other"])
    contribution_templates = category_templates.get(contribution, category_templates["increases_risk"])
    
    return contribution_templates.get(severity, contribution_templates["medium"])


def format_insight(
    template: str,
    feature_name: str,
    feature_info: dict[str, Any],
    value: Any,
    value_interpretation: str = ""
) -> str:
    """
    Format insight template with actual values.
    
    Args:
        template: Template string with placeholders
        feature_name: Technical feature name
        feature_info: Feature metadata from mapping
        value: Actual feature value
        value_interpretation: Business interpretation of value
        
    Returns:
        Formatted insight string
    """
    # Prepare format values
    format_values = {
        "feature_name": feature_name,
        "display_name": feature_info.get("display_name", feature_name),
        "business_concept": feature_info.get("business_concept", feature_name),
        "value": value,
        "unit": feature_info.get("unit", ""),
        "value_interpretation": value_interpretation,
    }
    
    try:
        return template.format(**format_values)
    except KeyError:
        # Fallback if template has unrecognized placeholders
        return f"{format_values['display_name']}: {value}"


def get_overall_severity(insight_severities: list[str]) -> str:
    """
    Determine overall insight severity from individual severities.
    
    Maps to: critical, warning, info, positive
    """
    severity_scores = {"high": 3, "medium": 2, "low": 1}
    
    if not insight_severities:
        return "info"
    
    avg_score = sum(severity_scores.get(s, 1) for s in insight_severities) / len(insight_severities)
    max_score = max(severity_scores.get(s, 1) for s in insight_severities)
    
    # If any critical issues, overall is critical
    if max_score >= 3:
        return "critical"
    elif avg_score >= 2:
        return "warning"
    else:
        return "info"
