"""Recommendations package (Phase 4)."""

from app.recommendations.action_catalog import (
    Action,
    ACTION_CATALOG,
    HR_ATTRITION_CATALOG,
    TELCO_ACTION_CATALOG,
    action_from_custom,
    build_feature_importance,
    feature_lookup,
    get_action,
    get_all_actions,
    get_applicable_actions,
    get_catalog,
)
from app.recommendations.cost_calculator import CostCalculator
from app.recommendations.decision_scorer import DecisionScorer
from app.recommendations.domains import (
    DOMAIN_HR_ATTRITION,
    DOMAIN_TELCO,
    detect_domain,
)
from app.recommendations.impact_calculator import ImpactCalculator

__all__ = [
    "Action",
    "ACTION_CATALOG",
    "HR_ATTRITION_CATALOG",
    "TELCO_ACTION_CATALOG",
    "action_from_custom",
    "build_feature_importance",
    "feature_lookup",
    "get_action",
    "get_all_actions",
    "get_applicable_actions",
    "get_catalog",
    "CostCalculator",
    "DecisionScorer",
    "ImpactCalculator",
    "DOMAIN_HR_ATTRITION",
    "DOMAIN_TELCO",
    "detect_domain",
]
