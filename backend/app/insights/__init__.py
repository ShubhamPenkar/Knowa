"""Insights package (Phase 3)."""

from app.insights.feature_mapping import (
    FEATURE_MAPPING,
    get_action_hint,
    get_feature_category,
    get_feature_info,
    get_value_interpretation,
)
from app.insights.nlp_generator import InsightGenerator
from app.insights.templates import (
    format_insight,
    get_severity_from_importance,
    get_template,
)

__all__ = [
    "get_feature_info",
    "get_value_interpretation",
    "get_feature_category",
    "get_action_hint",
    "FEATURE_MAPPING",
    "get_template",
    "format_insight",
    "get_severity_from_importance",
    "InsightGenerator",
]
