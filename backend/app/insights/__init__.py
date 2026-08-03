"""Insights package."""

from app.insights.feature_mapping import (
    get_feature_info,
    get_value_interpretation,
    get_feature_category,
    FEATURE_MAPPING,
)
from app.insights.templates import (
    get_template,
    format_insight,
    get_severity_from_importance,
)
from app.insights.nlp_generator import InsightGenerator

__all__ = [
    "get_feature_info",
    "get_value_interpretation",
    "get_feature_category",
    "FEATURE_MAPPING",
    "get_template",
    "format_insight",
    "get_severity_from_importance",
    "InsightGenerator",
]
