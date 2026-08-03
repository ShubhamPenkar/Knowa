"""Feature to business concept mapping."""

from typing import Any


# Mapping of technical feature names to business-friendly names and descriptions
FEATURE_MAPPING = {
    # Tenure and engagement
    "tenure": {
        "display_name": "Customer Tenure",
        "business_concept": "Customer loyalty duration",
        "category": "engagement",
        "unit": "months",
        "direction_impact": {
            "high": "Long-term customer loyalty reduces churn risk",
            "low": "New customers are more likely to churn",
        },
    },
    "days_since_last_interaction": {
        "display_name": "Days Since Last Contact",
        "business_concept": "Customer engagement recency",
        "category": "engagement",
        "unit": "days",
        "direction_impact": {
            "high": "Lack of recent engagement increases churn risk",
            "low": "Recent engagement indicates active customer",
        },
    },
    
    # Financial
    "monthly_charges": {
        "display_name": "Monthly Charges",
        "business_concept": "Monthly spending",
        "category": "financial",
        "unit": "currency",
        "direction_impact": {
            "high": "Higher bills may lead to cost-driven churn",
            "low": "Lower bills indicate basic service usage",
        },
    },
    "total_charges": {
        "display_name": "Total Charges",
        "business_concept": "Lifetime value",
        "category": "financial",
        "unit": "currency",
        "direction_impact": {
            "high": "High lifetime value customers are valuable to retain",
            "low": "Low lifetime value may indicate new or disengaged customer",
        },
    },
    
    # Contract and payment
    "contract_type": {
        "display_name": "Contract Type",
        "business_concept": "Commitment level",
        "category": "contract",
        "value_mapping": {
            "month-to-month": "No long-term commitment - highest churn risk",
            "one_year": "Medium commitment - moderate stability",
            "two_year": "Strong commitment - lowest churn risk",
        },
    },
    "payment_method": {
        "display_name": "Payment Method",
        "business_concept": "Payment convenience",
        "category": "contract",
        "value_mapping": {
            "electronic_check": "Manual payment - higher churn indicator",
            "mailed_check": "Traditional payment - moderate engagement",
            "bank_transfer": "Auto-payment - indicates stability",
            "credit_card": "Auto-payment - indicates stability",
        },
    },
    
    # Services
    "internet_service": {
        "display_name": "Internet Service",
        "business_concept": "Internet subscription type",
        "category": "services",
        "value_mapping": {
            "fiber_optic": "Premium service - high value customer",
            "dsl": "Standard service",
            "no": "No internet service - limited engagement",
        },
    },
    "online_security": {
        "display_name": "Online Security",
        "business_concept": "Security add-on subscription",
        "category": "services",
        "value_mapping": {
            "yes": "Has security service - deeper product engagement",
            "no": "No security service - opportunity for upsell",
            "no_internet": "No internet service",
        },
    },
    "tech_support": {
        "display_name": "Tech Support",
        "business_concept": "Technical support subscription",
        "category": "services",
        "value_mapping": {
            "yes": "Has tech support - values assistance",
            "no": "No tech support - may struggle with issues",
            "no_internet": "No internet service",
        },
    },
    "streaming_tv": {
        "display_name": "Streaming TV",
        "business_concept": "TV streaming service",
        "category": "services",
        "value_mapping": {
            "yes": "Uses TV streaming - engaged with entertainment",
            "no": "No TV streaming",
            "no_internet": "No internet service",
        },
    },
    "streaming_movies": {
        "display_name": "Streaming Movies",
        "business_concept": "Movie streaming service",
        "category": "services",
        "value_mapping": {
            "yes": "Uses movie streaming - engaged with entertainment",
            "no": "No movie streaming",
            "no_internet": "No internet service",
        },
    },
    
    # Support and satisfaction
    "num_support_tickets": {
        "display_name": "Support Tickets",
        "business_concept": "Support request frequency",
        "category": "support",
        "unit": "tickets",
        "direction_impact": {
            "high": "Frequent support needs may indicate product issues",
            "low": "Few support needs - either satisfied or disengaged",
        },
    },
    "num_complaints": {
        "display_name": "Complaints",
        "business_concept": "Customer complaint frequency",
        "category": "support",
        "unit": "complaints",
        "direction_impact": {
            "high": "High complaints strongly indicate dissatisfaction",
            "low": "Few complaints suggest adequate satisfaction",
        },
    },
    "satisfaction_score": {
        "display_name": "Satisfaction Score",
        "business_concept": "Customer satisfaction rating",
        "category": "support",
        "unit": "score (1-5)",
        "direction_impact": {
            "high": "High satisfaction reduces churn risk",
            "low": "Low satisfaction strongly predicts churn",
        },
    },
}


def get_feature_info(feature_name: str) -> dict[str, Any]:
    """Get business-friendly information for a feature."""
    if feature_name in FEATURE_MAPPING:
        return FEATURE_MAPPING[feature_name]
    
    # Default for unknown features
    return {
        "display_name": feature_name.replace("_", " ").title(),
        "business_concept": feature_name.replace("_", " "),
        "category": "other",
    }


def get_value_interpretation(feature_name: str, value: Any) -> str:
    """Get business interpretation of a feature value."""
    info = get_feature_info(feature_name)
    
    # Check for categorical value mapping
    if "value_mapping" in info:
        str_value = str(value).lower().replace(" ", "_")
        if str_value in info["value_mapping"]:
            return info["value_mapping"][str_value]
    
    # Check for numeric direction impact
    if "direction_impact" in info:
        # Determine if value is high or low based on typical thresholds
        thresholds = get_feature_thresholds(feature_name)
        if value >= thresholds["high"]:
            return info["direction_impact"]["high"]
        elif value <= thresholds["low"]:
            return info["direction_impact"]["low"]
    
    return f"{info['display_name']}: {value}"


def get_feature_thresholds(feature_name: str) -> dict[str, float]:
    """Get typical thresholds for numeric features."""
    thresholds = {
        "tenure": {"low": 6, "high": 36},
        "monthly_charges": {"low": 30, "high": 80},
        "total_charges": {"low": 500, "high": 3000},
        "num_support_tickets": {"low": 1, "high": 5},
        "num_complaints": {"low": 0, "high": 2},
        "days_since_last_interaction": {"low": 7, "high": 60},
        "satisfaction_score": {"low": 2.5, "high": 4.0},
    }
    
    return thresholds.get(feature_name, {"low": 0, "high": 100})


def get_feature_category(feature_name: str) -> str:
    """Get the category of a feature."""
    info = get_feature_info(feature_name)
    return info.get("category", "other")


def get_all_categories() -> list[str]:
    """Get all feature categories."""
    return list(set(info["category"] for info in FEATURE_MAPPING.values()))
