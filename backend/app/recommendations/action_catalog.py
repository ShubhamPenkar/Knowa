"""Action catalog with business actions for churn prevention."""

from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class Action:
    """Business action definition."""
    code: str
    name: str
    description: str
    base_cost: float  # 0-1 normalized
    impact_potential: float  # 0-1 expected impact
    implementation_time: str  # immediate, short, medium, long
    applicable_conditions: dict[str, Any]
    target_features: list[str]


# Default action catalog
ACTION_CATALOG = {
    # Immediate retention actions
    "discount_10": Action(
        code="discount_10",
        name="10% Discount Offer",
        description="Offer 10% discount on next 3 months of service",
        base_cost=0.3,
        impact_potential=0.5,
        implementation_time="immediate",
        applicable_conditions={"min_churn_probability": 0.4},
        target_features=["monthly_charges"],
    ),
    "discount_20": Action(
        code="discount_20",
        name="20% Discount Offer",
        description="Offer 20% discount on next 6 months of service",
        base_cost=0.5,
        impact_potential=0.7,
        implementation_time="immediate",
        applicable_conditions={"min_churn_probability": 0.6},
        target_features=["monthly_charges"],
    ),
    
    # Contract upgrades
    "upgrade_contract_1yr": Action(
        code="upgrade_contract_1yr",
        name="1-Year Contract Upgrade",
        description="Offer incentive to switch to 1-year contract",
        base_cost=0.4,
        impact_potential=0.6,
        implementation_time="short",
        applicable_conditions={
            "contract_type": ["month-to-month"],
            "min_tenure": 3,
        },
        target_features=["contract_type"],
    ),
    "upgrade_contract_2yr": Action(
        code="upgrade_contract_2yr",
        name="2-Year Contract Upgrade",
        description="Offer significant incentive to switch to 2-year contract",
        base_cost=0.6,
        impact_potential=0.8,
        implementation_time="short",
        applicable_conditions={
            "contract_type": ["month-to-month", "one_year"],
            "min_tenure": 6,
        },
        target_features=["contract_type"],
    ),
    
    # Service additions
    "add_tech_support": Action(
        code="add_tech_support",
        name="Free Tech Support Trial",
        description="Offer 3 months free tech support service",
        base_cost=0.25,
        impact_potential=0.4,
        implementation_time="immediate",
        applicable_conditions={"tech_support": "no"},
        target_features=["tech_support"],
    ),
    "add_security": Action(
        code="add_security",
        name="Free Security Service Trial",
        description="Offer 3 months free online security service",
        base_cost=0.25,
        impact_potential=0.4,
        implementation_time="immediate",
        applicable_conditions={"online_security": "no"},
        target_features=["online_security"],
    ),
    "add_streaming_bundle": Action(
        code="add_streaming_bundle",
        name="Streaming Bundle Discount",
        description="Offer discounted streaming TV and movies bundle",
        base_cost=0.3,
        impact_potential=0.3,
        implementation_time="immediate",
        applicable_conditions={
            "streaming_tv": "no",
            "streaming_movies": "no",
        },
        target_features=["streaming_tv", "streaming_movies"],
    ),
    
    # Customer service actions
    "proactive_support_call": Action(
        code="proactive_support_call",
        name="Proactive Support Call",
        description="Schedule personal call from customer success team",
        base_cost=0.2,
        impact_potential=0.5,
        implementation_time="short",
        applicable_conditions={"min_churn_probability": 0.3},
        target_features=["satisfaction_score", "num_complaints"],
    ),
    "satisfaction_followup": Action(
        code="satisfaction_followup",
        name="Satisfaction Follow-up",
        description="Send satisfaction survey and address concerns",
        base_cost=0.1,
        impact_potential=0.3,
        implementation_time="immediate",
        applicable_conditions={"max_satisfaction_score": 3.5},
        target_features=["satisfaction_score"],
    ),
    "complaint_resolution": Action(
        code="complaint_resolution",
        name="Priority Complaint Resolution",
        description="Escalate and prioritize complaint resolution",
        base_cost=0.3,
        impact_potential=0.6,
        implementation_time="short",
        applicable_conditions={"min_complaints": 1},
        target_features=["num_complaints"],
    ),
    
    # Engagement actions
    "loyalty_program": Action(
        code="loyalty_program",
        name="Loyalty Program Enrollment",
        description="Enroll in exclusive loyalty rewards program",
        base_cost=0.2,
        impact_potential=0.4,
        implementation_time="immediate",
        applicable_conditions={"min_tenure": 6},
        target_features=["tenure"],
    ),
    "engagement_campaign": Action(
        code="engagement_campaign",
        name="Re-engagement Campaign",
        description="Targeted campaign to re-engage inactive customer",
        base_cost=0.15,
        impact_potential=0.35,
        implementation_time="immediate",
        applicable_conditions={"min_days_since_interaction": 30},
        target_features=["days_since_last_interaction"],
    ),
    "vip_upgrade": Action(
        code="vip_upgrade",
        name="VIP Status Upgrade",
        description="Upgrade to VIP customer status with priority service",
        base_cost=0.35,
        impact_potential=0.5,
        implementation_time="immediate",
        applicable_conditions={
            "min_tenure": 12,
            "min_total_charges": 1000,
        },
        target_features=["tenure", "total_charges"],
    ),
    
    # Payment method optimization
    "autopay_incentive": Action(
        code="autopay_incentive",
        name="Auto-Pay Incentive",
        description="Offer discount for switching to automatic payment",
        base_cost=0.15,
        impact_potential=0.35,
        implementation_time="immediate",
        applicable_conditions={
            "payment_method": ["electronic_check", "mailed_check"],
        },
        target_features=["payment_method"],
    ),
    
    # Plan optimization
    "plan_optimization": Action(
        code="plan_optimization",
        name="Plan Optimization Review",
        description="Review and optimize service plan for better value",
        base_cost=0.1,
        impact_potential=0.4,
        implementation_time="short",
        applicable_conditions={"min_monthly_charges": 60},
        target_features=["monthly_charges"],
    ),
}


def get_action(code: str) -> Optional[Action]:
    """Get action by code."""
    return ACTION_CATALOG.get(code)


def get_all_actions() -> list[Action]:
    """Get all available actions."""
    return list(ACTION_CATALOG.values())


def get_applicable_actions(
    features: dict[str, Any],
    churn_probability: float
) -> list[Action]:
    """
    Get actions applicable to given customer features.
    
    Args:
        features: Customer features
        churn_probability: Predicted churn probability
        
    Returns:
        List of applicable actions
    """
    applicable = []
    
    for action in ACTION_CATALOG.values():
        if _check_conditions(action, features, churn_probability):
            applicable.append(action)
    
    return applicable


def _check_conditions(
    action: Action,
    features: dict[str, Any],
    churn_probability: float
) -> bool:
    """Check if action conditions are met."""
    conditions = action.applicable_conditions
    
    for condition, value in conditions.items():
        if condition == "min_churn_probability":
            if churn_probability < value:
                return False
        
        elif condition == "max_churn_probability":
            if churn_probability > value:
                return False
        
        elif condition == "min_tenure":
            if features.get("tenure", 0) < value:
                return False
        
        elif condition == "min_total_charges":
            if features.get("total_charges", 0) < value:
                return False
        
        elif condition == "min_monthly_charges":
            if features.get("monthly_charges", 0) < value:
                return False
        
        elif condition == "max_satisfaction_score":
            if features.get("satisfaction_score", 5) > value:
                return False
        
        elif condition == "min_complaints":
            if features.get("num_complaints", 0) < value:
                return False
        
        elif condition == "min_days_since_interaction":
            if features.get("days_since_last_interaction", 0) < value:
                return False
        
        elif condition in features:
            # Direct feature value match
            feature_value = features[condition]
            if isinstance(value, list):
                if feature_value not in value:
                    return False
            elif feature_value != value:
                return False
    
    return True
