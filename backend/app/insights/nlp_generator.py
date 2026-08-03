"""Natural language insight generator."""

from typing import Any

from app.insights.feature_mapping import (
    get_feature_info,
    get_value_interpretation,
    get_feature_category,
)
from app.insights.templates import (
    get_template,
    format_insight,
    get_severity_from_importance,
    get_overall_severity,
)


class InsightGenerator:
    """
    Generates business-friendly insights from ML explanations.
    
    Converts technical SHAP/LIME outputs into actionable business language.
    """
    
    def __init__(self, max_insights: int = 10):
        """
        Initialize insight generator.
        
        Args:
            max_insights: Maximum number of insights to generate
        """
        self.max_insights = max_insights
    
    def generate_insights(
        self,
        explanations: list[dict[str, Any]],
        features: dict[str, Any],
        churn_probability: float
    ) -> dict[str, Any]:
        """
        Generate business insights from explanations.
        
        Args:
            explanations: List of feature explanations with importance
            features: Original feature values
            churn_probability: Predicted churn probability
            
        Returns:
            Dictionary with insights, summary, and metadata
        """
        if not explanations:
            return {
                "insights": [],
                "summary": "No significant factors identified.",
                "risk_level": self._get_risk_level(churn_probability),
            }
        
        # Get max importance for relative severity calculation
        max_importance = max(e["importance"] for e in explanations)
        
        # Generate insights for top features
        insights = []
        risk_factors = []
        protective_factors = []
        
        for explanation in explanations[:self.max_insights]:
            feature_name = explanation["feature"]
            importance = explanation["importance"]
            contribution = explanation["contribution"]
            value = features.get(feature_name, explanation.get("value"))
            
            # Get feature metadata
            feature_info = get_feature_info(feature_name)
            category = feature_info.get("category", "other")
            
            # Calculate severity
            severity = get_severity_from_importance(importance, max_importance)
            
            # Get value interpretation
            value_interpretation = get_value_interpretation(feature_name, value)
            
            # Get and format template
            template = get_template(category, contribution, severity)
            insight_text = format_insight(
                template,
                feature_name,
                feature_info,
                value,
                value_interpretation
            )
            
            # Determine display severity
            if contribution == "increases_risk":
                if severity == "high":
                    display_severity = "critical"
                elif severity == "medium":
                    display_severity = "warning"
                else:
                    display_severity = "info"
                risk_factors.append(feature_info.get("display_name", feature_name))
            else:
                display_severity = "positive"
                protective_factors.append(feature_info.get("display_name", feature_name))
            
            insights.append({
                "text": insight_text,
                "severity": display_severity,
                "feature": feature_name,
                "display_name": feature_info.get("display_name", feature_name),
                "category": category,
                "importance": importance,
                "contribution": contribution,
                "value": value,
            })
        
        # Generate summary
        summary = self._generate_summary(
            churn_probability,
            risk_factors[:3],
            protective_factors[:3]
        )
        
        return {
            "insights": insights,
            "summary": summary,
            "risk_level": self._get_risk_level(churn_probability),
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "overall_severity": get_overall_severity([i["severity"] for i in insights]),
        }
    
    def _generate_summary(
        self,
        churn_probability: float,
        risk_factors: list[str],
        protective_factors: list[str]
    ) -> str:
        """Generate executive summary of insights."""
        risk_level = self._get_risk_level(churn_probability)
        prob_percent = round(churn_probability * 100)
        
        # Build summary parts
        parts = []
        
        # Risk level statement
        risk_statements = {
            "critical": f"This customer has a critical churn risk ({prob_percent}% probability).",
            "high": f"This customer shows high churn risk ({prob_percent}% probability).",
            "medium": f"This customer has moderate churn risk ({prob_percent}% probability).",
            "low": f"This customer has low churn risk ({prob_percent}% probability).",
        }
        parts.append(risk_statements.get(risk_level, f"Churn probability: {prob_percent}%"))
        
        # Key drivers
        if risk_factors:
            if len(risk_factors) == 1:
                parts.append(f"The primary risk driver is {risk_factors[0]}.")
            else:
                factors_str = ", ".join(risk_factors[:-1]) + f" and {risk_factors[-1]}"
                parts.append(f"Key risk drivers are {factors_str}.")
        
        # Protective factors
        if protective_factors:
            if len(protective_factors) == 1:
                parts.append(f"{protective_factors[0]} is helping retain this customer.")
            else:
                factors_str = ", ".join(protective_factors[:-1]) + f" and {protective_factors[-1]}"
                parts.append(f"Retention is supported by {factors_str}.")
        
        return " ".join(parts)
    
    def _get_risk_level(self, churn_probability: float) -> str:
        """Map probability to risk level."""
        if churn_probability >= 0.8:
            return "critical"
        elif churn_probability >= 0.6:
            return "high"
        elif churn_probability >= 0.4:
            return "medium"
        else:
            return "low"
    
    def generate_action_context(
        self,
        insights: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Generate context for recommendation engine.
        
        Identifies which factors are addressable and their priority.
        """
        addressable_factors = []
        
        # Factors that can be addressed by business actions
        addressable_features = {
            "contract_type": ["upgrade_contract", "loyalty_program"],
            "tech_support": ["add_tech_support", "proactive_support_call"],
            "online_security": ["add_security_service"],
            "monthly_charges": ["discount_offer", "plan_optimization"],
            "satisfaction_score": ["customer_service_outreach", "feedback_followup"],
            "num_complaints": ["complaint_resolution", "escalation_review"],
            "days_since_last_interaction": ["engagement_campaign", "check_in_call"],
        }
        
        for insight in insights:
            feature = insight["feature"]
            if feature in addressable_features and insight["contribution"] == "increases_risk":
                addressable_factors.append({
                    "feature": feature,
                    "severity": insight["severity"],
                    "importance": insight["importance"],
                    "suggested_actions": addressable_features[feature],
                })
        
        return {
            "addressable_factors": addressable_factors,
            "total_risk_factors": sum(1 for i in insights if i["contribution"] == "increases_risk"),
            "total_protective_factors": sum(1 for i in insights if i["contribution"] == "decreases_risk"),
        }
