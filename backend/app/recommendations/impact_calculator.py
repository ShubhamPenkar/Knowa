"""Impact calculator for estimating action effects."""

from typing import Any

from app.recommendations.action_catalog import Action


class ImpactCalculator:
    """
    Calculates expected impact of actions on churn probability.
    
    Uses historical effectiveness data when available, otherwise
    uses base impact potential with adjustments.
    """
    
    def __init__(self, effectiveness_data: dict[str, dict] = None):
        """
        Initialize impact calculator.
        
        Args:
            effectiveness_data: Historical action effectiveness metrics
                {action_code: {success_rate: float, avg_reduction: float}}
        """
        self.effectiveness_data = effectiveness_data or {}
        
        # Feature impact weights (how much addressing feature reduces risk)
        self.feature_impact_weights = {
            "contract_type": 0.25,
            "monthly_charges": 0.15,
            "tech_support": 0.10,
            "online_security": 0.10,
            "satisfaction_score": 0.20,
            "num_complaints": 0.15,
            "payment_method": 0.08,
            "days_since_last_interaction": 0.12,
            "tenure": 0.10,
        }
    
    def calculate_impact(
        self,
        action: Action,
        features: dict[str, Any],
        churn_probability: float,
        feature_importance: dict[str, float] = None
    ) -> dict[str, Any]:
        """
        Calculate expected impact of action.
        
        Args:
            action: Action to evaluate
            features: Customer features
            churn_probability: Current churn probability
            feature_importance: SHAP/LIME feature importance scores
            
        Returns:
            Impact metrics including score and expected reduction
        """
        # Start with base impact potential
        base_impact = action.impact_potential
        
        # Adjust based on historical effectiveness
        if action.code in self.effectiveness_data:
            hist = self.effectiveness_data[action.code]
            historical_factor = hist.get("success_rate", 0.5) * 1.2  # Slight boost for proven actions
            base_impact = (base_impact + historical_factor) / 2
        
        # Adjust based on feature importance
        relevance_multiplier = self._calculate_relevance_multiplier(
            action, feature_importance or {}
        )
        
        # Adjust based on current risk level
        risk_multiplier = self._calculate_risk_multiplier(churn_probability)
        
        # Calculate final impact score
        impact_score = min(1.0, base_impact * relevance_multiplier * risk_multiplier)
        
        # Estimate probability reduction
        max_possible_reduction = churn_probability * 0.5  # Max 50% reduction assumption
        expected_reduction = max_possible_reduction * impact_score
        
        new_probability = max(0.05, churn_probability - expected_reduction)  # Floor at 5%
        
        return {
            "impact_score": round(impact_score, 4),
            "expected_probability_reduction": round(expected_reduction, 4),
            "new_probability_estimate": round(new_probability, 4),
            "probability_reduction_percent": round((expected_reduction / churn_probability) * 100, 1),
            "components": {
                "base_impact": round(base_impact, 4),
                "relevance_multiplier": round(relevance_multiplier, 4),
                "risk_multiplier": round(risk_multiplier, 4),
            },
        }
    
    def _calculate_relevance_multiplier(
        self,
        action: Action,
        feature_importance: dict[str, float]
    ) -> float:
        """
        Calculate how relevant action is based on important features.
        
        Higher multiplier if action targets features that are driving churn.
        """
        if not feature_importance or not action.target_features:
            return 1.0
        
        # Normalize feature importance
        total_importance = sum(abs(v) for v in feature_importance.values()) or 1
        
        # Sum importance of features targeted by this action
        target_importance = sum(
            abs(feature_importance.get(f, 0))
            for f in action.target_features
        )
        
        relevance_ratio = target_importance / total_importance
        
        # Convert to multiplier (0.5 to 1.5 range)
        return 0.5 + relevance_ratio
    
    def _calculate_risk_multiplier(self, churn_probability: float) -> float:
        """
        Adjust impact based on current risk level.
        
        Higher risk customers may be harder to retain (diminishing returns)
        but also have more room for improvement.
        """
        if churn_probability >= 0.9:
            return 0.7  # Very high risk - harder to retain
        elif churn_probability >= 0.7:
            return 0.9
        elif churn_probability >= 0.5:
            return 1.0  # Sweet spot
        elif churn_probability >= 0.3:
            return 1.1  # Good chance of retention
        else:
            return 0.8  # Already low risk - less room for improvement
    
    def update_effectiveness(
        self,
        action_code: str,
        was_successful: bool,
        probability_reduction: float
    ) -> None:
        """
        Update effectiveness data based on feedback.
        
        Args:
            action_code: Action that was taken
            was_successful: Whether customer was retained
            probability_reduction: Observed probability reduction
        """
        if action_code not in self.effectiveness_data:
            self.effectiveness_data[action_code] = {
                "success_rate": 0.5,
                "avg_reduction": 0.1,
                "sample_count": 0,
            }
        
        data = self.effectiveness_data[action_code]
        n = data["sample_count"]
        
        # Exponential moving average update
        alpha = 1 / (n + 1) if n < 100 else 0.01  # More weight to recent for small samples
        
        data["success_rate"] = (1 - alpha) * data["success_rate"] + alpha * (1 if was_successful else 0)
        data["avg_reduction"] = (1 - alpha) * data["avg_reduction"] + alpha * probability_reduction
        data["sample_count"] = n + 1
