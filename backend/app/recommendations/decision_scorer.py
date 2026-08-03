"""Decision scorer for ranking action recommendations."""

from typing import Any

from app.recommendations.action_catalog import Action, get_applicable_actions
from app.recommendations.impact_calculator import ImpactCalculator
from app.recommendations.cost_calculator import CostCalculator


class DecisionScorer:
    """
    Scores and ranks actions based on impact, cost, and relevance.
    
    Final Score = α × Impact + β × (1 - Cost) + γ × Relevance
    """
    
    def __init__(
        self,
        impact_weight: float = 0.5,
        cost_weight: float = 0.3,
        relevance_weight: float = 0.2,
        effectiveness_data: dict = None
    ):
        """
        Initialize decision scorer.
        
        Args:
            impact_weight: Weight for impact score (α)
            cost_weight: Weight for cost score (β)
            relevance_weight: Weight for relevance score (γ)
            effectiveness_data: Historical effectiveness data
        """
        self.impact_weight = impact_weight
        self.cost_weight = cost_weight
        self.relevance_weight = relevance_weight
        
        self.impact_calculator = ImpactCalculator(effectiveness_data)
        self.cost_calculator = CostCalculator()
    
    def score_action(
        self,
        action: Action,
        features: dict[str, Any],
        churn_probability: float,
        feature_importance: dict[str, float] = None,
        customer_value: float = None
    ) -> dict[str, Any]:
        """
        Score a single action.
        
        Args:
            action: Action to score
            features: Customer features
            churn_probability: Current churn probability
            feature_importance: Feature importance from explainability
            customer_value: Customer lifetime value
            
        Returns:
            Comprehensive action score
        """
        # Calculate impact
        impact_result = self.impact_calculator.calculate_impact(
            action, features, churn_probability, feature_importance
        )
        impact_score = impact_result["impact_score"]
        
        # Calculate cost
        cost_result = self.cost_calculator.calculate_cost(
            action, features, customer_value
        )
        cost_score = cost_result["cost_score"]
        
        # Calculate relevance
        relevance_score = self._calculate_relevance(
            action, features, feature_importance
        )
        
        # Final weighted score
        final_score = (
            self.impact_weight * impact_score +
            self.cost_weight * (1 - cost_score) +  # Invert cost (lower is better)
            self.relevance_weight * relevance_score
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            action, impact_score, cost_score, relevance_score, features
        )
        
        return {
            "action_code": action.code,
            "action_name": action.name,
            "description": action.description,
            "final_score": round(final_score, 4),
            "impact_score": round(impact_score, 4),
            "cost_score": round(cost_score, 4),
            "relevance_score": round(relevance_score, 4),
            "expected_probability_reduction": impact_result["expected_probability_reduction"],
            "new_probability_estimate": impact_result["new_probability_estimate"],
            "reasoning": reasoning,
            "implementation_time": action.implementation_time,
        }
    
    def score_all_applicable(
        self,
        features: dict[str, Any],
        churn_probability: float,
        feature_importance: dict[str, float] = None,
        customer_value: float = None,
        top_n: int = 5
    ) -> list[dict[str, Any]]:
        """
        Score all applicable actions and return top N.
        
        Args:
            features: Customer features
            churn_probability: Current churn probability
            feature_importance: Feature importance scores
            customer_value: Customer lifetime value
            top_n: Number of top recommendations to return
            
        Returns:
            List of scored actions, ranked by final score
        """
        # Get applicable actions
        applicable_actions = get_applicable_actions(features, churn_probability)
        
        if not applicable_actions:
            return []
        
        # Score each action
        scored_actions = []
        for action in applicable_actions:
            score = self.score_action(
                action,
                features,
                churn_probability,
                feature_importance,
                customer_value
            )
            scored_actions.append(score)
        
        # Sort by final score descending
        scored_actions.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Add rank
        for i, action in enumerate(scored_actions[:top_n]):
            action["rank"] = i + 1
        
        return scored_actions[:top_n]
    
    def _calculate_relevance(
        self,
        action: Action,
        features: dict[str, Any],
        feature_importance: dict[str, float] = None
    ) -> float:
        """
        Calculate relevance score based on feature alignment.
        
        High relevance when action targets important churn drivers.
        """
        if not feature_importance:
            # Default relevance based on action's base potential
            return action.impact_potential
        
        # Calculate how much of the feature importance is covered by this action
        total_importance = sum(abs(v) for v in feature_importance.values()) or 1
        
        # Sum importance of features addressed by this action
        addressed_importance = sum(
            abs(feature_importance.get(f, 0))
            for f in action.target_features
        )
        
        relevance = addressed_importance / total_importance
        
        # Boost relevance if action directly addresses a high-importance feature
        max_target_importance = max(
            (abs(feature_importance.get(f, 0)) for f in action.target_features),
            default=0
        )
        max_overall_importance = max(abs(v) for v in feature_importance.values())
        
        if max_overall_importance > 0 and max_target_importance >= max_overall_importance * 0.5:
            relevance = min(1.0, relevance * 1.3)
        
        return relevance
    
    def _generate_reasoning(
        self,
        action: Action,
        impact_score: float,
        cost_score: float,
        relevance_score: float,
        features: dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        parts = []
        
        # Impact reasoning
        if impact_score >= 0.7:
            parts.append(f"High expected impact on retention")
        elif impact_score >= 0.4:
            parts.append(f"Moderate expected impact on retention")
        else:
            parts.append(f"Limited but positive impact expected")
        
        # Cost reasoning
        if cost_score <= 0.3:
            parts.append("cost-effective to implement")
        elif cost_score <= 0.6:
            parts.append("reasonable cost relative to benefit")
        else:
            parts.append("higher cost but justified by impact")
        
        # Relevance reasoning
        if relevance_score >= 0.6:
            parts.append(f"directly addresses key churn factors ({', '.join(action.target_features[:2])})")
        elif relevance_score >= 0.3:
            parts.append("addresses relevant factors")
        
        # Feature-specific reasoning
        feature_notes = []
        for target in action.target_features:
            if target in features:
                value = features[target]
                if target == "contract_type" and value == "month-to-month":
                    feature_notes.append("customer is on month-to-month contract")
                elif target == "tech_support" and value == "no":
                    feature_notes.append("customer lacks tech support")
                elif target == "satisfaction_score" and value < 3.5:
                    feature_notes.append(f"satisfaction score is low ({value})")
        
        if feature_notes:
            parts.append(f"given that {feature_notes[0]}")
        
        return ". ".join(parts).capitalize() + "."
    
    def get_decision_summary(
        self,
        recommendations: list[dict[str, Any]],
        churn_probability: float
    ) -> dict[str, Any]:
        """
        Generate summary of recommended decision strategy.
        
        Args:
            recommendations: Scored recommendations
            churn_probability: Current churn probability
            
        Returns:
            Summary with strategy and expected outcomes
        """
        if not recommendations:
            return {
                "strategy": "Monitor",
                "description": "No specific actions recommended at this time.",
                "expected_outcome": "Maintain current engagement",
            }
        
        top = recommendations[0]
        total_expected_reduction = sum(
            r["expected_probability_reduction"] for r in recommendations[:3]
        ) / 2  # Diminishing returns for multiple actions
        
        new_probability = max(0.05, churn_probability - total_expected_reduction)
        
        if churn_probability >= 0.7:
            strategy = "Urgent Intervention"
        elif churn_probability >= 0.5:
            strategy = "Active Retention"
        elif churn_probability >= 0.3:
            strategy = "Preventive Engagement"
        else:
            strategy = "Relationship Building"
        
        return {
            "strategy": strategy,
            "description": f"Primary recommendation: {top['action_name']}. {top['reasoning']}",
            "current_probability": round(churn_probability, 2),
            "expected_new_probability": round(new_probability, 2),
            "expected_reduction_percent": round(
                (churn_probability - new_probability) / churn_probability * 100, 1
            ),
            "recommended_actions_count": len(recommendations),
            "total_estimated_cost": sum(r["cost_score"] for r in recommendations[:3]) / 3,
        }
