"""Cost calculator for action resource requirements."""

from typing import Any

from app.recommendations.action_catalog import Action


class CostCalculator:
    """
    Calculates cost scores for actions.
    
    Considers multiple cost dimensions:
    - Monetary cost (discounts, free services)
    - Time/effort (staff time, implementation)
    - Opportunity cost
    """
    
    def __init__(
        self,
        cost_weights: dict[str, float] = None,
        budget_constraint: float = None
    ):
        """
        Initialize cost calculator.
        
        Args:
            cost_weights: Weights for different cost types
            budget_constraint: Optional budget limit (0-1 normalized)
        """
        self.cost_weights = cost_weights or {
            "monetary": 0.5,
            "effort": 0.3,
            "time": 0.2,
        }
        self.budget_constraint = budget_constraint
        
        # Effort scores by implementation time
        self.effort_scores = {
            "immediate": 0.1,
            "short": 0.3,
            "medium": 0.5,
            "long": 0.8,
        }
        
        # Time scores (inverse of implementation time value)
        self.time_scores = {
            "immediate": 0.1,
            "short": 0.25,
            "medium": 0.5,
            "long": 0.9,
        }
    
    def calculate_cost(
        self,
        action: Action,
        features: dict[str, Any] = None,
        customer_value: float = None
    ) -> dict[str, Any]:
        """
        Calculate comprehensive cost score for action.
        
        Args:
            action: Action to evaluate
            features: Customer features (for context)
            customer_value: Optional customer lifetime value
            
        Returns:
            Cost metrics including normalized score
        """
        # Base monetary cost from action definition
        monetary_cost = action.base_cost
        
        # Effort cost based on implementation time
        effort_cost = self.effort_scores.get(action.implementation_time, 0.5)
        
        # Time cost (opportunity cost of waiting)
        time_cost = self.time_scores.get(action.implementation_time, 0.5)
        
        # Adjust monetary cost based on customer value
        if customer_value is not None:
            # Higher value customers justify higher costs
            value_factor = min(2.0, 0.5 + (customer_value / 2000))  # Normalize around $2000
            monetary_cost = monetary_cost / value_factor
        
        # Combined cost score
        total_cost = (
            self.cost_weights["monetary"] * monetary_cost +
            self.cost_weights["effort"] * effort_cost +
            self.cost_weights["time"] * time_cost
        )
        
        # Check budget constraint
        within_budget = True
        if self.budget_constraint is not None:
            within_budget = monetary_cost <= self.budget_constraint
        
        return {
            "cost_score": round(total_cost, 4),
            "monetary_cost": round(monetary_cost, 4),
            "effort_cost": round(effort_cost, 4),
            "time_cost": round(time_cost, 4),
            "within_budget": within_budget,
            "implementation_time": action.implementation_time,
        }
    
    def calculate_roi_adjusted_cost(
        self,
        action: Action,
        impact_score: float,
        features: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Calculate cost adjusted for expected ROI.
        
        Lower adjusted cost for high-impact actions.
        """
        base_cost = self.calculate_cost(action, features)
        cost_score = base_cost["cost_score"]
        
        # ROI adjustment: high impact justifies higher cost
        # impact_score of 0.8 with cost_score of 0.5 -> adjusted = 0.3
        roi_factor = max(0.1, 1 - impact_score)
        adjusted_cost = cost_score * roi_factor
        
        return {
            **base_cost,
            "roi_adjusted_cost": round(adjusted_cost, 4),
            "roi_factor": round(roi_factor, 4),
        }
    
    def filter_by_budget(
        self,
        actions: list[Action],
        budget: float
    ) -> list[Action]:
        """Filter actions that fit within budget constraint."""
        return [
            action for action in actions
            if action.base_cost <= budget
        ]
    
    def get_cost_breakdown(self, action: Action) -> dict[str, str]:
        """Get human-readable cost breakdown."""
        cost = self.calculate_cost(action)
        
        def level(score: float) -> str:
            if score < 0.3:
                return "Low"
            elif score < 0.6:
                return "Medium"
            else:
                return "High"
        
        return {
            "overall": level(cost["cost_score"]),
            "monetary": level(cost["monetary_cost"]),
            "effort": level(cost["effort_cost"]),
            "time": level(cost["time_cost"]),
            "implementation": action.implementation_time.replace("_", " ").title(),
        }
