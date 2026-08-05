"""Cost calculator for action resource requirements."""

from __future__ import annotations

from typing import Any, Optional

from app.recommendations.action_catalog import Action


class CostCalculator:
    """Multi-dimension cost score in [0, 1] (higher = more expensive/heavier)."""

    def __init__(
        self,
        cost_weights: Optional[dict[str, float]] = None,
        budget_constraint: Optional[float] = None,
    ):
        self.cost_weights = cost_weights or {
            "monetary": 0.5,
            "effort": 0.3,
            "time": 0.2,
        }
        self.budget_constraint = budget_constraint
        self.effort_scores = {
            "immediate": 0.1,
            "short": 0.3,
            "medium": 0.5,
            "long": 0.8,
        }
        self.time_scores = {
            "immediate": 0.1,
            "short": 0.25,
            "medium": 0.5,
            "long": 0.9,
        }

    def calculate_cost(
        self,
        action: Action,
        features: Optional[dict[str, Any]] = None,
        customer_value: Optional[float] = None,
    ) -> dict[str, Any]:
        monetary_cost = float(action.base_cost)
        effort_cost = self.effort_scores.get(action.implementation_time, 0.5)
        time_cost = self.time_scores.get(action.implementation_time, 0.5)

        if customer_value is not None and customer_value > 0:
            # Higher LTV justifies paying more (lowers effective monetary cost)
            value_factor = min(2.0, 0.5 + (float(customer_value) / 2000.0))
            monetary_cost = monetary_cost / value_factor

        total_cost = (
            self.cost_weights["monetary"] * monetary_cost
            + self.cost_weights["effort"] * effort_cost
            + self.cost_weights["time"] * time_cost
        )
        total_cost = min(1.0, max(0.0, total_cost))

        within_budget = True
        if self.budget_constraint is not None:
            within_budget = monetary_cost <= self.budget_constraint

        level = (
            "low" if total_cost < 0.3 else "medium" if total_cost < 0.6 else "high"
        )

        return {
            "cost_score": round(total_cost, 4),
            "monetary_cost": round(monetary_cost, 4),
            "effort_cost": round(effort_cost, 4),
            "time_cost": round(time_cost, 4),
            "within_budget": within_budget,
            "implementation_time": action.implementation_time,
            "cost_label": level,
        }
