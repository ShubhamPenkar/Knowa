"""Recommendation service for action scoring and ranking."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Prediction, Explanation, Recommendation, ActionCatalog
from app.recommendations import DecisionScorer, get_action, get_all_actions

settings = get_settings()


class RecommendationService:
    """
    Generates scored and ranked action recommendations.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.scorer = DecisionScorer(
            impact_weight=settings.impact_weight,
            cost_weight=settings.cost_weight,
            relevance_weight=settings.relevance_weight,
        )
        self._ensure_action_catalog()
    
    def _ensure_action_catalog(self) -> None:
        """Ensure action catalog is populated in database."""
        existing = self.db.query(ActionCatalog).count()
        if existing == 0:
            # Populate from code catalog
            for action in get_all_actions():
                catalog_entry = ActionCatalog(
                    action_code=action.code,
                    action_name=action.name,
                    description=action.description,
                    base_cost=action.base_cost,
                    applicable_conditions=action.applicable_conditions,
                )
                self.db.add(catalog_entry)
            self.db.commit()
    
    def generate_recommendations(
        self,
        prediction_id: str,
        top_n: int = 5
    ) -> dict[str, Any]:
        """
        Generate recommendations for a prediction.
        
        Args:
            prediction_id: ID of prediction
            top_n: Number of top recommendations
            
        Returns:
            Recommendation response with ranked actions
        """
        # Get prediction
        prediction = self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()
        
        if not prediction:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        # Get explanation for feature importance
        explanation = self.db.query(Explanation).filter(
            Explanation.prediction_id == prediction_id
        ).first()
        
        feature_importance = explanation.shap_values if explanation else None
        features = prediction.features_snapshot
        
        # Calculate customer value (simple estimate)
        customer_value = features.get("total_charges", 0)
        
        # Score all applicable actions
        scored_actions = self.scorer.score_all_applicable(
            features=features,
            churn_probability=prediction.churn_probability,
            feature_importance=feature_importance,
            customer_value=customer_value,
            top_n=top_n
        )
        
        # Store recommendations
        self._store_recommendations(prediction_id, scored_actions)
        
        return {
            "prediction_id": prediction_id,
            "current_churn_probability": prediction.churn_probability,
            "recommendations": scored_actions,
            "generated_at": datetime.utcnow(),
        }
    
    def get_recommendations(
        self,
        prediction_id: str,
        top_n: int = 5
    ) -> Optional[dict[str, Any]]:
        """Get stored recommendations for prediction."""
        recommendations = (
            self.db.query(Recommendation)
            .filter(Recommendation.prediction_id == prediction_id)
            .order_by(Recommendation.rank)
            .limit(top_n)
            .all()
        )
        
        if not recommendations:
            # Try to generate
            try:
                return self.generate_recommendations(prediction_id, top_n)
            except Exception:
                return None
        
        # Get prediction for probability
        prediction = self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()
        
        # Format stored recommendations
        formatted = []
        for rec in recommendations:
            action = self.db.query(ActionCatalog).filter(
                ActionCatalog.id == rec.action_id
            ).first()
            
            formatted.append({
                "action_code": action.action_code if action else "unknown",
                "action_name": action.action_name if action else "Unknown Action",
                "description": action.description if action else "",
                "impact_score": rec.impact_score,
                "cost_score": rec.cost_score,
                "relevance_score": rec.relevance_score,
                "final_score": rec.final_score,
                "rank": rec.rank,
                "reasoning": rec.reasoning,
                "expected_probability_reduction": 0.0,  # Not stored
            })
        
        return {
            "prediction_id": prediction_id,
            "current_churn_probability": prediction.churn_probability if prediction else 0,
            "recommendations": formatted,
            "generated_at": datetime.utcnow(),
        }
    
    def _store_recommendations(
        self,
        prediction_id: str,
        recommendations: list[dict[str, Any]]
    ) -> None:
        """Store recommendations in database."""
        # Clear existing
        self.db.query(Recommendation).filter(
            Recommendation.prediction_id == prediction_id
        ).delete()
        
        for rec in recommendations:
            # Get or create action catalog entry
            action_catalog = self.db.query(ActionCatalog).filter(
                ActionCatalog.action_code == rec["action_code"]
            ).first()
            
            if not action_catalog:
                action_def = get_action(rec["action_code"])
                if action_def:
                    action_catalog = ActionCatalog(
                        action_code=action_def.code,
                        action_name=action_def.name,
                        description=action_def.description,
                        base_cost=action_def.base_cost,
                        applicable_conditions=action_def.applicable_conditions,
                    )
                    self.db.add(action_catalog)
                    self.db.flush()
            
            if action_catalog:
                recommendation = Recommendation(
                    prediction_id=prediction_id,
                    action_id=action_catalog.id,
                    impact_score=rec["impact_score"],
                    cost_score=rec["cost_score"],
                    relevance_score=rec["relevance_score"],
                    final_score=rec["final_score"],
                    rank=rec["rank"],
                    reasoning=rec["reasoning"],
                )
                self.db.add(recommendation)
        
        self.db.commit()
