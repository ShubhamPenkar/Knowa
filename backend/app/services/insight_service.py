"""Insight service for generating business insights."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.db.models import Prediction, Explanation, Insight
from app.insights import InsightGenerator


class InsightService:
    """
    Generates business-friendly insights from explanations.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.generator = InsightGenerator()
    
    def generate_insights(self, prediction_id: str) -> dict[str, Any]:
        """
        Generate insights for a prediction.
        
        Args:
            prediction_id: ID of prediction
            
        Returns:
            Insight response with business-friendly text
        """
        # Get prediction and explanation
        prediction = self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()
        
        if not prediction:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        explanation = self.db.query(Explanation).filter(
            Explanation.prediction_id == prediction_id
        ).first()
        
        if not explanation:
            raise ValueError(f"No explanation found for prediction {prediction_id}")
        
        # Build explanations list from SHAP values
        shap_values = explanation.shap_values
        features = prediction.features_snapshot
        
        explanations = []
        for feature, importance in sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            explanations.append({
                "feature": feature,
                "value": features.get(feature),
                "importance": abs(importance),
                "contribution": "increases_risk" if importance > 0 else "decreases_risk",
            })
        
        # Generate insights
        result = self.generator.generate_insights(
            explanations=explanations,
            features=features,
            churn_probability=prediction.churn_probability
        )
        
        # Store insights
        insight_record = self.db.query(Insight).filter(
            Insight.prediction_id == prediction_id
        ).first()
        
        if insight_record:
            insight_record.insights = result["insights"]
            insight_record.generated_at = datetime.utcnow()
        else:
            insight_record = Insight(
                prediction_id=prediction_id,
                insights=result["insights"],
            )
            self.db.add(insight_record)
        
        self.db.commit()
        
        return {
            "prediction_id": prediction_id,
            "insights": result["insights"],
            "summary": result["summary"],
            "generated_at": insight_record.generated_at,
        }
    
    def get_insights(self, prediction_id: str) -> Optional[dict[str, Any]]:
        """Get stored insights for prediction."""
        insight = self.db.query(Insight).filter(
            Insight.prediction_id == prediction_id
        ).first()
        
        if not insight:
            # Try to generate
            try:
                return self.generate_insights(prediction_id)
            except Exception:
                return None
        
        # Get prediction for summary regeneration
        prediction = self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()
        
        # Regenerate summary from stored insights
        risk_factors = [
            i["display_name"] for i in insight.insights
            if i.get("contribution") == "increases_risk"
        ][:3]
        protective_factors = [
            i["display_name"] for i in insight.insights
            if i.get("contribution") == "decreases_risk"
        ][:3]
        
        summary = self._generate_summary(
            prediction.churn_probability if prediction else 0.5,
            risk_factors,
            protective_factors
        )
        
        return {
            "prediction_id": prediction_id,
            "insights": insight.insights,
            "summary": summary,
            "generated_at": insight.generated_at,
        }
    
    def _generate_summary(
        self,
        probability: float,
        risk_factors: list[str],
        protective_factors: list[str]
    ) -> str:
        """Generate summary text."""
        prob_percent = round(probability * 100)
        
        if probability >= 0.8:
            risk_text = f"Critical churn risk ({prob_percent}%)"
        elif probability >= 0.6:
            risk_text = f"High churn risk ({prob_percent}%)"
        elif probability >= 0.4:
            risk_text = f"Moderate churn risk ({prob_percent}%)"
        else:
            risk_text = f"Low churn risk ({prob_percent}%)"
        
        parts = [risk_text + "."]
        
        if risk_factors:
            if len(risk_factors) == 1:
                parts.append(f"Primary driver: {risk_factors[0]}.")
            else:
                parts.append(f"Key drivers: {', '.join(risk_factors)}.")
        
        if protective_factors:
            parts.append(f"Positive factors: {', '.join(protective_factors)}.")
        
        return " ".join(parts)
