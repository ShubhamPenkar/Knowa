"""Explainability service for SHAP and LIME explanations."""

import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Prediction, Explanation
from app.ml.models import EnsembleModel, get_model
from app.ml.explainers import SHAPExplainer, LIMEExplainer, ConsistencyScorer
from app.ml.pipelines.preprocessing import preprocess_features

settings = get_settings()


class ExplainabilityService:
    """
    Orchestrates explainability workflow:
    1. Generate SHAP explanations
    2. Generate LIME explanations
    3. Calculate consistency score
    4. Store and return results
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.model = None
        self.training_data = None
        self.consistency_scorer = ConsistencyScorer(
            consistency_threshold=settings.explanation_consistency_threshold
        )
        self._load_model_and_data()
    
    def _load_model_and_data(self) -> None:
        """Load model and training data for explainers."""
        model_path = os.path.join(settings.model_path, "ensemble")
        
        if os.path.exists(model_path):
            self.model = EnsembleModel()
            self.model.load(model_path)
        else:
            single_model_path = os.path.join(settings.model_path, f"{settings.default_model}.joblib")
            if os.path.exists(single_model_path):
                self.model = get_model(settings.default_model)
                self.model.load(single_model_path)
        
        # Load training data for LIME
        training_data_path = os.path.join(settings.model_path, "training_data.parquet")
        if os.path.exists(training_data_path):
            self.training_data = pd.read_parquet(training_data_path)
    
    def generate_explanation(self, prediction_id: str) -> dict[str, Any]:
        """
        Generate explanations for a prediction.
        
        Args:
            prediction_id: ID of prediction to explain
            
        Returns:
            Explanation response with SHAP, LIME, and consistency
        """
        if self.model is None or not self.model.is_trained:
            raise ValueError("No trained model available")
        
        # Get prediction
        prediction = self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()
        
        if not prediction:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        # Prepare features
        features = prediction.features_snapshot
        feature_df = preprocess_features(features)
        
        # Get model for SHAP (use primary model from ensemble)
        if isinstance(self.model, EnsembleModel):
            shap_model = self.model.get_primary_model()
        else:
            shap_model = self.model
        
        # Generate SHAP explanations
        shap_explainer = SHAPExplainer(shap_model)
        shap_result = shap_explainer.explain_instance(feature_df)
        
        # Generate LIME explanations
        lime_result = {"explanations": [], "feature_importance": {}}
        if self.training_data is not None:
            lime_explainer = LIMEExplainer(shap_model, self.training_data)
            lime_result = lime_explainer.explain_instance(feature_df)
        
        # Calculate consistency
        consistency_result = self.consistency_scorer.calculate_consistency(
            shap_result["feature_importance"],
            lime_result["feature_importance"]
        )
        
        # Get top factors
        shap_risk, shap_protective = shap_explainer.get_top_factors(feature_df)
        
        # Store explanation
        explanation = Explanation(
            prediction_id=prediction_id,
            shap_values=shap_result["feature_importance"],
            lime_values=lime_result["feature_importance"],
            consistency_score=consistency_result["consistency_score"],
            trust_level=consistency_result["trust_level"],
        )
        
        # Check if explanation already exists
        existing = self.db.query(Explanation).filter(
            Explanation.prediction_id == prediction_id
        ).first()
        
        if existing:
            existing.shap_values = explanation.shap_values
            existing.lime_values = explanation.lime_values
            existing.consistency_score = explanation.consistency_score
            existing.trust_level = explanation.trust_level
        else:
            self.db.add(explanation)
        
        self.db.commit()
        
        return self._format_explanation(
            prediction_id,
            shap_result,
            lime_result,
            consistency_result,
            shap_risk,
            shap_protective
        )
    
    def get_explanation(self, prediction_id: str) -> Optional[dict[str, Any]]:
        """Get stored explanation for prediction."""
        explanation = self.db.query(Explanation).filter(
            Explanation.prediction_id == prediction_id
        ).first()
        
        if not explanation:
            # Try to generate
            try:
                return self.generate_explanation(prediction_id)
            except Exception:
                return None
        
        # Reconstruct response from stored data
        shap_explanations = self._format_feature_explanations(
            explanation.shap_values, "shap"
        )
        lime_explanations = self._format_feature_explanations(
            explanation.lime_values, "lime"
        )
        
        # Get top factors from SHAP
        sorted_shap = sorted(
            explanation.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        risk_factors = [f for f, v in sorted_shap if v > 0][:5]
        protective_factors = [f for f, v in sorted_shap if v < 0][:5]
        
        return {
            "prediction_id": prediction_id,
            "shap_explanations": shap_explanations,
            "lime_explanations": lime_explanations,
            "consistency_score": explanation.consistency_score,
            "trust_level": explanation.trust_level,
            "top_risk_factors": risk_factors,
            "top_protective_factors": protective_factors,
        }
    
    def _format_explanation(
        self,
        prediction_id: str,
        shap_result: dict,
        lime_result: dict,
        consistency_result: dict,
        risk_factors: list,
        protective_factors: list
    ) -> dict[str, Any]:
        """Format explanation for API response."""
        shap_explanations = [
            {
                "feature": e["feature"],
                "value": e["value"],
                "importance": e["importance"],
                "direction": e["direction"],
                "contribution": e["contribution"],
            }
            for e in shap_result["explanations"]
        ]
        
        lime_explanations = [
            {
                "feature": e["feature"],
                "value": e["value"],
                "importance": e["importance"],
                "direction": e["direction"],
                "contribution": e["contribution"],
            }
            for e in lime_result.get("explanations", [])
        ]
        
        return {
            "prediction_id": prediction_id,
            "shap_explanations": shap_explanations,
            "lime_explanations": lime_explanations,
            "consistency_score": consistency_result["consistency_score"],
            "trust_level": consistency_result["trust_level"],
            "top_risk_factors": risk_factors,
            "top_protective_factors": protective_factors,
        }
    
    def _format_feature_explanations(
        self,
        importance_dict: dict[str, float],
        source: str
    ) -> list[dict[str, Any]]:
        """Format stored importance dict as explanation list."""
        explanations = []
        for feature, importance in sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            explanations.append({
                "feature": feature,
                "value": None,  # Not stored
                "importance": abs(importance),
                "direction": "positive" if importance > 0 else "negative",
                "contribution": "increases_risk" if importance > 0 else "decreases_risk",
            })
        return explanations
