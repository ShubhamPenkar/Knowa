"""Simulation service for what-if analysis."""

import os
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Prediction
from app.ml.models import EnsembleModel, get_model
from app.ml.explainers import SHAPExplainer
from app.ml.pipelines.preprocessing import preprocess_features

settings = get_settings()


class SimulationService:
    """
    Handles what-if simulation:
    1. Modify features
    2. Re-run prediction
    3. Compare results
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model."""
        model_path = os.path.join(settings.model_path, "ensemble")
        
        if os.path.exists(model_path):
            self.model = EnsembleModel()
            self.model.load(model_path)
        else:
            single_model_path = os.path.join(settings.model_path, f"{settings.default_model}.joblib")
            if os.path.exists(single_model_path):
                self.model = get_model(settings.default_model)
                self.model.load(single_model_path)
    
    def simulate(
        self,
        base_features: dict[str, Any],
        modified_features: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Run what-if simulation.
        
        Args:
            base_features: Original feature values
            modified_features: Features to change
            
        Returns:
            Simulation comparison results
        """
        if self.model is None or not self.model.is_trained:
            raise ValueError("No trained model available")
        
        # Validate modified features
        invalid_features = set(modified_features.keys()) - set(base_features.keys())
        if invalid_features:
            raise ValueError(f"Invalid features: {invalid_features}")
        
        # Create modified feature set
        modified_full = {**base_features, **modified_features}
        
        # Prepare DataFrames
        base_df = preprocess_features(base_features)
        modified_df = preprocess_features(modified_full)
        
        # Get predictions
        original_prob = float(self.model.predict_proba(base_df)[0])
        modified_prob = float(self.model.predict_proba(modified_df)[0])
        
        # Calculate change
        prob_change = modified_prob - original_prob
        prob_change_percent = (prob_change / original_prob * 100) if original_prob > 0 else 0
        
        # Determine risk level change
        if prob_change < -0.1:
            risk_change = "improved"
        elif prob_change > 0.1:
            risk_change = "worsened"
        else:
            risk_change = "unchanged"
        
        # Get feature importance for comparison
        feature_comparisons = self._compare_feature_impacts(
            base_features,
            modified_full,
            base_df,
            modified_df
        )
        
        # Identify key changes
        key_changes = self._identify_key_changes(
            modified_features,
            base_features,
            prob_change
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            prob_change,
            modified_features
        )
        
        return {
            "original_probability": round(original_prob, 4),
            "modified_probability": round(modified_prob, 4),
            "probability_change": round(prob_change, 4),
            "probability_change_percent": round(prob_change_percent, 1),
            "risk_level_change": risk_change,
            "feature_comparisons": feature_comparisons,
            "key_changes": key_changes,
            "recommendation": recommendation,
        }
    
    def simulate_from_prediction(
        self,
        prediction_id: str,
        modified_features: dict[str, Any]
    ) -> dict[str, Any]:
        """Run simulation based on existing prediction."""
        prediction = self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()
        
        if not prediction:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        return self.simulate(
            base_features=prediction.features_snapshot,
            modified_features=modified_features
        )
    
    def _compare_feature_impacts(
        self,
        base_features: dict,
        modified_features: dict,
        base_df: pd.DataFrame,
        modified_df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Compare feature importance before/after."""
        comparisons = []
        
        try:
            # Get SHAP explainer
            if isinstance(self.model, EnsembleModel):
                shap_model = self.model.get_primary_model()
            else:
                shap_model = self.model
            
            explainer = SHAPExplainer(shap_model)
            
            base_exp = explainer.explain_instance(base_df)
            modified_exp = explainer.explain_instance(modified_df)
            
            base_importance = base_exp["feature_importance"]
            modified_importance = modified_exp["feature_importance"]
            
            # Compare changed features
            for feature in modified_features.keys():
                if feature in base_features:
                    comparisons.append({
                        "feature": feature,
                        "original_value": base_features[feature],
                        "modified_value": modified_features[feature],
                        "original_importance": base_importance.get(feature, 0),
                        "modified_importance": modified_importance.get(feature, 0),
                        "impact_change": modified_importance.get(feature, 0) - base_importance.get(feature, 0),
                    })
        except Exception:
            # Fallback without SHAP
            for feature in modified_features.keys():
                if feature in base_features:
                    comparisons.append({
                        "feature": feature,
                        "original_value": base_features[feature],
                        "modified_value": modified_features[feature],
                        "original_importance": 0,
                        "modified_importance": 0,
                        "impact_change": 0,
                    })
        
        return comparisons
    
    def _identify_key_changes(
        self,
        modified_features: dict,
        base_features: dict,
        prob_change: float
    ) -> list[str]:
        """Identify key insights from the changes."""
        changes = []
        
        for feature, new_value in modified_features.items():
            old_value = base_features.get(feature)
            
            if feature == "contract_type":
                if old_value == "month-to-month" and new_value in ["one_year", "two_year"]:
                    changes.append(f"Upgrading to {new_value.replace('_', ' ')} contract significantly reduces risk")
            
            elif feature == "tenure":
                if new_value > old_value:
                    changes.append(f"Increased tenure from {old_value} to {new_value} months improves loyalty")
            
            elif feature == "monthly_charges":
                if new_value < old_value:
                    reduction = round((old_value - new_value) / old_value * 100, 1)
                    changes.append(f"Reducing monthly charges by {reduction}% decreases price sensitivity")
            
            elif feature == "tech_support" and new_value == "yes" and old_value == "no":
                changes.append("Adding tech support improves service satisfaction")
            
            elif feature == "satisfaction_score":
                if new_value > old_value:
                    changes.append(f"Improving satisfaction from {old_value} to {new_value} reduces churn risk")
        
        if prob_change < -0.15:
            changes.append(f"These changes could reduce churn probability by {abs(round(prob_change * 100))}%")
        elif prob_change > 0.15:
            changes.append(f"Warning: These changes may increase churn risk by {round(prob_change * 100)}%")
        
        return changes
    
    def _generate_recommendation(
        self,
        prob_change: float,
        modified_features: dict
    ) -> str:
        """Generate actionable recommendation."""
        if prob_change < -0.2:
            return "Highly recommended changes - implement these modifications to significantly reduce churn risk."
        elif prob_change < -0.1:
            return "Recommended changes - these modifications would meaningfully improve retention."
        elif prob_change < -0.05:
            return "Beneficial changes - modest improvement in retention expected."
        elif prob_change < 0.05:
            return "Neutral impact - these changes have minimal effect on churn probability."
        else:
            return "Not recommended - these changes may increase churn risk. Consider alternative approaches."
