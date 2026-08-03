"""Prediction pipeline for inference."""

import os
from typing import Any, Optional

import pandas as pd

from app.ml.models import get_model, EnsembleModel
from app.ml.pipelines.preprocessing import preprocess_features


class PredictionPipeline:
    """
    Inference pipeline for making predictions.
    """
    
    def __init__(
        self,
        model_path: str = "./data/models",
        model_type: str = "ensemble"
    ):
        """
        Initialize prediction pipeline.
        
        Args:
            model_path: Directory with saved models
            model_type: Type of model to load
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model."""
        if self.model_type == "ensemble":
            ensemble_path = os.path.join(self.model_path, "ensemble")
            if os.path.exists(ensemble_path):
                self.model = EnsembleModel()
                self.model.load(ensemble_path)
        else:
            model_file = os.path.join(self.model_path, f"{self.model_type}.joblib")
            if os.path.exists(model_file):
                self.model = get_model(self.model_type)
                self.model.load(model_file)
    
    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Make prediction for single instance.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Prediction results
        """
        if self.model is None or not self.model.is_trained:
            raise ValueError("No trained model available")
        
        # Preprocess
        X = preprocess_features(features)
        
        # Predict
        probability = float(self.model.predict_proba(X)[0])
        confidence = float(self.model.get_confidence(X)[0])
        prediction = int(probability >= 0.5)
        
        return {
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "risk_level": self._get_risk_level(probability),
        }
    
    def predict_batch(
        self,
        features_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Make predictions for multiple instances.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        return [self.predict(features) for features in features_list]
    
    def _get_risk_level(self, probability: float) -> str:
        """Map probability to risk level."""
        if probability >= 0.8:
            return "critical"
        elif probability >= 0.6:
            return "high"
        elif probability >= 0.4:
            return "medium"
        else:
            return "low"
