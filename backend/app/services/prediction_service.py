"""Prediction service orchestrating the prediction pipeline."""

import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Customer, Prediction
from app.ml.models import get_model, EnsembleModel
from app.ml.pipelines.preprocessing import preprocess_features

settings = get_settings()


class PredictionService:
    """
    Orchestrates prediction workflow:
    1. Load/prepare features
    2. Run prediction
    3. Calculate confidence
    4. Store results
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model from disk."""
        model_path = os.path.join(settings.model_path, "ensemble")
        
        if os.path.exists(model_path):
            self.model = EnsembleModel()
            self.model.load(model_path)
        else:
            # Try to load single model
            single_model_path = os.path.join(settings.model_path, f"{settings.default_model}.joblib")
            if os.path.exists(single_model_path):
                self.model = get_model(settings.default_model)
                self.model.load(single_model_path)
            else:
                # Model not trained yet - will fail on predict
                self.model = None
    
    def predict(
        self,
        customer_id: Optional[str] = None,
        features: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Make churn prediction.
        
        Args:
            customer_id: Existing customer ID
            features: Feature dictionary for new/anonymous prediction
            
        Returns:
            Prediction response with probability and confidence
        """
        if self.model is None or not self.model.is_trained:
            raise ValueError("No trained model available. Please train a model first.")
        
        # Get features
        if customer_id:
            customer = self.db.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                raise ValueError(f"Customer {customer_id} not found")
            features = customer.features
        elif features is None:
            raise ValueError("Either customer_id or features must be provided")

        # P0: reject incomplete / empty feature values (demo schema path)
        from app.ml.feature_validation import validate_required_features
        from app.ml.pipelines.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

        required = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
        validate_required_features(features, required)
        
        # Preprocess features
        feature_df = preprocess_features(features)
        
        # Make prediction with conformal uncertainty (Phase 1a)
        probability = float(self.model.predict_proba(feature_df)[0])
        confidence = float(self.model.get_confidence(feature_df)[0])
        confidence_interval = None
        low_confidence = False
        abstention_reason = None

        if hasattr(self.model, "predict_with_uncertainty"):
            uncertainty = self.model.predict_with_uncertainty(feature_df)[0]
            confidence_interval = uncertainty.as_interval_dict()
            low_confidence = uncertainty.low_confidence
            abstention_reason = uncertainty.abstention_reason
            # Prefer agreement-based confidence but dampen when abstaining
            if low_confidence:
                confidence = min(confidence, 0.5)

        risk_level = self._get_risk_level(probability)

        prediction = Prediction(
            customer_id=customer_id,
            model_version=self.model.version,
            churn_probability=probability,
            confidence_score=confidence,
            features_snapshot=features,
        )
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)

        return {
            "id": prediction.id,
            "customer_id": customer_id,
            "churn_probability": probability,
            "churn_risk_level": risk_level,
            "confidence_score": confidence,
            "model_version": self.model.version,
            "prediction_timestamp": prediction.prediction_timestamp,
            "features_used": features,
            "confidence_interval": confidence_interval,
            "low_confidence": low_confidence,
            "abstention_reason": abstention_reason,
        }
    
    def get_prediction(self, prediction_id: str) -> Optional[dict[str, Any]]:
        """Get prediction by ID."""
        prediction = self.db.query(Prediction).filter(Prediction.id == prediction_id).first()
        
        if not prediction:
            return None
        
        return self._format_prediction(prediction)
    
    def get_latest_prediction(self, customer_id: str) -> Optional[dict[str, Any]]:
        """Get most recent prediction for customer."""
        prediction = (
            self.db.query(Prediction)
            .filter(Prediction.customer_id == customer_id)
            .order_by(Prediction.prediction_timestamp.desc())
            .first()
        )
        
        if not prediction:
            return None
        
        return self._format_prediction(prediction)
    
    def get_customer_predictions(
        self,
        customer_id: str,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get prediction history for customer."""
        predictions = (
            self.db.query(Prediction)
            .filter(Prediction.customer_id == customer_id)
            .order_by(Prediction.prediction_timestamp.desc())
            .limit(limit)
            .all()
        )
        
        return [self._format_prediction(p) for p in predictions]
    
    def create_customer(
        self,
        external_id: Optional[str],
        features: dict[str, Any]
    ) -> dict[str, Any]:
        """Create new customer record."""
        if external_id:
            existing = self.db.query(Customer).filter(
                Customer.external_id == external_id
            ).first()
            if existing:
                raise ValueError(f"Customer with external_id {external_id} already exists")
        
        customer = Customer(
            external_id=external_id,
            features=features,
        )
        self.db.add(customer)
        self.db.commit()
        self.db.refresh(customer)
        
        return {
            "id": customer.id,
            "external_id": customer.external_id,
            "features": customer.features,
            "created_at": customer.created_at,
        }
    
    def _format_prediction(self, prediction: Prediction) -> dict[str, Any]:
        """Format prediction for API response."""
        return {
            "id": prediction.id,
            "customer_id": prediction.customer_id,
            "churn_probability": prediction.churn_probability,
            "churn_risk_level": self._get_risk_level(prediction.churn_probability),
            "confidence_score": prediction.confidence_score,
            "model_version": prediction.model_version,
            "prediction_timestamp": prediction.prediction_timestamp,
            "features_used": prediction.features_snapshot,
            "confidence_interval": None,
            "low_confidence": prediction.confidence_score < settings.confidence_threshold,
            "abstention_reason": None,
        }
    
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
