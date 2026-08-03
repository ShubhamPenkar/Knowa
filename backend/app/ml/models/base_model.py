"""Abstract base class for ML models."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, model_name: str, version: str = "1.0"):
        self.model_name = model_name
        self.version = version
        self.model = None
        self.feature_names: list[str] = []
        self.is_trained = False
        self.training_metrics: dict[str, float] = {}
    
    @abstractmethod
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None
    ) -> dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional (X_val, y_val) tuple
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Binary predictions (0 or 1)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Probability of positive class (churn)
        """
        pass
    
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate prediction confidence.
        
        Default implementation based on probability distance from 0.5
        Override for model-specific confidence measures.
        
        Args:
            X: Features
            
        Returns:
            Confidence scores (0 to 1)
        """
        proba = self.predict_proba(X)
        # Confidence = distance from decision boundary (0.5)
        # Scaled to 0-1 where 0.5 -> 0 and 0 or 1 -> 1
        confidence = 2 * np.abs(proba - 0.5)
        return confidence
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass
    
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        
        return {}
    
    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata."""
        return {
            "name": self.model_name,
            "version": self.version,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names),
            "training_metrics": self.training_metrics,
        }
