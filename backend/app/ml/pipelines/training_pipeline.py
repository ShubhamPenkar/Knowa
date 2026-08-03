"""Training pipeline for model training and evaluation."""

import os
from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from app.ml.models import get_model, EnsembleModel
from app.ml.pipelines.preprocessing import preprocess_dataframe, get_feature_names


class TrainingPipeline:
    """
    End-to-end training pipeline.
    
    Handles:
    - Data splitting
    - Model training
    - Evaluation
    - Model saving
    """
    
    def __init__(
        self,
        model_type: str = "ensemble",
        model_path: str = "./data/models",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize training pipeline.
        
        Args:
            model_type: Type of model to train
            model_path: Directory to save models
            test_size: Fraction for test set
            random_state: Random seed
        """
        self.model_type = model_type
        self.model_path = model_path
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
    
    def train(
        self,
        df: pd.DataFrame,
        target_column: str = "churn"
    ) -> dict[str, Any]:
        """
        Train model on dataset.
        
        Args:
            df: Training data
            target_column: Name of target column
            
        Returns:
            Training results including metrics
        """
        # Preprocess
        X, y, _ = preprocess_dataframe(df, target_column=target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Initialize model
        if self.model_type == "ensemble":
            self.model = EnsembleModel()
        else:
            self.model = get_model(self.model_type)
        
        # Train
        metrics = self.model.train(
            X_train, y_train,
            validation_data=(X_test, y_test)
        )
        
        # Save model
        self._save_model()
        
        # Save training data for LIME
        self._save_training_data(X_train)
        
        return {
            "model_type": self.model_type,
            "version": self.model.version,
            "metrics": metrics,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_names": get_feature_names(),
        }
    
    def _save_model(self) -> None:
        """Save trained model."""
        os.makedirs(self.model_path, exist_ok=True)
        
        if self.model_type == "ensemble":
            save_path = os.path.join(self.model_path, "ensemble")
        else:
            save_path = os.path.join(self.model_path, f"{self.model_type}.joblib")
        
        self.model.save(save_path)
    
    def _save_training_data(self, X_train: pd.DataFrame) -> None:
        """Save training data for LIME explainer."""
        save_path = os.path.join(self.model_path, "training_data.parquet")
        X_train.to_parquet(save_path)
    
    def evaluate(
        self,
        df: pd.DataFrame,
        target_column: str = "churn"
    ) -> dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            df: Evaluation data
            target_column: Name of target column
            
        Returns:
            Evaluation metrics
        """
        if self.model is None or not self.model.is_trained:
            raise ValueError("Model must be trained first")
        
        X, y, _ = preprocess_dataframe(df, target_column=target_column)
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )
        
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_proba),
            "sample_size": len(y),
        }
