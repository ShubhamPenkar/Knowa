"""Training pipeline for model training and evaluation."""

import os
from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from app.ml.models import get_model
from app.ml.model_loader import build_model_for_strategy, write_route_meta
from app.ml.router import route_training
from app.ml.pipelines.preprocessing import preprocess_dataframe, get_feature_names
from app.config import get_settings


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
        model_type: str = "auto",
        model_path: str = "./data/models",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize training pipeline.
        
        Args:
            model_type: Type of model to train ("auto" uses Phase 1b router)
            model_path: Directory to save models
            test_size: Fraction for test set
            random_state: Random seed
        """
        self.model_type = model_type
        self.model_path = model_path
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.routing_decision = None
    
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
        settings = get_settings()

        # Preprocess
        X, y, _ = preprocess_dataframe(df, target_column=target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Initialize model (auto → router; ensemble/foundation explicit)
        if self.model_type in ("auto", "routed"):
            force = None
            if settings.routing_mode in ("foundation_model", "ensemble"):
                force = settings.routing_mode
            self.routing_decision = route_training(
                X_train,
                max_foundation_rows=settings.foundation_max_rows,
                max_foundation_features=settings.foundation_max_features,
                force_strategy=force,
            )
            resolved = self.routing_decision.strategy
            self.model = build_model_for_strategy(resolved)
            self.model_type = resolved
        elif self.model_type in ("ensemble", "foundation", "foundation_model"):
            strategy = (
                "foundation_model"
                if self.model_type in ("foundation", "foundation_model")
                else "ensemble"
            )
            self.model = build_model_for_strategy(strategy)
            self.model_type = strategy
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
            "routing": (
                self.routing_decision.to_dict() if self.routing_decision else None
            ),
        }
    
    def _save_model(self) -> None:
        """Save trained model."""
        os.makedirs(self.model_path, exist_ok=True)
        
        if self.model_type in ("ensemble", "foundation_model", "foundation"):
            dirname = "foundation" if self.model_type in ("foundation", "foundation_model") else "ensemble"
            save_path = os.path.join(self.model_path, dirname)
            self.model.save(save_path)
            if self.routing_decision:
                write_route_meta(
                    save_path,
                    self.routing_decision.strategy,
                    reason=self.routing_decision.reason,
                    extra=self.routing_decision.to_dict(),
                )
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
