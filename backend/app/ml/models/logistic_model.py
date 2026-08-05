"""Logistic/Linear Regression model for baseline and calibration."""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from app.ml.models.base_model import BaseModel


class LogisticModel(BaseModel):
    """Logistic Regression for classification, Ridge for regression baseline."""
    
    def __init__(self, version: str = "1.0", problem_type: str = "binary_classification", **params):
        super().__init__("logistic", version)
        self.problem_type = problem_type
        
        if problem_type == "regression":
            self.default_params = {
                "alpha": 1.0,
                "random_state": 42,
            }
            self.params = {**self.default_params, **params}
            self.model = Ridge(**self.params)
        else:
            self.default_params = {
                "penalty": "l2",
                "C": 0.5,
                "solver": "lbfgs",
                "max_iter": 2000,
                "class_weight": "balanced",  # Handle class imbalance
                "random_state": 42,
                "n_jobs": -1,
            }
            self.params = {**self.default_params, **params}
            self.model = LogisticRegression(**self.params)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None
    ) -> dict[str, float]:
        """Train model."""
        self.feature_names = list(X.columns)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        if self.problem_type == "regression":
            y_pred = self.predict(X)
            self.training_metrics = {
                "mae": mean_absolute_error(y, y_pred),
                "mse": mean_squared_error(y, y_pred),
                "rmse": np.sqrt(mean_squared_error(y, y_pred)),
                "r2_score": r2_score(y, y_pred),
            }
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                self.training_metrics.update({
                    "val_mae": mean_absolute_error(y_val, y_val_pred),
                    "val_mse": mean_squared_error(y_val, y_val_pred),
                    "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
                    "val_r2_score": r2_score(y_val, y_val_pred),
                })
        else:
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)
            self.training_metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1_score": f1_score(y, y_pred, zero_division=0),
                "auc_roc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
            }
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                y_val_proba = self.predict_proba(X_val)
                self.training_metrics.update({
                    "val_accuracy": accuracy_score(y_val, y_val_pred),
                    "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                    "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                    "val_f1_score": f1_score(y_val, y_val_pred, zero_division=0),
                    "val_auc_roc": roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.5,
                })
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability of positive class (classification only)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if self.problem_type == "regression":
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get coefficients as feature importance."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Use absolute coefficients as importance
        if self.problem_type == "regression":
            coefs = np.abs(self.model.coef_)
        else:
            coefs = np.abs(self.model.coef_[0])
        return dict(zip(self.feature_names, coefs))
    
    def save(self, path: str) -> None:
        """Save model and metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "version": self.version,
            "params": self.params,
            "problem_type": self.problem_type,
            "training_metrics": self.training_metrics,
        }
        joblib.dump(save_data, path)
    
    def load(self, path: str) -> None:
        """Load model and metadata."""
        save_data = joblib.load(path)
        
        self.model = save_data["model"]
        self.feature_names = save_data["feature_names"]
        self.version = save_data["version"]
        self.params = save_data["params"]
        self.problem_type = save_data.get("problem_type", "binary_classification")
        self.training_metrics = save_data["training_metrics"]
        self.is_trained = True
