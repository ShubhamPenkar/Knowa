"""XGBoost model implementation."""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from app.ml.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model for classification or regression."""
    
    def __init__(self, version: str = "1.0", problem_type: str = "binary_classification", **params):
        super().__init__("xgboost", version)
        self.problem_type = problem_type
        
        if problem_type == "regression":
            self.default_params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "random_state": 42,
                "n_jobs": -1,
            }
            self.params = {**self.default_params, **params}
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.default_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "scale_pos_weight": 1,  # Will be set dynamically for class imbalance
                "random_state": 42,
                "n_jobs": -1,
            }
            self.params = {**self.default_params, **params}
            self.model = xgb.XGBClassifier(**self.params)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None
    ) -> dict[str, float]:
        """Train XGBoost model."""
        self.feature_names = list(X.columns)
        
        # Handle class imbalance for classification
        if self.problem_type != "regression":
            # Calculate scale_pos_weight for imbalanced classes
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            if pos_count > 0 and neg_count > 0:
                scale_pos_weight = neg_count / pos_count
                self.model.set_params(scale_pos_weight=scale_pos_weight)
        
        # Prepare eval set
        eval_set = [(X, y)]
        if validation_data:
            eval_set.append(validation_data)
        
        # Train
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        
        # Calculate metrics based on problem type
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
            # For regression, return predictions directly
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]
    
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
    
    def get_booster(self):
        """Get underlying XGBoost booster for SHAP."""
        return self.model.get_booster() if self.is_trained else None
