"""Ensemble model for robust predictions and confidence scoring."""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from app.ml.models.base_model import BaseModel
from app.ml.models.xgboost_model import XGBoostModel
from app.ml.models.lightgbm_model import LightGBMModel
from app.ml.models.random_forest_model import RandomForestModel
from app.ml.models.logistic_model import LogisticModel


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models for robust prediction and confidence scoring.
    
    Uses model agreement as confidence measure - high agreement = high confidence.
    Supports both classification and regression.
    """
    
    def __init__(self, version: str = "1.0", problem_type: str = "binary_classification", weights: Optional[dict[str, float]] = None):
        super().__init__("ensemble", version)
        self.problem_type = problem_type
        
        # Default weights for each model
        self.weights = weights or {
            "xgboost": 0.35,
            "lightgbm": 0.35,
            "random_forest": 0.20,
            "logistic": 0.10,
        }
        
        # Initialize component models with problem type
        self.models = {
            "xgboost": XGBoostModel(version=version, problem_type=problem_type),
            "lightgbm": LightGBMModel(version=version, problem_type=problem_type),
            "random_forest": RandomForestModel(version=version, problem_type=problem_type),
            "logistic": LogisticModel(version=version, problem_type=problem_type),
        }
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None
    ) -> dict[str, float]:
        """Train all component models."""
        self.feature_names = list(X.columns)
        
        all_metrics = {}
        
        for name, model in self.models.items():
            metrics = model.train(X, y, validation_data)
            # Prefix metrics with model name
            for key, value in metrics.items():
                all_metrics[f"{name}_{key}"] = value
        
        self.is_trained = True
        
        # Calculate ensemble metrics based on problem type
        if self.problem_type == "regression":
            y_pred = self.predict(X)
            
            self.training_metrics = {
                "ensemble_mae": mean_absolute_error(y, y_pred),
                "ensemble_mse": mean_squared_error(y, y_pred),
                "ensemble_rmse": np.sqrt(mean_squared_error(y, y_pred)),
                "ensemble_r2_score": r2_score(y, y_pred),
                **all_metrics,
            }
            
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                
                self.training_metrics.update({
                    "ensemble_val_mae": mean_absolute_error(y_val, y_val_pred),
                    "ensemble_val_mse": mean_squared_error(y_val, y_val_pred),
                    "ensemble_val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
                    "ensemble_val_r2_score": r2_score(y_val, y_val_pred),
                })
        else:
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)
            
            self.training_metrics = {
                "ensemble_accuracy": accuracy_score(y, y_pred),
                "ensemble_precision": precision_score(y, y_pred, zero_division=0),
                "ensemble_recall": recall_score(y, y_pred, zero_division=0),
                "ensemble_f1_score": f1_score(y, y_pred, zero_division=0),
                "ensemble_auc_roc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
                **all_metrics,
            }
            
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                y_val_proba = self.predict_proba(X_val)
                
                self.training_metrics.update({
                    "ensemble_val_accuracy": accuracy_score(y_val, y_val_pred),
                    "ensemble_val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                    "ensemble_val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                    "ensemble_val_f1_score": f1_score(y_val, y_val_pred, zero_division=0),
                    "ensemble_val_auc_roc": roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.5,
                })
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.problem_type == "regression":
            # For regression, weighted average of predictions
            weighted_sum = np.zeros(len(X))
            for name, model in self.models.items():
                weight = self.weights.get(name, 0)
                weighted_sum += weight * model.predict(X)
            return weighted_sum
        else:
            # For classification, threshold probability
            proba = self.predict_proba(X)
            return (proba >= 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get weighted ensemble probability (classification) or prediction (regression)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        weighted_sum = np.zeros(len(X))
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 0)
            weighted_sum += weight * model.predict_proba(X)
        
        return weighted_sum
    
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate confidence based on model agreement.
        
        High agreement among models = high confidence.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get predictions from all models
        if self.problem_type == "regression":
            predictions = np.array([
                model.predict(X) for model in self.models.values()
            ])
            # For regression, use coefficient of variation
            mean_pred = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
            cv = np.where(mean_pred != 0, std / np.abs(mean_pred), std)
            # Convert to confidence: low cv = high confidence
            confidence = 1 / (1 + cv)
        else:
            predictions = np.array([
                model.predict_proba(X) for model in self.models.values()
            ])
            # Standard deviation across models - lower std = higher agreement
            std = np.std(predictions, axis=0)
            # Convert to confidence: 0 std -> 1.0 confidence, high std -> low confidence
            confidence = 1 - (std / 0.5)
            confidence = np.clip(confidence, 0, 1)
        
        return confidence
    
    def get_model_predictions(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Get individual model predictions for analysis."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.problem_type == "regression":
            return {
                name: model.predict(X)
                for name, model in self.models.items()
            }
        else:
            return {
                name: model.predict_proba(X)
                for name, model in self.models.items()
            }
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get weighted average feature importance across models."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importances = {}
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 0)
            model_importance = model.get_feature_importance()
            
            for feature, imp in model_importance.items():
                if feature not in importances:
                    importances[feature] = 0
                importances[feature] += weight * imp
        
        return importances
    
    def get_primary_model(self) -> BaseModel:
        """Get the primary (highest weight) model for SHAP explanations."""
        primary = max(self.weights.items(), key=lambda x: x[1])
        return self.models[primary[0]]
    
    def save(self, path: str) -> None:
        """Save all models and ensemble metadata."""
        os.makedirs(path, exist_ok=True)
        
        # Save each component model
        for name, model in self.models.items():
            model.save(os.path.join(path, f"{name}.joblib"))
        
        # Save ensemble metadata
        metadata = {
            "weights": self.weights,
            "version": self.version,
            "problem_type": self.problem_type,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
        }
        joblib.dump(metadata, os.path.join(path, "ensemble_meta.joblib"))
    
    def load(self, path: str) -> None:
        """Load all models and ensemble metadata."""
        # Load ensemble metadata first to get problem_type
        metadata = joblib.load(os.path.join(path, "ensemble_meta.joblib"))
        
        self.weights = metadata["weights"]
        self.version = metadata["version"]
        self.problem_type = metadata.get("problem_type", "binary_classification")
        self.feature_names = metadata["feature_names"]
        self.training_metrics = metadata["training_metrics"]
        
        # Re-initialize models with correct problem_type then load
        self.models = {
            "xgboost": XGBoostModel(version=self.version, problem_type=self.problem_type),
            "lightgbm": LightGBMModel(version=self.version, problem_type=self.problem_type),
            "random_forest": RandomForestModel(version=self.version, problem_type=self.problem_type),
            "logistic": LogisticModel(version=self.version, problem_type=self.problem_type),
        }
        
        # Load each component model
        for name, model in self.models.items():
            model.load(os.path.join(path, f"{name}.joblib"))
        
        self.is_trained = True
