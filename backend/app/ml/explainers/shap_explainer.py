"""SHAP explainer for global and local explanations."""

from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from app.ml.models.base_model import BaseModel


class SHAPExplainer:
    """
    SHAP-based explainer for tree models.
    
    Provides both global (dataset-wide) and local (instance-specific) explanations.
    """
    
    def __init__(self, model: BaseModel, background_data: Optional[pd.DataFrame] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain
            background_data: Sample data for SHAP background (for KernelSHAP fallback)
        """
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self) -> None:
        """Initialize appropriate SHAP explainer based on model type."""
        if not self.model.is_trained:
            raise ValueError("Model must be trained before creating explainer")
        
        model_name = self.model.model_name.lower()
        
        if model_name in ["xgboost", "lightgbm", "random_forest"]:
            # TreeExplainer is fast and exact for tree-based models
            self.explainer = shap.TreeExplainer(self.model.model)
        else:
            # Fallback to KernelSHAP for other models
            if self.background_data is None:
                raise ValueError("Background data required for non-tree models")
            
            # Sample background data if too large
            if len(self.background_data) > 100:
                background = shap.sample(self.background_data, 100)
            else:
                background = self.background_data
            
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
    
    def explain_instance(self, instance: pd.DataFrame) -> dict[str, Any]:
        """
        Get local explanation for a single instance.
        
        Args:
            instance: Single row DataFrame with features
            
        Returns:
            Dictionary with feature names, values, SHAP values, and directions
        """
        # Get SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification returns list [class_0, class_1]
            shap_values = shap_values[1]  # Use class 1 (churn) values
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Get first instance
        
        # Get base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.5
        
        feature_names = self.model.feature_names
        feature_values = instance.iloc[0].values if len(instance) > 0 else instance.values
        
        # Build explanation
        explanations = []
        for i, (name, value, shap_val) in enumerate(zip(feature_names, feature_values, shap_values)):
            explanations.append({
                "feature": name,
                "value": float(value) if isinstance(value, (int, float, np.number)) else value,
                "importance": float(abs(shap_val)),
                "shap_value": float(shap_val),
                "direction": "positive" if shap_val > 0 else "negative",
                "contribution": "increases_risk" if shap_val > 0 else "decreases_risk",
            })
        
        # Sort by absolute importance
        explanations.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "base_value": float(base_value),
            "explanations": explanations,
            "feature_importance": {e["feature"]: e["importance"] for e in explanations},
        }
    
    def explain_global(self, data: pd.DataFrame, max_samples: int = 500) -> dict[str, Any]:
        """
        Get global feature importance across dataset.
        
        Args:
            data: Dataset to explain
            max_samples: Maximum samples to use (for performance)
            
        Returns:
            Dictionary with global feature importance
        """
        # Sample if needed
        if len(data) > max_samples:
            sample_data = data.sample(n=max_samples, random_state=42)
        else:
            sample_data = data
        
        # Get SHAP values for all samples
        shap_values = self.explainer.shap_values(sample_data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        feature_names = self.model.feature_names
        
        # Build global importance
        global_importance = {}
        for name, importance in zip(feature_names, mean_abs_shap):
            global_importance[name] = float(importance)
        
        # Sort by importance
        sorted_importance = dict(
            sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "feature_importance": sorted_importance,
            "sample_size": len(sample_data),
            "shap_values_summary": {
                "mean": float(np.mean(shap_values)),
                "std": float(np.std(shap_values)),
            }
        }
    
    def get_top_factors(
        self, 
        instance: pd.DataFrame, 
        n: int = 5
    ) -> tuple[list[str], list[str]]:
        """
        Get top risk and protective factors.
        
        Args:
            instance: Single instance to explain
            n: Number of top factors to return
            
        Returns:
            Tuple of (risk_factors, protective_factors) as feature names
        """
        explanation = self.explain_instance(instance)
        explanations = explanation["explanations"]
        
        risk_factors = [
            e["feature"] for e in explanations 
            if e["direction"] == "positive"
        ][:n]
        
        protective_factors = [
            e["feature"] for e in explanations 
            if e["direction"] == "negative"
        ][:n]
        
        return risk_factors, protective_factors
