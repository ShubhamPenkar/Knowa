"""LIME explainer for local explanations."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from app.ml.models.base_model import BaseModel


class LIMEExplainer:
    """
    LIME-based explainer for local interpretable explanations.
    
    Creates local linear approximations around individual predictions.
    """
    
    def __init__(
        self, 
        model: BaseModel, 
        training_data: pd.DataFrame,
        categorical_features: Optional[list[int]] = None,
        feature_names: Optional[list[str]] = None
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model to explain
            training_data: Training data for LIME's kernel
            categorical_features: Indices of categorical features
            feature_names: Feature names (uses model's if not provided)
        """
        self.model = model
        self.training_data = training_data
        
        if not self.model.is_trained:
            raise ValueError("Model must be trained before creating explainer")
        
        self.feature_names = feature_names or self.model.feature_names
        self.categorical_features = categorical_features or []
        
        # Identify categorical features by dtype if not provided
        if not self.categorical_features:
            self.categorical_features = [
                i for i, col in enumerate(training_data.columns)
                if training_data[col].dtype == 'object' or training_data[col].dtype.name == 'category'
            ]
        
        # Initialize LIME explainer
        self.explainer = LimeTabularExplainer(
            training_data=training_data.values,
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
            class_names=["No Churn", "Churn"],
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
    
    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Prediction function wrapper for LIME."""
        df = pd.DataFrame(X, columns=self.feature_names)
        proba = self.model.predict_proba(df)
        # Return both classes for LIME
        return np.column_stack([1 - proba, proba])
    
    def explain_instance(
        self, 
        instance: pd.DataFrame,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> dict[str, Any]:
        """
        Get local explanation for a single instance.
        
        Args:
            instance: Single row DataFrame with features
            num_features: Number of top features to include
            num_samples: Number of samples for LIME's local approximation
            
        Returns:
            Dictionary with feature explanations and model fidelity
        """
        # Convert to numpy array
        instance_array = instance.values[0] if len(instance.shape) > 1 else instance.values
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            instance_array,
            self._predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,)  # Explain churn class
        )
        
        # Extract feature contributions for churn class
        lime_exp = explanation.as_list(label=1)
        
        # Build structured explanation
        explanations = []
        feature_values = dict(zip(self.feature_names, instance_array))
        
        for feature_expr, weight in lime_exp:
            # Parse feature name from LIME's expression (e.g., "tenure <= 5.00")
            feature_name = self._parse_feature_name(feature_expr)
            
            if feature_name in feature_values:
                value = feature_values[feature_name]
            else:
                value = feature_expr  # Use expression if can't find exact value
            
            explanations.append({
                "feature": feature_name,
                "expression": feature_expr,
                "value": float(value) if isinstance(value, (int, float, np.number)) else value,
                "importance": float(abs(weight)),
                "lime_weight": float(weight),
                "direction": "positive" if weight > 0 else "negative",
                "contribution": "increases_risk" if weight > 0 else "decreases_risk",
            })
        
        # Sort by absolute importance
        explanations.sort(key=lambda x: x["importance"], reverse=True)
        
        # Get model fidelity (how well LIME's linear model approximates locally)
        local_pred = explanation.local_pred[0] if hasattr(explanation, 'local_pred') else None
        score = explanation.score if hasattr(explanation, 'score') else None
        
        return {
            "explanations": explanations,
            "feature_importance": {e["feature"]: e["importance"] for e in explanations},
            "intercept": float(explanation.intercept[1]) if hasattr(explanation, 'intercept') else 0,
            "local_prediction": float(local_pred) if local_pred is not None else None,
            "model_fidelity": float(score) if score is not None else None,
        }
    
    def _parse_feature_name(self, expression: str) -> str:
        """
        Extract feature name from LIME's expression.
        
        LIME returns expressions like "tenure <= 5.00" or "contract_type=month-to-month"
        """
        # Common operators in LIME expressions
        operators = [" <= ", " < ", " >= ", " > ", " = ", "="]
        
        for op in operators:
            if op in expression:
                return expression.split(op)[0].strip()
        
        return expression
    
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
