"""ML Models package."""

from app.ml.models.base_model import BaseModel
from app.ml.models.xgboost_model import XGBoostModel
from app.ml.models.lightgbm_model import LightGBMModel
from app.ml.models.random_forest_model import RandomForestModel
from app.ml.models.logistic_model import LogisticModel
from app.ml.models.ensemble_model import EnsembleModel
from app.ml.models.foundation_model import FoundationModel

__all__ = [
    "BaseModel",
    "XGBoostModel",
    "LightGBMModel",
    "RandomForestModel",
    "LogisticModel",
    "EnsembleModel",
    "FoundationModel",
]


def get_model(model_type: str, version: str = "1.0") -> BaseModel:
    """Factory function to get model by type."""
    models = {
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "random_forest": RandomForestModel,
        "logistic": LogisticModel,
        "ensemble": EnsembleModel,
        "foundation": FoundationModel,
        "foundation_model": FoundationModel,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](version=version)
