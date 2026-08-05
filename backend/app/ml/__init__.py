"""ML package."""

from app.ml.models import (
    BaseModel,
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    LogisticModel,
    EnsembleModel,
    FoundationModel,
    get_model,
)
from app.ml.explainers import (
    SHAPExplainer,
    LIMEExplainer,
    ConsistencyScorer,
)
from app.ml.calibration import ConformalCalibrator, UncertaintyResult
from app.ml.router import RoutingDecision, route_training
from app.ml.model_loader import load_routed_model, build_model_for_strategy

__all__ = [
    # Models
    "BaseModel",
    "XGBoostModel",
    "LightGBMModel",
    "RandomForestModel",
    "LogisticModel",
    "EnsembleModel",
    "FoundationModel",
    "get_model",
    # Explainers
    "SHAPExplainer",
    "LIMEExplainer",
    "ConsistencyScorer",
    # Calibration
    "ConformalCalibrator",
    "UncertaintyResult",
    # Routing
    "RoutingDecision",
    "route_training",
    "load_routed_model",
    "build_model_for_strategy",
]
