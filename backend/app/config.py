"""Application configuration management."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Decision Intelligence Platform"
    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    
    # Database
    database_url: str = "sqlite:///./data/decision_intelligence.db"
    database_echo: bool = False
    
    # Data paths
    data_path: str = "./data"
    model_path: str = "./data/models"
    default_model: str = "ensemble"
    confidence_threshold: float = 0.7
    explanation_consistency_threshold: float = 0.7

    # Conformal prediction / abstention (Phase 1a)
    conformal_alpha: float = 0.1  # 1 - alpha = target coverage (0.9)
    disagreement_threshold: float = 0.25  # base-model proba/prediction std
    # Classification residual intervals are often 0.4–0.8 wide; use a high bar
    # so "Low confidence" means genuinely uninformative, not typical width.
    interval_width_threshold: float = 0.85
    stacking_n_folds: int = 5

    # Phase 1.5 quality training
    test_size: float = 0.2
    calib_size: float = 0.2  # of remaining train pool → ~16% overall when test=0.2
    enable_optuna: bool = True
    optuna_trials: int = 12
    optuna_timeout_seconds: float = 90.0
    probability_calibration: Literal["isotonic", "sigmoid", "none"] = "isotonic"
    early_stopping_rounds: int = 40
    drop_leakage_columns: bool = True
    add_missing_indicators: bool = True

    # Model routing (Phase 1b)
    # auto: router inspects n_rows / n_features; force_* overrides for debugging
    routing_mode: Literal["auto", "foundation_model", "ensemble"] = "auto"
    foundation_max_rows: int = 10_000
    foundation_max_features: int = 500
    prefer_tabpfn: bool = True

    # Recommendation Settings
    impact_weight: float = 0.5
    cost_weight: float = 0.3
    relevance_weight: float = 0.2
    top_recommendations: int = 5

    # Feedback Settings
    feedback_window_days: int = 90
    retrain_threshold_samples: int = 100
    accuracy_drop_threshold: float = 0.05

    # API Settings
    api_prefix: str = "/api"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Auth
    jwt_secret: str = "dev-secret-change-in-production"

    # Celery / scheduled B3 rechecks
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_recheck_interval_minutes: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
