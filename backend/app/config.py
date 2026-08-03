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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
