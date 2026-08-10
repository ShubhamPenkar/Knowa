"""SQLAlchemy database models."""

import uuid
import secrets
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, Index, LargeBinary
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def generate_uuid() -> str:
    """Generate UUID string."""
    return str(uuid.uuid4())


def generate_api_key() -> str:
    """Generate secure API key."""
    return f"di_{secrets.token_urlsafe(32)}"


# =============================================================================
# MULTI-TENANT SAAS MODELS
# =============================================================================

class Organization(Base):
    """Multi-tenant organization."""
    
    __tablename__ = "organizations"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    industry: Mapped[Optional[str]] = mapped_column(String(50))  # saas, ecommerce, finance, etc.
    plan: Mapped[str] = mapped_column(String(20), default="free")  # free, starter, pro, enterprise
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users: Mapped[list["User"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    api_keys: Mapped[list["APIKey"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    datasets: Mapped[list["Dataset"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    projects: Mapped[list["Project"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    custom_actions: Mapped[list["CustomAction"]] = relationship(back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    """User accounts with organization membership."""
    
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="member")  # owner, admin, member, viewer
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="users")


class APIKey(Base):
    """API keys for programmatic access."""
    
    __tablename__ = "api_keys"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True, default=generate_api_key)
    name: Mapped[str] = mapped_column(String(100), nullable=False)  # "Production", "Development"
    scopes: Mapped[list] = mapped_column(JSON, default=list)  # ["predict", "train", "admin"]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="api_keys")


class Dataset(Base):
    """Uploaded datasets for training."""
    
    __tablename__ = "datasets"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)  # Storage path
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)  # Bytes
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[list] = mapped_column(JSON, nullable=False)  # [{name, dtype, sample_values}]
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="datasets")
    projects: Mapped[list["Project"]] = relationship(back_populates="dataset")


class Project(Base):
    """Prediction project configuration."""
    
    __tablename__ = "projects"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"))
    
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Prediction configuration
    target_column: Mapped[str] = mapped_column(String(100), nullable=False)  # Column to predict
    target_positive_label: Mapped[str] = mapped_column(String(100), default="1")  # What value = positive
    target_description: Mapped[str] = mapped_column(String(200), default="outcome")  # "churn", "conversion"
    problem_type: Mapped[str] = mapped_column(String(50), default="binary_classification")  # binary_classification, regression
    
    # Feature configuration
    feature_columns: Mapped[list] = mapped_column(JSON, nullable=False)  # List of column names to use
    feature_config: Mapped[Optional[dict]] = mapped_column(JSON)  # {column: {type, categories, etc}}
    
    # Model configuration
    model_type: Mapped[str] = mapped_column(String(50), default="ensemble")
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft, training, ready, error (legacy DBs may have "trained")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="projects")
    dataset: Mapped["Dataset"] = relationship(back_populates="projects")
    trained_models: Mapped[list["TrainedModel"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    project_predictions: Mapped[list["ProjectPrediction"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    decisions: Mapped[list["Decision"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class CustomAction(Base):
    """Organization-specific custom actions."""
    
    __tablename__ = "custom_actions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    
    code: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Scoring
    estimated_cost: Mapped[float] = mapped_column(Float, default=0)  # Dollar cost
    estimated_impact: Mapped[float] = mapped_column(Float, default=0.5)  # 0-1 expected impact
    
    # Applicability rules (optional)
    applicable_when: Mapped[Optional[dict]] = mapped_column(JSON)  # {feature: {condition: value}}
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="custom_actions")


class TrainedModel(Base):
    """Trained model artifacts for a project."""
    
    __tablename__ = "trained_models"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), index=True)
    
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Metrics - Classification
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    precision_score: Mapped[Optional[float]] = mapped_column(Float)
    recall_score: Mapped[Optional[float]] = mapped_column(Float)
    f1_score: Mapped[Optional[float]] = mapped_column(Float)
    auc_roc: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metrics - Regression
    mae: Mapped[Optional[float]] = mapped_column(Float)  # Mean Absolute Error
    mse: Mapped[Optional[float]] = mapped_column(Float)  # Mean Squared Error
    rmse: Mapped[Optional[float]] = mapped_column(Float)  # Root Mean Squared Error
    r2_score: Mapped[Optional[float]] = mapped_column(Float)  # R-squared
    
    # Feature importance (for insights)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Training info
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    training_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # Current production model
    
    # Relationships
    project: Mapped["Project"] = relationship(back_populates="trained_models")


class ProjectPrediction(Base):
    """Predictions made through a project."""
    
    __tablename__ = "project_predictions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Input/Output
    entity_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)  # Customer ID, user ID, etc.
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    
    # Classification output
    probability: Mapped[Optional[float]] = mapped_column(Float)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    risk_level: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Regression output
    predicted_value: Mapped[Optional[float]] = mapped_column(Float)
    
    # Explanations (stored inline for SaaS simplicity)
    shap_values: Mapped[Optional[dict]] = mapped_column(JSON)
    top_factors: Mapped[Optional[list]] = mapped_column(JSON)  # [{feature, impact, direction}]
    
    # Recommendations
    recommendations: Mapped[Optional[list]] = mapped_column(JSON)  # [{action, score, reasoning}]
    
    # Feedback
    actual_outcome: Mapped[Optional[str]] = mapped_column(String(50))
    action_taken: Mapped[Optional[str]] = mapped_column(String(100))
    feedback_date: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Trust / abstention (A2) — persisted for Don't-act queues
    low_confidence: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project: Mapped["Project"] = relationship(back_populates="project_predictions")
    decisions: Mapped[list["Decision"]] = relationship(back_populates="prediction")


class Decision(Base):
    """B3 decision ledger — accountable action committed from a case.

    Distinct from A7 feedback stamps on ProjectPrediction. A decision is a
    first-class object with expected lift, status, and scheduled recheck
    (30/60/90). Autopsy narratives deepen in later B3 slices.
    """

    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organizations.id"), index=True
    )
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), index=True)
    prediction_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("project_predictions.id"), index=True
    )

    entity_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)

    # proposed | committed | checking | closed | cancelled
    status: Mapped[str] = mapped_column(String(20), default="committed", index=True)

    action_code: Mapped[str] = mapped_column(String(100), nullable=False)
    action_name: Mapped[str] = mapped_column(String(200), nullable=False)
    action_description: Mapped[Optional[str]] = mapped_column(Text)

    # Case snapshot at commit time
    probability_at_commit: Mapped[Optional[float]] = mapped_column(Float)
    risk_level_at_commit: Mapped[Optional[str]] = mapped_column(String(20))
    expected_probability_after: Mapped[Optional[float]] = mapped_column(Float)
    expected_lift: Mapped[Optional[float]] = mapped_column(Float)  # negative = risk down
    decision_summary: Mapped[Optional[str]] = mapped_column(Text)
    case_snapshot: Mapped[Optional[dict]] = mapped_column(JSON)  # features, drivers, recs meta

    # Recheck schedule (B3 backbone)
    recheck_interval_days: Mapped[int] = mapped_column(Integer, default=30)
    recheck_at: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    last_checkin_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    checkin_count: Mapped[int] = mapped_column(Integer, default=0)

    # Outcome / autopsy (filled on check-in)
    actual_outcome: Mapped[Optional[str]] = mapped_column(String(50))
    outcome_notes: Mapped[Optional[str]] = mapped_column(Text)
    autopsy_narrative: Mapped[Optional[str]] = mapped_column(Text)

    committed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="decisions")
    prediction: Mapped[Optional["ProjectPrediction"]] = relationship(back_populates="decisions")


# =============================================================================
# ORIGINAL DEMO MODELS (kept for backward compatibility)
# =============================================================================


class Customer(Base):
    """Customer data with features for prediction."""
    
    __tablename__ = "customers"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    external_id: Mapped[Optional[str]] = mapped_column(String(100), unique=True, index=True)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="customer")


class Prediction(Base):
    """Prediction record with probability and confidence."""
    
    __tablename__ = "predictions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    customer_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("customers.id"), nullable=True, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    churn_probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    features_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    
    # Relationships
    customer: Mapped["Customer"] = relationship(back_populates="predictions")
    explanation: Mapped[Optional["Explanation"]] = relationship(back_populates="prediction", uselist=False)
    insights: Mapped[Optional["Insight"]] = relationship(back_populates="prediction", uselist=False)
    recommendations: Mapped[list["Recommendation"]] = relationship(back_populates="prediction")
    feedback: Mapped[Optional["Feedback"]] = relationship(back_populates="prediction", uselist=False)


class Explanation(Base):
    """SHAP and LIME explanation values with consistency score."""
    
    __tablename__ = "explanations"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    prediction_id: Mapped[str] = mapped_column(String(36), ForeignKey("predictions.id"), unique=True)
    shap_values: Mapped[dict] = mapped_column(JSON, nullable=False)
    lime_values: Mapped[dict] = mapped_column(JSON, nullable=False)
    consistency_score: Mapped[float] = mapped_column(Float, nullable=False)
    trust_level: Mapped[str] = mapped_column(String(20), nullable=False)  # high, medium, low
    
    # Relationships
    prediction: Mapped["Prediction"] = relationship(back_populates="explanation")


class Insight(Base):
    """Business-friendly insights generated from explanations."""
    
    __tablename__ = "insights"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    prediction_id: Mapped[str] = mapped_column(String(36), ForeignKey("predictions.id"), unique=True)
    insights: Mapped[list] = mapped_column(JSON, nullable=False)  # [{text, severity, feature}]
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prediction: Mapped["Prediction"] = relationship(back_populates="insights")


class ActionCatalog(Base):
    """Reference table of available business actions."""
    
    __tablename__ = "action_catalog"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    action_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    action_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    base_cost: Mapped[float] = mapped_column(Float, nullable=False)
    applicable_conditions: Mapped[Optional[dict]] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    recommendations: Mapped[list["Recommendation"]] = relationship(back_populates="action")
    feedback_records: Mapped[list["Feedback"]] = relationship(back_populates="action_taken")
    effectiveness_records: Mapped[list["ActionEffectiveness"]] = relationship(back_populates="action")


class Recommendation(Base):
    """Scored action recommendations for a prediction."""
    
    __tablename__ = "recommendations"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    prediction_id: Mapped[str] = mapped_column(String(36), ForeignKey("predictions.id"), index=True)
    action_id: Mapped[str] = mapped_column(String(36), ForeignKey("action_catalog.id"))
    impact_score: Mapped[float] = mapped_column(Float, nullable=False)
    cost_score: Mapped[float] = mapped_column(Float, nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    
    # Relationships
    prediction: Mapped["Prediction"] = relationship(back_populates="recommendations")
    action: Mapped["ActionCatalog"] = relationship(back_populates="recommendations")


class Feedback(Base):
    """Outcome feedback for predictions and actions taken."""
    
    __tablename__ = "feedback"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    prediction_id: Mapped[str] = mapped_column(String(36), ForeignKey("predictions.id"), unique=True)
    action_taken_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("action_catalog.id"))
    actual_outcome: Mapped[str] = mapped_column(String(50), nullable=False)  # churned, retained, unknown
    outcome_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    
    # Relationships
    prediction: Mapped["Prediction"] = relationship(back_populates="feedback")
    action_taken: Mapped[Optional["ActionCatalog"]] = relationship(back_populates="feedback_records")


class ModelPerformance(Base):
    """Model performance metrics over time."""
    
    __tablename__ = "model_performance"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    evaluation_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    precision_score: Mapped[Optional[float]] = mapped_column(Float)
    recall_score: Mapped[Optional[float]] = mapped_column(Float)
    f1_score: Mapped[Optional[float]] = mapped_column(Float)
    auc_roc: Mapped[Optional[float]] = mapped_column(Float)
    explanation_consistency_avg: Mapped[Optional[float]] = mapped_column(Float)
    sample_size: Mapped[Optional[int]] = mapped_column(Integer)


class ActionEffectiveness(Base):
    """Track action effectiveness over time periods."""
    
    __tablename__ = "action_effectiveness"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    action_id: Mapped[str] = mapped_column(String(36), ForeignKey("action_catalog.id"), index=True)
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    times_recommended: Mapped[int] = mapped_column(Integer, default=0)
    times_taken: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[Optional[float]] = mapped_column(Float)
    avg_probability_reduction: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    action: Mapped["ActionCatalog"] = relationship(back_populates="effectiveness_records")
