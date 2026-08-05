"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============== Customer Schemas ==============

class CustomerFeatures(BaseModel):
    """Customer features for prediction."""
    tenure: int = Field(..., ge=0, description="Months with company")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill amount")
    total_charges: float = Field(..., ge=0, description="Total charges to date")
    contract_type: str = Field(..., description="month-to-month, one_year, two_year")
    payment_method: str = Field(..., description="Payment method used")
    internet_service: str = Field(..., description="DSL, fiber_optic, no")
    online_security: str = Field(..., description="yes, no, no_internet")
    tech_support: str = Field(..., description="yes, no, no_internet")
    streaming_tv: str = Field(..., description="yes, no, no_internet")
    streaming_movies: str = Field(..., description="yes, no, no_internet")
    num_support_tickets: int = Field(0, ge=0, description="Support tickets filed")
    days_since_last_interaction: int = Field(0, ge=0, description="Days since last contact")
    num_complaints: int = Field(0, ge=0, description="Number of complaints")
    satisfaction_score: float = Field(3.0, ge=1, le=5, description="Customer satisfaction 1-5")


class CustomerCreate(BaseModel):
    """Create customer request."""
    external_id: Optional[str] = None
    features: CustomerFeatures


class CustomerResponse(BaseModel):
    """Customer response."""
    id: str
    external_id: Optional[str]
    features: dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============== Prediction Schemas ==============

class PredictionRequest(BaseModel):
    """Request for making a prediction."""
    customer_id: Optional[str] = None
    features: Optional[CustomerFeatures] = None
    
    def model_post_init(self, __context: Any) -> None:
        if not self.customer_id and not self.features:
            raise ValueError("Either customer_id or features must be provided")


class ConfidenceInterval(BaseModel):
    """Conformal prediction interval around the point estimate."""
    lower: float
    upper: float
    level: float = Field(0.9, description="Target coverage level, e.g. 0.9")
    width: Optional[float] = None


class PredictionResponse(BaseModel):
    """Prediction result with confidence and conformal uncertainty."""
    id: str
    customer_id: Optional[str]
    churn_probability: float = Field(..., ge=0, le=1)
    churn_risk_level: str  # low, medium, high, critical
    confidence_score: float = Field(..., ge=0, le=1)
    model_version: str
    prediction_timestamp: datetime
    features_used: dict[str, Any]
    confidence_interval: Optional[ConfidenceInterval] = None
    low_confidence: bool = False
    abstention_reason: Optional[str] = None

    class Config:
        from_attributes = True


# ============== Explanation Schemas ==============

class FeatureExplanation(BaseModel):
    """Single feature explanation."""
    feature: str
    value: Any
    importance: float
    direction: str  # positive, negative
    contribution: str  # increases_risk, decreases_risk


class ExplanationResponse(BaseModel):
    """Full explanation with SHAP, LIME, and consistency."""
    prediction_id: str
    shap_explanations: list[FeatureExplanation]
    lime_explanations: list[FeatureExplanation]
    consistency_score: float
    trust_level: str  # high, medium, low
    top_risk_factors: list[str]
    top_protective_factors: list[str]


# ============== Insight Schemas ==============

class InsightItem(BaseModel):
    """Single business insight."""
    text: str
    severity: str  # critical, warning, info, positive
    feature: str
    impact_magnitude: float


class InsightResponse(BaseModel):
    """Collection of business insights."""
    prediction_id: str
    insights: list[InsightItem]
    summary: str
    generated_at: datetime


# ============== Recommendation Schemas ==============

class ActionRecommendation(BaseModel):
    """Single action recommendation with scores."""
    action_code: str
    action_name: str
    description: str
    impact_score: float = Field(..., ge=0, le=1)
    cost_score: float = Field(..., ge=0, le=1)
    relevance_score: float = Field(..., ge=0, le=1)
    final_score: float = Field(..., ge=0, le=1)
    rank: int
    reasoning: str
    expected_probability_reduction: float


class RecommendationResponse(BaseModel):
    """Ranked action recommendations."""
    prediction_id: str
    current_churn_probability: float
    recommendations: list[ActionRecommendation]
    generated_at: datetime


# ============== Simulation Schemas ==============

class SimulationRequest(BaseModel):
    """Request for what-if simulation."""
    base_features: CustomerFeatures
    modified_features: dict[str, Any]  # Only fields to change


class SimulationComparison(BaseModel):
    """Before/after comparison for a feature."""
    feature: str
    original_value: Any
    modified_value: Any
    original_importance: float
    modified_importance: float
    impact_change: float


class SimulationResponse(BaseModel):
    """What-if simulation results."""
    original_probability: float
    modified_probability: float
    probability_change: float
    probability_change_percent: float
    risk_level_change: str  # improved, worsened, unchanged
    feature_comparisons: list[SimulationComparison]
    key_changes: list[str]
    recommendation: str


# ============== Feedback Schemas ==============

class FeedbackCreate(BaseModel):
    """Submit feedback for a prediction."""
    prediction_id: str
    action_taken_code: Optional[str] = None
    actual_outcome: str  # churned, retained, unknown
    outcome_date: Optional[datetime] = None
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Feedback record response."""
    id: str
    prediction_id: str
    action_taken: Optional[str]
    actual_outcome: str
    outcome_date: Optional[datetime]
    recorded_at: datetime
    
    class Config:
        from_attributes = True


# ============== Analytics Schemas ==============

class ModelMetricsResponse(BaseModel):
    """Model performance metrics."""
    model_version: str
    evaluation_date: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    explanation_consistency_avg: float
    sample_size: int


class ActionEffectivenessResponse(BaseModel):
    """Action effectiveness statistics."""
    action_code: str
    action_name: str
    times_recommended: int
    times_taken: int
    adoption_rate: float
    success_rate: float
    avg_probability_reduction: float
