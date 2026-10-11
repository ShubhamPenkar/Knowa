"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

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
    model_config = ConfigDict(protected_namespaces=())

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


# ============== Semantic Understanding Schemas (Phase 1) ==============

class ColumnSemanticUpdate(BaseModel):
    """User correction for a column's inferred semantic meaning."""
    semantic_type: Literal[
        "id", "numeric_measure", "categorical", "boolean",
        "datetime", "geographic", "text", "unknown"
    ] = Field(..., description="id, numeric_measure, categorical, boolean, datetime, geographic, text, unknown")
    role: Literal[
        "identifier", "measure", "dimension", "timestamp", "unknown"
    ] = Field(..., description="identifier, measure, dimension, timestamp, unknown")
    notes: Optional[str] = Field(None, description="Optional reasoning for the user correction")


class BatchSemanticUpdate(BaseModel):
    """Batch user corrections for column semantics."""
    corrections: dict[str, ColumnSemanticUpdate] = Field(..., description="Map of column name to correction")


class ColumnSemanticDetail(BaseModel):
    """Detailed column semantic metadata with heuristic score labeling."""
    name: str
    semantic_type: str
    role: str
    confidence: float = Field(..., description="Rule-based heuristic score in [0.0, 1.0], not a calibrated probability")
    confidence_type: str = Field("heuristic", description="'heuristic' for inferred rules, 'user_assigned' for manual corrections")
    evidence: list[str] = Field(default_factory=list, description="Privacy-hardened pattern explanations without raw values")
    is_user_corrected: bool = False


# ============== Multi-Dataset Intelligence Schemas (Phase 2) ==============

class DatasetGrainResponse(BaseModel):
    """Inferred dataset grain / level-of-detail."""
    dataset_id: str
    dataset_name: str
    grain_type: str = Field(..., description="entity, transaction, periodic_aggregate, composite_key, unknown")
    primary_key: list[str] = Field(default_factory=list, description="Primary or composite key identifying the grain")
    confidence: float = Field(..., description="Rule-based heuristic score in [0.0, 1.0], not a calibrated probability")
    confidence_type: str = Field("heuristic", description="Explicit confidence type label")
    evidence: list[str] = Field(default_factory=list, description="Pattern evidence establishing dataset grain")
    potential_measures: list[str] = Field(default_factory=list, description="Measures vulnerable to double counting on fan-out")
    entity_identifiers: list[str] = Field(default_factory=list, description="Foreign key links to business entities")
    timestamp_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetRelationshipResponse(BaseModel):
    """Candidate or validated relationship between two datasets."""
    id: str
    source_dataset_id: str
    source_dataset_name: str
    source_column: str
    target_dataset_id: str
    target_dataset_name: str
    target_column: str
    cardinality: str = Field(..., description="one_to_one, one_to_many, many_to_one, many_to_many, unknown")
    source_uniqueness_ratio: float
    target_uniqueness_ratio: float
    containment_ratio: float
    jaccard_similarity: float
    overlapping_values_count: int
    confidence: float = Field(..., description="Rule-based heuristic score in [0.0, 1.0], not a calibrated probability")
    confidence_type: str = Field("heuristic", description="Explicit confidence type label")
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validation_status: str = Field(..., description="candidate, validated, ambiguous, rejected")
    is_ambiguous: bool = False
    fan_out_ratio: float = Field(1.0, description="Joined row expansion multiplier")
    double_counting_risk: bool = Field(False, description="Flag indicating potential measure inflation")
    double_counting_measures: list[str] = Field(default_factory=list)


class RelationshipDiscoveryRequest(BaseModel):
    """Request parameters for discovering relationships."""
    dataset_ids: Optional[list[str]] = Field(None, description="Subset of dataset IDs to analyze (defaults to all)")
    refresh: bool = Field(False, description="Force re-discovery rather than returning cached candidates")


class RelationshipDiscoveryResponse(BaseModel):
    """Discovered candidate join relationships."""
    relationships: list[DatasetRelationshipResponse]
    total_found: int


class RelationshipValidationRequest(BaseModel):
    """Request to validate a proposed join relationship against actual data."""
    source_dataset_id: str
    source_column: str
    target_dataset_id: str
    target_column: str


# ============== Dynamic Product Intelligence Schemas (Phase 3) ==============

class DatasetCoverageItem(BaseModel):
    """Dataset summary within product intelligence overview."""
    dataset_id: str
    dataset_name: str
    row_count: int
    column_count: int
    grain_type: str = "unknown"
    primary_key: list[str] = Field(default_factory=list)
    measures_count: int = 0
    dimensions_count: int = 0
    timestamps_count: int = 0
    readiness_status: str = "ready"
    notes: list[str] = Field(default_factory=list)


class ReadinessSummary(BaseModel):
    """Overall dataset coverage and analysis readiness."""
    total_datasets: int
    ready_datasets: int
    total_rows: int
    dataset_summaries: list[DatasetCoverageItem] = Field(default_factory=list)


class DataDrivenMetricResponse(BaseModel):
    """Computed data-driven metric or breakdown."""
    id: str
    label: str
    category: str = Field(..., description="summary, distribution, trend, breakdown, top_performers, growth, quality, cross_dataset")
    dataset_id: str
    dataset_name: str
    column_name: Optional[str] = None
    secondary_column: Optional[str] = None
    aggregation: str
    status: str = "available"
    reason_unavailable: Optional[str] = None
    value: Optional[Any] = None
    summary_stats: Optional[dict[str, Any]] = None
    time_range: Optional[dict[str, Any]] = None
    grouping: Optional[dict[str, Any]] = None
    evidence: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class UnavailableMetricItem(BaseModel):
    """Reason why an expected analytical metric is unsupported."""
    category: str
    label: str
    reason: str


class ExplainableInsightResponse(BaseModel):
    """Explainable business observation backed by metric data."""
    id: str
    title: str
    finding: str
    evidence: list[str] = Field(default_factory=list)
    relevant_entity: Optional[str] = None
    relevant_dimension: Optional[str] = None
    relevant_time_period: Optional[str] = None
    why_it_matters: str
    confidence: float = Field(..., description="Rule-based heuristic score in [0.0, 1.0], not a calibrated probability")
    confidence_type: str = Field("heuristic", description="Explicit confidence type label")
    limitations: list[str] = Field(default_factory=list)
    is_causal_claim: bool = False


class InvestigationOpportunityResponse(BaseModel):
    """Candidate opportunity for further business investigation."""
    id: str
    title: str
    candidate_area: str
    linked_metric_id: str
    linked_insight_id: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    action_type: str = "investigation_candidate"


class DataQualityWarningResponse(BaseModel):
    """Data quality notice such as missingness or high cardinality."""
    dataset_id: str
    dataset_name: str
    column_name: Optional[str] = None
    warning_type: str
    message: str
    impact: str


class RelationshipSafetyWarningResponse(BaseModel):
    """Safety notice regarding joins, fan-out, or ambiguity."""
    source_dataset_id: str
    target_dataset_id: str
    warning_type: str
    message: str
    risk: str
    affected_measures: list[str] = Field(default_factory=list)


class ProductIntelligenceOverviewResponse(BaseModel):
    """Complete Dynamic Product Intelligence overview."""
    organization_id: str
    readiness: ReadinessSummary
    metrics: list[DataDrivenMetricResponse] = Field(default_factory=list)
    unavailable_metrics: list[UnavailableMetricItem] = Field(default_factory=list)
    insights: list[ExplainableInsightResponse] = Field(default_factory=list)
    opportunities: list[InvestigationOpportunityResponse] = Field(default_factory=list)
    data_quality_warnings: list[DataQualityWarningResponse] = Field(default_factory=list)
    safety_warnings: list[RelationshipSafetyWarningResponse] = Field(default_factory=list)


class ProductIntelligenceRequest(BaseModel):
    """Optional parameters for overview query."""
    dataset_ids: Optional[list[str]] = Field(None, description="Subset of dataset IDs to analyze")
    refresh: bool = Field(False, description="Bypass cache and recompute overview")
