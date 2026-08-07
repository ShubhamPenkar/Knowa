"""Project API routes for SaaS."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.auth_service import AuthContext, get_auth_context
from app.services.project_service import ProjectService
from app.ml.dataset_profiler import ProfilingError
from app.ml.feature_validation import FeatureValidationError

router = APIRouter()


# =============================================================================
# Request Models
# =============================================================================

class CreateProjectRequest(BaseModel):
    name: str
    dataset_id: str
    target_column: str
    feature_columns: list[str]
    target_positive_label: str = "1"
    target_description: str = "outcome"
    problem_type: str = "binary_classification"  # binary_classification or regression
    description: str | None = None


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    target_positive_label: str | None = None
    target_description: str | None = None
    feature_columns: list[str] | None = None
    model_type: str | None = None


class PredictRequest(BaseModel):
    features: dict
    entity_id: str | None = None
    include_explanations: bool = True
    include_recommendations: bool = True


class SimulateRequest(BaseModel):
    base_features: dict
    modified_features: dict


class ScenarioLeversRequest(BaseModel):
    """Rank what-if dials by actual score movement for this case."""
    base_features: dict


class BatchPredictRequest(BaseModel):
    """Triage scoring for many rows (no per-row SHAP/LIME)."""
    rows: list[dict]
    entity_ids: list[str | None] | None = None
    max_rows: int = 1000


class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_outcome: str
    action_taken: str | None = None
    notes: str | None = None


class CreateDecisionRequest(BaseModel):
    """B3: commit an action from a scored case onto the decision ledger."""
    action_code: str
    prediction_id: str | None = None
    action_name: str | None = None
    action_description: str | None = None
    entity_id: str | None = None
    probability: float | None = None
    risk_level: str | None = None
    expected_probability_after: float | None = None
    expected_lift: float | None = None
    decision_summary: str | None = None
    case_snapshot: dict | None = None
    recheck_interval_days: int = 30
    status: str = "committed"


class DecisionCheckInRequest(BaseModel):
    actual_outcome: str | None = None
    notes: str | None = None
    close: bool = False
    schedule_next: bool = True


# =============================================================================
# Routes
# =============================================================================

@router.post("")
async def create_project(
    request: CreateProjectRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Create new prediction project."""
    service = ProjectService(db, auth.org_id)
    
    # Validate problem_type
    if request.problem_type not in ["binary_classification", "regression"]:
        raise HTTPException(status_code=400, detail="problem_type must be 'binary_classification' or 'regression'")
    
    try:
        project = service.create_project(
            name=request.name,
            dataset_id=request.dataset_id,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            target_positive_label=request.target_positive_label,
            target_description=request.target_description,
            problem_type=request.problem_type,
            description=request.description,
        )
        
        return {
            "id": project.id,
            "name": project.name,
            "status": project.status,
            "problem_type": project.problem_type,
            "target_column": project.target_column,
            "target_description": project.target_description,
            "feature_count": len(project.feature_columns),
            "created_at": project.created_at.isoformat(),
        }
    except ProfilingError as e:
        raise HTTPException(status_code=400, detail=e.as_detail())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
async def list_projects(
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """List all projects."""
    service = ProjectService(db, auth.org_id)
    projects = service.list_projects()
    
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "status": p.status,
            "problem_type": p.problem_type,
            "target_column": p.target_column,
            "target_description": p.target_description,
            "feature_count": len(p.feature_columns),
            "created_at": p.created_at.isoformat(),
            "updated_at": p.updated_at.isoformat(),
        }
        for p in projects
    ]


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Get project details."""
    service = ProjectService(db, auth.org_id)
    project = service.get_project(project_id)
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get active model
    active_model = service.get_active_model(project_id)
    
    # Build active_model response based on problem type
    active_model_response = None
    if active_model:
        if project.problem_type == "regression":
            active_model_response = {
                "version": active_model.version,
                "mae": active_model.mae,
                "mse": active_model.mse,
                "rmse": active_model.rmse,
                "r2_score": active_model.r2_score,
                "trained_at": active_model.trained_at.isoformat(),
                "feature_importance": active_model.feature_importance,
            }
        else:
            active_model_response = {
                "version": active_model.version,
                "accuracy": active_model.accuracy,
                "precision": active_model.precision_score,
                "recall": active_model.recall_score,
                "f1_score": active_model.f1_score,
                "auc_roc": active_model.auc_roc,
                "trained_at": active_model.trained_at.isoformat(),
                "feature_importance": active_model.feature_importance,
            }
    
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "status": project.status,
        "problem_type": project.problem_type,
        "dataset_id": project.dataset_id,
        "target_column": project.target_column,
        "target_positive_label": project.target_positive_label,
        "target_description": project.target_description,
        "feature_columns": project.feature_columns,
        "feature_config": project.feature_config,
        "model_type": project.model_type,
        "active_model": active_model_response,
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat(),
    }


@router.patch("/{project_id}")
async def update_project(
    project_id: str,
    request: UpdateProjectRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Update project configuration."""
    service = ProjectService(db, auth.org_id)
    
    try:
        updates = request.model_dump(exclude_unset=True)
        project = service.update_project(project_id, updates)
        
        return {
            "id": project.id,
            "name": project.name,
            "status": project.status,
            "updated_at": project.updated_at.isoformat(),
        }
    except ProfilingError as e:
        raise HTTPException(status_code=400, detail=e.as_detail())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/test-data")
async def get_test_data(
    project_id: str,
    limit: int = 50,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Get test set rows for prediction validation."""
    service = ProjectService(db, auth.org_id)
    
    try:
        rows = service.get_test_data(project_id, limit)
        return {"rows": rows, "count": len(rows)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Delete a project."""
    if not auth.has_scope("admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    service = ProjectService(db, auth.org_id)
    if not service.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    
    return {"message": "Project deleted"}


@router.post("/{project_id}/train")
async def train_model(
    project_id: str,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Train ML model for project."""
    service = ProjectService(db, auth.org_id)
    project = service.get_project(project_id)
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        # Train synchronously for now (could be async with background_tasks)
        trained_model = service.train_model(project_id)
        
        return {
            "message": "Model trained successfully",
            "model": {
                "version": trained_model.version,
                "accuracy": trained_model.accuracy,
                "precision": trained_model.precision_score,
                "recall": trained_model.recall_score,
                "f1_score": trained_model.f1_score,
                "auc_roc": trained_model.auc_roc,
                "training_samples": trained_model.training_samples,
                "training_time_seconds": trained_model.training_time_seconds,
            },
        }
    except ProfilingError as e:
        raise HTTPException(status_code=400, detail=e.as_detail())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/spot-check")
async def spot_check(
    project_id: str,
    limit: int = 50,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Compare held-out known outcomes to model predictions (trust strip)."""
    service = ProjectService(db, auth.org_id)
    try:
        return service.spot_check(project_id, limit=min(limit, 100))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_id}/predict")
async def predict(
    project_id: str,
    request: PredictRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Make prediction using trained model."""
    from app.ml.feature_validation import FeatureValidationError

    service = ProjectService(db, auth.org_id)
    
    try:
        result = service.predict(
            project_id=project_id,
            features=request.features,
            entity_id=request.entity_id,
            include_explanations=request.include_explanations,
            include_recommendations=request.include_recommendations,
            persist=True,
            source="api",
        )
        return result
    except FeatureValidationError as e:
        raise HTTPException(status_code=400, detail=e.as_detail())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_id}/predict/batch")
async def predict_batch(
    project_id: str,
    request: BatchPredictRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """
    Batch triage scores (probability / risk / soft) without SHAP/LIME.

    Use single-case /predict for full explanations and recommendations.
    """
    from app.ml.feature_validation import FeatureValidationError

    service = ProjectService(db, auth.org_id)
    try:
        return service.predict_batch(
            project_id,
            rows=request.rows,
            entity_ids=request.entity_ids,
            max_rows=min(request.max_rows or 1000, 2000),
        )
    except FeatureValidationError as e:
        raise HTTPException(status_code=400, detail=e.as_detail())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_id}/simulate")
async def simulate(
    project_id: str,
    request: SimulateRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Run what-if simulation (does not persist prediction history)."""
    from app.ml.feature_validation import FeatureValidationError

    service = ProjectService(db, auth.org_id)
    
    try:
        result = service.simulate(
            project_id=project_id,
            base_features=request.base_features,
            modified_features=request.modified_features,
        )
        return result
    except FeatureValidationError as e:
        raise HTTPException(status_code=400, detail=e.as_detail())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_id}/scenario-levers")
async def scenario_levers(
    project_id: str,
    request: ScenarioLeversRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Levers ranked by how much they move *this* case (not SHAP-only)."""
    from app.ml.feature_validation import FeatureValidationError

    service = ProjectService(db, auth.org_id)
    try:
        return service.scenario_levers(project_id, request.base_features)
    except FeatureValidationError as e:
        raise HTTPException(status_code=400, detail=e.as_detail())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_id}/feedback")
async def record_feedback(
    project_id: str,
    request: FeedbackRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Record or update real-world outcome for a project prediction (A7 log)."""
    service = ProjectService(db, auth.org_id)
    try:
        result = service.record_feedback(
            prediction_id=request.prediction_id,
            actual_outcome=request.actual_outcome,
            action_taken=request.action_taken,
            project_id=project_id,
            notes=request.notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not result:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return result


@router.get("/{project_id}/feedback/{prediction_id}")
async def get_prediction_feedback(
    project_id: str,
    prediction_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Get feedback for one prediction if logged."""
    service = ProjectService(db, auth.org_id)
    result = service.get_feedback(project_id, prediction_id)
    if not result:
        raise HTTPException(status_code=404, detail="No feedback for this prediction")
    return result


@router.get("/{project_id}/feedback-summary")
async def feedback_summary(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """A7 aggregate: coverage, model match rate, action effectiveness."""
    service = ProjectService(db, auth.org_id)
    try:
        return service.get_feedback_summary(project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{project_id}/decisions")
async def create_decision(
    project_id: str,
    request: CreateDecisionRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """B3: commit a decision from a case onto the ledger."""
    from app.services.decision_service import DecisionService

    service = DecisionService(db, auth.org_id)
    try:
        return service.create_from_case(
            project_id,
            action_code=request.action_code,
            prediction_id=request.prediction_id,
            action_name=request.action_name,
            action_description=request.action_description,
            entity_id=request.entity_id,
            probability=request.probability,
            risk_level=request.risk_level,
            expected_probability_after=request.expected_probability_after,
            expected_lift=request.expected_lift,
            decision_summary=request.decision_summary,
            case_snapshot=request.case_snapshot,
            recheck_interval_days=request.recheck_interval_days,
            status=request.status,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/decisions")
async def list_decisions(
    project_id: str,
    status: str | None = None,
    limit: int = 50,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """B3: list decisions on the project ledger."""
    from app.services.decision_service import DecisionService

    service = DecisionService(db, auth.org_id)
    try:
        return service.list_decisions(project_id, status=status, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{project_id}/decisions/{decision_id}")
async def get_decision(
    project_id: str,
    decision_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """B3: fetch one decision (incl. case snapshot)."""
    from app.services.decision_service import DecisionService

    service = DecisionService(db, auth.org_id)
    try:
        return service.get_decision(project_id, decision_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{project_id}/decisions/{decision_id}/check-in")
async def check_in_decision(
    project_id: str,
    decision_id: str,
    request: DecisionCheckInRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """B3: 30/60/90 check-in — log notes/outcome, optionally close or reschedule."""
    from app.services.decision_service import DecisionService

    service = DecisionService(db, auth.org_id)
    try:
        return service.check_in(
            project_id,
            decision_id,
            actual_outcome=request.actual_outcome,
            notes=request.notes,
            close=request.close,
            schedule_next=request.schedule_next,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/predictions")
async def list_predictions(
    project_id: str,
    limit: int = 50,
    entity_id: str | None = None,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """List predictions for project."""
    from app.db.models import ProjectPrediction
    
    service = ProjectService(db, auth.org_id)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    query = db.query(ProjectPrediction).filter(ProjectPrediction.project_id == project_id)
    
    if entity_id:
        query = query.filter(ProjectPrediction.entity_id == entity_id)
    
    predictions = query.order_by(ProjectPrediction.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": p.id,
            "entity_id": p.entity_id,
            "probability": p.probability,
            "risk_level": p.risk_level,
            "confidence": p.confidence,
            "actual_outcome": p.actual_outcome,
            "action_taken": p.action_taken,
            "feedback_date": p.feedback_date.isoformat() if p.feedback_date else None,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in predictions
    ]
