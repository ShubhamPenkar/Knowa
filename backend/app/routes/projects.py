"""Project API routes for SaaS."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.auth_service import AuthContext, get_auth_context
from app.services.project_service import ProjectService

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


class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_outcome: str
    action_taken: str | None = None


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
    service = ProjectService(db, auth.org_id)
    
    try:
        result = service.predict(
            project_id=project_id,
            features=request.features,
            entity_id=request.entity_id,
            include_explanations=request.include_explanations,
            include_recommendations=request.include_recommendations,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_id}/simulate")
async def simulate(
    project_id: str,
    request: SimulateRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Run what-if simulation."""
    service = ProjectService(db, auth.org_id)
    
    try:
        result = service.simulate(
            project_id=project_id,
            base_features=request.base_features,
            modified_features=request.modified_features,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_id}/feedback")
async def record_feedback(
    project_id: str,
    request: FeedbackRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Record outcome feedback."""
    service = ProjectService(db, auth.org_id)
    
    success = service.record_feedback(
        prediction_id=request.prediction_id,
        actual_outcome=request.actual_outcome,
        action_taken=request.action_taken,
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return {"message": "Feedback recorded"}


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
            "created_at": p.created_at.isoformat(),
        }
        for p in predictions
    ]
