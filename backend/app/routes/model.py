"""Model management API routes."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import ModelMetricsResponse, ActionEffectivenessResponse
from app.services.model_service import ModelService

router = APIRouter()


@router.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    model_type: str = "ensemble",
    db: Session = Depends(get_db)
):
    """
    Trigger model training.
    
    Runs in background to not block API.
    Supported types: xgboost, lightgbm, random_forest, logistic, ensemble
    """
    service = ModelService(db)
    
    valid_types = ["xgboost", "lightgbm", "random_forest", "logistic", "ensemble"]
    if model_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model type. Must be one of: {valid_types}"
        )
    
    # Queue training in background
    background_tasks.add_task(service.train_model, model_type)
    
    return {
        "status": "training_started",
        "model_type": model_type,
        "message": "Model training has been queued"
    }


@router.get("/metrics", response_model=list[ModelMetricsResponse])
async def get_model_metrics(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get recent model performance metrics."""
    service = ModelService(db)
    return service.get_metrics(limit=limit)


@router.get("/metrics/{model_version}", response_model=ModelMetricsResponse)
async def get_version_metrics(
    model_version: str,
    db: Session = Depends(get_db)
):
    """Get metrics for a specific model version."""
    service = ModelService(db)
    result = service.get_version_metrics(model_version)
    if not result:
        raise HTTPException(status_code=404, detail="Model version not found")
    return result


@router.get("/actions/effectiveness", response_model=list[ActionEffectivenessResponse])
async def get_action_effectiveness(
    db: Session = Depends(get_db)
):
    """Get effectiveness statistics for all actions."""
    service = ModelService(db)
    return service.get_action_effectiveness()


@router.get("/status")
async def get_model_status(
    db: Session = Depends(get_db)
):
    """Get current model status and info."""
    service = ModelService(db)
    return service.get_status()
