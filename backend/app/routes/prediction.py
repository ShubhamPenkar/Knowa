"""Prediction API routes (demo / fixed-schema churn path)."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    CustomerCreate,
    CustomerResponse,
)
from app.services.prediction_service import PredictionService
from app.ml.feature_validation import FeatureValidationError

router = APIRouter()


@router.post("", response_model=PredictionResponse)
async def create_prediction(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new churn prediction.

    Provide either:
    - `customer_id`: For existing customer
    - `features`: For new/anonymous prediction
    """
    service = PredictionService(db)
    try:
        result = service.predict(
            customer_id=request.customer_id,
            features=request.features.model_dump() if request.features else None
        )
        return result
    except FeatureValidationError as e:
        raise HTTPException(status_code=400, detail=e.as_detail())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Static path segments MUST be registered before /{prediction_id}
@router.post("/customer", response_model=CustomerResponse)
async def create_customer(
    customer: CustomerCreate,
    db: Session = Depends(get_db)
):
    """Create a new customer with features."""
    service = PredictionService(db)
    try:
        return service.create_customer(
            external_id=customer.external_id,
            features=customer.features.model_dump()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/customer/{customer_id}", response_model=list[PredictionResponse])
async def get_customer_predictions(
    customer_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get prediction history for a customer."""
    service = PredictionService(db)
    return service.get_customer_predictions(customer_id, limit=limit)


@router.get("/customer/{customer_id}/latest", response_model=PredictionResponse)
async def get_latest_prediction(
    customer_id: str,
    db: Session = Depends(get_db)
):
    """Get the most recent prediction for a customer."""
    service = PredictionService(db)
    result = service.get_latest_prediction(customer_id)
    if not result:
        raise HTTPException(status_code=404, detail="No predictions found for customer")
    return result


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific prediction by ID."""
    service = PredictionService(db)
    result = service.get_prediction(prediction_id)
    if not result:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result
