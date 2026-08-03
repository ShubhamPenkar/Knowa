"""Explanation API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import ExplanationResponse
from app.services.explainability_service import ExplainabilityService

router = APIRouter()


@router.get("/{prediction_id}", response_model=ExplanationResponse)
async def get_explanation(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    """
    Get SHAP and LIME explanations for a prediction.
    
    Returns:
    - Feature importance from SHAP (global context)
    - Feature importance from LIME (local approximation)
    - Consistency score between SHAP and LIME
    - Trust level (high/medium/low)
    - Top risk and protective factors
    """
    service = ExplainabilityService(db)
    result = service.get_explanation(prediction_id)
    if not result:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return result


@router.post("/{prediction_id}/generate", response_model=ExplanationResponse)
async def generate_explanation(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    """Generate explanations for an existing prediction."""
    service = ExplainabilityService(db)
    try:
        result = service.generate_explanation(prediction_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")
