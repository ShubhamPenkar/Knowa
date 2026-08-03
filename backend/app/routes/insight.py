"""Insight API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import InsightResponse
from app.services.insight_service import InsightService

router = APIRouter()


@router.get("/{prediction_id}", response_model=InsightResponse)
async def get_insights(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    """
    Get business-friendly insights for a prediction.
    
    Transforms technical explanations into actionable business language.
    Each insight includes:
    - Human-readable text
    - Severity level (critical, warning, info, positive)
    - Associated feature
    - Impact magnitude
    """
    service = InsightService(db)
    result = service.get_insights(prediction_id)
    if not result:
        raise HTTPException(status_code=404, detail="Insights not found")
    return result


@router.post("/{prediction_id}/generate", response_model=InsightResponse)
async def generate_insights(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    """Generate insights for an existing prediction with explanation."""
    service = InsightService(db)
    try:
        result = service.generate_insights(prediction_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")
