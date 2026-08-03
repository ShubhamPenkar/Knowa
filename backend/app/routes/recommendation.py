"""Recommendation API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import RecommendationResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter()


@router.get("/{prediction_id}", response_model=RecommendationResponse)
async def get_recommendations(
    prediction_id: str,
    top_n: int = 5,
    db: Session = Depends(get_db)
):
    """
    Get ranked action recommendations for a prediction.
    
    Each recommendation includes:
    - Action details
    - Impact score (expected effect on churn)
    - Cost score (resource requirements)
    - Relevance score (fit for customer profile)
    - Final weighted score
    - Reasoning explanation
    """
    service = RecommendationService(db)
    result = service.get_recommendations(prediction_id, top_n=top_n)
    if not result:
        raise HTTPException(status_code=404, detail="Recommendations not found")
    return result


@router.post("/{prediction_id}/generate", response_model=RecommendationResponse)
async def generate_recommendations(
    prediction_id: str,
    top_n: int = 5,
    db: Session = Depends(get_db)
):
    """Generate recommendations for an existing prediction."""
    service = RecommendationService(db)
    try:
        result = service.generate_recommendations(prediction_id, top_n=top_n)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")
