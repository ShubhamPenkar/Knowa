"""Feedback API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import FeedbackCreate, FeedbackResponse
from app.services.feedback_service import FeedbackService

router = APIRouter()


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackCreate,
    db: Session = Depends(get_db)
):
    """
    Submit outcome feedback for a prediction.
    
    This feeds into the learning loop to:
    - Track model performance over time
    - Measure action effectiveness
    - Trigger model retraining when needed
    """
    service = FeedbackService(db)
    try:
        result = service.submit_feedback(feedback)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@router.get("/{prediction_id}", response_model=FeedbackResponse)
async def get_feedback(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    """Get feedback for a specific prediction."""
    service = FeedbackService(db)
    result = service.get_feedback(prediction_id)
    if not result:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return result


@router.get("/stats/summary")
async def get_feedback_summary(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get feedback statistics summary."""
    service = FeedbackService(db)
    return service.get_summary_stats(days=days)
