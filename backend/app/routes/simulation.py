"""Simulation API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import SimulationRequest, SimulationResponse
from app.services.simulation_service import SimulationService

router = APIRouter()


@router.post("", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    db: Session = Depends(get_db)
):
    """
    Run a what-if simulation.
    
    Modify feature values and see how the churn prediction changes.
    
    Returns:
    - Original vs modified probability
    - Percentage change
    - Feature-by-feature impact comparison
    - Key insights about the changes
    - Actionable recommendation
    """
    service = SimulationService(db)
    try:
        result = service.simulate(
            base_features=request.base_features.model_dump(),
            modified_features=request.modified_features
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.post("/from-prediction/{prediction_id}", response_model=SimulationResponse)
async def simulate_from_prediction(
    prediction_id: str,
    modified_features: dict,
    db: Session = Depends(get_db)
):
    """Run simulation starting from an existing prediction's features."""
    service = SimulationService(db)
    try:
        result = service.simulate_from_prediction(
            prediction_id=prediction_id,
            modified_features=modified_features
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
