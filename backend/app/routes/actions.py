"""Custom Actions API routes for SaaS."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.db.models import CustomAction
from app.services.auth_service import AuthContext, get_auth_context

router = APIRouter()


class CreateActionRequest(BaseModel):
    code: str
    name: str
    description: str | None = None
    estimated_cost: float = 0
    estimated_impact: float = 0.5
    applicable_when: dict | None = None


class UpdateActionRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    estimated_cost: float | None = None
    estimated_impact: float | None = None
    applicable_when: dict | None = None


@router.post("")
async def create_action(
    request: CreateActionRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Create custom action for organization."""
    # Check if code exists
    existing = db.query(CustomAction).filter(
        CustomAction.organization_id == auth.org_id,
        CustomAction.code == request.code
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail=f"Action code '{request.code}' already exists")
    
    action = CustomAction(
        organization_id=auth.org_id,
        code=request.code,
        name=request.name,
        description=request.description,
        estimated_cost=request.estimated_cost,
        estimated_impact=request.estimated_impact,
        applicable_when=request.applicable_when,
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    
    return {
        "id": action.id,
        "code": action.code,
        "name": action.name,
        "description": action.description,
        "estimated_cost": action.estimated_cost,
        "estimated_impact": action.estimated_impact,
        "applicable_when": action.applicable_when,
        "created_at": action.created_at.isoformat(),
    }


@router.get("")
async def list_actions(
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """List all custom actions."""
    actions = db.query(CustomAction).filter(
        CustomAction.organization_id == auth.org_id,
        CustomAction.is_active == True
    ).order_by(CustomAction.created_at.desc()).all()
    
    return [
        {
            "id": a.id,
            "code": a.code,
            "name": a.name,
            "description": a.description,
            "estimated_cost": a.estimated_cost,
            "estimated_impact": a.estimated_impact,
            "applicable_when": a.applicable_when,
            "created_at": a.created_at.isoformat(),
        }
        for a in actions
    ]


@router.get("/{action_id}")
async def get_action(
    action_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Get action details."""
    action = db.query(CustomAction).filter(
        CustomAction.id == action_id,
        CustomAction.organization_id == auth.org_id
    ).first()
    
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")
    
    return {
        "id": action.id,
        "code": action.code,
        "name": action.name,
        "description": action.description,
        "estimated_cost": action.estimated_cost,
        "estimated_impact": action.estimated_impact,
        "applicable_when": action.applicable_when,
        "is_active": action.is_active,
        "created_at": action.created_at.isoformat(),
    }


@router.patch("/{action_id}")
async def update_action(
    action_id: str,
    request: UpdateActionRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Update action."""
    action = db.query(CustomAction).filter(
        CustomAction.id == action_id,
        CustomAction.organization_id == auth.org_id
    ).first()
    
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")
    
    updates = request.model_dump(exclude_unset=True)
    for key, value in updates.items():
        setattr(action, key, value)
    
    db.commit()
    db.refresh(action)
    
    return {
        "id": action.id,
        "code": action.code,
        "name": action.name,
        "description": action.description,
        "estimated_cost": action.estimated_cost,
        "estimated_impact": action.estimated_impact,
    }


@router.delete("/{action_id}")
async def delete_action(
    action_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Delete (deactivate) action."""
    action = db.query(CustomAction).filter(
        CustomAction.id == action_id,
        CustomAction.organization_id == auth.org_id
    ).first()
    
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")
    
    action.is_active = False
    db.commit()
    
    return {"message": "Action deleted"}


# =============================================================================
# Action Templates (Starter actions for different industries)
# =============================================================================

ACTION_TEMPLATES = {
    "saas": [
        {"code": "discount_10", "name": "10% Discount", "description": "Offer 10% off next renewal", "estimated_cost": 100, "estimated_impact": 0.4},
        {"code": "discount_20", "name": "20% Discount", "description": "Offer 20% off next renewal", "estimated_cost": 200, "estimated_impact": 0.6},
        {"code": "upgrade_free", "name": "Free Upgrade", "description": "Upgrade to higher tier free for 3 months", "estimated_cost": 300, "estimated_impact": 0.7},
        {"code": "personal_call", "name": "Success Manager Call", "description": "Schedule call with customer success", "estimated_cost": 50, "estimated_impact": 0.5},
        {"code": "training_session", "name": "Free Training", "description": "Offer personalized training session", "estimated_cost": 100, "estimated_impact": 0.4},
    ],
    "ecommerce": [
        {"code": "coupon_10", "name": "10% Coupon", "description": "Send 10% off coupon code", "estimated_cost": 20, "estimated_impact": 0.3},
        {"code": "coupon_20", "name": "20% Coupon", "description": "Send 20% off coupon code", "estimated_cost": 40, "estimated_impact": 0.5},
        {"code": "free_shipping", "name": "Free Shipping", "description": "Offer free shipping on next order", "estimated_cost": 15, "estimated_impact": 0.4},
        {"code": "loyalty_bonus", "name": "Loyalty Points", "description": "Award bonus loyalty points", "estimated_cost": 10, "estimated_impact": 0.3},
        {"code": "personal_email", "name": "Personal Email", "description": "Send personalized recommendation email", "estimated_cost": 5, "estimated_impact": 0.2},
    ],
    "finance": [
        {"code": "rate_reduction", "name": "Rate Reduction", "description": "Offer reduced interest rate", "estimated_cost": 500, "estimated_impact": 0.6},
        {"code": "fee_waiver", "name": "Fee Waiver", "description": "Waive annual fee for one year", "estimated_cost": 100, "estimated_impact": 0.5},
        {"code": "advisor_call", "name": "Advisor Call", "description": "Schedule call with financial advisor", "estimated_cost": 75, "estimated_impact": 0.4},
        {"code": "credit_increase", "name": "Credit Increase", "description": "Offer credit limit increase", "estimated_cost": 0, "estimated_impact": 0.3},
        {"code": "rewards_bonus", "name": "Rewards Bonus", "description": "Award bonus reward points", "estimated_cost": 50, "estimated_impact": 0.4},
    ],
}


@router.post("/templates/{industry}")
async def apply_industry_templates(
    industry: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Apply starter action templates for an industry."""
    if industry not in ACTION_TEMPLATES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown industry. Available: {list(ACTION_TEMPLATES.keys())}"
        )
    
    templates = ACTION_TEMPLATES[industry]
    created = []
    
    for template in templates:
        # Skip if exists
        existing = db.query(CustomAction).filter(
            CustomAction.organization_id == auth.org_id,
            CustomAction.code == template["code"]
        ).first()
        
        if existing:
            continue
        
        action = CustomAction(
            organization_id=auth.org_id,
            **template
        )
        db.add(action)
        created.append(template["name"])
    
    db.commit()
    
    return {
        "message": f"Created {len(created)} actions",
        "created": created,
    }
