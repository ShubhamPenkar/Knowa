"""Authentication API routes for SaaS."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.auth_service import AuthService, AuthContext, get_auth_context

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class SignupRequest(BaseModel):
    organization_name: str
    organization_slug: str
    industry: str | None = None
    email: EmailStr
    password: str
    name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict
    organization: dict


class CreateAPIKeyRequest(BaseModel):
    name: str
    scopes: list[str] = ["predict"]


class APIKeyResponse(BaseModel):
    id: str
    key: str
    name: str
    scopes: list[str]
    created_at: str


class UpdateProfileRequest(BaseModel):
    name: str


class UpdateOrganizationRequest(BaseModel):
    name: str | None = None
    industry: str | None = None


def _org_payload(org) -> dict:
    return {
        "id": org.id,
        "name": org.name,
        "slug": org.slug,
        "plan": org.plan,
        "industry": org.industry,
    }


def _user_payload(user) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
    }


# =============================================================================
# Routes
# =============================================================================

@router.post("/signup", response_model=TokenResponse)
async def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """
    Create new organization and admin user.
    Returns JWT token for immediate login.
    """
    auth_service = AuthService(db)
    
    try:
        # Create organization
        org = auth_service.create_organization(
            name=request.organization_name,
            slug=request.organization_slug,
            industry=request.industry,
        )
        
        # Create owner user
        user = auth_service.create_user(
            org_id=org.id,
            email=request.email,
            password=request.password,
            name=request.name,
            role="owner",
        )
        
        # Generate token
        token = auth_service.create_token(user.id, org.id)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": _user_payload(user),
            "organization": _org_payload(org),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Login and get JWT token."""
    from app.db.models import Organization

    auth_service = AuthService(db)
    user, token = auth_service.authenticate_user(request.email, request.password)

    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    if not org:
        raise HTTPException(status_code=401, detail="Organization not found")

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": _user_payload(user),
        "organization": _org_payload(org),
    }


@router.get("/me")
async def get_current_user(auth: AuthContext = Depends(get_auth_context)):
    """Get current authenticated user/organization."""
    return {
        "organization": _org_payload(auth.organization),
        "user": _user_payload(auth.user) if auth.user else None,
        "api_key": {
            "id": auth.api_key.id if auth.api_key else None,
            "name": auth.api_key.name if auth.api_key else None,
        } if auth.api_key else None,
        "scopes": auth.scopes,
    }


@router.patch("/me")
async def update_current_user(
    request: UpdateProfileRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Update the signed-in user's profile (display name)."""
    if not auth.user:
        raise HTTPException(status_code=403, detail="User session required")
    auth_service = AuthService(db)
    try:
        user = auth_service.update_user_profile(auth.user.id, name=request.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"user": _user_payload(user)}


@router.patch("/organization")
async def update_organization(
    request: UpdateOrganizationRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Update workspace name / industry (owner or admin)."""
    if not auth.user:
        raise HTTPException(status_code=403, detail="User session required")
    if auth.user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Owner or admin access required")
    if request.name is None and request.industry is None:
        raise HTTPException(status_code=400, detail="Nothing to update")

    auth_service = AuthService(db)
    try:
        org = auth_service.update_organization(
            auth.org_id,
            name=request.name,
            industry=request.industry,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"organization": _org_payload(org)}


@router.get("/members")
async def list_org_members(
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Active org members for assignee pickers."""
    from app.db.models import User

    rows = (
        db.query(User)
        .filter(
            User.organization_id == auth.org_id,
            User.is_active == True,  # noqa: E712
        )
        .order_by(User.name.asc())
        .all()
    )
    return {
        "n": len(rows),
        "members": [
            {
                "id": u.id,
                "name": u.name,
                "email": u.email,
                "role": u.role,
            }
            for u in rows
        ],
    }


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Create new API key for organization."""
    if not auth.has_scope("admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    auth_service = AuthService(db)
    api_key = auth_service.create_api_key(
        org_id=auth.org_id,
        name=request.name,
        scopes=request.scopes,
    )
    
    return {
        "id": api_key.id,
        "key": api_key.key,  # Only shown once!
        "name": api_key.name,
        "scopes": api_key.scopes,
        "created_at": api_key.created_at.isoformat(),
    }


@router.get("/api-keys")
async def list_api_keys(
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """List API keys (without revealing full key)."""
    from app.db.models import APIKey
    
    keys = db.query(APIKey).filter(
        APIKey.organization_id == auth.org_id,
        APIKey.is_active == True
    ).all()
    
    return [
        {
            "id": k.id,
            "name": k.name,
            "key_prefix": k.key[:10] + "...",
            "scopes": k.scopes,
            "last_used": k.last_used.isoformat() if k.last_used else None,
            "created_at": k.created_at.isoformat(),
        }
        for k in keys
    ]


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Revoke an API key."""
    if not auth.has_scope("admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from app.db.models import APIKey
    
    key = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.organization_id == auth.org_id
    ).first()
    
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    key.is_active = False
    db.commit()
    
    return {"message": "API key revoked"}
