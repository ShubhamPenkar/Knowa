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
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "role": user.role,
            },
            "organization": {
                "id": org.id,
                "name": org.name,
                "slug": org.slug,
                "plan": org.plan,
            },
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
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
        },
        "organization": {
            "id": org.id,
            "name": org.name,
            "slug": org.slug,
            "plan": org.plan,
        },
    }


@router.get("/me")
async def get_current_user(auth: AuthContext = Depends(get_auth_context)):
    """Get current authenticated user/organization."""
    return {
        "organization": {
            "id": auth.organization.id,
            "name": auth.organization.name,
            "slug": auth.organization.slug,
            "plan": auth.organization.plan,
        },
        "user": {
            "id": auth.user.id if auth.user else None,
            "email": auth.user.email if auth.user else None,
            "name": auth.user.name if auth.user else None,
            "role": auth.user.role if auth.user else None,
        } if auth.user else None,
        "api_key": {
            "id": auth.api_key.id if auth.api_key else None,
            "name": auth.api_key.name if auth.api_key else None,
        } if auth.api_key else None,
        "scopes": auth.scopes,
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
