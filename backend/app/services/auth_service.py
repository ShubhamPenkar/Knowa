"""Authentication service for SaaS multi-tenant access."""

import os
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.db.models import Organization, User, APIKey

settings = get_settings()

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

JWT_SECRET = os.getenv("JWT_SECRET", settings.jwt_secret if hasattr(settings, "jwt_secret") else "dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


class AuthService:
    """Handle authentication and authorization."""
    
    def __init__(self, db: Session):
        self.db = db
    
    # =========================================================================
    # Password Hashing
    # =========================================================================
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with bcrypt."""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode(), hashed.encode())
    
    # =========================================================================
    # JWT Tokens
    # =========================================================================
    
    @staticmethod
    def create_token(user_id: str, org_id: str) -> str:
        """Create JWT token for user."""
        payload = {
            "user_id": user_id,
            "org_id": org_id,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def decode_token(token: str) -> dict:
        """Decode and validate JWT token."""
        try:
            return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    # =========================================================================
    # User Management
    # =========================================================================
    
    def create_organization(self, name: str, slug: str, industry: Optional[str] = None) -> Organization:
        """Create new organization."""
        existing = self.db.query(Organization).filter(Organization.slug == slug).first()
        if existing:
            raise ValueError(f"Organization slug '{slug}' already exists")
        
        org = Organization(name=name, slug=slug, industry=industry)
        self.db.add(org)
        self.db.commit()
        self.db.refresh(org)
        return org
    
    def create_user(
        self,
        org_id: str,
        email: str,
        password: str,
        name: str,
        role: str = "member"
    ) -> User:
        """Create new user in organization."""
        existing = self.db.query(User).filter(User.email == email).first()
        if existing:
            raise ValueError(f"User with email '{email}' already exists")
        
        user = User(
            organization_id=org_id,
            email=email,
            password_hash=self.hash_password(password),
            name=name,
            role=role,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def authenticate_user(self, email: str, password: str) -> tuple[User, str]:
        """Authenticate user and return token."""
        user = self.db.query(User).filter(User.email == email).first()
        if not user or not self.verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        if not user.is_active:
            raise HTTPException(status_code=401, detail="Account disabled")
        
        # Update last login
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        token = self.create_token(user.id, user.organization_id)
        return user, token
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def update_user_profile(self, user_id: str, *, name: str) -> User:
        """Update the authenticated user's display name."""
        user = self.get_user_by_id(user_id)
        if not user or not user.is_active:
            raise ValueError("User not found")
        cleaned = (name or "").strip()
        if not cleaned:
            raise ValueError("Name is required")
        if len(cleaned) > 100:
            raise ValueError("Name must be 100 characters or fewer")
        user.name = cleaned
        self.db.commit()
        self.db.refresh(user)
        return user

    def update_organization(
        self,
        org_id: str,
        *,
        name: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> Organization:
        """Update organization profile fields."""
        org = (
            self.db.query(Organization)
            .filter(Organization.id == org_id, Organization.is_active == True)  # noqa: E712
            .first()
        )
        if not org:
            raise ValueError("Organization not found")

        if name is not None:
            cleaned = name.strip()
            if not cleaned:
                raise ValueError("Workspace name is required")
            if len(cleaned) > 100:
                raise ValueError("Workspace name must be 100 characters or fewer")
            org.name = cleaned

        if industry is not None:
            cleaned_ind = industry.strip().lower() if industry.strip() else None
            allowed = {None, "saas", "ecommerce", "finance", "healthcare", "other"}
            if cleaned_ind not in allowed:
                raise ValueError("Invalid industry")
            org.industry = cleaned_ind

        self.db.commit()
        self.db.refresh(org)
        return org
    
    # =========================================================================
    # API Key Management
    # =========================================================================
    
    def create_api_key(
        self,
        org_id: str,
        name: str,
        scopes: list[str] = None
    ) -> APIKey:
        """Create new API key for organization."""
        api_key = APIKey(
            organization_id=org_id,
            name=name,
            scopes=scopes or ["predict"],
        )
        self.db.add(api_key)
        self.db.commit()
        self.db.refresh(api_key)
        return api_key
    
    def validate_api_key(self, key: str) -> tuple[APIKey, Organization]:
        """Validate API key and return org."""
        api_key = self.db.query(APIKey).filter(
            APIKey.key == key,
            APIKey.is_active == True
        ).first()
        
        if not api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            raise HTTPException(status_code=401, detail="API key expired")
        
        org = self.db.query(Organization).filter(
            Organization.id == api_key.organization_id,
            Organization.is_active == True
        ).first()
        
        if not org:
            raise HTTPException(status_code=401, detail="Organization not found or inactive")
        
        # Update last used
        api_key.last_used = datetime.utcnow()
        self.db.commit()
        
        return api_key, org


# =============================================================================
# FastAPI Dependencies
# =============================================================================

class AuthContext:
    """Authentication context with user/org info."""
    
    def __init__(
        self,
        organization: Organization,
        user: Optional[User] = None,
        api_key: Optional[APIKey] = None,
        scopes: list[str] = None
    ):
        self.organization = organization
        self.user = user
        self.api_key = api_key
        self.scopes = scopes or []
    
    @property
    def org_id(self) -> str:
        return self.organization.id
    
    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "admin" in self.scopes


async def get_auth_context(
    bearer: HTTPAuthorizationCredentials = Security(bearer_scheme),
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> AuthContext:
    """
    Get authentication context from either:
    - Bearer token (JWT)
    - X-API-Key header
    """
    auth_service = AuthService(db)
    
    # Try API key first
    if api_key:
        key_obj, org = auth_service.validate_api_key(api_key)
        return AuthContext(
            organization=org,
            api_key=key_obj,
            scopes=key_obj.scopes
        )
    
    # Try bearer token
    if bearer:
        payload = auth_service.decode_token(bearer.credentials)
        user = auth_service.get_user_by_id(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        org = db.query(Organization).filter(Organization.id == user.organization_id).first()
        if not org or not org.is_active:
            raise HTTPException(status_code=401, detail="Organization not found or inactive")
        
        return AuthContext(
            organization=org,
            user=user,
            scopes=["admin"] if user.role in ["owner", "admin"] else ["predict", "view"]
        )
    
    raise HTTPException(status_code=401, detail="Authentication required")


async def get_optional_auth(
    bearer: HTTPAuthorizationCredentials = Security(bearer_scheme),
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> Optional[AuthContext]:
    """Optional auth - returns None if not authenticated."""
    if not bearer and not api_key:
        return None
    return await get_auth_context(bearer, api_key, db)
