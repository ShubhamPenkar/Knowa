"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db

# Original demo routes
from app.routes import (
    prediction,
    explanation,
    insight,
    recommendation,
    simulation,
    feedback,
    model,
)

# SaaS routes
from app.routes import (
    auth,
    datasets,
    projects,
    actions,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    import os

    # Ensure data directories exist (models, uploads, sqlite)
    for sub in ("", "models", "raw", "processed", "uploads"):
        path = os.path.join(settings.data_path, sub) if sub else settings.data_path
        os.makedirs(path, exist_ok=True)

    init_db()
    # Ensure all ORM tables (incl. B3 decisions) are registered
    import app.db.models  # noqa: F401
    from app.database import Base, engine
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
# Explainable Business Decision Intelligence Platform

A multi-tenant SaaS platform that enables any business to:

1. **Upload data** - Import your customer/transaction data
2. **Train models** - ML models trained on YOUR data
3. **Predict outcomes** - Churn, conversion, default, etc.
4. **Understand why** - SHAP/LIME explanations
5. **Get recommendations** - Scored, actionable recommendations
6. **Simulate decisions** - What-if analysis
7. **Learn from outcomes** - Feedback loop for continuous improvement

## Authentication

Use either:
- **JWT Token**: Login to get bearer token
- **API Key**: Create API key for programmatic access

## Quick Start

1. `POST /api/auth/signup` - Create organization
2. `POST /api/datasets` - Upload CSV data
3. `POST /api/projects` - Create prediction project
4. `POST /api/projects/{id}/train` - Train model
5. `POST /api/projects/{id}/predict` - Make predictions!
""",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# SaaS Routes (Multi-tenant)
# =============================================================================
app.include_router(auth.router, prefix=f"{settings.api_prefix}/auth", tags=["🔐 Authentication"])
app.include_router(datasets.router, prefix=f"{settings.api_prefix}/datasets", tags=["📊 Datasets"])
app.include_router(projects.router, prefix=f"{settings.api_prefix}/projects", tags=["🎯 Projects"])
app.include_router(actions.router, prefix=f"{settings.api_prefix}/actions", tags=["⚡ Actions"])

# =============================================================================
# Original Demo Routes (Backward compatibility)
# =============================================================================
app.include_router(prediction.router, prefix=f"{settings.api_prefix}/predict", tags=["Demo - Prediction"])
app.include_router(explanation.router, prefix=f"{settings.api_prefix}/explain", tags=["Demo - Explanation"])
app.include_router(insight.router, prefix=f"{settings.api_prefix}/insights", tags=["Demo - Insights"])
app.include_router(recommendation.router, prefix=f"{settings.api_prefix}/recommend", tags=["Demo - Recommendations"])
app.include_router(simulation.router, prefix=f"{settings.api_prefix}/simulate", tags=["Demo - Simulation"])
app.include_router(feedback.router, prefix=f"{settings.api_prefix}/feedback", tags=["Demo - Feedback"])
app.include_router(model.router, prefix=f"{settings.api_prefix}/model", tags=["Demo - Model"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Explainable Business Decision Intelligence Platform",
        "docs": "/docs",
        "health": "/health",
        "saas_endpoints": {
            "signup": "/api/auth/signup",
            "login": "/api/auth/login",
            "datasets": "/api/datasets",
            "projects": "/api/projects",
            "actions": "/api/actions",
        },
        "demo_endpoints": {
            "predict": "/api/predict",
            "explain": "/api/explain/{id}",
            "recommend": "/api/recommend/{id}",
            "simulate": "/api/simulate",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.app_version}
