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
    # Startup
    init_db()
    yield
    # Shutdown


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

1. `POST /api/v1/auth/signup` - Create organization
2. `POST /api/v1/datasets` - Upload CSV data
3. `POST /api/v1/projects` - Create prediction project
4. `POST /api/v1/projects/{id}/train` - Train model
5. `POST /api/v1/projects/{id}/predict` - Make predictions!
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
            "signup": "/api/v1/auth/signup",
            "login": "/api/v1/auth/login",
            "datasets": "/api/v1/datasets",
            "projects": "/api/v1/projects",
            "actions": "/api/v1/actions",
        },
        "demo_endpoints": {
            "predict": "/api/v1/predict",
            "explain": "/api/v1/explain/{id}",
            "recommend": "/api/v1/recommend/{id}",
            "simulate": "/api/v1/simulate",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.app_version}
