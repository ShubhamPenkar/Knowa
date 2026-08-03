"""API Routes package."""

from app.routes import (
    prediction,
    explanation,
    insight,
    recommendation,
    simulation,
    feedback,
    model,
    # SaaS routes
    auth,
    datasets,
    projects,
    actions,
)

__all__ = [
    "prediction",
    "explanation",
    "insight",
    "recommendation",
    "simulation",
    "feedback",
    "model",
    "auth",
    "datasets",
    "projects",
    "actions",
]
