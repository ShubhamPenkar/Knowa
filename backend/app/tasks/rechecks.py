"""B3 scheduled recheck tasks."""

from __future__ import annotations

import logging

from app.celery_app import celery
from app.database import SessionLocal
from app.services.decision_service import flag_due_rechecks

logger = logging.getLogger(__name__)


@celery.task(name="app.tasks.rechecks.flag_due_decision_rechecks")
def flag_due_decision_rechecks(limit: int = 500) -> dict:
    """
    Periodic sweep: mark due committed decisions as checking.

    Safe to run often — idempotent for already-flagged rows.
    """
    db = SessionLocal()
    try:
        result = flag_due_rechecks(db, org_id=None, limit=limit)
        logger.info(
            "recheck sweep flagged=%s already_checking=%s",
            result.get("flagged"),
            result.get("already_checking"),
        )
        return result
    finally:
        db.close()
