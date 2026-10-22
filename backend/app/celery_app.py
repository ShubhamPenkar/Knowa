"""Celery application for scheduled B3 follow-up rechecks.

Local run (Redis required):
  docker compose up -d redis
  # Linux/macOS:
  cd backend && celery -A app.celery_app.celery worker -l info
  cd backend && celery -A app.celery_app.celery beat -l info

  # Windows (PowerShell):
  cd backend; .\venv\Scripts\celery.exe -A app.celery_app.celery worker -l info -P solo
  cd backend; .\venv\Scripts\celery.exe -A app.celery_app.celery beat -l info

One-shot without Beat:
  celery -A app.celery_app.celery call app.tasks.rechecks.flag_due_decision_rechecks

Manual API (no Redis): POST /api/projects/decisions/recheck-sweep
"""

from __future__ import annotations

import sys
from celery import Celery

from app.config import get_settings

settings = get_settings()

celery = Celery(
    "knowa",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.rechecks"],
)

# Beat interval in seconds (default hourly). Override via CELERY_RECHECK_INTERVAL_MINUTES.
_interval_sec = max(60, int(settings.celery_recheck_interval_minutes) * 60)

celery_conf = {
    "task_serializer": "json",
    "accept_content": ["json"],
    "result_serializer": "json",
    "timezone": "UTC",
    "enable_utc": True,
    "task_track_started": True,
    "worker_prefetch_multiplier": 1,
    "beat_schedule": {
        "flag-due-decision-rechecks": {
            "task": "app.tasks.rechecks.flag_due_decision_rechecks",
            "schedule": float(_interval_sec),
        },
    },
}

# Windows does not support Unix prefork; auto-select solo pool on Windows
if sys.platform == "win32":
    celery_conf["worker_pool"] = "solo"

celery.conf.update(**celery_conf)
