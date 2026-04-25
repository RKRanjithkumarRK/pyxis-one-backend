from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "nexusai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.services.memory.tasks",
        "app.services.rag.tasks",
        "app.services.research.tasks",
        "app.services.image.tasks",
        "app.services.workflows.tasks",
        "app.services.export.tasks",
        "app.services.observability.tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    beat_schedule={
        "workflow-schedule-check": {
            "task": "workflows.schedule_check",
            "schedule": 60.0,
        },
        "slo-burn-check": {
            "task": "observability.check_slo_burn",
            "schedule": 300.0,  # every 5 minutes
        },
    },
)
