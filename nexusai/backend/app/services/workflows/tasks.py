"""Celery tasks for workflow execution."""
from __future__ import annotations
import asyncio

from app.core.celery_app import celery_app
from app.core.config import settings


@celery_app.task(bind=True, name="workflows.run", max_retries=0)
def run_workflow_task(self, workflow_id: str, inputs: dict) -> dict:
    """Execute a workflow DAG synchronously inside Celery worker."""
    from app.services.workflows.executor import run_workflow
    return asyncio.run(run_workflow(workflow_id, inputs, settings.DATABASE_URL))


@celery_app.task(bind=True, name="workflows.schedule_check")
def schedule_check(self) -> None:
    """Check for scheduled workflows and fire them. Runs every minute via beat."""
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy import select
    from datetime import datetime, timezone
    from app.models.workflow import Workflow

    async def _check():
        engine = create_async_engine(settings.DATABASE_URL, pool_size=2, max_overflow=2)
        Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        now = datetime.now(timezone.utc)
        async with Session() as db:
            result = await db.execute(
                select(Workflow).where(
                    Workflow.is_active == True,
                    Workflow.trigger_type == "schedule",
                )
            )
            workflows = result.scalars().all()
            for wf in workflows:
                cfg = wf.trigger_config or {}
                cron = cfg.get("cron", "")
                if _should_fire(cron, now):
                    run_workflow_task.delay(str(wf.id), {})
        await engine.dispose()

    asyncio.run(_check())


def _should_fire(cron: str, now) -> bool:
    """Very basic cron check — matches @hourly/@daily/@weekly or H:M pattern."""
    if not cron:
        return False
    if cron == "@hourly":
        return now.minute == 0
    if cron == "@daily":
        return now.hour == 0 and now.minute == 0
    if cron == "@weekly":
        return now.weekday() == 0 and now.hour == 0 and now.minute == 0
    if ":" in cron:
        try:
            h, m = map(int, cron.split(":"))
            return now.hour == h and now.minute == m
        except ValueError:
            return False
    return False
