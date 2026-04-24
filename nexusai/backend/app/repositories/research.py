from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research import ResearchReport


class ResearchRepository:

    @staticmethod
    async def create(
        db: AsyncSession,
        user_id: uuid.UUID,
        query: str,
        depth: str = "standard",
    ) -> ResearchReport:
        report = ResearchReport(user_id=user_id, query=query, depth=depth, status="pending")
        db.add(report)
        await db.flush()
        await db.refresh(report)
        return report

    @staticmethod
    async def get(db: AsyncSession, report_id: uuid.UUID) -> Optional[ResearchReport]:
        result = await db.execute(
            select(ResearchReport).where(ResearchReport.id == report_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_for_user(
        db: AsyncSession, user_id: uuid.UUID, *, limit: int = 50
    ) -> list[ResearchReport]:
        result = await db.execute(
            select(ResearchReport)
            .where(ResearchReport.user_id == user_id)
            .order_by(ResearchReport.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def set_task_id(db: AsyncSession, report: ResearchReport, task_id: str) -> None:
        report.task_id = task_id
        report.status = "running"
        db.add(report)
        await db.flush()

    @staticmethod
    async def complete(
        db: AsyncSession,
        report: ResearchReport,
        *,
        title: str,
        report_data: dict,
        sources_count: int,
    ) -> ResearchReport:
        report.status = "complete"
        report.title = title
        report.report = report_data
        report.sources_count = sources_count
        report.completed_at = datetime.now(timezone.utc)
        db.add(report)
        await db.flush()
        await db.refresh(report)
        return report

    @staticmethod
    async def fail(db: AsyncSession, report: ResearchReport, error: str) -> ResearchReport:
        report.status = "error"
        report.error = error
        report.completed_at = datetime.now(timezone.utc)
        db.add(report)
        await db.flush()
        await db.refresh(report)
        return report
