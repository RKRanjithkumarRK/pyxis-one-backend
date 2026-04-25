from __future__ import annotations
import uuid
from typing import Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.project import Project, ProjectMember


ROLE_OWNER = "owner"
ROLE_EDITOR = "editor"
ROLE_VIEWER = "viewer"
VALID_ROLES = {ROLE_OWNER, ROLE_EDITOR, ROLE_VIEWER}


class ProjectRepository:

    @staticmethod
    async def create(
        db: AsyncSession,
        owner_id: uuid.UUID,
        name: str,
        description: str | None = None,
        system_prompt: str | None = None,
        icon_url: str | None = None,
    ) -> Project:
        project = Project(
            owner_id=owner_id,
            name=name,
            description=description,
            system_prompt=system_prompt,
            icon_url=icon_url,
        )
        db.add(project)
        await db.flush()
        await db.refresh(project)

        # Owner is automatically a member with role "owner"
        member = ProjectMember(
            project_id=project.id,
            user_id=owner_id,
            role=ROLE_OWNER,
        )
        db.add(member)
        await db.flush()
        return project

    @staticmethod
    async def get(
        db: AsyncSession,
        project_id: uuid.UUID,
        *,
        with_members: bool = False,
    ) -> Optional[Project]:
        q = select(Project).where(Project.id == project_id)
        if with_members:
            q = q.options(selectinload(Project.members))
        result = await db.execute(q)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_member(
        db: AsyncSession,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Optional[ProjectMember]:
        result = await db.execute(
            select(ProjectMember).where(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def is_member(
        db: AsyncSession,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        m = await ProjectRepository.get_member(db, project_id, user_id)
        return m is not None

    @staticmethod
    async def list_for_user(
        db: AsyncSession,
        user_id: uuid.UUID,
    ) -> list[Project]:
        result = await db.execute(
            select(Project)
            .join(ProjectMember, ProjectMember.project_id == Project.id)
            .where(ProjectMember.user_id == user_id)
            .order_by(Project.created_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def update(
        db: AsyncSession,
        project: Project,
        *,
        name: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
        icon_url: str | None = None,
    ) -> Project:
        if name is not None:
            project.name = name
        if description is not None:
            project.description = description
        if system_prompt is not None:
            project.system_prompt = system_prompt
        if icon_url is not None:
            project.icon_url = icon_url
        db.add(project)
        await db.flush()
        await db.refresh(project)
        return project

    @staticmethod
    async def delete(db: AsyncSession, project: Project) -> None:
        await db.delete(project)
        await db.flush()

    @staticmethod
    async def add_member(
        db: AsyncSession,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
        role: str = ROLE_VIEWER,
    ) -> ProjectMember:
        existing = await ProjectRepository.get_member(db, project_id, user_id)
        if existing:
            existing.role = role
            db.add(existing)
            await db.flush()
            return existing

        member = ProjectMember(project_id=project_id, user_id=user_id, role=role)
        db.add(member)
        await db.flush()
        return member

    @staticmethod
    async def remove_member(
        db: AsyncSession,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        await db.execute(
            delete(ProjectMember).where(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == user_id,
            )
        )
        await db.flush()

    @staticmethod
    async def list_members(
        db: AsyncSession,
        project_id: uuid.UUID,
    ) -> list[ProjectMember]:
        result = await db.execute(
            select(ProjectMember).where(ProjectMember.project_id == project_id)
        )
        return list(result.scalars().all())
