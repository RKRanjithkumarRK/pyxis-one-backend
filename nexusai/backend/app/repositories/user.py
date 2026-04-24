from __future__ import annotations
import uuid
from typing import Optional
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User, AuthProvider


class UserRepository:

    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
        result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_email(db: AsyncSession, email: str) -> Optional[User]:
        result = await db.execute(
            select(User).where(User.email == email.lower(), User.is_active == True)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_provider(
        db: AsyncSession, provider: AuthProvider, provider_id: str
    ) -> Optional[User]:
        result = await db.execute(
            select(User).where(
                User.provider == provider,
                User.provider_id == provider_id,
                User.is_active == True,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        email: str | None = None,
        name: str | None = None,
        hashed_password: str | None = None,
        provider: AuthProvider = AuthProvider.email,
        provider_id: str | None = None,
        avatar_url: str | None = None,
        is_verified: bool = False,
    ) -> User:
        user = User(
            email=email.lower() if email else None,
            name=name,
            hashed_password=hashed_password,
            provider=provider,
            provider_id=provider_id,
            avatar_url=avatar_url,
            is_verified=is_verified,
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)
        return user

    @staticmethod
    async def update(
        db: AsyncSession,
        user_id: uuid.UUID,
        **kwargs,
    ) -> Optional[User]:
        await db.execute(
            update(User).where(User.id == user_id).values(**kwargs)
        )
        return await UserRepository.get_by_id(db, user_id)

    @staticmethod
    async def create_guest(db: AsyncSession) -> User:
        user = User(
            provider=AuthProvider.guest,
            is_active=True,
            is_verified=False,
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)
        return user
