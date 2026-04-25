"""Shared FastAPI dependencies."""
from __future__ import annotations
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import bearer, decode_token, is_revoked
from app.models.user import User


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = decode_token(credentials.credentials)
    if await is_revoked(payload.get("jti", "")):
        raise HTTPException(status_code=401, detail="Token revoked")
    user_id = payload["sub"]
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


__all__ = ["get_current_user", "get_db"]
