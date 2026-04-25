from __future__ import annotations
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import require_bearer, decode_token
from app.repositories.memory import MemoryRepository

router = APIRouter(prefix="/memory", tags=["memory"])


def _user_id(creds: HTTPAuthorizationCredentials) -> uuid.UUID:
    return uuid.UUID(decode_token(creds.credentials)["sub"])


class MemoryOut(BaseModel):
    id: str
    fact: str
    use_count: int
    created_at: str
    last_used_at: str | None

    @classmethod
    def from_orm(cls, m) -> "MemoryOut":
        return cls(
            id=str(m.id),
            fact=m.fact,
            use_count=m.use_count or 0,
            created_at=m.created_at.isoformat(),
            last_used_at=m.last_used_at.isoformat() if m.last_used_at and not isinstance(m.last_used_at, str) else m.last_used_at,
        )


@router.get("", response_model=list[MemoryOut])
async def list_memories(
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    mems = await MemoryRepository.list_for_user(db, user_id)
    return [MemoryOut.from_orm(m) for m in mems]


@router.get("/stats")
async def memory_stats(
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    count = await MemoryRepository.count(db, user_id)
    return {"count": count}


@router.delete("", status_code=status.HTTP_200_OK)
async def clear_all_memories(
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    deleted = await MemoryRepository.delete_all_for_user(db, user_id)
    await db.commit()
    return {"deleted": deleted}


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    mem = await MemoryRepository.get(db, memory_id, user_id)
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")
    await MemoryRepository.delete(db, mem)
    await db.commit()
