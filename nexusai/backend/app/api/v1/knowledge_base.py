"""Knowledge Base REST API — upload files, track ingest status, search."""
from __future__ import annotations
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.repositories.knowledge_base import KBRepository
from app.services.rag.parser import file_type_from_name, SUPPORTED_TYPES
from app.services.storage import gcs

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


# ── Schemas ──────────────────────────────────────────────────────────────────

class KBCreate(BaseModel):
    name: str
    description: str | None = None
    project_id: str | None = None


class KBUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class FileOut(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    error_msg: str | None

    class Config:
        from_attributes = True


class KBOut(BaseModel):
    id: str
    name: str
    description: str | None
    project_id: str | None
    files: list[FileOut]

    class Config:
        from_attributes = True


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _get_kb_or_404(kb_id: str, user: User, db: AsyncSession):
    kb = await KBRepository.get(db, uuid.UUID(kb_id), user.id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("", response_model=KBOut, status_code=status.HTTP_201_CREATED)
async def create_kb(
    body: KBCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project_id = uuid.UUID(body.project_id) if body.project_id else None
    kb = await KBRepository.create(
        db, current_user.id, body.name, body.description, project_id
    )
    return kb


@router.get("", response_model=list[KBOut])
async def list_kbs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await KBRepository.list_for_user(db, current_user.id)


@router.get("/{kb_id}", response_model=KBOut)
async def get_kb(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_kb_or_404(kb_id, current_user, db)


@router.patch("/{kb_id}", response_model=KBOut)
async def update_kb(
    kb_id: str,
    body: KBUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_kb_or_404(kb_id, current_user, db)
    return await KBRepository.update(db, kb, name=body.name, description=body.description)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_kb_or_404(kb_id, current_user, db)
    from app.services.rag.qdrant_store import delete_kb_chunks
    delete_kb_chunks(kb_id)
    await KBRepository.delete(db, kb)


@router.post("/{kb_id}/files", response_model=FileOut, status_code=status.HTTP_201_CREATED)
async def upload_file(
    kb_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_kb_or_404(kb_id, current_user, db)

    ft = file_type_from_name(file.filename or "")
    if ft not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ft}'. Allowed: {', '.join(sorted(SUPPORTED_TYPES))}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    file_id = str(uuid.uuid4())
    storage_path = gcs.kb_file_path(kb_id, file_id, file.filename or "upload")
    gcs.upload(storage_path, content, content_type=file.content_type or "application/octet-stream")

    kb_file = await KBRepository.create_file(
        db,
        kb_id=kb.id,
        filename=file.filename or "upload",
        file_type=ft,
        file_size=len(content),
        storage_path=storage_path,
    )

    from app.services.rag.tasks import ingest_file
    ingest_file.delay(str(kb_file.id))

    return kb_file


@router.delete("/{kb_id}/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    kb_id: str,
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_kb_or_404(kb_id, current_user, db)
    f = await KBRepository.get_file(db, uuid.UUID(file_id), uuid.UUID(kb_id))
    if not f:
        raise HTTPException(status_code=404, detail="File not found")

    from app.services.rag.qdrant_store import delete_file_chunks
    delete_file_chunks(kb_id, file_id)
    gcs.delete(f.storage_path)
    await KBRepository.delete_file(db, f)


@router.get("/{kb_id}/files/{file_id}", response_model=FileOut)
async def get_file_status(
    kb_id: str,
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_kb_or_404(kb_id, current_user, db)
    f = await KBRepository.get_file(db, uuid.UUID(file_id), uuid.UUID(kb_id))
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    return f
