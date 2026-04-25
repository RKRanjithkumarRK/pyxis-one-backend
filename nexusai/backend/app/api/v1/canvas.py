from __future__ import annotations
import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import require_bearer, decode_token, optional_bearer
from app.repositories.canvas import CanvasRepository
from app.services.canvas.service import ai_edit, export_docx, export_markdown, export_html

router = APIRouter(prefix="/canvas", tags=["canvas"])


def _user_id(credentials: HTTPAuthorizationCredentials) -> uuid.UUID:
    return uuid.UUID(decode_token(credentials.credentials)["sub"])


class DocOut(BaseModel):
    id: str
    title: str
    content: dict | None
    version: int
    is_public: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_orm(cls, d) -> "DocOut":
        return cls(
            id=str(d.id),
            title=d.title,
            content=d.content,
            version=d.version,
            is_public=d.is_public,
            created_at=d.created_at.isoformat(),
            updated_at=d.updated_at.isoformat(),
        )


class DocListItem(BaseModel):
    id: str
    title: str
    version: int
    is_public: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_orm(cls, d) -> "DocListItem":
        return cls(
            id=str(d.id),
            title=d.title,
            version=d.version,
            is_public=d.is_public,
            created_at=d.created_at.isoformat(),
            updated_at=d.updated_at.isoformat(),
        )


class VersionOut(BaseModel):
    id: str
    document_id: str
    version: int
    title: str | None
    created_at: str

    @classmethod
    def from_orm(cls, v) -> "VersionOut":
        return cls(
            id=str(v.id),
            document_id=str(v.document_id),
            version=v.version,
            title=v.title,
            created_at=v.created_at.isoformat(),
        )


class CreateDocRequest(BaseModel):
    title: str = Field(default="Untitled", max_length=512)


class UpdateDocRequest(BaseModel):
    title: str | None = Field(default=None, max_length=512)
    content: dict | None = None
    content_text: str | None = None
    is_public: bool | None = None
    save_version: bool = False


class AIEditRequest(BaseModel):
    selected_text: str = Field(min_length=1, max_length=20000)
    instruction: str = Field(min_length=1, max_length=1000)
    context: str = Field(default="", max_length=2000)


class AIEditResponse(BaseModel):
    original: str
    suggested: str


@router.post("", response_model=DocOut, status_code=status.HTTP_201_CREATED)
async def create_doc(
    payload: CreateDocRequest,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    doc = await CanvasRepository.create(db, user_id, payload.title)
    await db.commit()
    await db.refresh(doc)
    return DocOut.from_orm(doc)


@router.get("", response_model=list[DocListItem])
async def list_docs(
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    docs = await CanvasRepository.list_for_user(db, user_id)
    return [DocListItem.from_orm(d) for d in docs]


@router.get("/{doc_id}", response_model=DocOut)
async def get_doc(
    doc_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_bearer),
    db: AsyncSession = Depends(get_db),
):
    doc = await CanvasRepository.get(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.is_public:
        if not credentials:
            raise HTTPException(status_code=403, detail="Access denied")
        user_id = _user_id(credentials)
        if doc.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
    return DocOut.from_orm(doc)


@router.patch("/{doc_id}", response_model=DocOut)
async def update_doc(
    doc_id: uuid.UUID,
    payload: UpdateDocRequest,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    doc = await CanvasRepository.get(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    if payload.is_public is not None:
        doc.is_public = payload.is_public

    doc = await CanvasRepository.update(
        db, doc,
        title=payload.title,
        content=payload.content,
        content_text=payload.content_text,
        save_version=payload.save_version,
    )
    await db.commit()
    await db.refresh(doc)
    return DocOut.from_orm(doc)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_doc(
    doc_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    doc = await CanvasRepository.get(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    await CanvasRepository.delete(db, doc)
    await db.commit()


@router.post("/{doc_id}/ai-edit", response_model=AIEditResponse)
async def ai_edit_endpoint(
    doc_id: uuid.UUID,
    payload: AIEditRequest,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    doc = await CanvasRepository.get(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        suggested = await ai_edit(
            payload.selected_text,
            payload.instruction,
            context=payload.context,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI edit failed: {exc}")

    return AIEditResponse(original=payload.selected_text, suggested=suggested)


@router.get("/{doc_id}/versions", response_model=list[VersionOut])
async def list_versions(
    doc_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    doc = await CanvasRepository.get(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    versions = await CanvasRepository.list_versions(db, doc_id)
    return [VersionOut.from_orm(v) for v in versions]


@router.post("/{doc_id}/versions/{version}/restore", response_model=DocOut)
async def restore_version(
    doc_id: uuid.UUID,
    version: int,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    doc = await CanvasRepository.get(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    ver = await CanvasRepository.get_version(db, doc_id, version)
    if not ver:
        raise HTTPException(status_code=404, detail="Version not found")

    doc = await CanvasRepository.update(
        db, doc,
        title=ver.title,
        content=ver.content,
        save_version=True,
    )
    await db.commit()
    await db.refresh(doc)
    return DocOut.from_orm(doc)


@router.get("/{doc_id}/export")
async def export_doc(
    doc_id: uuid.UUID,
    format: Literal["md", "html", "docx"] = Query(default="md"),
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_bearer),
    db: AsyncSession = Depends(get_db),
):
    doc = await CanvasRepository.get(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.is_public:
        if not credentials:
            raise HTTPException(status_code=403, detail="Access denied")
        user_id = _user_id(credentials)
        if doc.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

    html_content = doc.content_text or ""

    if format == "md":
        md = export_markdown(html_content) if "<" in html_content else html_content
        filename = f"{doc.title}.md"
        return Response(
            content=md,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    elif format == "html":
        full_html = export_html(doc.title, html_content)
        filename = f"{doc.title}.html"
        return Response(
            content=full_html,
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    elif format == "docx":
        docx_bytes = export_docx(doc.title, html_content)
        filename = f"{doc.title}.docx"
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
