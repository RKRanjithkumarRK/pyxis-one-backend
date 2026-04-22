"""
File upload and management.

Endpoints:
  POST /api/files/upload    multipart upload, extract text, return file_id
  GET  /api/files/{id}      get file metadata
  DELETE /api/files/{id}    delete file
"""

from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.models import FileUpload
from tools.file_analyzer import extract, get_file_metadata

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024   # 20 MB


@router.post("/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    conversation_id: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    # Size check
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")

    filename = file.filename or "upload"
    meta = get_file_metadata(filename, len(contents))

    # Extract content
    extracted = extract(contents, filename)

    record = FileUpload(
        session_id=session_id,
        conversation_id=conversation_id,
        filename=filename,
        content_type=meta["kind"],
        file_size=len(contents),
        extracted_text=extracted.get("content"),
        image_b64=extracted.get("image_b64"),
        page_count=extracted.get("page_count"),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return {
        "file_id": record.id,
        "filename": filename,
        "content_type": meta["kind"],
        "file_size": meta["size"],
        "extension": meta["extension"],
        "page_count": extracted.get("page_count"),
        "truncated": extracted.get("truncated", False),
        "has_image": extracted.get("image_b64") is not None,
        "preview": (extracted.get("content") or "")[:300],
    }


@router.get("/files/{file_id}")
async def get_file(file_id: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(FileUpload).where(FileUpload.id == file_id))
    f = res.scalar_one_or_none()
    if not f:
        raise HTTPException(404, "File not found")
    return {
        "file_id": f.id,
        "filename": f.filename,
        "content_type": f.content_type,
        "file_size": f.file_size,
        "page_count": f.page_count,
        "has_text": bool(f.extracted_text),
        "has_image": bool(f.image_b64),
        "created_at": f.created_at.isoformat(),
    }


@router.delete("/files/{file_id}")
async def delete_file(file_id: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(FileUpload).where(FileUpload.id == file_id))
    f = res.scalar_one_or_none()
    if not f:
        raise HTTPException(404, "File not found")
    await db.delete(f)
    await db.commit()
    return {"deleted": file_id}
