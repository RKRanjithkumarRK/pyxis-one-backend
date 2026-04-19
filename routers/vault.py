from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.schemas import (
    VaultStoreRequest, VaultStoreResponse,
    VaultSearchRequest, VaultSearchResponse,
    VaultTimelineResponse,
)
from engines.vault import vault_engine

router = APIRouter()


@router.post("/vault/store", response_model=VaultStoreResponse)
async def vault_store(request: VaultStoreRequest, db: AsyncSession = Depends(get_db)):
    try:
        entry_id = await vault_engine.store(
            session_id=request.session_id,
            content=request.content,
            concept_tags=request.concept_tags,
            emotion_tags=request.emotion_tags,
        )
        return VaultStoreResponse(entry_id=entry_id, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vault store failed: {e}")


@router.get("/vault/timeline/{session_id}", response_model=VaultTimelineResponse)
async def vault_timeline(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        entries = await vault_engine.get_timeline(session_id)
        return VaultTimelineResponse(session_id=session_id, entries=entries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vault timeline failed: {e}")


@router.post("/vault/search", response_model=VaultSearchResponse)
async def vault_search(request: VaultSearchRequest, db: AsyncSession = Depends(get_db)):
    try:
        results = await vault_engine.search(request.session_id, request.query)
        return VaultSearchResponse(session_id=request.session_id, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vault search failed: {e}")


@router.get("/vault/export/{session_id}")
async def vault_export(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        content_bytes = await vault_engine.export_pdf(session_id)
        return Response(
            content=content_bytes,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=pyxis_vault_{session_id[:8]}.txt"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vault export failed: {e}")
