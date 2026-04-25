"""Image Studio API — generate, upscale, remove background, history."""
from __future__ import annotations
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.image import ImageRequest
from app.models.user import User
from app.services.image.service import MODELS, remove_background, upscale_image

router = APIRouter(prefix="/image", tags=["image"])

VALID_MODELS = set(MODELS.keys())
VALID_SIZES = {512, 768, 1024, 1280, 1536, 1792}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = ""
    model: str = "flux-schnell"
    width: int = 1024
    height: int = 1024
    num_images: int = Field(default=4, ge=1, le=4)


class UpscaleRequest(BaseModel):
    image_url: str
    scale: int = Field(default=4, ge=2, le=4)


class RemoveBgRequest(BaseModel):
    image_url: str


class ImageRequestOut(BaseModel):
    id: str
    prompt: str
    model: str
    width: int
    height: int
    num_images: int
    status: str
    result_urls: list[str] | None
    error_msg: str | None
    created_at: str

    class Config:
        from_attributes = True


@router.post("/generate", response_model=ImageRequestOut, status_code=status.HTTP_202_ACCEPTED)
async def generate(
    body: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.model not in VALID_MODELS:
        raise HTTPException(status_code=422, detail=f"Unknown model: {body.model}")
    if body.width not in VALID_SIZES or body.height not in VALID_SIZES:
        raise HTTPException(status_code=422, detail="Invalid image size")

    req = ImageRequest(
        user_id=current_user.id,
        prompt=body.prompt,
        negative_prompt=body.negative_prompt or None,
        model=body.model,
        width=body.width,
        height=body.height,
        num_images=body.num_images,
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    db.add(req)
    await db.commit()
    await db.refresh(req)

    from app.services.image.tasks import generate_images_task
    generate_images_task.delay(str(req.id), str(current_user.id))

    return req


@router.get("/generate/{request_id}", response_model=ImageRequestOut)
async def get_request(
    request_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    req = (
        await db.execute(
            select(ImageRequest).where(
                ImageRequest.id == uuid.UUID(request_id),
                ImageRequest.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    return req


@router.get("/history", response_model=list[ImageRequestOut])
async def history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = await db.execute(
        select(ImageRequest)
        .where(ImageRequest.user_id == current_user.id)
        .order_by(ImageRequest.created_at.desc())
        .limit(100)
    )
    return list(rows.scalars())


@router.post("/upscale")
async def upscale(
    body: UpscaleRequest,
    current_user: User = Depends(get_current_user),
):
    url = await upscale_image(body.image_url, body.scale)
    return {"url": url}


@router.post("/remove-background")
async def remove_bg(
    body: RemoveBgRequest,
    current_user: User = Depends(get_current_user),
):
    data_uri = await remove_background(body.image_url)
    return {"url": data_uri}


@router.get("/models")
async def list_models():
    return {"models": [{"id": k, **v} for k, v in MODELS.items()]}
