from __future__ import annotations
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_token, get_current_user_or_guest, require_bearer
from app.repositories.user import UserRepository
from app.schemas.auth import (
    AuthResponse,
    GuestSessionResponse,
    LoginRequest,
    MagicLinkRequest,
    MagicLinkVerifyRequest,
    OAuthCallbackRequest,
    RefreshRequest,
    RegisterRequest,
    UpdateProfileRequest,
    UserResponse,
)
from app.services.auth.service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse, status_code=201)
async def register(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    return await AuthService.register(
        db, email=payload.email, password=payload.password, name=payload.name
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    return await AuthService.login(db, email=payload.email, password=payload.password)


@router.post("/logout", status_code=204)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
):
    payload = decode_token(credentials.credentials)
    await AuthService.logout(jti=payload.get("jti", ""))
    return Response(status_code=204)


@router.post("/refresh", response_model=AuthResponse)
async def refresh(
    payload: RefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    return await AuthService.refresh(db, refresh_token=payload.refresh_token)


@router.post("/magic-link", status_code=202)
async def send_magic_link(
    payload: MagicLinkRequest,
    db: AsyncSession = Depends(get_db),
):
    await AuthService.send_magic_link_email(db, email=payload.email)
    return {"message": "If that email exists, a sign-in link has been sent"}


@router.post("/magic-link/verify", response_model=AuthResponse)
async def verify_magic_link(
    payload: MagicLinkVerifyRequest,
    db: AsyncSession = Depends(get_db),
):
    return await AuthService.verify_magic_link(db, token=payload.token)


@router.post("/verify-email", response_model=AuthResponse)
async def verify_email(
    payload: MagicLinkVerifyRequest,
    db: AsyncSession = Depends(get_db),
):
    return await AuthService.verify_email(db, token=payload.token)


@router.post("/oauth/callback", response_model=AuthResponse)
async def oauth_callback(
    payload: OAuthCallbackRequest,
    db: AsyncSession = Depends(get_db),
):
    return await AuthService.oauth_callback(
        db,
        provider=payload.provider,
        code=payload.code,
        redirect_uri=payload.redirect_uri,
    )


@router.post("/guest", response_model=GuestSessionResponse, status_code=201)
async def create_guest(
    db: AsyncSession = Depends(get_db),
):
    return await AuthService.create_guest_session(db)


@router.get("/me", response_model=UserResponse)
async def get_me(
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    payload = decode_token(credentials.credentials)
    user = await UserRepository.get_by_id(db, uuid.UUID(payload["sub"]))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.model_validate(user)


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: UpdateProfileRequest,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    payload = decode_token(credentials.credentials)
    user_id = uuid.UUID(payload["sub"])
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")
    user = await UserRepository.update(db, user_id, **updates)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.model_validate(user)
