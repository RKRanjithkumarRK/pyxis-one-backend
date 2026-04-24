from __future__ import annotations
import secrets
import uuid
from datetime import timedelta
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.redis import redis_client
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    revoke_token,
)
from app.models.user import AuthProvider, User
from app.repositories.user import UserRepository
from app.schemas.auth import AuthResponse, UserResponse
from app.services.auth.email import send_magic_link, send_verification_email
from app.services.auth.oauth import OAUTH_HANDLERS

GUEST_MSG_LIMIT = 10
MAGIC_LINK_TTL = 60 * 15         # 15 minutes
EMAIL_VERIFY_TTL = 60 * 60 * 24  # 24 hours


def _build_auth_response(user: User) -> AuthResponse:
    access_token = create_access_token(
        str(user.id),
        extra={"email": user.email, "plan": user.plan, "is_admin": user.is_admin},
    )
    refresh_token = create_refresh_token(str(user.id))
    return AuthResponse(
        user=UserResponse.model_validate(user),
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


class AuthService:

    # ─── Email + password ─────────────────────────────────────────────────────

    @staticmethod
    async def register(
        db: AsyncSession,
        email: str,
        password: str,
        name: str,
    ) -> AuthResponse:
        existing = await UserRepository.get_by_email(db, email)
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

        hashed = hash_password(password)
        user = await UserRepository.create(
            db,
            email=email,
            name=name,
            hashed_password=hashed,
            provider=AuthProvider.email,
            is_verified=False,
        )

        # Send verification email (non-blocking; failure doesn't break signup)
        verify_token = secrets.token_urlsafe(32)
        await redis_client.setex(
            f"verify_email:{verify_token}", EMAIL_VERIFY_TTL, str(user.id)
        )
        await send_verification_email(email, name, verify_token)

        return _build_auth_response(user)

    @staticmethod
    async def login(db: AsyncSession, email: str, password: str) -> AuthResponse:
        user = await UserRepository.get_by_email(db, email)
        if not user or not user.hashed_password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account deactivated")
        return _build_auth_response(user)

    # ─── Token lifecycle ──────────────────────────────────────────────────────

    @staticmethod
    async def refresh(db: AsyncSession, refresh_token: str) -> AuthResponse:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Not a refresh token")

        jti = payload.get("jti", "")
        if await redis_client.exists(f"revoked:jti:{jti}"):
            raise HTTPException(status_code=401, detail="Token revoked")

        # Rotate: revoke old refresh token immediately
        await revoke_token(jti, ttl_seconds=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400)

        user = await UserRepository.get_by_id(db, uuid.UUID(payload["sub"]))
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found")

        return _build_auth_response(user)

    @staticmethod
    async def logout(jti: str) -> None:
        await revoke_token(jti, ttl_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60 + 60)

    @staticmethod
    async def verify_email(db: AsyncSession, token: str) -> AuthResponse:
        user_id = await redis_client.get(f"verify_email:{token}")
        if not user_id:
            raise HTTPException(status_code=400, detail="Invalid or expired verification link")

        await redis_client.delete(f"verify_email:{token}")
        user = await UserRepository.update(db, uuid.UUID(user_id), is_verified=True)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return _build_auth_response(user)

    # ─── Magic link ───────────────────────────────────────────────────────────

    @staticmethod
    async def send_magic_link_email(db: AsyncSession, email: str) -> None:
        user = await UserRepository.get_by_email(db, email)
        if not user:
            # Create a new account automatically for magic-link signups
            user = await UserRepository.create(
                db,
                email=email,
                provider=AuthProvider.magic_link,
                is_verified=True,
            )

        token = secrets.token_urlsafe(32)
        await redis_client.setex(f"magic_link:{token}", MAGIC_LINK_TTL, str(user.id))
        await send_magic_link(email, user.name or "", token)

    @staticmethod
    async def verify_magic_link(db: AsyncSession, token: str) -> AuthResponse:
        user_id = await redis_client.get(f"magic_link:{token}")
        if not user_id:
            raise HTTPException(status_code=400, detail="Invalid or expired magic link")

        await redis_client.delete(f"magic_link:{token}")
        user = await UserRepository.get_by_id(db, uuid.UUID(user_id))
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not user.is_verified:
            await UserRepository.update(db, user.id, is_verified=True)
            await db.refresh(user)

        return _build_auth_response(user)

    # ─── OAuth ────────────────────────────────────────────────────────────────

    @staticmethod
    async def oauth_callback(
        db: AsyncSession, provider: str, code: str, redirect_uri: str
    ) -> AuthResponse:
        handler = OAUTH_HANDLERS.get(provider)
        if not handler:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

        try:
            info = await handler(code, redirect_uri)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"OAuth exchange failed: {exc}")

        auth_provider = AuthProvider(provider)

        # Look up by provider_id first, then email
        user = await UserRepository.get_by_provider(db, auth_provider, info.provider_id)
        if not user and info.email:
            user = await UserRepository.get_by_email(db, info.email)
            if user:
                # Link provider to existing account
                await UserRepository.update(
                    db, user.id, provider_id=info.provider_id
                )
                await db.refresh(user)

        if not user:
            user = await UserRepository.create(
                db,
                email=info.email,
                name=info.name,
                avatar_url=info.avatar_url,
                provider=auth_provider,
                provider_id=info.provider_id,
                is_verified=True,
            )

        return _build_auth_response(user)

    # ─── Guest mode ───────────────────────────────────────────────────────────

    @staticmethod
    async def create_guest_session(db: AsyncSession) -> dict:
        user = await UserRepository.create_guest(db)
        token = create_access_token(
            str(user.id),
            extra={"plan": "free", "is_guest": True},
        )
        # Track message count in Redis
        key = f"guest:msgs:{user.id}"
        await redis_client.setex(key, 60 * 60 * 24 * 7, "0")  # 7-day TTL
        return {
            "guest_id": str(user.id),
            "messages_remaining": GUEST_MSG_LIMIT,
            "token": token,
        }

    @staticmethod
    async def get_guest_messages_remaining(user_id: str) -> int:
        count_str = await redis_client.get(f"guest:msgs:{user_id}")
        if count_str is None:
            return 0
        used = int(count_str)
        return max(0, GUEST_MSG_LIMIT - used)

    @staticmethod
    async def increment_guest_message_count(user_id: str) -> int:
        key = f"guest:msgs:{user_id}"
        count = await redis_client.incr(key)
        if count == 1:
            await redis_client.expire(key, 60 * 60 * 24 * 7)
        return max(0, GUEST_MSG_LIMIT - count)
