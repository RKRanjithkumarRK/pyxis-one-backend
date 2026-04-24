from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings
from app.core.redis import redis_client

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        "jti": str(uuid.uuid4()),
        **(extra or {}),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        "jti": str(uuid.uuid4()),
        "type": "refresh",
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def revoke_token(jti: str, ttl_seconds: int = 86400 * 30) -> None:
    await redis_client.setex(f"revoked:jti:{jti}", ttl_seconds, "1")


async def is_revoked(jti: str) -> bool:
    return bool(await redis_client.exists(f"revoked:jti:{jti}"))


class _GuestUser:
    """Minimal user-like object for unauthenticated guest sessions."""
    id = "guest"
    email = None
    plan = "free"
    is_admin = False
    memory_enabled = False


async def get_current_user_or_guest(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: "AsyncSession | None" = None,  # injected by router-level override; None for health
) -> Any:
    """Returns authenticated User row (or GuestUser for bearer-less requests)."""
    if credentials is None:
        return _GuestUser()

    payload = decode_token(credentials.credentials)

    if await is_revoked(payload.get("jti", "")):
        raise HTTPException(status_code=401, detail="Token has been revoked")

    # Inline import to avoid circular dependency at module load time
    if db is not None:
        from app.repositories.user import UserRepository
        import uuid as _uuid
        user = await UserRepository.get_by_id(db, _uuid.UUID(payload["sub"]))
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user

    # Fallback for endpoints that don't inject db (e.g. health check)
    class _TokenUser:
        id = payload["sub"]
        email = payload.get("email")
        plan = payload.get("plan", "free")
        is_admin = payload.get("is_admin", False)
        memory_enabled = True
        is_guest = payload.get("is_guest", False)

    return _TokenUser()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> Any:
    """Strict version — returns 401 if no token provided."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return await get_current_user_or_guest(credentials)
