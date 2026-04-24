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
) -> Any:
    """Returns the authenticated user, or a GuestUser for unauthenticated requests.
    Full DB lookup is wired in Phase 2; this stub handles the dependency for Phase 1.
    """
    if credentials is None:
        return _GuestUser()

    payload = decode_token(credentials.credentials)

    if await is_revoked(payload.get("jti", "")):
        raise HTTPException(status_code=401, detail="Token has been revoked")

    # Phase 2 will do: user = await UserRepository.get(db, uuid.UUID(payload["sub"]))
    # For now return a minimal object so chat/health endpoints don't break
    class _AuthUser:
        id = payload["sub"]
        email = payload.get("email")
        plan = payload.get("plan", "free")
        is_admin = payload.get("is_admin", False)
        memory_enabled = True

    return _AuthUser()
