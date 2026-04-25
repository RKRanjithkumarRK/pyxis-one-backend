"""Settings API — profile, BYOK, 2FA TOTP, sessions, WebAuthn."""
from __future__ import annotations
import base64
import json
import os
import secrets
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.api.deps import get_current_user, get_db
from app.core.redis import redis_client
from app.models.user import User

router = APIRouter(prefix="/settings", tags=["settings"])

# ─── Encryption helpers (envelope: AES-256-GCM) ──────────

def _derive_key() -> bytes:
    from app.core.config import settings
    import hashlib
    return hashlib.sha256(settings.SECRET_KEY.encode()).digest()


def _encrypt(plaintext: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = _derive_key()
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext.encode(), None)
    return base64.b64encode(nonce + ct).decode()


def _decrypt(ciphertext: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    data = base64.b64decode(ciphertext)
    nonce, ct = data[:12], data[12:]
    return AESGCM(_derive_key()).decrypt(nonce, ct, None).decode()


# ─── Profile ──────────────────────────────────────────────

class UpdateProfileRequest(BaseModel):
    name: str | None = Field(default=None, max_length=256)
    email: str | None = Field(default=None, max_length=320)
    avatar_url: str | None = Field(default=None, max_length=2048)


@router.get("/profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "name": current_user.name,
        "avatar_url": current_user.avatar_url,
        "plan": current_user.plan.value if hasattr(current_user.plan, "value") else current_user.plan,
    }


@router.patch("/profile")
async def update_profile(
    payload: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(current_user, field, value)
    await db.commit()
    await db.refresh(current_user)
    return {"status": "updated"}


# ─── BYOK (Bring Your Own Key) ────────────────────────────

BYOK_KEY = "byok:{user_id}"
SUPPORTED_PROVIDERS = {"openai", "anthropic", "google", "groq", "mistral", "cerebras", "sambanova"}


class BYOKRequest(BaseModel):
    provider: str
    api_key: str = Field(min_length=8)


@router.get("/byok")
async def get_byok_keys(current_user: User = Depends(get_current_user)):
    """Return list of configured BYOK providers (keys masked)."""
    data = await redis_client.get(BYOK_KEY.format(user_id=current_user.id))
    if not data:
        return {"providers": []}
    keys = json.loads(data)
    return {"providers": list(keys.keys())}


@router.put("/byok")
async def set_byok_key(
    payload: BYOKRequest,
    current_user: User = Depends(get_current_user),
):
    """Store encrypted BYOK key for a provider."""
    if payload.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {payload.provider}")
    redis_key = BYOK_KEY.format(user_id=current_user.id)
    data = await redis_client.get(redis_key)
    keys = json.loads(data) if data else {}
    keys[payload.provider] = _encrypt(payload.api_key)
    await redis_client.set(redis_key, json.dumps(keys))
    return {"status": "saved", "provider": payload.provider}


@router.delete("/byok/{provider}")
async def delete_byok_key(
    provider: str,
    current_user: User = Depends(get_current_user),
):
    redis_key = BYOK_KEY.format(user_id=current_user.id)
    data = await redis_client.get(redis_key)
    if not data:
        raise HTTPException(status_code=404, detail="No BYOK keys configured")
    keys = json.loads(data)
    keys.pop(provider, None)
    await redis_client.set(redis_key, json.dumps(keys))
    return {"status": "deleted", "provider": provider}


async def get_byok_api_key(user_id: str, provider: str) -> str | None:
    """Used by LiteLLM router to get user's own key."""
    data = await redis_client.get(BYOK_KEY.format(user_id=user_id))
    if not data:
        return None
    keys = json.loads(data)
    encrypted = keys.get(provider)
    return _decrypt(encrypted) if encrypted else None


# ─── 2FA TOTP ─────────────────────────────────────────────

TOTP_SECRET_KEY = "totp:secret:{user_id}"
TOTP_ENABLED_KEY = "totp:enabled:{user_id}"


@router.post("/2fa/setup")
async def setup_2fa(current_user: User = Depends(get_current_user)):
    """Generate a TOTP secret and QR code URI."""
    import pyotp
    secret = pyotp.random_base32()
    await redis_client.setex(TOTP_SECRET_KEY.format(user_id=current_user.id), 600, secret)
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(
        name=current_user.email or str(current_user.id),
        issuer_name="NexusAI",
    )
    return {"secret": secret, "uri": uri}


class VerifyTOTPRequest(BaseModel):
    code: str = Field(min_length=6, max_length=6)
    secret: str | None = None


@router.post("/2fa/verify")
async def verify_2fa(
    payload: VerifyTOTPRequest,
    current_user: User = Depends(get_current_user),
):
    import pyotp
    secret = payload.secret or (await redis_client.get(TOTP_SECRET_KEY.format(user_id=current_user.id)) or b"").decode()
    if not secret:
        raise HTTPException(status_code=400, detail="No pending 2FA setup")
    totp = pyotp.TOTP(secret)
    if not totp.verify(payload.code, valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid code")
    # Persist the verified secret
    await redis_client.set(TOTP_ENABLED_KEY.format(user_id=current_user.id), _encrypt(secret))
    return {"status": "2fa_enabled"}


@router.delete("/2fa")
async def disable_2fa(
    payload: VerifyTOTPRequest,
    current_user: User = Depends(get_current_user),
):
    import pyotp
    enc = await redis_client.get(TOTP_ENABLED_KEY.format(user_id=current_user.id))
    if not enc:
        raise HTTPException(status_code=400, detail="2FA not enabled")
    secret = _decrypt(enc.decode())
    totp = pyotp.TOTP(secret)
    if not totp.verify(payload.code, valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid code")
    await redis_client.delete(TOTP_ENABLED_KEY.format(user_id=current_user.id))
    return {"status": "2fa_disabled"}


@router.get("/2fa/status")
async def get_2fa_status(current_user: User = Depends(get_current_user)):
    enabled = bool(await redis_client.exists(TOTP_ENABLED_KEY.format(user_id=current_user.id)))
    return {"enabled": enabled}


# ─── Session management ───────────────────────────────────

@router.get("/sessions")
async def list_sessions(current_user: User = Depends(get_current_user)):
    """List active sessions stored in Redis (jti keys)."""
    pattern = f"session:{current_user.id}:*"
    keys = await redis_client.keys(pattern)
    sessions = []
    for key in keys[:50]:
        data = await redis_client.get(key)
        if data:
            try:
                sessions.append(json.loads(data))
            except Exception:
                pass
    return {"sessions": sessions}


@router.delete("/sessions/{jti}")
async def revoke_session(
    jti: str,
    current_user: User = Depends(get_current_user),
):
    from app.core.security import revoke_token
    await revoke_token(jti)
    return {"status": "revoked", "jti": jti}


@router.delete("/sessions")
async def revoke_all_sessions(current_user: User = Depends(get_current_user)):
    """Revoke all active sessions for the current user."""
    pattern = f"session:{current_user.id}:*"
    keys = await redis_client.keys(pattern)
    for key in keys:
        data = await redis_client.get(key)
        if data:
            try:
                info = json.loads(data)
                from app.core.security import revoke_token
                await revoke_token(info["jti"])
            except Exception:
                pass
    return {"status": "all_revoked", "count": len(keys)}


# ─── Delete account ───────────────────────────────────────

@router.delete("/account")
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Permanently delete account and all data. Irreversible."""
    await db.delete(current_user)
    await db.commit()
    return {"status": "account_deleted"}
