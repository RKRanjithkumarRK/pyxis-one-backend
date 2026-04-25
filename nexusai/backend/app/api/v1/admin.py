"""Admin Console API — org dashboard, user management, audit logs, SSO, content filters."""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, delete, text

from app.api.deps import get_current_user, get_db
from app.core.redis import redis_client
from app.models.user import User, SubscriptionPlan

router = APIRouter(prefix="/admin", tags=["admin"])

AUDIT_LOG_TTL = 86400 * 90  # 90 days


def _require_admin(user: User = Depends(get_current_user)) -> User:
    if not getattr(user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ─── Audit logging ────────────────────────────────────────

async def audit_log(action: str, actor_id: str, target: str | None = None, metadata: dict | None = None) -> None:
    entry = {
        "action": action,
        "actor_id": actor_id,
        "target": target,
        "metadata": metadata or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await redis_client.lpush("audit:log", json.dumps(entry))
    await redis_client.ltrim("audit:log", 0, 9999)


@router.get("/audit-logs")
async def get_audit_logs(
    limit: int = Query(default=50, ge=1, le=500),
    admin: User = Depends(_require_admin),
):
    raw = await redis_client.lrange("audit:log", 0, limit - 1)
    return [json.loads(r) for r in raw]


# ─── Org dashboard ────────────────────────────────────────

@router.get("/dashboard")
async def get_dashboard(
    admin: User = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
):
    total_users = (await db.execute(select(func.count(User.id)))).scalar_one()
    active_today = (await db.execute(
        text("SELECT COUNT(*) FROM users WHERE updated_at > NOW() - INTERVAL '1 day'")
    )).scalar_one()
    plan_dist = {}
    for plan in SubscriptionPlan:
        count = (await db.execute(
            select(func.count(User.id)).where(User.plan == plan)
        )).scalar_one()
        plan_dist[plan.value] = count

    return {
        "total_users": total_users,
        "active_today": active_today,
        "plan_distribution": plan_dist,
    }


# ─── User management ─────────────────────────────────────

@router.get("/users")
async def list_users(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    search: str | None = None,
    admin: User = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
):
    q = select(User).order_by(User.created_at.desc()).offset((page - 1) * limit).limit(limit)
    if search:
        q = q.where(User.email.ilike(f"%{search}%"))
    users = (await db.execute(q)).scalars().all()
    return [{"id": str(u.id), "email": u.email, "name": u.name, "plan": u.plan.value if hasattr(u.plan, "value") else u.plan, "created_at": u.created_at.isoformat()} for u in users]


class UpdateUserAdminRequest(BaseModel):
    plan: str | None = None
    is_admin: bool | None = None
    is_active: bool | None = None


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    payload: UpdateUserAdminRequest,
    admin: User = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    target = result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    if payload.plan:
        target.plan = SubscriptionPlan(payload.plan)
    if payload.is_admin is not None:
        target.is_admin = payload.is_admin
    await db.commit()
    await audit_log("update_user", str(admin.id), user_id, {"changes": payload.model_dump(exclude_none=True)})
    return {"status": "updated"}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    admin: User = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    target = result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    await db.delete(target)
    await db.commit()
    await audit_log("delete_user", str(admin.id), user_id)
    return {"status": "deleted"}


# ─── Content filters ──────────────────────────────────────

CONTENT_FILTER_KEY = "admin:content_filters"


class ContentFilterRequest(BaseModel):
    action: Literal["add", "remove"]
    pattern: str


@router.post("/content-filters")
async def manage_content_filters(
    payload: ContentFilterRequest,
    admin: User = Depends(_require_admin),
):
    raw = await redis_client.get(CONTENT_FILTER_KEY)
    filters: list[str] = json.loads(raw) if raw else []
    if payload.action == "add" and payload.pattern not in filters:
        filters.append(payload.pattern)
    elif payload.action == "remove":
        filters = [f for f in filters if f != payload.pattern]
    await redis_client.set(CONTENT_FILTER_KEY, json.dumps(filters))
    await audit_log("update_content_filters", str(admin.id), metadata={"action": payload.action, "pattern": payload.pattern})
    return {"filters": filters}


@router.get("/content-filters")
async def get_content_filters(admin: User = Depends(_require_admin)):
    raw = await redis_client.get(CONTENT_FILTER_KEY)
    return {"filters": json.loads(raw) if raw else []}


# ─── SSO configuration ────────────────────────────────────

SSO_CONFIG_KEY = "admin:sso_config"


class SSOConfigRequest(BaseModel):
    type: Literal["saml", "oidc", "scim"]
    config: dict


@router.get("/sso")
async def get_sso_config(admin: User = Depends(_require_admin)):
    raw = await redis_client.get(SSO_CONFIG_KEY)
    return json.loads(raw) if raw else {}


@router.put("/sso")
async def set_sso_config(
    payload: SSOConfigRequest,
    admin: User = Depends(_require_admin),
):
    config_data = {"type": payload.type, "config": payload.config, "updated_at": datetime.now(timezone.utc).isoformat()}
    await redis_client.set(SSO_CONFIG_KEY, json.dumps(config_data))
    await audit_log("update_sso", str(admin.id), metadata={"type": payload.type})
    return {"status": "saved", "type": payload.type}


# ─── Data retention ───────────────────────────────────────

RETENTION_KEY = "admin:retention_days"


class RetentionRequest(BaseModel):
    days: int


@router.get("/retention")
async def get_retention(admin: User = Depends(_require_admin)):
    raw = await redis_client.get(RETENTION_KEY)
    return {"retention_days": int(raw) if raw else 365}


@router.put("/retention")
async def set_retention(
    payload: RetentionRequest,
    admin: User = Depends(_require_admin),
):
    await redis_client.set(RETENTION_KEY, str(payload.days))
    await audit_log("update_retention", str(admin.id), metadata={"days": payload.days})
    return {"status": "saved", "retention_days": payload.days}
