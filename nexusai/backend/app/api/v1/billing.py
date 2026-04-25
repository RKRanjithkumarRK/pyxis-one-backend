"""Billing API — Stripe Checkout, Portal, webhooks, feature gates."""
from __future__ import annotations
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.redis import redis_client
from app.models.user import User, SubscriptionPlan

router = APIRouter(prefix="/billing", tags=["billing"])

# ─── Plan configuration ───────────────────────────────────

PLANS = {
    "free": {
        "name": "Free",
        "price_usd": 0,
        "features": {
            "messages_per_day": 20,
            "models": ["groq/llama-3.3-70b-versatile", "gemini-2.0-flash"],
            "image_gen_per_day": 5,
            "research_per_day": 1,
            "projects": 3,
            "kb_mb": 50,
        },
    },
    "plus": {
        "name": "Plus",
        "price_usd": 20,
        "stripe_price_id": settings.STRIPE_PLUS_PRICE_ID if hasattr(settings, "STRIPE_PLUS_PRICE_ID") else None,
        "features": {
            "messages_per_day": 500,
            "models": "all",
            "image_gen_per_day": 50,
            "research_per_day": 10,
            "projects": 20,
            "kb_mb": 1000,
            "voice": True,
            "computer_use": True,
        },
    },
    "team": {
        "name": "Team",
        "price_usd": 30,
        "stripe_price_id": settings.STRIPE_TEAM_PRICE_ID if hasattr(settings, "STRIPE_TEAM_PRICE_ID") else None,
        "features": {
            "messages_per_day": 2000,
            "models": "all",
            "image_gen_per_day": 200,
            "research_per_day": 50,
            "projects": 100,
            "kb_mb": 10000,
            "voice": True,
            "computer_use": True,
            "shared_projects": True,
            "sso": True,
        },
    },
    "enterprise": {
        "name": "Enterprise",
        "price_usd": -1,
        "features": {"messages_per_day": -1, "models": "all"},
    },
}


@router.get("/plans")
async def list_plans():
    return PLANS


@router.get("/subscription")
async def get_subscription(current_user: User = Depends(get_current_user)):
    plan_name = current_user.plan.value if hasattr(current_user.plan, "value") else str(current_user.plan)
    return {
        "plan": plan_name,
        "features": PLANS.get(plan_name, PLANS["free"])["features"],
    }


# ─── Stripe Checkout ──────────────────────────────────────

class CheckoutRequest(BaseModel):
    plan: str
    success_url: str
    cancel_url: str


@router.post("/checkout")
async def create_checkout(
    payload: CheckoutRequest,
    current_user: User = Depends(get_current_user),
):
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Billing not configured")
    if payload.plan not in PLANS:
        raise HTTPException(status_code=400, detail="Unknown plan")
    plan_cfg = PLANS[payload.plan]
    price_id = plan_cfg.get("stripe_price_id")
    if not price_id:
        raise HTTPException(status_code=400, detail="No Stripe price ID configured for this plan")

    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY
    session = stripe.checkout.Session.create(
        customer_email=current_user.email,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url=payload.success_url + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=payload.cancel_url,
        metadata={"user_id": str(current_user.id), "plan": payload.plan},
    )
    return {"checkout_url": session.url, "session_id": session.id}


@router.post("/portal")
async def customer_portal(
    current_user: User = Depends(get_current_user),
):
    """Stripe Billing Portal for subscription management."""
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Billing not configured")

    customer_id = await redis_client.get(f"stripe:customer:{current_user.id}")
    if not customer_id:
        raise HTTPException(status_code=400, detail="No Stripe customer found. Subscribe first.")

    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY
    portal = stripe.billing_portal.Session.create(
        customer=customer_id.decode(),
        return_url=f"{settings.FRONTEND_URL}/settings/billing",
    )
    return {"portal_url": portal.url}


# ─── Stripe webhook ──────────────────────────────────────

@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Stripe event webhook — updates plan on subscription events."""
    if not settings.STRIPE_SECRET_KEY:
        return {"status": "ignored"}

    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig, settings.STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    etype = event["type"]

    if etype in ("checkout.session.completed", "customer.subscription.updated"):
        data = event["data"]["object"]
        user_id = data.get("metadata", {}).get("user_id") or data.get("client_reference_id")
        plan = data.get("metadata", {}).get("plan", "plus")
        customer_id = data.get("customer")

        if user_id:
            new_plan = SubscriptionPlan(plan) if plan in SubscriptionPlan.__members__ else SubscriptionPlan.plus
            await db.execute(
                update(User).where(User.id == uuid.UUID(user_id)).values(plan=new_plan)
            )
            await db.commit()
            if customer_id:
                await redis_client.set(f"stripe:customer:{user_id}", customer_id)

    elif etype in ("customer.subscription.deleted", "customer.subscription.paused"):
        data = event["data"]["object"]
        customer_id = data.get("customer")
        if customer_id:
            uid_raw = await redis_client.get(f"stripe:customer_to_user:{customer_id}")
            if uid_raw:
                await db.execute(
                    update(User).where(User.id == uuid.UUID(uid_raw.decode())).values(plan=SubscriptionPlan.free)
                )
                await db.commit()

    return {"status": "ok"}


# ─── Feature gate helper ──────────────────────────────────

async def check_feature_gate(user: User, feature: str) -> bool:
    """Return True if the user's plan includes the given feature."""
    plan_name = user.plan.value if hasattr(user.plan, "value") else str(user.plan)
    plan_features = PLANS.get(plan_name, PLANS["free"])["features"]
    return bool(plan_features.get(feature, False))
