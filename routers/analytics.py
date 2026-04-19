from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.models import Message, ConceptMastery, ForgeProgress, PsycheState
from core.schemas import (
    PsycheStateResponse, PsycheVisualizationResponse,
    DashboardResponse, WeeklyReportResponse,
)
from engines.psyche import psyche_engine
import engines.anthropic_client as ac

router = APIRouter()


@router.get("/psyche/state/{session_id}", response_model=PsycheStateResponse)
async def get_psyche_state(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(
            select(PsycheState).where(PsycheState.session_id == session_id)
        )
        states = result.scalars().all()

        dimensions = {s.dimension: s.value for s in states}
        latest_update = max((s.updated_at for s in states), default=None) if states else None

        return PsycheStateResponse(
            session_id=session_id,
            dimensions=dimensions,
            updated_at=latest_update,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/psyche/visualization/{session_id}", response_model=PsycheVisualizationResponse)
async def get_psyche_visualization(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        viz = await psyche_engine.get_visualization_data(session_id)
        return PsycheVisualizationResponse(
            session_id=session_id,
            dimensions=viz["dimensions"],
            trends=viz["trends"],
            organism_health=viz["organism_health"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/dashboard/{session_id}", response_model=DashboardResponse)
async def get_dashboard(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        msg_count_result = await db.execute(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        )
        message_count = msg_count_result.scalar() or 0

        mastery_result = await db.execute(
            select(ConceptMastery).where(ConceptMastery.session_id == session_id)
        )
        masteries = mastery_result.scalars().all()

        concepts_mastered = sum(1 for m in masteries if m.mastery_score >= 0.75)
        active_concepts = len(masteries)

        psyche_result = await db.execute(
            select(PsycheState).where(PsycheState.session_id == session_id)
        )
        psyche_states = psyche_result.scalars().all()
        psyche_summary = {s.dimension: s.value for s in psyche_states}

        forge_result = await db.execute(
            select(ForgeProgress).where(ForgeProgress.session_id == session_id)
        )
        forge_rows = forge_result.scalars().all()
        forge_stages: dict[str, int] = {}
        for fp in forge_rows:
            forge_stages[fp.stage] = forge_stages.get(fp.stage, 0) + 1

        top_concepts = sorted(
            [{"concept": m.concept, "mastery": m.mastery_score, "stage": m.helix_revolution} for m in masteries],
            key=lambda x: x["mastery"],
            reverse=True,
        )[:10]

        return DashboardResponse(
            session_id=session_id,
            message_count=message_count,
            concepts_mastered=concepts_mastered,
            active_concepts=active_concepts,
            psyche_summary=psyche_summary,
            forge_stages=forge_stages,
            top_concepts=top_concepts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/weekly/{session_id}", response_model=WeeklyReportResponse)
async def get_weekly_report(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        now = datetime.utcnow()
        week_start = now - timedelta(days=7)

        msg_result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id, Message.timestamp >= week_start)
            .order_by(Message.timestamp)
        )
        weekly_messages = msg_result.scalars().all()

        concepts_this_week: list[str] = []
        for msg in weekly_messages:
            if msg.feature_mode and msg.feature_mode not in ["standard", "general"]:
                concepts_this_week.append(msg.feature_mode)
        concepts_this_week = list(set(concepts_this_week))

        mastery_result = await db.execute(
            select(ConceptMastery)
            .where(
                ConceptMastery.session_id == session_id,
                ConceptMastery.last_encounter >= week_start,
            )
        )
        recent_masteries = mastery_result.scalars().all()
        mastery_gains = {m.concept: m.mastery_score for m in recent_masteries}

        if weekly_messages and mastery_gains:
            history_text = "\n".join(f"{m.role}: {m.content[:150]}" for m in weekly_messages[-10:])
            mastery_str = str(mastery_gains)
            prompt = (
                f"Based on this week's learning session, identify:\n"
                f"1. 3 breakthrough moments\n"
                f"2. 3 actionable recommendations for next week\n\n"
                f"Messages sample:\n{history_text}\n\n"
                f"Mastery gains: {mastery_str}\n\n"
                "Return plain text, two sections: BREAKTHROUGHS and RECOMMENDATIONS."
            )
            system = "You are a weekly learning analyst. Be specific and actionable."
            api_messages = [{"role": "user", "content": prompt}]
            try:
                analysis = await ac.complete_response(api_messages, system, max_tokens=768)
                lines = analysis.split("\n")
                breakthroughs = [l.strip("- ").strip() for l in lines if l.strip() and "breakthrough" not in l.lower() and "recommendation" not in l.lower()][:3]
                recommendations = [l.strip("- ").strip() for l in lines if l.strip()][-3:]
            except Exception:
                breakthroughs = ["Active learning session this week"]
                recommendations = ["Continue current study pace", "Review weakest concepts", "Attempt one new topic"]
        else:
            breakthroughs = ["Session started — first steps taken"]
            recommendations = ["Begin with foundational concepts", "Set a daily study goal", "Use the Forge engine for a key concept"]

        return WeeklyReportResponse(
            session_id=session_id,
            week_start=week_start,
            week_end=now,
            messages_this_week=len(weekly_messages),
            concepts_encountered=concepts_this_week,
            mastery_gains=mastery_gains,
            breakthroughs=breakthroughs,
            recommendations=recommendations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
