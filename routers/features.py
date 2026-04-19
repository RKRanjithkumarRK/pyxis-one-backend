from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.schemas import (
    ForgeAdvanceRequest, ForgeStatusRequest, ForgeStatusResponse,
    CurriculumNextRequest, CurriculumNextResponse,
    OracleTimelineRequest, OracleTimelineResponse,
    NemisChallengeRequest, NemesisChallengeResponse,
    NemesisOutcomeRequest, NemesisOutcomeResponse,
    HelixNextRequest, HelixNextResponse,
    HelixDueRequest, HelixDueResponse,
    TideReadingRequest, TideReadingResponse,
    TideChartRequest, TideChartResponse,
    GravityMapRequest, GravityMapResponse,
    DarkKnowledgeDetectRequest, DarkKnowledgeDetectResponse,
    MirrorReportRequest, MirrorReportResponse,
    CivilizationDecisionRequest, CivilizationDecisionResponse,
    SymphonyMotifRequest, SymphonyMotifResponse,
    BlindSpotsAnalyzeRequest, BlindSpotsAnalyzeResponse,
    PrecognitionMapRequest, PrecognitionMapResponse,
    ShadowSelfRequest, ShadowSelfResponse,
    TemporalWavesRequest, SynapticSprintRequest, FinalBossRequest,
    BabelMindRequest, AlienModeRequest, CivilizationInitRequest,
)
from core.config import RESPONSE_STRUCTURE
from engines.forge import forge_engine
from engines.curriculum import curriculum_engine
from engines.oracle import oracle_engine
from engines.nemesis import nemesis_engine
from engines.helix import helix_engine
from engines.tides import tide_engine
from engines.gravity import gravity_engine
from engines.dark_knowledge import dark_knowledge_engine
from engines.mirror import mirror_engine
from engines.civilization import civilization_engine
from engines.symphony import symphony_engine
from engines.blind_spots import blind_spot_engine
from engines.precognition import precognition_engine
from engines.shadow_self import shadow_self_engine
from engines.psyche import psyche_engine
import engines.anthropic_client as ac
from datetime import datetime

router = APIRouter()


# ── Forge ─────────────────────────────────────────────────────────────────────

@router.post("/forge/advance")
async def forge_advance(request: ForgeAdvanceRequest, db: AsyncSession = Depends(get_db)):
    try:
        new_stage = await forge_engine.advance_stage(request.session_id, request.concept)
        psyche = await psyche_engine.get_context_block(request.session_id)
        prompt_text = await forge_engine.get_stage_prompt(new_stage, request.concept, psyche)
        return {
            "session_id": request.session_id,
            "concept": request.concept,
            "new_stage": new_stage,
            "stage_prompt": prompt_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forge/status", response_model=ForgeStatusResponse)
async def forge_status(request: ForgeStatusRequest, db: AsyncSession = Depends(get_db)):
    try:
        stage = await forge_engine.get_stage(request.session_id, request.concept)
        psyche = await psyche_engine.get_context_block(request.session_id)
        prompt_text = await forge_engine.get_stage_prompt(stage, request.concept, psyche)
        return ForgeStatusResponse(
            session_id=request.session_id,
            concept=request.concept,
            stage=stage,
            prompt=prompt_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Curriculum ────────────────────────────────────────────────────────────────

@router.post("/curriculum/next", response_model=CurriculumNextResponse)
async def curriculum_next(request: CurriculumNextRequest, db: AsyncSession = Depends(get_db)):
    try:
        moves = await curriculum_engine.generate_next_moves(request.session_id, request.topic or "")
        sequence = await curriculum_engine.get_sequence(request.session_id)
        return CurriculumNextResponse(
            session_id=request.session_id,
            moves=moves,
            sequence=[s["concept"] for s in sequence],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Oracle ────────────────────────────────────────────────────────────────────

@router.post("/oracle/timeline", response_model=OracleTimelineResponse)
async def oracle_timeline(request: OracleTimelineRequest, db: AsyncSession = Depends(get_db)):
    try:
        wall_concepts = await oracle_engine.predict_wall_concepts(request.session_id)
        timeline = await oracle_engine.get_timeline(request.session_id)
        return OracleTimelineResponse(
            session_id=request.session_id,
            timeline=timeline,
            wall_concepts=wall_concepts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Nemesis ───────────────────────────────────────────────────────────────────

@router.post("/nemesis/challenge", response_model=NemesisChallengeResponse)
async def nemesis_challenge(request: NemisChallengeRequest, db: AsyncSession = Depends(get_db)):
    try:
        if not request.weakness:
            weaknesses = await nemesis_engine.analyze_weaknesses(request.session_id)
            weakness = weaknesses[0]["weakness"] if weaknesses else "conceptual precision"
        else:
            weakness = request.weakness

        challenge_text = await nemesis_engine.generate_challenge(request.session_id, weakness)

        record = await nemesis_engine.get_record(request.session_id)
        challenge_id = record["weaknesses"][0]["id"] if record["weaknesses"] else request.session_id

        return NemesisChallengeResponse(
            session_id=request.session_id,
            challenge=challenge_text,
            weakness=weakness,
            challenge_id=challenge_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nemesis/outcome", response_model=NemesisOutcomeResponse)
async def nemesis_outcome(request: NemesisOutcomeRequest, db: AsyncSession = Depends(get_db)):
    try:
        await nemesis_engine.record_outcome(request.session_id, request.challenge_id, request.passed)
        record = await nemesis_engine.get_record(request.session_id)
        return NemesisOutcomeResponse(session_id=request.session_id, record=record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Helix ─────────────────────────────────────────────────────────────────────

@router.post("/helix/next", response_model=HelixNextResponse)
async def helix_next(request: HelixNextRequest, db: AsyncSession = Depends(get_db)):
    try:
        revolution = await helix_engine.get_revolution(request.session_id, request.concept)
        psyche = await psyche_engine.get_context_block(request.session_id)
        prompt_text = await helix_engine.get_prompt(revolution, request.concept, psyche)
        await helix_engine.schedule_next(request.session_id, request.concept, quality=4)
        return HelixNextResponse(
            session_id=request.session_id,
            concept=request.concept,
            revolution=revolution,
            prompt=prompt_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/helix/due", response_model=HelixDueResponse)
async def helix_due(request: HelixDueRequest, db: AsyncSession = Depends(get_db)):
    try:
        due = await helix_engine.get_due_concepts(request.session_id)
        return HelixDueResponse(session_id=request.session_id, due_concepts=due)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Tides ─────────────────────────────────────────────────────────────────────

@router.post("/tides/reading", response_model=TideReadingResponse)
async def tides_reading(request: TideReadingRequest, db: AsyncSession = Depends(get_db)):
    try:
        await tide_engine.record_reading(request.session_id, request.concept, request.message)
        chart = await tide_engine.get_chart(request.session_id, request.concept)
        readings = chart.get("readings", [])
        last = readings[-1] if readings else {}
        return TideReadingResponse(
            session_id=request.session_id,
            concept=request.concept,
            vocabulary_precision=last.get("vocabulary_precision", 0.5),
            confidence_score=last.get("confidence_score", 0.5),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tides/chart", response_model=TideChartResponse)
async def tides_chart(request: TideChartRequest, db: AsyncSession = Depends(get_db)):
    try:
        chart = await tide_engine.get_chart(request.session_id, request.concept)
        return TideChartResponse(**chart)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Gravity ───────────────────────────────────────────────────────────────────

@router.post("/gravity/map", response_model=GravityMapResponse)
async def gravity_map(request: GravityMapRequest, db: AsyncSession = Depends(get_db)):
    try:
        if request.concept:
            orbital = await gravity_engine.get_orbital_system(request.session_id, request.concept)
            universe = {"orbital_system": orbital}
        else:
            universe = await gravity_engine.get_universe_map(request.session_id)
        return GravityMapResponse(session_id=request.session_id, universe_map=universe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dark Knowledge ────────────────────────────────────────────────────────────

@router.post("/dark-knowledge/detect", response_model=DarkKnowledgeDetectResponse)
async def dark_knowledge_detect(request: DarkKnowledgeDetectRequest, db: AsyncSession = Depends(get_db)):
    try:
        contradictions = await dark_knowledge_engine.detect_contradictions(request.session_id, request.message)
        blind_spots = await blind_spot_engine.analyze(request.session_id, request.message)
        return DarkKnowledgeDetectResponse(
            session_id=request.session_id,
            contradictions=contradictions,
            blind_spots=blind_spots,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Mirror ────────────────────────────────────────────────────────────────────

@router.post("/mirror/report", response_model=MirrorReportResponse)
async def mirror_report(request: MirrorReportRequest, db: AsyncSession = Depends(get_db)):
    try:
        is_due = await mirror_engine.check_due(request.session_id)
        if is_due:
            report_text = await mirror_engine.generate_report(request.session_id)
        else:
            last = await mirror_engine.get_last_report(request.session_id)
            if last.get("report"):
                return MirrorReportResponse(
                    session_id=request.session_id,
                    report=last["report"],
                    key_insights=last["key_insights"],
                    generated_at=datetime.fromisoformat(last["generated_at"]),
                )
            report_text = await mirror_engine.generate_report(request.session_id)

        last = await mirror_engine.get_last_report(request.session_id)
        return MirrorReportResponse(
            session_id=request.session_id,
            report=report_text,
            key_insights=last.get("key_insights", []),
            generated_at=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Civilization ──────────────────────────────────────────────────────────────

@router.post("/civilization/init")
async def civilization_init(request: CivilizationInitRequest, db: AsyncSession = Depends(get_db)):
    try:
        state = await civilization_engine.initialize(request.session_id, request.subject)
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/civilization/decision", response_model=CivilizationDecisionResponse)
async def civilization_decision(request: CivilizationDecisionRequest, db: AsyncSession = Depends(get_db)):
    try:
        result = await civilization_engine.make_decision(request.session_id, request.decision)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return CivilizationDecisionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Symphony ──────────────────────────────────────────────────────────────────

@router.post("/symphony/motif", response_model=SymphonyMotifResponse)
async def symphony_motif(request: SymphonyMotifRequest, db: AsyncSession = Depends(get_db)):
    try:
        motif = await symphony_engine.generate_motif(request.concept)
        await symphony_engine.store_motif(request.session_id, request.concept, motif)
        symphony = await symphony_engine.get_symphony(request.session_id)
        return SymphonyMotifResponse(
            session_id=request.session_id,
            concept=request.concept,
            motif=motif,
            symphony=symphony,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Blind Spots ───────────────────────────────────────────────────────────────

@router.post("/blind-spots/analyze", response_model=BlindSpotsAnalyzeResponse)
async def blind_spots_analyze(request: BlindSpotsAnalyzeRequest, db: AsyncSession = Depends(get_db)):
    try:
        spots = await blind_spot_engine.analyze(request.session_id, request.message)
        tree = await blind_spot_engine.build_assumption_tree(request.session_id)
        return BlindSpotsAnalyzeResponse(
            session_id=request.session_id,
            blind_spots=spots,
            assumption_tree=tree,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Precognition ──────────────────────────────────────────────────────────────

@router.post("/precognition/map", response_model=PrecognitionMapResponse)
async def precognition_map(request: PrecognitionMapRequest, db: AsyncSession = Depends(get_db)):
    try:
        constellation = await precognition_engine.get_constellation_map(request.session_id)
        return PrecognitionMapResponse(
            session_id=request.session_id,
            trajectory=constellation["trajectory"],
            struggles=constellation["predicted_struggles"],
            constellation_map=constellation,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Shadow Self ───────────────────────────────────────────────────────────────

@router.post("/shadow/prompt", response_model=ShadowSelfResponse)
async def shadow_prompt(request: ShadowSelfRequest, db: AsyncSession = Depends(get_db)):
    try:
        system_prompt = await shadow_self_engine.get_system_prompt(request.session_id)
        profile = await shadow_self_engine.build_profile(request.session_id)
        return ShadowSelfResponse(
            session_id=request.session_id,
            system_prompt=system_prompt,
            profile=profile,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Temporal Waves ────────────────────────────────────────────────────────────

@router.post("/temporal-waves/response")
async def temporal_waves(request: TemporalWavesRequest, db: AsyncSession = Depends(get_db)):
    try:
        psyche = await psyche_engine.get_context_block(request.session_id)
        system = (
            "You are a temporal learning guide.\n"
            "Explain concepts by showing how human understanding of them evolved across eras.\n"
            "Structure: Ancient understanding → Medieval → Renaissance → Modern → Cutting Edge → Unresolved.\n"
            "Show how each era's confusion led to the next breakthrough.\n\n"
            f"{RESPONSE_STRUCTURE}\n\n{psyche}"
        )
        messages = [{"role": "user", "content": f"Explain '{request.concept}' through temporal waves across history."}]
        response = await ac.complete_response(messages, system)
        return {"session_id": request.session_id, "concept": request.concept, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Synaptic Sprint ───────────────────────────────────────────────────────────

@router.post("/synaptic-sprint/generate")
async def synaptic_sprint(request: SynapticSprintRequest, db: AsyncSession = Depends(get_db)):
    try:
        psyche = await psyche_engine.get_context_block(request.session_id)
        system = (
            "You are the Synaptic Sprint engine.\n"
            f"Generate a rapid-fire {request.duration_minutes}-minute learning sprint.\n"
            "Format: 8-12 ultra-short questions/prompts, each 30-60 seconds to answer.\n"
            "Increase difficulty with each question. No explanations — just prompts.\n"
            "Mark each with its cognitive type: [RECALL] [APPLY] [INFER] [CREATE]\n\n"
            f"{psyche}"
        )
        messages = [{"role": "user", "content": f"Generate a synaptic sprint on: {request.topic}"}]
        response = await ac.complete_response(messages, system)
        return {
            "session_id": request.session_id,
            "topic": request.topic,
            "duration_minutes": request.duration_minutes,
            "sprint": response,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Final Boss ────────────────────────────────────────────────────────────────

@router.post("/final-boss/examine")
async def final_boss(request: FinalBossRequest, db: AsyncSession = Depends(get_db)):
    try:
        psyche = await psyche_engine.get_context_block(request.session_id)
        system = (
            "You are the Final Boss — the ultimate intellectual examination.\n"
            "Generate ONE question of maximum possible difficulty for this concept.\n"
            "The question must:\n"
            "1. Combine the concept with two unrelated domains\n"
            "2. Require formal precise language to answer correctly\n"
            "3. Have exactly one correct answer (not opinion-based)\n"
            "4. Be impossible to answer without genuine deep understanding\n"
            "5. Take 20-40 minutes for an expert to answer fully\n\n"
            "After the question, provide the complete model answer.\n\n"
            f"{RESPONSE_STRUCTURE}\n\n{psyche}"
        )
        messages = [{"role": "user", "content": f"Generate the Final Boss exam for: {request.concept}"}]
        response = await ac.complete_response(messages, system, max_tokens=2048)
        return {"session_id": request.session_id, "concept": request.concept, "final_boss": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Babel Mind ────────────────────────────────────────────────────────────────

@router.post("/babel-mind/reframe")
async def babel_mind(request: BabelMindRequest, db: AsyncSession = Depends(get_db)):
    try:
        psyche = await psyche_engine.get_context_block(request.session_id)
        target = request.target_framework or "five different frameworks"
        system = (
            "You are the Babel Mind — a cross-paradigm translator.\n"
            f"Reframe the given concept through {target}.\n"
            "For each framework: explain how that paradigm would understand, use, or criticise the concept.\n"
            "Show the student that the concept exists differently in each cognitive language.\n\n"
            f"{RESPONSE_STRUCTURE}\n\n{psyche}"
        )
        messages = [{"role": "user", "content": f"Reframe '{request.concept}' through {target}."}]
        response = await ac.complete_response(messages, system)
        return {
            "session_id": request.session_id,
            "concept": request.concept,
            "target_framework": target,
            "reframed": response,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Alien Mode ────────────────────────────────────────────────────────────────

@router.post("/alien-mode/translate")
async def alien_mode(request: AlienModeRequest, db: AsyncSession = Depends(get_db)):
    try:
        psyche = await psyche_engine.get_context_block(request.session_id)
        system = (
            "You are the Alien Mode translator.\n"
            "Explain the concept as if to an alien intelligence with no human context whatsoever.\n"
            "No cultural references. No assumed shared experience. No human metaphors.\n"
            "Build from pure logic and observable patterns. Then translate back to human understanding.\n"
            "This reveals hidden assumptions in how we normally explain things.\n\n"
            f"{RESPONSE_STRUCTURE}\n\n{psyche}"
        )
        messages = [{"role": "user", "content": f"Translate '{request.concept}' for an alien with no human context."}]
        response = await ac.complete_response(messages, system)
        return {
            "session_id": request.session_id,
            "concept": request.concept,
            "alien_translation": response,
            "human_revelation": "The alien translation reveals these hidden human assumptions...",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
