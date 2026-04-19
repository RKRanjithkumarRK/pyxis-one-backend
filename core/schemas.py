from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict


# ── Session ──────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    student_name: Optional[str] = None


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at: datetime
    last_active: datetime
    student_name: Optional[str] = None


# ── Message ───────────────────────────────────────────────────────────────────

class MessageCreate(BaseModel):
    session_id: str
    role: str
    content: str
    feature_mode: Optional[str] = "standard"


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime
    feature_mode: Optional[str] = None
    psyche_snapshot: Optional[dict] = None


# ── Psyche ────────────────────────────────────────────────────────────────────

class PsycheStateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    session_id: str
    dimensions: dict[str, float]
    updated_at: Optional[datetime] = None


class PsycheVisualizationResponse(BaseModel):
    session_id: str
    dimensions: dict[str, float]
    trends: dict[str, str]
    organism_health: float


# ── Forge ─────────────────────────────────────────────────────────────────────

class ForgeAdvanceRequest(BaseModel):
    session_id: str
    concept: str


class ForgeStatusRequest(BaseModel):
    session_id: str
    concept: str


class ForgeStatusResponse(BaseModel):
    session_id: str
    concept: str
    stage: str
    stage_entered_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    prompt: Optional[str] = None


# ── Curriculum ────────────────────────────────────────────────────────────────

class CurriculumNextRequest(BaseModel):
    session_id: str
    topic: Optional[str] = None


class CurriculumNextResponse(BaseModel):
    session_id: str
    moves: list[dict]
    sequence: list[str]


# ── Oracle ────────────────────────────────────────────────────────────────────

class OracleTimelineRequest(BaseModel):
    session_id: str


class OracleTimelineResponse(BaseModel):
    session_id: str
    timeline: list[dict]
    wall_concepts: list[dict]


# ── Nemesis ───────────────────────────────────────────────────────────────────

class NemisChallengeRequest(BaseModel):
    session_id: str
    weakness: Optional[str] = None


class NemesisChallengeResponse(BaseModel):
    session_id: str
    challenge: str
    weakness: str
    challenge_id: str


class NemesisOutcomeRequest(BaseModel):
    session_id: str
    challenge_id: str
    passed: bool


class NemesisOutcomeResponse(BaseModel):
    session_id: str
    record: dict


# ── Helix ─────────────────────────────────────────────────────────────────────

class HelixNextRequest(BaseModel):
    session_id: str
    concept: str


class HelixNextResponse(BaseModel):
    session_id: str
    concept: str
    revolution: str
    prompt: str
    next_due: Optional[datetime] = None


class HelixDueRequest(BaseModel):
    session_id: str


class HelixDueResponse(BaseModel):
    session_id: str
    due_concepts: list[dict]


# ── Tides ─────────────────────────────────────────────────────────────────────

class TideReadingRequest(BaseModel):
    session_id: str
    concept: str
    message: str


class TideReadingResponse(BaseModel):
    session_id: str
    concept: str
    vocabulary_precision: float
    confidence_score: float


class TideChartRequest(BaseModel):
    session_id: str
    concept: str


class TideChartResponse(BaseModel):
    session_id: str
    concept: str
    readings: list[dict]
    trend: str
    alert: Optional[str] = None


# ── Gravity ───────────────────────────────────────────────────────────────────

class GravityMapRequest(BaseModel):
    session_id: str
    concept: Optional[str] = None


class GravityMapResponse(BaseModel):
    session_id: str
    universe_map: dict


# ── Dark Knowledge ────────────────────────────────────────────────────────────

class DarkKnowledgeDetectRequest(BaseModel):
    session_id: str
    message: str


class DarkKnowledgeDetectResponse(BaseModel):
    session_id: str
    contradictions: list[dict]
    blind_spots: list[dict]


# ── Mirror ────────────────────────────────────────────────────────────────────

class MirrorReportRequest(BaseModel):
    session_id: str


class MirrorReportResponse(BaseModel):
    session_id: str
    report: str
    key_insights: list[str]
    generated_at: datetime


# ── Civilization ──────────────────────────────────────────────────────────────

class CivilizationInitRequest(BaseModel):
    session_id: str
    subject: str


class CivilizationDecisionRequest(BaseModel):
    session_id: str
    decision: str


class CivilizationDecisionResponse(BaseModel):
    session_id: str
    consequences: str
    new_state: dict
    turn_number: int


# ── Symphony ──────────────────────────────────────────────────────────────────

class SymphonyMotifRequest(BaseModel):
    session_id: str
    concept: str


class SymphonyMotifResponse(BaseModel):
    session_id: str
    concept: str
    motif: dict
    symphony: list[dict]


# ── Vault ─────────────────────────────────────────────────────────────────────

class VaultStoreRequest(BaseModel):
    session_id: str
    content: str
    concept_tags: list[str] = []
    emotion_tags: list[str] = []


class VaultStoreResponse(BaseModel):
    entry_id: str
    session_id: str


class VaultSearchRequest(BaseModel):
    session_id: str
    query: str


class VaultSearchResponse(BaseModel):
    session_id: str
    results: list[dict]


class VaultTimelineResponse(BaseModel):
    session_id: str
    entries: list[dict]


# ── Blind Spots ───────────────────────────────────────────────────────────────

class BlindSpotsAnalyzeRequest(BaseModel):
    session_id: str
    message: str


class BlindSpotsAnalyzeResponse(BaseModel):
    session_id: str
    blind_spots: list[dict]
    assumption_tree: dict


# ── Precognition ──────────────────────────────────────────────────────────────

class PrecognitionMapRequest(BaseModel):
    session_id: str


class PrecognitionMapResponse(BaseModel):
    session_id: str
    trajectory: list[dict]
    struggles: list[dict]
    constellation_map: dict


# ── Shadow Self ───────────────────────────────────────────────────────────────

class ShadowSelfRequest(BaseModel):
    session_id: str


class ShadowSelfResponse(BaseModel):
    session_id: str
    system_prompt: str
    profile: dict


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    feature_mode: Optional[str] = "standard"
    student_name: Optional[str] = None


# ── Trident ───────────────────────────────────────────────────────────────────

class TridentRequest(BaseModel):
    session_id: str
    question: str


class TridentResponse(BaseModel):
    session_id: str
    architect: str
    street_fighter: str
    heretic: str


# ── Assessment ────────────────────────────────────────────────────────────────

class AssessmentGenerateRequest(BaseModel):
    session_id: str
    topic: Optional[str] = None


class AssessmentGenerateResponse(BaseModel):
    session_id: str
    questions: list[dict]
    assessment_id: str


class AssessmentAutopsyRequest(BaseModel):
    session_id: str
    assessment_id: str
    answers: list[dict]


class AssessmentAutopsyResponse(BaseModel):
    session_id: str
    forensic_report: dict
    wrong_answer_origins: list[dict]
    score: float


# ── Parliament ────────────────────────────────────────────────────────────────

class ParliamentConveneRequest(BaseModel):
    session_id: str
    question: str


class PhilosopherResponse(BaseModel):
    philosopher: str
    response: str
    era: str


class ParliamentConveneResponse(BaseModel):
    session_id: str
    responses: list[PhilosopherResponse]


class ParliamentDuelRequest(BaseModel):
    session_id: str
    philosopher_a: str
    philosopher_b: str
    topic: str


class ParliamentDuelResponse(BaseModel):
    session_id: str
    philosopher_a: str
    philosopher_b: str
    exchange: list[dict]
    verdict: str


class ParliamentSubpoenaRequest(BaseModel):
    session_id: str
    philosopher: str
    question: str


class ParliamentSubpoenaResponse(BaseModel):
    session_id: str
    philosopher: str
    testimony: str


class ParliamentVoteRequest(BaseModel):
    session_id: str
    proposition: str


class ParliamentVoteResponse(BaseModel):
    session_id: str
    proposition: str
    votes: list[dict]
    verdict: str


# ── Voice ─────────────────────────────────────────────────────────────────────

class VoiceAnalysisResponse(BaseModel):
    session_id: str
    soul_report: str
    tempo: float
    avg_volume: float
    pause_count: int
    speech_rate_wpm: float
    confidence_indicators: dict


# ── Analytics ─────────────────────────────────────────────────────────────────

class DashboardResponse(BaseModel):
    session_id: str
    message_count: int
    concepts_mastered: int
    active_concepts: int
    psyche_summary: dict
    forge_stages: dict
    top_concepts: list[dict]


class WeeklyReportResponse(BaseModel):
    session_id: str
    week_start: datetime
    week_end: datetime
    messages_this_week: int
    concepts_encountered: list[str]
    mastery_gains: dict
    breakthroughs: list[str]
    recommendations: list[str]


# ── Feature Extras ────────────────────────────────────────────────────────────

class TemporalWavesRequest(BaseModel):
    session_id: str
    concept: str


class SynapticSprintRequest(BaseModel):
    session_id: str
    topic: str
    duration_minutes: int = 5


class FinalBossRequest(BaseModel):
    session_id: str
    concept: str


class BabelMindRequest(BaseModel):
    session_id: str
    concept: str
    target_framework: Optional[str] = None


class AlienModeRequest(BaseModel):
    session_id: str
    concept: str
