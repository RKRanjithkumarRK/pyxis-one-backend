from datetime import datetime
from uuid import uuid4
from typing import Optional, Any
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base


def _uuid() -> str:
    return str(uuid4())


# ── Conversations (ChatGPT-style history) ─────────────────────────────────────

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    model: Mapped[str] = mapped_column(String(100), default="claude-sonnet-4-6")
    feature_mode: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    session: Mapped["Session"] = relationship("Session", back_populates="conversations")
    messages: Mapped[list["ConversationMessage"]] = relationship(
        "ConversationMessage", back_populates="conversation", cascade="all, delete-orphan",
        order_by="ConversationMessage.created_at"
    )


class ConversationMessage(Base):
    """
    Messages with branching support.
    parent_id + branch_index enables conversation tree:
      message A (parent_id=None, branch_index=0)
      ├── message B (parent_id=A, branch_index=0)  ← main branch
      └── message B' (parent_id=A, branch_index=1) ← edited/regenerated
    """
    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String, ForeignKey("conversations.id"), nullable=False
    )
    parent_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("conversation_messages.id"), nullable=True
    )
    branch_index: Mapped[int] = mapped_column(Integer, default=0)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    finish_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    tool_calls: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    tool_results: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    usage: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    feature_mode: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )


class FileUpload(Base):
    """Uploaded files with extracted text for tool use."""
    __tablename__ = "file_uploads"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    conversation_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_b64: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ── Session (existing — extended with conversations) ──────────────────────────

class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_active: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    student_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tier: Mapped[str] = mapped_column(String(20), default="free")

    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="session", cascade="all, delete-orphan"
    )
    psyche_states: Mapped[list["PsycheState"]] = relationship(
        "PsycheState", back_populates="session", cascade="all, delete-orphan"
    )
    forge_progress: Mapped[list["ForgeProgress"]] = relationship(
        "ForgeProgress", back_populates="session", cascade="all, delete-orphan"
    )
    concept_masteries: Mapped[list["ConceptMastery"]] = relationship(
        "ConceptMastery", back_populates="session", cascade="all, delete-orphan"
    )
    vault_entries: Mapped[list["VaultEntry"]] = relationship(
        "VaultEntry", back_populates="session", cascade="all, delete-orphan"
    )
    blind_spots: Mapped[list["BlindSpot"]] = relationship(
        "BlindSpot", back_populates="session", cascade="all, delete-orphan"
    )
    oracle_timelines: Mapped[list["OracleTimeline"]] = relationship(
        "OracleTimeline", back_populates="session", cascade="all, delete-orphan"
    )
    civilization_states: Mapped[list["CivilizationState"]] = relationship(
        "CivilizationState", back_populates="session", cascade="all, delete-orphan"
    )
    mirror_reports: Mapped[list["MirrorReport"]] = relationship(
        "MirrorReport", back_populates="session", cascade="all, delete-orphan"
    )
    nemesis_records: Mapped[list["NemesisRecord"]] = relationship(
        "NemesisRecord", back_populates="session", cascade="all, delete-orphan"
    )
    helix_revolutions: Mapped[list["HelixRevolution"]] = relationship(
        "HelixRevolution", back_populates="session", cascade="all, delete-orphan"
    )
    tide_readings: Mapped[list["TideReading"]] = relationship(
        "TideReading", back_populates="session", cascade="all, delete-orphan"
    )


# ── Legacy Message (existing engines use this) ───────────────────────────────

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    feature_mode: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    psyche_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    session: Mapped["Session"] = relationship("Session", back_populates="messages")


# ── Psyche ────────────────────────────────────────────────────────────────────

class PsycheState(Base):
    __tablename__ = "psyche_states"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    dimension: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[float] = mapped_column(Float, default=0.5)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    session: Mapped["Session"] = relationship("Session", back_populates="psyche_states")


class ForgeProgress(Base):
    __tablename__ = "forge_progress"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    concept: Mapped[str] = mapped_column(String(255), nullable=False)
    stage: Mapped[str] = mapped_column(String(50), default="RAW_ORE")
    stage_entered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    session: Mapped["Session"] = relationship("Session", back_populates="forge_progress")


class ConceptMastery(Base):
    __tablename__ = "concept_mastery"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    concept: Mapped[str] = mapped_column(String(255), nullable=False)
    mastery_score: Mapped[float] = mapped_column(Float, default=0.0)
    helix_revolution: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    last_encounter: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_encounter: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    tide_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    session: Mapped["Session"] = relationship("Session", back_populates="concept_masteries")


class VaultEntry(Base):
    __tablename__ = "vault_entries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    content_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    concept_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    emotion_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped["Session"] = relationship("Session", back_populates="vault_entries")


class BlindSpot(Base):
    __tablename__ = "blind_spots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    assumption: Mapped[str] = mapped_column(Text, nullable=False)
    origin_message_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    affected_concepts: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    excavated: Mapped[bool] = mapped_column(Boolean, default=False)
    excavated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    session: Mapped["Session"] = relationship("Session", back_populates="blind_spots")


class OracleTimeline(Base):
    __tablename__ = "oracle_timelines"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    concept: Mapped[str] = mapped_column(String(255), nullable=False)
    predicted_wall_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    scaffolding_injected: Mapped[bool] = mapped_column(Boolean, default=False)
    wall_avoided: Mapped[bool] = mapped_column(Boolean, default=False)

    session: Mapped["Session"] = relationship("Session", back_populates="oracle_timelines")


class CivilizationState(Base):
    __tablename__ = "civilization_states"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    era: Mapped[str] = mapped_column(String(100), default="Ancient")
    population: Mapped[int] = mapped_column(Integer, default=1000)
    resources: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    decisions_history: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    current_crisis: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    turn_number: Mapped[int] = mapped_column(Integer, default=0)

    session: Mapped["Session"] = relationship("Session", back_populates="civilization_states")


class MirrorReport(Base):
    __tablename__ = "mirror_reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    report_content: Mapped[str] = mapped_column(Text, nullable=False)
    key_insights: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    session: Mapped["Session"] = relationship("Session", back_populates="mirror_reports")


class NemesisRecord(Base):
    __tablename__ = "nemesis_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    weakness: Mapped[str] = mapped_column(Text, nullable=False)
    challenges_issued: Mapped[int] = mapped_column(Integer, default=0)
    challenges_passed: Mapped[int] = mapped_column(Integer, default=0)
    challenges_failed: Mapped[int] = mapped_column(Integer, default=0)

    session: Mapped["Session"] = relationship("Session", back_populates="nemesis_records")


class HelixRevolution(Base):
    __tablename__ = "helix_revolutions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    concept: Mapped[str] = mapped_column(String(255), nullable=False)
    revolution: Mapped[str] = mapped_column(String(50), default="SURFACE")
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_due: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    session: Mapped["Session"] = relationship("Session", back_populates="helix_revolutions")


class TideReading(Base):
    __tablename__ = "tide_readings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    concept: Mapped[str] = mapped_column(String(255), nullable=False)
    vocabulary_precision: Mapped[float] = mapped_column(Float, default=0.5)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)
    reading_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped["Session"] = relationship("Session", back_populates="tide_readings")
