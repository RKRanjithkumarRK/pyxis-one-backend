from __future__ import annotations
import re
import uuid
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator


CATEGORIES = Literal[
    "code", "writing", "productivity", "education", "business", "creative", "data", "general"
]

VISIBILITY = Literal["public", "private", "unlisted"]


class AgentCapabilities(BaseModel):
    vision: bool = False
    tool_use: bool = False
    web_search: bool = False


class AgentBase(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=2000)
    icon: str | None = Field(default=None, max_length=256)
    category: CATEGORIES = "general"
    instructions: str | None = Field(default=None, max_length=16000)
    starters: list[str] | None = Field(default=None, max_length=4)
    capabilities: AgentCapabilities | None = None
    default_model: str = "claude-sonnet-4"


class AgentCreate(AgentBase):
    slug: str | None = Field(default=None, max_length=128)
    visibility: VISIBILITY = "private"

    @field_validator("slug", mode="before")
    @classmethod
    def normalize_slug(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.lower().strip()
        v = re.sub(r"[^a-z0-9-]", "-", v)
        v = re.sub(r"-+", "-", v).strip("-")
        if len(v) < 2:
            raise ValueError("Slug must be at least 2 characters")
        return v


class AgentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=2000)
    icon: str | None = Field(default=None, max_length=256)
    category: CATEGORIES | None = None
    instructions: str | None = Field(default=None, max_length=16000)
    starters: list[str] | None = None
    capabilities: AgentCapabilities | None = None
    default_model: str | None = None
    visibility: VISIBILITY | None = None


class AgentVersionOut(BaseModel):
    id: uuid.UUID
    agent_id: uuid.UUID
    version: int
    snapshot: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentOut(BaseModel):
    id: uuid.UUID
    slug: str
    name: str
    description: str | None
    icon: str | None
    icon_url: str | None
    category: str
    instructions: str | None
    starters: list | None
    capabilities: dict | None
    default_model: str
    visibility: str
    version: int
    is_builtin: bool
    usage_count: int
    rating: float | None
    rating_count: int
    creator_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AgentListResponse(BaseModel):
    agents: list[AgentOut]
    total: int
    page: int
    page_size: int
    pages: int


class RateAgentRequest(BaseModel):
    rating: float = Field(ge=1.0, le=5.0)
