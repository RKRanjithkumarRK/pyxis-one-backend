from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, EmailStr, Field, field_validator
import re


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(min_length=1, max_length=256)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class MagicLinkRequest(BaseModel):
    email: EmailStr


class MagicLinkVerifyRequest(BaseModel):
    token: str


class OAuthCallbackRequest(BaseModel):
    provider: Literal["google", "github", "apple", "microsoft"]
    code: str
    redirect_uri: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserResponse(BaseModel):
    id: str
    email: str | None
    name: str | None
    avatar_url: str | None
    provider: str
    plan: str
    is_admin: bool
    is_verified: bool
    memory_enabled: bool
    custom_instructions: str | None
    created_at: str

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    user: UserResponse
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UpdateProfileRequest(BaseModel):
    name: str | None = Field(None, max_length=256)
    avatar_url: str | None = Field(None, max_length=2048)
    custom_instructions: str | None = Field(None, max_length=8000)
    memory_enabled: bool | None = None


class GuestSessionResponse(BaseModel):
    guest_id: str
    messages_remaining: int
    token: str
