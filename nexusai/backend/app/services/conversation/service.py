from __future__ import annotations
import uuid
import re
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.conversation import Conversation
from app.models.agent import Agent
from app.models.project import Project
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository


class ConversationService:

    @staticmethod
    async def get_or_create(
        db: AsyncSession,
        conversation_id: str | None,
        user_id: uuid.UUID,
        *,
        model_id: str = "claude-sonnet-4",
        agent_id: str | None = None,
        project_id: str | None = None,
    ) -> Conversation:
        if conversation_id:
            conv = await ConversationRepository.get(db, uuid.UUID(conversation_id), user_id)
            if conv:
                return conv

        return await ConversationRepository.create(
            db,
            user_id,
            model_id=model_id,
            agent_id=uuid.UUID(agent_id) if agent_id else None,
            project_id=uuid.UUID(project_id) if project_id else None,
        )

    @staticmethod
    async def get_history(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> list[dict]:
        conv = await ConversationRepository.get(db, conversation_id, user_id)
        if not conv or not conv.active_branch_id:
            return []
        messages = await MessageRepository.list_branch(
            db, conversation_id, conv.active_branch_id
        )
        result = []
        for m in messages:
            if m.role == "system":
                continue
            content: Any = m.content
            if m.attachments and m.role == "user":
                parts: list[dict] = [{"type": "text", "text": m.content}]
                for att in m.attachments:
                    if att.get("type") == "image":
                        parts.append({"type": "image_url", "image_url": {"url": att["url"]}})
                    elif att.get("type") == "document":
                        parts.append({"type": "text", "text": f"[File: {att['name']}]\n{att.get('extracted_text', '')}"})
                content = parts
            result.append({"role": m.role, "content": content})
        return result

    @staticmethod
    async def append(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        *,
        model: str | None = None,
        usage: dict | None = None,
        citations: list | None = None,
        attachments: list | None = None,
        user_id: uuid.UUID,
    ) -> uuid.UUID:
        conv = await ConversationRepository.get(db, conversation_id, user_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")

        branch_id = conv.active_branch_id or uuid.uuid4()
        if not conv.active_branch_id:
            await ConversationRepository.update(db, conversation_id, user_id, active_branch_id=branch_id)

        seq = await MessageRepository.next_sequence(db, conversation_id, branch_id)
        msg = await MessageRepository.create(
            db,
            conversation_id=conversation_id,
            branch_id=branch_id,
            sequence=seq,
            role=role,
            content=content,
            model_id=model,
            usage=usage,
            citations=citations,
            attachments=attachments,
        )

        # Auto-title from first user message
        if role == "user" and seq == 0:
            title = content[:80].strip()
            title = re.sub(r"\s+", " ", title)
            await ConversationRepository.update(db, conversation_id, user_id, title=title)

        # Touch updated_at on every append
        await ConversationRepository.update(db, conversation_id, user_id)
        return msg.id

    @staticmethod
    async def edit_message(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        message_id: uuid.UUID,
        new_content: str,
        user_id: uuid.UUID,
    ) -> uuid.UUID:
        """Edit a user message → create new branch from that point. Returns new branch_id."""
        conv = await ConversationRepository.get(db, conversation_id, user_id)
        if not conv:
            raise ValueError("Conversation not found")

        original = await MessageRepository.get(db, message_id)
        if not original or original.conversation_id != conversation_id:
            raise ValueError("Message not found")
        if original.role != "user":
            raise ValueError("Only user messages can be edited")

        new_branch_id = uuid.uuid4()

        # Copy messages up to (not including) the edited one
        await MessageRepository.copy_branch_up_to(
            db, conversation_id, original.branch_id, new_branch_id, original.sequence
        )

        # Append edited message
        await MessageRepository.create(
            db,
            conversation_id=conversation_id,
            branch_id=new_branch_id,
            sequence=original.sequence,
            role="user",
            content=new_content,
            parent_branch_id=original.branch_id,
        )

        # Switch active branch
        await ConversationRepository.update(
            db, conversation_id, user_id, active_branch_id=new_branch_id
        )

        return new_branch_id

    @staticmethod
    async def get_agent_prompt(db: AsyncSession, agent_id: str, user_id: uuid.UUID) -> str:
        result = await db.execute(select(Agent).where(Agent.id == uuid.UUID(agent_id)))
        agent = result.scalar_one_or_none()
        if not agent or not agent.instructions:
            return ""
        return agent.instructions

    @staticmethod
    async def get_project_prompt(db: AsyncSession, project_id: str, user_id: uuid.UUID) -> str:
        result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
        project = result.scalar_one_or_none()
        if not project or not project.system_prompt:
            return ""
        return project.system_prompt
