from __future__ import annotations
import uuid
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.message import Message


class MessageRepository:

    @staticmethod
    async def list_branch(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        branch_id: uuid.UUID,
        limit: int = 200,
    ) -> list[Message]:
        result = await db.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.branch_id == branch_id,
            )
            .order_by(Message.sequence)
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def next_sequence(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        branch_id: uuid.UUID,
    ) -> int:
        result = await db.execute(
            select(func.coalesce(func.max(Message.sequence), -1)).where(
                Message.conversation_id == conversation_id,
                Message.branch_id == branch_id,
            )
        )
        return (result.scalar() or -1) + 1

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        branch_id: uuid.UUID,
        sequence: int,
        role: str,
        content: str,
        model_id: str | None = None,
        usage: dict | None = None,
        citations: list | None = None,
        attachments: list | None = None,
        parent_branch_id: uuid.UUID | None = None,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            branch_id=branch_id,
            sequence=sequence,
            role=role,
            content=content,
            model_id=model_id,
            usage=usage,
            citations=citations,
            attachments=attachments,
            parent_branch_id=parent_branch_id,
        )
        db.add(msg)
        await db.flush()
        await db.refresh(msg)
        return msg

    @staticmethod
    async def copy_branch_up_to(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        from_branch_id: uuid.UUID,
        new_branch_id: uuid.UUID,
        up_to_sequence: int,
    ) -> None:
        msgs = await db.execute(
            select(Message).where(
                Message.conversation_id == conversation_id,
                Message.branch_id == from_branch_id,
                Message.sequence < up_to_sequence,
            ).order_by(Message.sequence)
        )
        for original in msgs.scalars():
            copy = Message(
                conversation_id=conversation_id,
                branch_id=new_branch_id,
                parent_branch_id=from_branch_id,
                sequence=original.sequence,
                role=original.role,
                content=original.content,
                model_id=original.model_id,
                usage=original.usage,
                citations=original.citations,
                attachments=original.attachments,
            )
            db.add(copy)
        await db.flush()

    @staticmethod
    async def get(db: AsyncSession, message_id: uuid.UUID) -> Message | None:
        result = await db.execute(select(Message).where(Message.id == message_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def set_feedback(db: AsyncSession, message_id: uuid.UUID, feedback: str) -> None:
        from sqlalchemy import update
        await db.execute(
            update(Message).where(Message.id == message_id).values(feedback=feedback)
        )
