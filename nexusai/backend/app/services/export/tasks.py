"""Celery export task — builds user data ZIP and emails download link."""
from __future__ import annotations
import asyncio
import io
import json
import zipfile

from app.core.celery_app import celery_app
from app.core.config import settings


@celery_app.task(bind=True, name="export.build_user_export", max_retries=1)
def build_user_export(self, user_id: str, user_email: str | None) -> str:
    """Build ZIP of all user data, upload to GCS, email download link."""
    return asyncio.run(_async_export(self.request.id, user_id, user_email))


async def _async_export(task_id: str, user_id: str, user_email: str | None) -> str:
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy import select
    from app.models.conversation import Conversation
    from app.models.message import Message
    from app.core.redis import redis_client

    engine = create_async_engine(settings.DATABASE_URL, pool_size=2, max_overflow=2)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        async with Session() as db:
            # Conversations + messages
            convs = (await db.execute(
                select(Conversation).where(Conversation.user_id == user_id)
            )).scalars().all()

            conv_data = []
            for conv in convs:
                msgs = (await db.execute(
                    select(Message)
                    .where(Message.conversation_id == conv.id)
                    .order_by(Message.sequence)
                )).scalars().all()
                conv_data.append({
                    "id": str(conv.id),
                    "title": conv.title,
                    "model": conv.model,
                    "created_at": conv.created_at.isoformat(),
                    "messages": [
                        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat() if m.created_at else None}
                        for m in msgs
                    ],
                })
            zf.writestr("conversations.json", json.dumps(conv_data, indent=2))
            zf.writestr("README.txt", f"NexusAI data export for user {user_id}\nGenerated at: {__import__('datetime').datetime.utcnow().isoformat()}\n")

    buf.seek(0)

    # Upload to GCS or local
    export_path = f"exports/{user_id}/{task_id}.zip"
    from app.services.storage.gcs import upload, _use_gcs, _LOCAL_ROOT
    upload(export_path, buf.read(), "application/zip")

    if _use_gcs():
        download_url = f"https://storage.googleapis.com/{settings.GCS_BUCKET_NAME}/{export_path}"
    else:
        download_url = f"file://{_LOCAL_ROOT}/{export_path}"

    # Store URL in Redis
    await redis_client.setex(f"export:{task_id}:url", 86400, download_url)
    await redis_client.setex(f"export:{task_id}:status", 86400, "ready")

    # Email link
    if user_email and settings.SENDGRID_API_KEY:
        _send_export_email(user_email, download_url)

    await engine.dispose()
    return download_url


def _send_export_email(email: str, url: str) -> None:
    import sendgrid
    from sendgrid.helpers.mail import Mail
    sg = sendgrid.SendGridAPIClient(api_key=settings.SENDGRID_API_KEY)
    msg = Mail(
        from_email="noreply@nexusai.dev",
        to_emails=email,
        subject="Your NexusAI data export is ready",
        html_content=f'<p>Your data export is ready. <a href="{url}">Download here</a> (link expires in 24 hours).</p>',
    )
    sg.send(msg)
