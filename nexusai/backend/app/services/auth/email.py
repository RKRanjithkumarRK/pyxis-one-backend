from __future__ import annotations
import logging
from app.core.config import settings

logger = logging.getLogger("nexusai.auth.email")

MAGIC_LINK_TEMPLATE = """
Hi {name},

Click the link below to sign in to NexusAI. This link expires in 15 minutes.

{link}

If you didn't request this, ignore this email.

— The NexusAI Team
"""


async def send_magic_link(email: str, name: str, token: str) -> bool:
    link = f"{settings.FRONTEND_URL}/auth/magic?token={token}"

    if not settings.SENDGRID_API_KEY:
        logger.warning("SENDGRID_API_KEY not set — magic link would be: %s", link)
        return True  # dev mode: log instead of fail

    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail, Email, To, Content

        sg = sendgrid.SendGridAPIClient(api_key=settings.SENDGRID_API_KEY)
        message = Mail(
            from_email=Email("noreply@nexusai.dev", "NexusAI"),
            to_emails=To(email),
            subject="Your NexusAI sign-in link",
            plain_text_content=Content(
                "text/plain",
                MAGIC_LINK_TEMPLATE.format(name=name or "there", link=link),
            ),
        )
        resp = sg.client.mail.send.post(request_body=message.get())
        return resp.status_code in (200, 202)
    except Exception as exc:
        logger.error("Failed to send magic link email to %s: %s", email, exc)
        return False


async def send_verification_email(email: str, name: str, token: str) -> bool:
    link = f"{settings.FRONTEND_URL}/auth/verify-email?token={token}"

    if not settings.SENDGRID_API_KEY:
        logger.warning("SENDGRID_API_KEY not set — verification link: %s", link)
        return True

    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail, Email, To, Content

        sg = sendgrid.SendGridAPIClient(api_key=settings.SENDGRID_API_KEY)
        message = Mail(
            from_email=Email("noreply@nexusai.dev", "NexusAI"),
            to_emails=To(email),
            subject="Verify your NexusAI email",
            plain_text_content=Content(
                "text/plain",
                f"Hi {name or 'there'},\n\nVerify your email:\n{link}\n\nThis link expires in 24 hours.\n\n— NexusAI",
            ),
        )
        resp = sg.client.mail.send.post(request_body=message.get())
        return resp.status_code in (200, 202)
    except Exception as exc:
        logger.error("Failed to send verification email to %s: %s", email, exc)
        return False
