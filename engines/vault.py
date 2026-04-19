import hashlib
import os
import base64
import json
from datetime import datetime
from sqlalchemy import select
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from core.database import AsyncSessionLocal
from core.models import VaultEntry
from core.config import settings
import engines.anthropic_client as ac


def _derive_key() -> bytes:
    return hashlib.sha256(settings.SECRET_KEY.encode()).digest()


def _encrypt(plaintext: str, key: bytes) -> str:
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def _decrypt_data(encrypted_b64: str, key: bytes) -> str:
    raw = base64.b64decode(encrypted_b64)
    nonce, ciphertext = raw[:12], raw[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")


class VaultEngine:
    async def store(
        self,
        session_id: str,
        content: str,
        concept_tags: list,
        emotion_tags: list,
    ) -> str:
        key = _derive_key()
        encrypted = _encrypt(content, key)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        async with AsyncSessionLocal() as db:
            entry = VaultEntry(
                session_id=session_id,
                content_encrypted=encrypted,
                content_hash=content_hash,
                concept_tags=concept_tags,
                emotion_tags=emotion_tags,
                timestamp=datetime.utcnow(),
            )
            db.add(entry)
            await db.commit()
            await db.refresh(entry)
            return entry.id

    async def search(self, session_id: str, query: str) -> list:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(VaultEntry)
                .where(VaultEntry.session_id == session_id)
                .order_by(VaultEntry.timestamp.desc())
            )
            entries = result.scalars().all()

        if not entries:
            return []

        key = _derive_key()
        decrypted_entries = []
        for entry in entries:
            try:
                content = _decrypt_data(entry.content_encrypted, key)
                decrypted_entries.append(
                    {
                        "id": entry.id,
                        "content": content,
                        "concept_tags": entry.concept_tags or [],
                        "emotion_tags": entry.emotion_tags or [],
                        "timestamp": entry.timestamp.isoformat(),
                    }
                )
            except Exception:
                continue

        if not decrypted_entries:
            return []

        entries_text = json.dumps(
            [{"id": e["id"], "content": e["content"][:300], "tags": e["concept_tags"]} for e in decrypted_entries]
        )

        prompt = (
            f"Find the most relevant vault entries for this search query.\n\n"
            f"Query: {query}\n\n"
            f"Entries: {entries_text}\n\n"
            "Return JSON array of matching entry IDs ordered by relevance:\n"
            '["id1", "id2", ...]\n'
            "Return ONLY valid JSON array."
        )
        system = "You are a semantic search engine. Return only valid JSON array of IDs."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=256)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            ranked_ids: list = json.loads(raw)
        except Exception:
            ranked_ids = [e["id"] for e in decrypted_entries[:5]]

        id_to_entry = {e["id"]: e for e in decrypted_entries}
        results = [id_to_entry[rid] for rid in ranked_ids if rid in id_to_entry]

        if not results:
            results = decrypted_entries[:5]

        return results

    async def decrypt(self, entry_id: str, session_id: str) -> str:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(VaultEntry).where(
                    VaultEntry.id == entry_id,
                    VaultEntry.session_id == session_id,
                )
            )
            entry = result.scalar_one_or_none()

        if entry is None:
            raise ValueError(f"Entry {entry_id} not found for session {session_id}")

        key = _derive_key()
        return _decrypt_data(entry.content_encrypted, key)

    async def get_timeline(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(VaultEntry)
                .where(VaultEntry.session_id == session_id)
                .order_by(VaultEntry.timestamp)
            )
            entries = result.scalars().all()

        key = _derive_key()
        timeline = []
        for entry in entries:
            try:
                content = _decrypt_data(entry.content_encrypted, key)
                preview = content[:120] + "..." if len(content) > 120 else content
            except Exception:
                preview = "[encrypted]"

            timeline.append(
                {
                    "id": entry.id,
                    "preview": preview,
                    "concept_tags": entry.concept_tags or [],
                    "emotion_tags": entry.emotion_tags or [],
                    "timestamp": entry.timestamp.isoformat(),
                }
            )

        return timeline

    async def export_pdf(self, session_id: str) -> bytes:
        timeline = await self.get_timeline(session_id)
        key = _derive_key()

        lines = [
            "PYXIS ONE — VAULT EXPORT",
            f"Session: {session_id}",
            f"Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
            "",
        ]

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(VaultEntry)
                .where(VaultEntry.session_id == session_id)
                .order_by(VaultEntry.timestamp)
            )
            entries = result.scalars().all()

        for entry in entries:
            try:
                content = _decrypt_data(entry.content_encrypted, key)
            except Exception:
                content = "[decryption error]"

            lines.append(f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M')}]")
            lines.append(f"Concepts: {', '.join(entry.concept_tags or [])}")
            lines.append(f"Emotions: {', '.join(entry.emotion_tags or [])}")
            lines.append(content)
            lines.append("-" * 40)
            lines.append("")

        return "\n".join(lines).encode("utf-8")


vault_engine = VaultEngine()
