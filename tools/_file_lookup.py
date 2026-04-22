"""DB lookup for uploaded files — used by the read_file tool."""

from __future__ import annotations


async def lookup_file(file_id: str, extract_type: str = "full") -> str:
    try:
        from core.database import AsyncSessionLocal
        from core.models import FileUpload
        from sqlalchemy import select

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(FileUpload).where(FileUpload.id == file_id)
            )
            file = result.scalar_one_or_none()

        if file is None:
            return f"File not found: {file_id}"

        content = file.extracted_text or ""
        if extract_type == "summary" and len(content) > 2000:
            content = content[:2000] + "... [truncated for summary]"

        return f"File: {file.filename}\n\n{content}" if content else f"File {file.filename} has no extractable text."

    except Exception as e:
        return f"File lookup error: {str(e)[:200]}"
