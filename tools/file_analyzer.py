"""
File content extractor — reads uploaded files and returns text for model context.
Supports: PDF, plain text, code files, images (base64 for vision models).
"""

from __future__ import annotations
import base64
import mimetypes
from pathlib import Path


_TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".scss",
    ".rs", ".go", ".java", ".kt", ".swift", ".c", ".cpp", ".h",
    ".sql", ".sh", ".bash", ".zsh", ".xml", ".env",
}

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def extract(file_bytes: bytes, filename: str, extract_type: str = "full") -> dict:
    """
    Returns:
        {
            "filename": str,
            "content_type": str,   # "text" | "image" | "pdf" | "unsupported"
            "content": str,        # extracted text content
            "image_b64": str | None,  # base64 for vision
            "page_count": int | None,
            "truncated": bool,
        }
    """
    suffix = Path(filename).suffix.lower()
    mime, _ = mimetypes.guess_type(filename)

    if suffix == ".pdf":
        return _extract_pdf(file_bytes, filename, extract_type)
    elif suffix in _IMAGE_EXTENSIONS:
        return _extract_image(file_bytes, filename, mime or "image/jpeg")
    elif suffix in _TEXT_EXTENSIONS:
        return _extract_text(file_bytes, filename)
    else:
        # Try as text
        try:
            text = file_bytes.decode("utf-8", errors="replace")
            return {
                "filename": filename,
                "content_type": "text",
                "content": _truncate(text, 32000),
                "image_b64": None,
                "page_count": None,
                "truncated": len(text) > 32000,
            }
        except Exception:
            return {
                "filename": filename,
                "content_type": "unsupported",
                "content": f"Cannot extract content from {filename} (unsupported format).",
                "image_b64": None,
                "page_count": None,
                "truncated": False,
            }


def _extract_pdf(file_bytes: bytes, filename: str, extract_type: str) -> dict:
    try:
        import pypdf
        import io

        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        page_count = len(reader.pages)

        if extract_type == "summary":
            # Extract first 2 + last 1 pages for summary
            pages_to_read = list(range(min(2, page_count))) + ([page_count - 1] if page_count > 2 else [])
        else:
            pages_to_read = range(page_count)

        text_parts = []
        for i in pages_to_read:
            page_text = reader.pages[i].extract_text() or ""
            if page_text.strip():
                text_parts.append(f"--- Page {i+1} ---\n{page_text.strip()}")

        full_text = "\n\n".join(text_parts)
        truncated = len(full_text) > 32000

        return {
            "filename": filename,
            "content_type": "pdf",
            "content": _truncate(full_text, 32000),
            "image_b64": None,
            "page_count": page_count,
            "truncated": truncated,
        }
    except ImportError:
        return {
            "filename": filename,
            "content_type": "pdf",
            "content": "PDF extraction requires pypdf package.",
            "image_b64": None,
            "page_count": None,
            "truncated": False,
        }
    except Exception as e:
        return {
            "filename": filename,
            "content_type": "pdf",
            "content": f"PDF extraction error: {str(e)[:200]}",
            "image_b64": None,
            "page_count": None,
            "truncated": False,
        }


def _extract_image(file_bytes: bytes, filename: str, mime: str) -> dict:
    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
    return {
        "filename": filename,
        "content_type": "image",
        "content": f"[Image: {filename}]",
        "image_b64": f"data:{mime};base64,{b64}",
        "page_count": None,
        "truncated": False,
    }


def _extract_text(file_bytes: bytes, filename: str) -> dict:
    text = file_bytes.decode("utf-8", errors="replace")
    truncated = len(text) > 32000
    return {
        "filename": filename,
        "content_type": "text",
        "content": _truncate(text, 32000),
        "image_b64": None,
        "page_count": None,
        "truncated": truncated,
    }


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated — {len(text) - limit} characters omitted ...]"


def get_file_metadata(filename: str, file_size: int) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        kind = "PDF document"
    elif suffix in _IMAGE_EXTENSIONS:
        kind = "Image"
    elif suffix in {".py", ".js", ".ts", ".jsx", ".tsx"}:
        kind = "Code file"
    elif suffix == ".csv":
        kind = "CSV data"
    elif suffix == ".json":
        kind = "JSON data"
    else:
        kind = "Text file"

    size_kb = file_size / 1024
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"

    return {"kind": kind, "size": size_str, "extension": suffix}
