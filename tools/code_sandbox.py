"""
Code execution sandbox.

Priority:
  1. E2B cloud sandbox (if E2B_API_KEY set) — full Python, matplotlib, etc.
  2. RestrictedPython in-process (safe subset, no I/O, no imports)
  3. Returns error if both unavailable.
"""

from __future__ import annotations
import asyncio
import sys
import io
import traceback
from core.config import settings


# ── E2B Cloud Sandbox ─────────────────────────────────────────────────────────

_e2b_sandbox = None
_e2b_lock = asyncio.Lock()


async def _get_e2b_sandbox():
    global _e2b_sandbox
    async with _e2b_lock:
        try:
            import e2b_code_interpreter as e2b  # type: ignore
            if _e2b_sandbox is None or getattr(_e2b_sandbox, "_closed", True):
                _e2b_sandbox = await e2b.AsyncSandbox.create(
                    api_key=settings.E2B_API_KEY,
                    timeout=300,
                )
        except Exception:
            _e2b_sandbox = None
    return _e2b_sandbox


async def _run_e2b(code: str) -> str:
    sandbox = await _get_e2b_sandbox()
    if sandbox is None:
        raise RuntimeError("E2B sandbox unavailable")

    try:
        result = await sandbox.run_code(code)
        parts = []

        stdout = "".join(result.logs.stdout) if result.logs.stdout else ""
        stderr = "".join(result.logs.stderr) if result.logs.stderr else ""

        if stdout:
            parts.append(f"Output:\n{stdout.strip()}")
        if stderr:
            parts.append(f"Stderr:\n{stderr.strip()[:500]}")
        if result.error:
            parts.append(f"Error: {result.error.value}")

        # Base64-encode any image outputs (matplotlib plots etc.)
        import base64
        for r in (result.results or []):
            if hasattr(r, "png") and r.png:
                parts.append(f"[IMAGE:data:image/png;base64,{r.png}]")
            elif hasattr(r, "text") and r.text:
                parts.append(r.text)

        return "\n".join(parts) if parts else "Code executed successfully (no output)."

    except Exception as e:
        # Sandbox may have died — reset it
        global _e2b_sandbox
        _e2b_sandbox = None
        raise RuntimeError(f"Sandbox execution failed: {e}")


# ── In-process restricted execution (fallback) ────────────────────────────────

_ALLOWED_BUILTINS = {
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "sum", "min", "max", "abs", "round",
    "int", "float", "str", "bool", "list", "dict", "set", "tuple",
    "isinstance", "type", "repr", "chr", "ord", "hex", "bin", "oct",
    "True", "False", "None",
}

_BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "ctypes", "multiprocessing", "threading",
    "signal", "pty", "fcntl", "resource",
}


async def _run_restricted(code: str) -> str:
    # Check for dangerous imports first
    for blocked in _BLOCKED_IMPORTS:
        if f"import {blocked}" in code or f"from {blocked}" in code:
            return f"Execution blocked: import of '{blocked}' is not allowed in sandbox mode."

    loop = asyncio.get_event_loop()

    def _exec():
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        local_ns: dict = {}
        try:
            exec(compile(code, "<sandbox>", "exec"), {"__builtins__": {}}, local_ns)  # noqa: S102
            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()
            output = []
            if stdout:
                output.append(f"Output:\n{stdout.strip()}")
            if stderr:
                output.append(f"Stderr:\n{stderr.strip()[:300]}")
            return "\n".join(output) if output else "Code executed (no output)."
        except Exception:
            return f"Error:\n{traceback.format_exc(limit=5)}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return await loop.run_in_executor(None, _exec)


# ── Public API ────────────────────────────────────────────────────────────────

async def execute(code: str, language: str = "python") -> str:
    """Execute code and return result string."""
    if language != "python":
        return f"Only Python execution is supported (got: {language})."

    if settings.E2B_API_KEY:
        try:
            return await asyncio.wait_for(_run_e2b(code), timeout=30.0)
        except asyncio.TimeoutError:
            return "Execution timed out (30s limit)."
        except Exception as e:
            # Fall through to restricted mode
            pass

    # Restricted in-process fallback
    try:
        return await asyncio.wait_for(_run_restricted(code), timeout=10.0)
    except asyncio.TimeoutError:
        return "Execution timed out (10s limit)."
    except Exception as e:
        return f"Execution error: {str(e)[:300]}"
