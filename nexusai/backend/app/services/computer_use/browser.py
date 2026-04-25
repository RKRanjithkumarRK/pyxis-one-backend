"""Computer Use — Playwright-in-E2B screenshot→model→action loop."""
from __future__ import annotations
import asyncio
import base64
import json
import logging
import re
from typing import AsyncIterator

from app.core.config import settings
from app.core.telemetry import tracer

logger = logging.getLogger("nexusai.computer_use")

# Domains blocked for SSRF protection
BLOCKED_DOMAINS = {
    "169.254.169.254",  # AWS metadata
    "metadata.google.internal",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
}


def _is_ssrf(url: str) -> bool:
    """Block internal IPs and metadata endpoints."""
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    if any(blocked in host for blocked in BLOCKED_DOMAINS):
        return True
    # Block raw IP ranges
    parts = host.split(".")
    if len(parts) == 4:
        try:
            o1 = int(parts[0])
            if o1 in (10, 127):
                return True
            if o1 == 172 and 16 <= int(parts[1]) <= 31:
                return True
            if o1 == 192 and parts[1] == "168":
                return True
        except ValueError:
            pass
    return False


async def run_computer_use_loop(
    sandbox_id: str,
    task: str,
    model: str = "claude-opus-4-20250514",
    max_steps: int = 10,
    approval_required: bool = True,
) -> AsyncIterator[dict]:
    """
    Main computer use loop:
    1. Take screenshot of current browser state
    2. Send to vision model with task
    3. Parse action from model response
    4. Execute action in Playwright (inside E2B sandbox)
    5. Repeat until done or max_steps reached
    """
    from app.services.sandbox.e2b_service import execute_command, write_file

    if not sandbox_id or sandbox_id.startswith("dev-"):
        yield {"type": "error", "message": "Computer use requires a real E2B sandbox"}
        return

    # Install Playwright in sandbox if needed
    yield {"type": "status", "message": "Setting up browser environment..."}
    await execute_command(sandbox_id, "pip install playwright && playwright install chromium --with-deps", "/workspace")

    # Write the browser controller script
    controller_script = _get_controller_script(task)
    await write_file(sandbox_id, "/workspace/__cu_controller.py", controller_script)

    screenshot_b64: str | None = None
    history: list[dict] = []
    step = 0

    while step < max_steps:
        step += 1
        yield {"type": "step", "step": step, "max_steps": max_steps}

        # Take screenshot
        screenshot_result = await execute_command(
            sandbox_id,
            "python3 -c \"from __cu_controller import screenshot; screenshot()\"",
            "/workspace",
        )

        if screenshot_result["exit_code"] != 0:
            yield {"type": "error", "message": screenshot_result["stderr"]}
            break

        # Read screenshot as base64
        read_result = await execute_command(sandbox_id, "base64 /tmp/screenshot.png", "/workspace")
        if read_result["exit_code"] != 0:
            yield {"type": "error", "message": "Could not read screenshot"}
            break

        screenshot_b64 = read_result["stdout"].strip()
        yield {"type": "screenshot", "data": screenshot_b64[:200] + "..."}  # truncated for SSE

        # Ask model what to do
        action = await _ask_model(task, screenshot_b64, history, model)
        if action is None:
            yield {"type": "done", "message": "Task complete — model indicated done"}
            break

        yield {"type": "action", "action": action, "requires_approval": approval_required}

        # In approval mode, caller must approve each action
        if approval_required:
            yield {"type": "waiting_approval", "action": action}
            return  # Caller will call again with approved action

        # Execute action
        error = await _execute_action(sandbox_id, action)
        if error:
            yield {"type": "error", "message": error}
            break

        history.append({"action": action, "screenshot": f"step_{step}"})

    yield {"type": "finished", "steps": step}


async def execute_approved_action(sandbox_id: str, action: dict) -> dict:
    """Execute a single action that was approved by the user."""
    error = await _execute_action(sandbox_id, action)
    if error:
        return {"status": "error", "error": error}
    return {"status": "ok"}


async def _execute_action(sandbox_id: str, action: dict) -> str | None:
    from app.services.sandbox.e2b_service import execute_command
    atype = action.get("type", "")
    cmd = None

    if atype == "navigate":
        url = action.get("url", "")
        if _is_ssrf(url):
            return f"SSRF protection: blocked URL {url}"
        cmd = f"python3 -c \"from __cu_controller import navigate; navigate('{url}')\" 2>&1"

    elif atype == "click":
        x, y = action.get("x", 0), action.get("y", 0)
        cmd = f"python3 -c \"from __cu_controller import click; click({x}, {y})\" 2>&1"

    elif atype == "type":
        text = action.get("text", "").replace("'", "\\'")
        cmd = f"python3 -c \"from __cu_controller import type_text; type_text('{text}')\" 2>&1"

    elif atype == "scroll":
        direction = action.get("direction", "down")
        amount = action.get("amount", 300)
        cmd = f"python3 -c \"from __cu_controller import scroll; scroll('{direction}', {amount})\" 2>&1"

    elif atype == "key":
        key = action.get("key", "Enter")
        cmd = f"python3 -c \"from __cu_controller import key_press; key_press('{key}')\" 2>&1"

    else:
        return f"Unknown action type: {atype}"

    if cmd:
        result = await execute_command(sandbox_id, cmd, "/workspace")
        if result["exit_code"] != 0:
            return result["stderr"]
    return None


async def _ask_model(task: str, screenshot_b64: str, history: list[dict], model: str) -> dict | None:
    """Ask the vision model what action to take next."""
    from app.services.llm.router import litellm_complete

    history_text = ""
    if history:
        history_text = "\nPrevious actions:\n" + "\n".join(
            f"- {h['action']['type']}: {h['action']}" for h in history[-5:]
        )

    prompt = f"""You are a computer use agent. Your task is: {task}
{history_text}

Look at the screenshot and decide the next action. Respond ONLY with a JSON object.

Available actions:
- {{"type": "navigate", "url": "https://..."}}
- {{"type": "click", "x": <number>, "y": <number>}}
- {{"type": "type", "text": "..."}}
- {{"type": "scroll", "direction": "down|up", "amount": 300}}
- {{"type": "key", "key": "Enter|Tab|Escape|..."}}
- {{"type": "done"}} — if task is complete

Return ONLY the JSON, nothing else."""

    try:
        response = await litellm_complete(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        action = json.loads(response.strip())
        if action.get("type") == "done":
            return None
        return action
    except Exception as exc:
        logger.warning("Model response parse failed: %s", exc)
        return None


def _get_controller_script(task: str) -> str:
    return '''"""Playwright browser controller for Computer Use."""
from playwright.sync_api import sync_playwright
import base64, io

_pw = None
_browser = None
_page = None


def _ensure():
    global _pw, _browser, _page
    if _page is None:
        _pw = sync_playwright().start()
        _browser = _pw.chromium.launch(headless=True, args=["--no-sandbox"])
        _page = _browser.new_page(viewport={"width": 1280, "height": 720})


def screenshot():
    _ensure()
    _page.screenshot(path="/tmp/screenshot.png")


def navigate(url: str):
    _ensure()
    _page.goto(url, wait_until="networkidle", timeout=30000)


def click(x: int, y: int):
    _ensure()
    _page.mouse.click(x, y)


def type_text(text: str):
    _ensure()
    _page.keyboard.type(text)


def scroll(direction: str, amount: int = 300):
    _ensure()
    dy = amount if direction == "down" else -amount
    _page.mouse.wheel(0, dy)


def key_press(key: str):
    _ensure()
    _page.keyboard.press(key)
'''
