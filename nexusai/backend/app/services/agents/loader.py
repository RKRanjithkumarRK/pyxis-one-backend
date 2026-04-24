from __future__ import annotations
import logging
import re
from pathlib import Path

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.agent import AgentRepository

logger = logging.getLogger("nexusai.agents.loader")

BUILTIN_YAML = Path(__file__).parent.parent.parent / "data" / "agents" / "builtin.yaml"


def _slug_from_name(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug


async def load_builtin_agents(db: AsyncSession) -> int:
    if not BUILTIN_YAML.exists():
        logger.warning("builtin.yaml not found at %s — skipping", BUILTIN_YAML)
        return 0

    with BUILTIN_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    agents_data: list[dict] = data.get("agents", [])
    loaded = 0

    for entry in agents_data:
        slug = entry.get("slug") or _slug_from_name(entry.get("name", "unknown"))
        capabilities = entry.get("capabilities") or {}
        if isinstance(capabilities, dict):
            cap_clean = {
                "vision": bool(capabilities.get("vision", False)),
                "tool_use": bool(capabilities.get("tool_use", False)),
                "web_search": bool(capabilities.get("web_search", False)),
            }
        else:
            cap_clean = {"vision": False, "tool_use": False, "web_search": False}

        starters = entry.get("starters") or []
        starters = [s for s in starters if isinstance(s, str)][:4]

        agent_data = {
            "slug": slug,
            "name": entry.get("name", slug),
            "description": entry.get("description"),
            "icon": entry.get("icon"),
            "category": entry.get("category", "general"),
            "instructions": entry.get("instructions"),
            "starters": starters or None,
            "capabilities": cap_clean,
            "default_model": entry.get("default_model", "claude-sonnet-4"),
            "visibility": entry.get("visibility", "public"),
            "is_builtin": True,
        }

        try:
            await AgentRepository.upsert_builtin(db, agent_data)
            loaded += 1
        except Exception as exc:
            logger.error("Failed to upsert built-in agent '%s': %s", slug, exc)

    await db.commit()
    logger.info("Loaded %d built-in agents", loaded)
    return loaded
