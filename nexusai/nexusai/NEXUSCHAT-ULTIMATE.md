# NEXUSCHAT ULTIMATE — Full ChatGPT Equivalent (Non-Coding Surface)
# Upgrade every chat/agent/research/memory/voice/image/canvas feature to real parity.
# ADD TO EXISTING NEXUSAI-ULTIMATE + NEXUSCODE-CLOUD BUILD.

---

# IDENTITY

You are the engineering org at NexusAI. The coding surface (NexusCode Cloud)
is already specified in NEXUSCODE-CLOUD.md. This document covers EVERYTHING
ELSE — the NexusChat surface, agents, research, memory, canvas, voice, image,
workflows, projects, knowledge bases, sharing, admin, billing.

We are not building "ChatGPT-like" features. We are building the same product
capability with our own multi-provider edge. Every feature listed below has
a named production analog. We match or exceed the production analog on:
  - Latency SLA (measured, not estimated)
  - Functional completeness (every sub-feature, not the 80% version)
  - Cross-platform parity (Windows, macOS, Linux, iOS, iPadOS, Android, Chromebook, web, PWA)
  - Reliability (graceful degradation, retries, offline behavior)
  - Accessibility (WCAG AA minimum, keyboard-only usable, screen reader clean)

No mocks. No placeholders. No "coming soon." If a sub-feature is listed, it ships.

---

# FEATURE MAP — WHAT "FULL" MEANS

| Surface | Production Analog | Must-Have Sub-Features |
|---|---|---|
| NexusChat Core | ChatGPT conversation | Streaming, stop, regenerate, edit message, branch conversation, model switch mid-conversation, attachments, voice input, voice output, copy, share, pin, archive, search across all chats, export |
| Multi-provider | — (our edge) | 7 providers, 20+ models, per-message switch, compare mode (2-3 models side-by-side streaming), cost/latency shown per model |
| Agents (GPT Store) | ChatGPT GPTs | Agent Store browse/search/category, 40+ built-in agents, user-created custom agents, per-agent tools (web, code, image, files, memory), per-agent knowledge base, per-agent system prompt, agent versioning, publish public/unlisted/private, rating, usage count |
| Deep Research | Perplexity / ChatGPT Deep Research | Multi-step plan → parallel search → source dedup → cite every claim → structured report with sections, tables, chart embeds → follow-up questions → export PDF/MD |
| Canvas | ChatGPT Canvas / Claude Artifacts | Split pane document/code editor, inline AI edits with diff view, comments, version history, restore, collaborative cursor positions, export |
| Memory | ChatGPT Memory | Auto-extracted facts, manual add/edit/delete, per-conversation toggle, full transparency panel, used-in-response indicator |
| Projects | ChatGPT Projects / Claude Projects | Project-scoped chats, shared files, shared system prompt, shared memory, invite members, role permissions |
| Knowledge Base (RAG) | ChatGPT custom GPT files / NotebookLM | Upload PDF/DOCX/TXT/MD/HTML/CSV/XLSX/PPTX, chunking, embedding, Qdrant vector store, hybrid search (BM25 + vector), rerank, citation-per-claim, source preview on hover |
| Image Studio | DALL-E / Midjourney | Flux Pro, Flux Schnell, SDXL, DALL-E 3, Gemini Imagen, prompt library, aspect ratios, batch generation, in-painting, out-painting, upscale, remove background, style transfer |
| Voice | ChatGPT Voice | Real-time push-to-talk, continuous mode, barge-in (interrupt assistant), 9+ voices via ElevenLabs, Whisper transcription, visible transcript, language auto-detect |
| Video Gen | Sora / Veo | Runway, Luma Dream Machine, Veo 3 via API, prompt-to-video, image-to-video, duration control |
| Workflows | Zapier / ChatGPT Actions | Visual builder (nodes + edges), triggers (schedule, webhook, new email, etc), actions across providers, per-step model choice, test run, run history |
| Computer Use | Claude Computer Use / Operator | Playwright-driven browser automation, screenshot-per-step, user-in-the-loop approval, safe-mode list of allowed domains |
| Sharing | ChatGPT share links | Public share link, snapshot at share time, read-only, optional password, optional expiry |
| Search | ChatGPT search | Full-text across all user chats + messages + canvas docs, filters (date, model, agent, project), instant results |
| Export | ChatGPT export | Full data export as ZIP (all chats, memory, files, knowledge bases) in under 10 min |
| Settings | ChatGPT settings | Account, appearance (theme, accent, density, font size), data controls (history toggle, export, delete), personalization (custom instructions, memory), notifications, connected apps, API keys (BYOK), billing, security (2FA, sessions, passkeys) |
| Auth | — | Email/password, Google, GitHub, Apple, Microsoft, Magic link, Guest mode (10 free messages, upgrade path), 2FA TOTP, passkeys (WebAuthn), session management |
| Billing | ChatGPT Plus/Team/Enterprise | Free, Plus ($20), Team ($30/user), Enterprise (sales), Stripe checkout, metered usage for API credits, invoices, seat management |
| Admin | ChatGPT admin console | Org dashboard, user management, SSO (SAML/OIDC), audit logs, usage analytics, content filters, data retention controls |
| Mobile | ChatGPT iOS/Android | Installable PWA on all platforms, push notifications (PWA+FCM), offline history browsing, voice button in home screen shortcut |

---

# ARCHITECTURE

## Shared with existing build
- Backend: FastAPI on Cloud Run
- Frontend: Next.js 14 App Router on Cloud Run (or Vercel for dev)
- DB: Cloud SQL Postgres 15 + pgvector (auxiliary) + Qdrant (primary vector store)
- Cache/Queue: Memorystore Redis + Redis Streams for real-time events
- Background jobs: Celery + Celery Beat
- Object storage: GCS
- Observability: OpenTelemetry → Cloud Trace + Cloud Logging; Sentry for errors; PostHog for product analytics
- Auth: NextAuth on frontend + FastAPI JWT resource server

## New subsystems introduced by this doc
- **LiteLLM Router** — unified completion+embedding+image+TTS+STT across all 7 providers with per-model routing, retry, fallback, cost tracking
- **Research Orchestrator** — plan → parallel search fan-out → dedup → summarize → cite
- **Memory Engine** — extract, store, retrieve, reinject; user-visible
- **RAG Pipeline** — ingest → chunk → embed → store → hybrid retrieve → rerank → cite
- **Canvas Engine** — Yjs CRDT for multi-cursor, version snapshots, inline-edit with diff
- **Workflow Runtime** — ReactFlow on frontend, Temporal-lite executor on backend (Celery-based)
- **Voice Pipeline** — Whisper streaming STT + ElevenLabs streaming TTS + interruption detection
- **Image Pipeline** — Flux/SDXL/DALL-E via Replicate and direct; ControlNet for in/out-painting
- **Computer Use Sandbox** — reuse E2B sandbox, headless Chromium + Playwright

---

# ============================================================
# PHASE NC-1 — NexusChat core at ChatGPT parity
# ============================================================

## Behaviors

1. **Streaming**
   - SSE from backend; frontend uses `useChat` hook with abort controller
   - First token SLA < 400 ms (cached routes < 150 ms)
   - Stop button aborts at any token; server-side cancels the provider stream
   - Backpressure: if client disconnects, kill upstream within 500 ms (save tokens)

2. **Message actions** (hover menu on every assistant message)
   - Copy (plain + markdown + rich)
   - Regenerate (same model | different model dropdown)
   - Edit my message → creates branch, old branch preserved, branch switcher in UI like ChatGPT's `< 2 / 3 >`
   - Good/Bad feedback → stored for eval
   - Read aloud (TTS)
   - Share this message only (creates permalink)

3. **Conversation list (left sidebar)**
   - Grouping: Today / Yesterday / Previous 7 Days / Previous 30 Days / By Month
   - Rename, pin (pinned section stays on top), archive, delete, export
   - Search within list (instant fuzzy)
   - Keyboard: ⌘K opens palette (new chat, search chats, switch model, open settings)

4. **Per-message model switcher**
   - Dropdown next to send button
   - Shows: model name, provider, est. cost per 1k tokens, est. latency (based on last 50 calls rolling)
   - Recently used models at top

5. **Compare mode**
   - Toggle: "Compare"
   - Pick 2 or 3 models → one prompt → all stream simultaneously in columns
   - After all finish: vote which was best (feeds eval dataset)
   - Shareable compare result link

6. **Attachments (drag-drop, paste, click)**
   - Images (PNG/JPG/WEBP/HEIC) → vision-capable model auto-selected if current lacks vision
   - PDFs → extract text + extract images per page → inline
   - Office docs (DOCX/XLSX/PPTX) → converted via libreoffice headless
   - Code files (any extension) → syntax-highlighted in message
   - Audio (MP3/WAV/M4A) → transcribed via Whisper → text attached
   - Per-file size limit: 50 MB; per-message total: 200 MB

7. **Artifact detection** (ChatGPT-style auto Canvas)
   - Heuristic: assistant produced ≥ 15 lines of code OR ≥ 400 words document → "Open in Canvas" button appears
   - Clicking opens right-side Canvas pane with that content, editable

8. **Voice output toggle** (per-conversation)
   - When on, every assistant message auto-reads via ElevenLabs streaming TTS
   - Barge-in: user presses mic → stops TTS mid-word

9. **Offline behavior**
   - Service Worker caches last 50 conversations (IndexedDB)
   - Offline banner; pending sends queued and fire on reconnect
   - Message composer still works (saves as draft)

## SLAs
| Action | Target |
|---|---|
| First token | < 400 ms (P95), < 150 ms on cache hit |
| Full response for 500 tokens | ≤ provider latency + 100 ms overhead |
| Conversation list open | < 80 ms (virtualized) |
| ⌘K palette open | < 50 ms |
| Search across 10k messages | < 200 ms (Postgres full-text GIN index) |
| Attachment upload (10 MB) | < 3 s on broadband |

## Backend files

### app/services/llm/router.py
```python
"""
LiteLLM-backed unified router. Single call signature for all 7 providers.
Adds: cost tracking, fallback, retry with exponential backoff, stream-safe
cancellation, rolling latency stats per model.
"""
from __future__ import annotations
import asyncio
import time
from typing import AsyncIterator, Optional
from dataclasses import dataclass
from litellm import acompletion, aembedding
from litellm.exceptions import RateLimitError, APIError, Timeout
from app.core.config import settings
from app.core.redis import redis_client
from app.core.telemetry import tracer

@dataclass
class ModelRoute:
    provider: str
    model_id: str           # the id we expose to frontend
    litellm_id: str         # the id litellm expects
    vision: bool
    tool_use: bool
    max_input: int
    cost_in_per_1k: float
    cost_out_per_1k: float

ROUTES: dict[str, ModelRoute] = {
    "claude-sonnet-4": ModelRoute("anthropic", "claude-sonnet-4", "claude-sonnet-4-20250514", True, True, 200_000, 0.003, 0.015),
    "claude-opus-4":   ModelRoute("anthropic", "claude-opus-4",   "claude-opus-4-20250514",   True, True, 200_000, 0.015, 0.075),
    "gpt-4o":          ModelRoute("openai",    "gpt-4o",          "gpt-4o",                    True, True, 128_000, 0.0025, 0.01),
    "gpt-4o-mini":     ModelRoute("openai",    "gpt-4o-mini",     "gpt-4o-mini",              True, True, 128_000, 0.00015, 0.0006),
    "gemini-2-pro":    ModelRoute("google",    "gemini-2-pro",    "gemini/gemini-2.0-pro",    True, True, 2_000_000, 0.00125, 0.005),
    "gemini-2-flash":  ModelRoute("google",    "gemini-2-flash",  "gemini/gemini-2.0-flash",  True, True, 1_000_000, 0.000075, 0.0003),
    "groq-llama-70b":  ModelRoute("groq",      "groq-llama-70b",  "groq/llama-3.3-70b-versatile", False, True, 128_000, 0.00059, 0.00079),
    "mistral-large":   ModelRoute("mistral",   "mistral-large",   "mistral/mistral-large-latest", False, True, 128_000, 0.002, 0.006),
    "cerebras-llama":  ModelRoute("cerebras",  "cerebras-llama",  "cerebras/llama3.1-70b",    False, False, 128_000, 0.0006, 0.0006),
    "sambanova-llama": ModelRoute("sambanova", "sambanova-llama", "sambanova/Meta-Llama-3.1-70B-Instruct", False, False, 8_000, 0.0006, 0.0012),
}

FALLBACK_CHAIN = {
    "claude-sonnet-4": ["claude-opus-4", "gpt-4o"],
    "gpt-4o": ["claude-sonnet-4", "gemini-2-pro"],
    "gemini-2-pro": ["claude-sonnet-4", "gpt-4o"],
    # every model has a chain; define for each
}

async def stream_chat(
    model_id: str,
    messages: list[dict],
    *,
    tools: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    user_id: str,
    conversation_id: str,
) -> AsyncIterator[dict]:
    """Yields: {type: 'token'|'tool_call'|'done'|'error', ...}"""
    if model_id not in ROUTES:
        raise ValueError(f"Unknown model: {model_id}")
    route = ROUTES[model_id]

    with tracer.start_as_current_span("llm.stream_chat", attributes={
        "llm.provider": route.provider, "llm.model": route.model_id,
        "user.id": user_id, "conversation.id": conversation_id,
    }) as span:
        attempt = 0
        tried = [model_id]
        current = route
        while True:
            try:
                t0 = time.perf_counter()
                stream = await acompletion(
                    model=current.litellm_id,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata={"user_id": user_id, "conversation_id": conversation_id},
                )
                first_token_time = None
                usage = {"prompt_tokens": 0, "completion_tokens": 0}
                async for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - t0
                        await _record_latency(current.model_id, first_token_time)
                        span.set_attribute("llm.first_token_ms", int(first_token_time * 1000))
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue
                    if delta.content:
                        yield {"type": "token", "content": delta.content}
                    if getattr(delta, "tool_calls", None):
                        yield {"type": "tool_call", "tool_calls": [tc.model_dump() for tc in delta.tool_calls]}
                    if getattr(chunk, "usage", None):
                        usage["prompt_tokens"] = chunk.usage.prompt_tokens
                        usage["completion_tokens"] = chunk.usage.completion_tokens
                yield {"type": "done", "usage": usage, "model": current.model_id}
                await _record_cost(user_id, current, usage)
                return
            except (RateLimitError, APIError, Timeout, asyncio.TimeoutError) as e:
                chain = FALLBACK_CHAIN.get(model_id, [])
                next_model = next((m for m in chain if m not in tried), None)
                if next_model is None or attempt >= 3:
                    yield {"type": "error", "message": str(e), "model": current.model_id}
                    return
                tried.append(next_model)
                current = ROUTES[next_model]
                attempt += 1
                await asyncio.sleep(min(2 ** attempt, 8))

async def _record_latency(model_id: str, seconds: float):
    key = f"latency:{model_id}"
    await redis_client.lpush(key, int(seconds * 1000))
    await redis_client.ltrim(key, 0, 49)   # rolling 50

async def _record_cost(user_id: str, route: ModelRoute, usage: dict):
    cost = (usage["prompt_tokens"] / 1000) * route.cost_in_per_1k + \
           (usage["completion_tokens"] / 1000) * route.cost_out_per_1k
    await redis_client.incrbyfloat(f"cost:user:{user_id}:daily", cost)
    await redis_client.expire(f"cost:user:{user_id}:daily", 86400 * 40)

async def embed(texts: list[str], model: str = "text-embedding-3-large") -> list[list[float]]:
    resp = await aembedding(model=model, input=texts)
    return [d["embedding"] for d in resp.data]
```

### app/api/v1/chat.py (streaming endpoint)
```python
"""SSE streaming chat endpoint with full abort / save / memory / RAG hooks."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from app.core.security import get_current_user_or_guest
from app.services.llm.router import stream_chat
from app.services.memory.service import MemoryService
from app.services.rag.service import RAGService
from app.services.conversation.service import ConversationService
import json, asyncio

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/stream")
async def chat_stream(payload: dict, request: Request, user = Depends(get_current_user_or_guest)):
    conversation_id = payload["conversation_id"]
    model = payload["model"]
    user_message = payload["message"]
    attachments = payload.get("attachments", [])
    project_id = payload.get("project_id")
    agent_id = payload.get("agent_id")
    kb_ids = payload.get("knowledge_base_ids", [])
    use_web_search = payload.get("web_search", False)
    use_memory = payload.get("use_memory", True)

    convo = ConversationService()
    history = await convo.get_history(conversation_id, user.id)

    # 1. Build system prompt: agent + project + custom instructions + memory
    system_parts = []
    if agent_id:
        system_parts.append(await convo.get_agent_prompt(agent_id, user.id))
    elif project_id:
        system_parts.append(await convo.get_project_prompt(project_id, user.id))
    if use_memory:
        memories = await MemoryService().retrieve_for(user.id, user_message)
        if memories:
            system_parts.append("Relevant memory about the user:\n" + "\n".join(f"- {m}" for m in memories))

    # 2. RAG retrieval
    rag_context = []
    if kb_ids:
        rag_context = await RAGService().hybrid_search(kb_ids, user_message, k=8)
        if rag_context:
            blob = "\n\n".join(f"[source #{i+1}: {c.title}]\n{c.text}" for i, c in enumerate(rag_context))
            system_parts.append(f"Reference documents. Cite with [#n]:\n{blob}")

    # 3. Web search (Serper)
    web_results = []
    if use_web_search:
        from app.services.research.web import search as web_search
        web_results = await web_search(user_message, k=8)
        if web_results:
            blob = "\n\n".join(f"[web #{i+1} {r.title} — {r.url}]\n{r.snippet}" for i, r in enumerate(web_results))
            system_parts.append(f"Live web results. Cite with [web #n]:\n{blob}")

    system_prompt = "\n\n".join(system_parts) if system_parts else None

    # 4. Attachments → expand into message content
    content = [{"type": "text", "text": user_message}]
    for att in attachments:
        if att["type"] == "image":
            content.append({"type": "image_url", "image_url": {"url": att["url"]}})
        elif att["type"] == "document":
            # text was extracted at upload time
            content.append({"type": "text", "text": f"[Attached file {att['name']}]\n{att['extracted_text']}"})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": content})

    # 5. Save user message (before stream so it's persisted even on error)
    user_msg_id = await convo.append(conversation_id, "user", user_message, attachments=attachments)

    async def event_stream():
        assistant_buf = ""
        model_used = model
        usage_final = None
        try:
            async for event in stream_chat(
                model_id=model, messages=messages,
                user_id=user.id, conversation_id=conversation_id,
            ):
                if await request.is_disconnected():
                    break
                if event["type"] == "token":
                    assistant_buf += event["content"]
                    yield f"data: {json.dumps({'type':'token','content':event['content']})}\n\n"
                elif event["type"] == "done":
                    model_used = event["model"]
                    usage_final = event["usage"]
                    yield f"data: {json.dumps({'type':'done','usage':event['usage'],'model':event['model']})}\n\n"
                elif event["type"] == "error":
                    yield f"data: {json.dumps({'type':'error','message':event['message']})}\n\n"
        finally:
            if assistant_buf:
                await convo.append(
                    conversation_id, "assistant", assistant_buf,
                    model=model_used, usage=usage_final,
                    citations=[{"type": "kb", **c.meta()} for c in rag_context] +
                              [{"type": "web", **r.dict()} for r in web_results],
                )
                if use_memory:
                    asyncio.create_task(MemoryService().extract_from_exchange(
                        user.id, user_message, assistant_buf
                    ))

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/compare")
async def chat_compare(payload: dict, user = Depends(get_current_user_or_guest)):
    """Stream 2-3 models in parallel. Returns multiplexed SSE."""
    models: list[str] = payload["models"]
    assert 2 <= len(models) <= 3
    prompt = payload["message"]

    async def multiplexed():
        queue: asyncio.Queue = asyncio.Queue()

        async def run(idx: int, m: str):
            async for ev in stream_chat(m, [{"role":"user","content":prompt}],
                                        user_id=user.id, conversation_id=f"compare-{idx}"):
                await queue.put({"column": idx, **ev})
            await queue.put({"column": idx, "type": "column_done"})

        tasks = [asyncio.create_task(run(i, m)) for i, m in enumerate(models)]
        done_cols = 0
        while done_cols < len(models):
            ev = await queue.get()
            if ev.get("type") == "column_done":
                done_cols += 1
                continue
            yield f"data: {json.dumps(ev)}\n\n"
        for t in tasks:
            t.cancel()

    return StreamingResponse(multiplexed(), media_type="text/event-stream")
```

### Conversation branching (edit message → new branch)
```python
# app/services/conversation/service.py
class ConversationService:
    async def edit_message(self, conversation_id: str, message_id: str, new_content: str, user_id: str) -> str:
        """Edit user message → creates new branch from that point. Returns new branch id."""
        async with db.transaction():
            original = await db.fetchrow("SELECT * FROM messages WHERE id=$1", message_id)
            assert original["user_id"] == user_id
            new_branch_id = uuid4()
            # Copy messages up to (but not including) original
            await db.execute("""
                INSERT INTO messages (conversation_id, branch_id, parent_branch_id, sequence, role, content, created_at)
                SELECT $1, $2, branch_id, sequence, role, content, created_at
                FROM messages
                WHERE conversation_id=$1 AND sequence < $3 AND branch_id = $4
            """, conversation_id, new_branch_id, original["sequence"], original["branch_id"])
            # Append edited message
            await db.execute("""
                INSERT INTO messages (conversation_id, branch_id, sequence, role, content)
                VALUES ($1, $2, $3, 'user', $4)
            """, conversation_id, new_branch_id, original["sequence"], new_content)
            # Update conversation's active branch
            await db.execute("UPDATE conversations SET active_branch_id=$1 WHERE id=$2",
                             new_branch_id, conversation_id)
        return str(new_branch_id)
```

## Database schema additions (migration)
```sql
-- Branches / alternate message threads per conversation (for edit)
ALTER TABLE conversations ADD COLUMN active_branch_id UUID;
ALTER TABLE messages ADD COLUMN branch_id UUID NOT NULL DEFAULT gen_random_uuid();
ALTER TABLE messages ADD COLUMN parent_branch_id UUID;
ALTER TABLE messages ADD COLUMN sequence INTEGER NOT NULL DEFAULT 0;
ALTER TABLE messages ADD COLUMN citations JSONB;
ALTER TABLE messages ADD COLUMN usage JSONB;
CREATE INDEX idx_messages_branch ON messages(conversation_id, branch_id, sequence);
CREATE INDEX idx_messages_fts ON messages USING GIN (to_tsvector('english', content));

-- Pinned / archived
ALTER TABLE conversations ADD COLUMN pinned_at TIMESTAMPTZ;
ALTER TABLE conversations ADD COLUMN archived_at TIMESTAMPTZ;
CREATE INDEX idx_conv_pinned ON conversations(user_id, pinned_at DESC NULLS LAST);
```

## Frontend components
- `components/chat/MessageList.tsx` — virtualized with react-window, keyboard nav, hover actions
- `components/chat/MessageBubble.tsx` — markdown via react-markdown + remark-gfm + rehype-katex + rehype-highlight
- `components/chat/Composer.tsx` — textarea with slash-command palette, attachment pills, voice button, model switcher
- `components/chat/ModelPicker.tsx` — grouped by provider, badges (vision, tools, speed, cost), live latency
- `components/chat/CompareLane.tsx` — 2-3 column parallel streaming
- `components/chat/BranchNav.tsx` — `< 2/3 >` arrows like ChatGPT
- `components/chat/SharedLink.tsx` — shareable conversation snapshot view

---

# ============================================================
# PHASE NC-2 — Agent Store (GPT Store equivalent)
# ============================================================

## Behavior
- **Store page** at `/agents` — grid of cards, categories (Writing, Productivity, Research, Programming, Education, Lifestyle, Other), sort (Popular / New / Trending), search
- **Agent detail page** — description, example prompts, tools enabled, creator, rating, usage count, "Start chat" button
- **Agent editor** (for creators) — GPT-Builder style: 2-pane (chat with builder | live preview)
  - Name, description, profile image (Flux-generated or upload)
  - Instructions (system prompt, up to 8000 chars)
  - Conversation starters (4 examples)
  - Knowledge (upload files → attached KB)
  - Capabilities (toggles): Web search, Code interpreter, Image gen, Memory access, Canvas, Voice
  - Actions (custom API tools via OpenAPI schema)
  - Visibility: Private / Unlisted (link-only) / Public (indexed in store)
- **Versioning** — save as draft, publish new version, users pinned to version they first used (opt-in to latest)
- **Ratings** — thumbs up/down + optional comment → average star shown

## Built-in agents (40, all ship Phase NC-2)
Research: WebResearcher, FactChecker, MarketAnalyst, AcademicScholar, NewsBriefer
Writing: CopyWriter, EmailDrafter, StoryWriter, ResumeBuilder, TechnicalWriter, Editor
Productivity: MeetingNotes, TaskPlanner, CalendarAssistant, TravelPlanner, BudgetAdvisor
Education: TutorAI, LanguageCoach, MathSolver, CodeExplainer, FlashcardMaker
Programming: CodeReviewer, BugHunter, TestWriter, DocWriter, SQLMaster, RegexWizard, APIDesigner, DevOpsAdvisor
Creative: ArtDirector, LogoDesigner, SongLyricist, PoemPoet
Analysis: DataAnalyst, ChartMaker, PDFSummarizer, CompetitorSpy
Lifestyle: Chef, FitnessCoach, TherapistLite (with safety filter), DreamInterpreter, GiftIdeator

Each agent defined as a YAML under `app/data/agents/` — loaded on startup into DB.

```yaml
# app/data/agents/webresearcher.yaml
slug: webresearcher
name: Web Researcher
category: research
description: Searches the web, synthesizes findings, cites every claim.
icon: https://cdn.nexusai.dev/agents/webresearcher.png
instructions: |
  You are WebResearcher. For every factual claim, search the web and cite
  the source inline as [1], [2], etc. At the end, list sources with URLs.
  Never fabricate a source. If uncertain, say so and search again with
  different keywords.
capabilities:
  web_search: true
  code_interpreter: false
  image_gen: false
  memory: true
  canvas: true
starters:
  - "Research the latest advances in solid-state batteries"
  - "What's the consensus on X among economists?"
  - "Compare the methodologies of [paper A] and [paper B]"
  - "Give me a brief on yesterday's market news"
default_model: claude-sonnet-4
visibility: public
```

## Custom actions (OpenAPI-driven tools)
```python
# app/services/agents/actions.py
class AgentAction:
    """Agent custom action = remote HTTP call as tool."""
    @classmethod
    def from_openapi(cls, spec: dict) -> list[ToolDefinition]:
        # Parse OpenAPI, produce one tool per operation
        ...
    async def execute(self, tool_call: dict, user: User) -> dict:
        # Add auth (API key / OAuth token stored per-user per-agent encrypted)
        # Call, parse, return
        ...
```

## Frontend
- `/agents` store page — SWR cached, virtualized grid
- `/agents/[slug]` detail page
- `/agents/new` and `/agents/[slug]/edit` — split-pane builder
- Sidebar: "Recent agents" list for fast recall

---

# ============================================================
# PHASE NC-3 — Deep Research (Perplexity Pro / ChatGPT Deep Research equivalent)
# ============================================================

## Pipeline
1. **Plan** — LLM produces a numbered research plan (3-10 sub-questions) as structured JSON
2. **Search fan-out** — For each sub-question: 2-3 parallel Serper queries + optional domain-specific APIs (arXiv, PubMed, GitHub, SEC EDGAR based on detected domain)
3. **Fetch + extract** — Top 8 URLs per query via web_fetch; readability extraction; drop ads / nav
4. **Dedup** — cluster near-duplicate sources (MinHash); keep canonical
5. **Summarize per source** — short summary + relevance score
6. **Synthesis** — LLM writes final report with inline `[n]` citations, sections, tables, optional chart suggestions (embed as Canvas chart)
7. **Verify** — second pass checks every citation appears in sources; flag uncited claims back to writer for one rewrite
8. **Follow-up suggestions** — 4 next questions

Runs in background (Celery task) with live progress streamed to frontend:
```
[▓░░░░░] Planning research
[▓▓░░░░] Searching: 3/12 queries done
[▓▓▓░░░] Fetched 18/36 sources
[▓▓▓▓░░] Summarizing 18 sources
[▓▓▓▓▓░] Writing report
[▓▓▓▓▓▓] Verifying citations
```

## SLA: 1-5 minutes end-to-end depending on depth setting (Quick / Standard / Deep)

```python
# app/services/research/deep.py
from celery import shared_task
from app.services.research.web import search, fetch
from app.services.llm.router import stream_chat
import asyncio, hashlib

@shared_task(bind=True)
def deep_research(self, query: str, user_id: str, conversation_id: str, depth: str = "standard"):
    async def _run():
        limits = {"quick": (3, 3), "standard": (6, 5), "deep": (10, 8)}
        n_subq, n_sources_per = limits[depth]

        await _progress(conversation_id, "planning", 5)
        plan = await _plan(query, n_subq)

        await _progress(conversation_id, "searching", 15)
        all_urls = await asyncio.gather(*[search(sq, k=n_sources_per) for sq in plan["sub_questions"]])

        await _progress(conversation_id, "fetching", 35)
        flat = [u for batch in all_urls for u in batch]
        unique = _dedup(flat)
        pages = await asyncio.gather(*[fetch(u.url) for u in unique], return_exceptions=True)
        pages = [p for p in pages if not isinstance(p, Exception)]

        await _progress(conversation_id, "summarizing", 60)
        summaries = await asyncio.gather(*[_summarize_source(query, p) for p in pages])

        await _progress(conversation_id, "writing", 80)
        report = await _synthesize(query, plan, summaries)

        await _progress(conversation_id, "verifying", 92)
        report = await _verify_citations(report, summaries)

        await _progress(conversation_id, "done", 100, report=report)
    asyncio.run(_run())

def _dedup(results):
    seen, out = set(), []
    for r in results:
        key = hashlib.md5(r.url.split("?")[0].encode()).hexdigest()
        if key not in seen:
            seen.add(key); out.append(r)
    return out
```

Frontend: `/research` page with progress bar, live sources list, final report renders into Canvas.

---

# ============================================================
# PHASE NC-4 — Canvas (ChatGPT Canvas / Claude Artifacts)
# ============================================================

## Behavior
- Right-side pane slides in when assistant produces long doc/code OR user clicks "Open in Canvas"
- **Text mode**: ProseMirror editor, markdown + headings + lists + tables + LaTeX
- **Code mode**: Monaco editor
- **AI actions** on selected text: Improve writing, Shorter, Longer, Fix grammar, Change tone, Translate, Custom instruction
- **Inline diff view**: AI proposes change → user sees red/green diff → accept/reject per hunk
- **Version history**: every AI edit + every 30s of user edits = snapshot; timeline on the right; restore any version
- **Comments**: highlight text → add comment (for collaboration)
- **Real-time collaboration** via Yjs WebSocket provider — multi-cursor, presence avatars
- **Export**: MD, PDF (via weasyprint on backend), DOCX (python-docx), HTML

## Backend
```python
# app/services/canvas/service.py
import y_py as Y
from fastapi import WebSocket
from app.core.redis import redis_client

class CanvasService:
    async def apply_ai_edit(self, canvas_id: str, selection: dict, instruction: str, user_id: str):
        # Load current doc from GCS, produce edited version, compute diff,
        # return proposed_patch for user to accept/reject
        ...

    async def collab_websocket(self, ws: WebSocket, canvas_id: str, user_id: str):
        """Yjs sync protocol over WS. Snapshot to GCS every 30s."""
        ...
```

## Frontend
- `components/canvas/CanvasPane.tsx` — split with chat
- `components/canvas/VersionTimeline.tsx`
- `components/canvas/DiffOverlay.tsx` — per-hunk accept/reject
- Uses `@codemirror/merge` for diff

---

# ============================================================
# PHASE NC-5 — Memory
# ============================================================

## Behavior (matches ChatGPT Memory)
- Auto-extracts memorable facts from every exchange (bg task after assistant replies)
- Shown in Settings → Personalization → Memory (full list, edit, delete)
- Per-conversation toggle: "Reference memory" on/off
- When a memory was used in a response, chip "Memory used" appears on that message; click → shows which memories
- Max 200 memories per user; when full, least-recently-used pruned (user alerted)

## Backend
```python
# app/services/memory/service.py
class MemoryService:
    EXTRACT_PROMPT = """Extract persistent, reusable facts about the user from the exchange below.
Return JSON list. Only include things that would help in future conversations
(preferences, ongoing projects, relationships, goals, constraints). Skip
anything sensitive (health details, financial specifics, political views)
unless the user explicitly asked to remember them.

Format: [{"fact": "...", "confidence": 0..1}]"""

    async def extract_from_exchange(self, user_id: str, user_msg: str, assistant_msg: str):
        facts = await self._llm_extract(user_msg, assistant_msg)
        for f in facts:
            if f["confidence"] >= 0.7:
                await self._store(user_id, f["fact"])

    async def retrieve_for(self, user_id: str, query: str, k: int = 5) -> list[str]:
        # embed query, cosine top-k against user's memory table (pgvector)
        ...
```

## Schema
```sql
CREATE TABLE user_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fact TEXT NOT NULL,
    embedding vector(1536),
    source_message_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ DEFAULT NOW(),
    use_count INT DEFAULT 0
);
CREATE INDEX idx_mem_user ON user_memories(user_id, last_used_at DESC);
CREATE INDEX idx_mem_vec ON user_memories USING ivfflat (embedding vector_cosine_ops);
```

## UI
- Settings → Personalization → Memory: searchable list, inline edit, delete-all button
- Message meta: "Memory used" popover

---

# ============================================================
# PHASE NC-6 — Projects (scoped workspace)
# ============================================================

## Behavior (matches ChatGPT / Claude Projects)
- Project = folder containing: custom system prompt, attached files (shared KB), attached agents, conversations, canvas docs
- Every chat inside project inherits: project system prompt + project KB + project memory namespace
- Share project with collaborators (role: Owner / Editor / Viewer)
- Move conversations in/out of project

## Schema
```sql
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    system_prompt TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE project_members (
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role TEXT CHECK (role IN ('owner','editor','viewer')),
    PRIMARY KEY (project_id, user_id)
);
ALTER TABLE conversations ADD COLUMN project_id UUID REFERENCES projects(id) ON DELETE SET NULL;
```

## UI
- `/projects` — list of projects
- `/projects/[id]` — project home: chats, files, agents, settings, members

---

# ============================================================
# PHASE NC-7 — Knowledge Base (RAG)
# ============================================================

## Behavior
- Upload files to a KB (PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, CSV, EPUB)
- Ingestion pipeline (Celery): parse → chunk (semantic, 512 tokens, 64 overlap) → embed (text-embedding-3-large) → upsert to Qdrant
- Attach KB to: project, agent, or single conversation
- **Hybrid retrieval**: BM25 (Postgres FTS) + vector (Qdrant) → RRF merge → Cohere rerank → top-k
- **Cite per claim** in answer; hovering citation shows source excerpt; clicking opens source at the chunk

## Pipeline
```python
# app/services/rag/ingest.py
@shared_task
def ingest_file(file_id: str):
    f = FileRepo.get(file_id)
    text = _extract(f)                    # PDF/DOCX/... → text
    chunks = _semantic_chunk(text, 512, 64)
    embeddings = asyncio.run(embed([c.text for c in chunks]))
    qdrant.upsert(collection=f.kb_id, points=[
        {"id": c.id, "vector": e, "payload": {**c.meta, "kb_id": f.kb_id, "file_id": file_id}}
        for c, e in zip(chunks, embeddings)
    ])
    # Also index in Postgres FTS
    ...
```

## Retrieval
```python
# app/services/rag/service.py
class RAGService:
    async def hybrid_search(self, kb_ids: list[str], query: str, k: int = 8) -> list[Chunk]:
        q_emb = (await embed([query]))[0]
        vec_hits = qdrant.search_batch(kb_ids, vector=q_emb, limit=30)
        bm25_hits = await db.fetch("""
            SELECT chunk_id, ts_rank(fts, plainto_tsquery('english', $1)) as score
            FROM chunks_fts WHERE kb_id = ANY($2) ORDER BY score DESC LIMIT 30
        """, query, kb_ids)
        merged = rrf_merge(vec_hits, bm25_hits)
        reranked = await cohere_rerank(query, merged, top_n=k)
        return reranked
```

---

# ============================================================
# PHASE NC-8 — Image Studio
# ============================================================

## Behavior
- `/image` route: prompt box + options (model, aspect, # images, seed, reference image for img2img)
- Models: Flux Pro (Replicate), Flux Schnell (Replicate), SDXL (Replicate), DALL-E 3 (OpenAI), Imagen 3 (Google), Stable Diffusion 3.5 (Replicate)
- Batch of 4 generates in parallel; streaming preview (low-res → high-res)
- History gallery (all generations, infinite scroll, virtualized)
- **Edit tools**: in-paint (mask + prompt), out-paint (extend), upscale (Real-ESRGAN), remove background (Rembg), style transfer (ControlNet)
- Right-click on chat attachment image → "Edit in Image Studio" deep-links with ref preloaded
- In chat: "/image {prompt}" slash command → inline generation in conversation

## Backend
```python
# app/services/image/service.py
from replicate import Client as Replicate
from openai import AsyncOpenAI

class ImageService:
    async def generate(self, model: str, prompt: str, *, aspect: str = "1:1",
                       n: int = 1, seed: int | None = None, ref_image: str | None = None):
        if model.startswith("flux"):
            return await self._replicate("black-forest-labs/" + model, ...)
        if model == "dall-e-3":
            return await self._openai(prompt, aspect, n)
        ...
    async def inpaint(self, image_url: str, mask_url: str, prompt: str): ...
    async def upscale(self, image_url: str, factor: int = 4): ...
    async def remove_bg(self, image_url: str): ...
```

---

# ============================================================
# PHASE NC-9 — Voice Mode (ChatGPT Voice equivalent)
# ============================================================

## Behavior
- Floating mic button on every page; fullscreen orb mode in voice-only route
- **Push-to-talk**: hold space (desktop) or tap-and-hold (mobile)
- **Continuous**: tap once, VAD (Silero VAD in browser via WASM) detects turn-taking
- **Barge-in**: user speaks → TTS stops within 200 ms
- **Streaming STT**: Whisper (OpenAI API) via chunked audio upload every 300 ms; live partial transcripts
- **Streaming TTS**: ElevenLabs turbo v2.5 via streaming endpoint; plays as it arrives
- **Voices**: 9 ElevenLabs voices (Rachel, Domi, Bella, Antoni, Elli, Josh, Arnold, Adam, Sam)
- Transcript visible and scrollable; after session, saved as normal conversation

## Pipeline diagram
```
[mic] → VAD (client) → chunked PCM16 → WS → Whisper stream → text
text → LLM stream → token → ElevenLabs stream → audio → [speaker]
                                        ↑
                                  interrupted if mic active
```

## Implementation
```python
# app/websocket/voice.py
from fastapi import WebSocket
from openai import AsyncOpenAI
import elevenlabs

async def voice_websocket(ws: WebSocket, user_id: str):
    await ws.accept()
    while True:
        frame = await ws.receive_bytes()
        # feed to Whisper realtime (transcribe_streaming)
        # when turn detected → call LLM → stream tokens
        # concurrently pipe tokens to ElevenLabs streaming TTS → ws.send_bytes(audio)
        # if ws receives {"type":"user_speaking"} → cancel TTS stream
        ...
```

---

# ============================================================
# PHASE NC-10 — Video Generation
# ============================================================
- Models: Runway Gen-3, Luma Dream Machine, Veo 3 (Google), Pika, Kling (via fal.ai)
- Prompt → video, image → video, video → video style transfer
- Duration: 5s / 10s / 20s; resolution up to 1080p
- Background Celery task with progress; push notification when done
- Gallery like Image Studio

---

# ============================================================
# PHASE NC-11 — Workflows (Zapier for AI)
# ============================================================

## Behavior
- `/workflows` — list, create, edit
- Editor: ReactFlow canvas with nodes
- **Trigger nodes**: Manual, Schedule (cron), Webhook, New email (Gmail connector), New calendar event, New file in Drive, Form submission
- **Action nodes**: Any agent call, any model call, web search, image gen, HTTP request, Gmail send, Drive upload, Slack message, condition (if/else), loop, merge, set variable
- **Test run** with mock input, shows each node's input/output
- Save as template, publish to template gallery
- Run history, rerun from any node

## Runtime
```python
# app/services/workflows/executor.py
# Lightweight DAG executor on Celery. Each node = a task.
# State passed via Redis (namespaced by run_id).
```

---

# ============================================================
# PHASE NC-12 — Computer Use (browser agent)
# ============================================================
- Playwright running in the E2B sandbox (already available from NexusCode Cloud)
- Claude Computer Use model (or GPT-4o with tool use fallback)
- Screenshot → model → proposed action (click x,y; type; navigate) → user approval (optional) → execute → loop
- Safety: allowlist of domains; never enters credit card / password fields (DOM-level block); pause-for-approval on form submission / purchase actions
- Use cases: book flights, fill forms, scrape tables, research-with-interaction

---

# ============================================================
# PHASE NC-13 — Sharing & Public Links
# ============================================================
- Any chat → Share → generates immutable snapshot stored as `shared_conversations` row
- URL: `https://nexusai.dev/s/{short_id}`
- Options: public / unlisted / password-protected / expires in N days
- Continue-from-share: anyone can click "Continue this chat" → forks into their account
- OG image auto-generated (serverless puppeteer → PNG of first exchange)

---

# ============================================================
# PHASE NC-14 — Search (across all user data)
# ============================================================
- ⌘K palette or `/search`
- Sources: conversations, messages, canvas docs, files, memories, projects, agents
- Filters: date range, model used, agent used, project, has attachments
- Results grouped by source type; click → jump to exact location
- Backend: Postgres FTS (GIN) + tsvector on messages.content, canvas.content, files.extracted_text

---

# ============================================================
# PHASE NC-15 — Export & Data Controls
# ============================================================
- Settings → Data Controls → Export my data
- Celery job: ZIP containing chats.json, memories.json, files/, canvas/, knowledge_bases/
- Email download link when ready (expires 7 days)
- Chat history toggle: off = new chats not saved, not used for training (we never train, but make it explicit)
- Delete account: 30-day grace period, then hard delete (CASCADE)

---

# ============================================================
# PHASE NC-16 — Settings (everything ChatGPT has)
# ============================================================

## Sections
1. **Account** — email, name, profile photo, delete account
2. **Subscription** — plan, billing portal (Stripe), invoices, usage meter
3. **Personalization** — custom instructions (How should NexusAI call you? What should it know?), memory list, default agent
4. **Data controls** — history toggle, chat export, training opt-out (already default off), shared links list, archived chats
5. **Appearance** — theme (system/light/dark), accent color, density, font size, code theme
6. **Notifications** — desktop push, email summaries, long-task-done alerts
7. **Connected apps** — Google Drive, Gmail, Calendar, Slack, Notion, GitHub, Zapier (OAuth flows)
8. **API keys (BYOK)** — user can paste their own Anthropic/OpenAI/Google keys to bypass our billing and use their rate limits (stored encrypted; KMS envelope encryption)
9. **Security** — change password, 2FA (TOTP), passkeys (WebAuthn), active sessions list with revoke, login history
10. **Billing** — Stripe portal link, credits balance (for metered API), auto-top-up

---

# ============================================================
# PHASE NC-17 — Auth (all providers + passkeys + guest)
# ============================================================
- NextAuth config with: Credentials (email/password + bcrypt), Google, GitHub, Apple, Microsoft, Magic link (SendGrid)
- WebAuthn passkeys via @simplewebauthn
- 2FA: TOTP (any authenticator app) + backup codes
- Guest mode: cookie-based temp user_id, 10 msg limit, banner "Sign up to save this chat"
- Session management: JTI-based JWT with revocation list in Redis
- Rate limiting: 20 req/min unauthenticated, 120 req/min Free, 600 req/min Plus

---

# ============================================================
# PHASE NC-18 — Billing (Stripe)
# ============================================================

## Plans
| Plan | Price | Quotas |
|---|---|---|
| Free | $0 | 40 msg/day on small models, 5 msg/day on frontier, 3 image/day, 1 GB KB, no Deep Research |
| Plus | $20/mo | 200 msg/day frontier, 50 image/day, 10 GB KB, Deep Research, Voice, Canvas, priority during peak |
| Team | $30/user/mo | Plus + team workspace, admin, SSO basic, 100 GB pooled KB |
| Enterprise | contact | SAML SSO, SCIM, audit logs, custom retention, on-prem option |

- Stripe Checkout for self-serve, Stripe Billing Portal for management
- Metered usage for API BYOK pass-through (1% surcharge or free if BYOK)
- Webhooks update `subscriptions` table; feature gates read from there

---

# ============================================================
# PHASE NC-19 — Admin console (enterprise)
# ============================================================
- `/admin` route for admins only
- Org overview: seats, usage, top users, top agents, cost
- User management: invite, role, deactivate, reset 2FA
- Audit log: every admin action + sign-in + data export
- Content filters: configurable blocklists, PII detection on uploads
- Retention policy: N days then auto-delete
- SSO config: SAML + OIDC + SCIM provisioning

---

# ============================================================
# PHASE NC-20 — Mobile PWA (iOS/Android/iPad/Chromebook native feel)
# ============================================================
- Installable on all platforms via `manifest.json` (already in NexusCode-Cloud spec)
- Home screen icon launches standalone (no browser chrome)
- Push notifications: PWA Push API + FCM for Android, APNs-compatible wrapper for iOS (now supported in iOS 16.4+ PWA)
- Voice button wired to home screen shortcut (quick action)
- Swipe left on conversation = archive; swipe right = pin (Gmail-style)
- Bottom sheet instead of modal on mobile
- Responsive breakpoints:
  - < 480px: single column, bottom tab bar
  - 480-768px: collapsible sidebar
  - 768-1024px: tablet layout (sidebar always visible, chat + canvas in tabs)
  - > 1024px: desktop with resizable panes
- Safe area insets for notch / gesture bar (env(safe-area-inset-*))
- iPad keyboard shortcuts when external keyboard attached
- Chromebook: Android app behavior via PWA; touch + keyboard hybrid

---

# ============================================================
# PHASE NC-21 — Cross-OS native wrappers (optional, for app store presence)
# ============================================================
- **Desktop** (Electron or Tauri): wraps PWA, adds global hotkey (⌘Space / Ctrl+Space) to open quick prompt HUD, system tray, auto-update (Tauri preferred for size)
- **iOS / iPadOS** (Capacitor): wraps PWA for App Store distribution, adds Siri Shortcuts, Shortcuts.app action, share extension (share anything to NexusChat)
- **Android** (Capacitor): Google Play, share target, Assistant integration, Quick Settings tile
- **Chromebook**: already installable PWA + Android app via Capacitor APK

All native wrappers share 100% of UI code via web. Platform-specific code only for: global hotkey, system tray, share targets, OS voice integration.

---

# ============================================================
# PHASE NC-22 — Observability & SLAs (enforced)
# ============================================================

## Instrumentation
- OpenTelemetry auto-instrument FastAPI, httpx, asyncpg, Redis, Celery
- Every LLM call span: provider, model, input_tokens, output_tokens, ttft_ms, total_ms, cost_usd
- Every user action event → PostHog
- Error capture → Sentry (PII scrubbed)

## Real User Monitoring (frontend)
- Web Vitals (LCP, CLS, INP, TTFB) reported to PostHog
- Custom events: chat_first_token_ms, model_switch, artifact_opened, research_completed_ms
- Alerting: PagerDuty hooked to SLO burn alerts

## SLOs (rolling 28-day)
| SLO | Target |
|---|---|
| Chat availability | 99.9% |
| Chat first-token < 400 ms | 95% |
| Sandbox cold start < 3 s | 95% |
| Research completion < 5 min | 90% |
| Voice round-trip < 1 s | 95% |
| Image gen < 10 s | 90% |
| Search latency < 200 ms | 99% |

SLO burn > 1× triggers PagerDuty; > 2× pages on-call immediately.

---

# ============================================================
# PHASE NC-23 — Testing & Quality
# ============================================================
- Backend: pytest + pytest-asyncio; min 80% coverage gating CI; contract tests for provider router (mock each provider's SSE response format)
- Frontend: Vitest + React Testing Library for units; Playwright for E2E on every critical flow (sign up, send message, switch model, upload file, voice, image, compare, research, canvas, share, settings)
- Load test: k6 scripts for chat endpoint (target: 5000 RPS sustained, P99 < 1.5 s non-streaming / < 400 ms TTFT streaming)
- Accessibility: axe-core in Playwright; block merge on any serious violation
- Visual regression: Chromatic or Percy for key pages across 6 viewports × 2 themes

---

# ============================================================
# PHASE NC-24 — Security & Compliance
# ============================================================
- OWASP Top 10 pass (ZAP scan in CI)
- CSP headers: strict, nonce-based for inline scripts
- HSTS + preload
- Encryption at rest: Cloud SQL CMEK, GCS CMEK, Secret Manager for all keys
- Encryption in transit: TLS 1.3 only
- PII detection on uploads (Presidio); flag before embedding into KB
- Rate limiting + WAF (Cloud Armor) + bot protection (reCAPTCHA Enterprise on sign-up and guest-mode)
- SOC 2 Type I readiness (audit logs, access reviews, change management docs)
- GDPR: right to export (PHASE NC-15), right to erasure (same), DPA template, cookie consent banner for EU
- SSRF protection on all fetch tools (IP range blocklist 10/8, 172.16/12, 192.168/16, 169.254, localhost)

---

# EXECUTION ORDER

Run sequentially. Do not skip. Each phase ends with a green E2E test.

```
NC-1   Chat core + branching + streaming + multi-provider
NC-2   Agent Store + 40 built-ins + custom builder + actions
NC-3   Deep Research with citations
NC-4   Canvas with versioning + diff + collab
NC-5   Memory
NC-6   Projects
NC-7   Knowledge Base (RAG)
NC-8   Image Studio
NC-9   Voice
NC-10  Video
NC-11  Workflows
NC-12  Computer Use
NC-13  Sharing
NC-14  Search
NC-15  Export / data controls
NC-16  Settings (all sections)
NC-17  Auth (all providers + passkeys + guest)
NC-18  Billing (Stripe)
NC-19  Admin console
NC-20  Mobile PWA
NC-21  Native wrappers (desktop / iOS / Android)
NC-22  Observability + SLOs
NC-23  Testing (unit/E2E/load/a11y)
NC-24  Security + compliance
```

After each phase:
1. Run its E2E test. Must pass.
2. Run Lighthouse on affected pages. Must score ≥ 90.
3. Run axe-core. Must have zero serious violations.
4. Measure SLA for the phase's primary metric. Must hit target.
5. Only then move to next phase.

---

# RULES

1. **No mocks.** Real providers, real sandboxes, real DB.
2. **No placeholders.** Every TODO is a bug.
3. **No partial files.** Every file complete, every config complete.
4. **SLAs are gates.** If a feature misses SLA, it doesn't ship until fixed.
5. **Every feature works on every OS and device.** Phase NC-20 is not optional.
6. **Accessibility is a gate.** WCAG AA or it doesn't ship.
7. **Tests must pass.** CI red = no merge. No exceptions.
8. **No commit without telemetry.** If you can't measure it, you can't ship it.
9. **Never fabricate a citation, a source, or a benchmark.**
10. **Don't stop until all 24 phases pass their gates.**

---

# EXECUTE NOW

After NEXUSCODE-CLOUD finishes (or in parallel if dependencies allow), begin Phase NC-1.

When all 24 phases pass, run the final end-to-end user journey test:

1. New user signs up with Google → lands on home
2. Starts a chat with default model → first token < 400ms
3. Switches model mid-conversation → seamless
4. Uploads a 20-page PDF → RAG ingests in background → asks question → cited answer
5. Enables web search → asks "what's new this week in AI" → cited live results
6. Clicks "Open in Canvas" on a long reply → edits inline → AI suggests changes with diff → accepts
7. Starts Deep Research "Compare approaches X, Y, Z" → full report in under 5 min with ≥ 15 citations
8. Opens Image Studio → generates 4 images with Flux → in-paints a region
9. Voice mode → asks question → interrupts mid-reply → assistant stops within 200 ms
10. Creates a custom agent "FitnessCoach v2" with its own KB → publishes unlisted → shares link with friend
11. Creates a workflow: "Every Monday 9am → search AI news → summarize → send to my Slack" → test run passes
12. Opens on iPhone via installed PWA → home screen shortcut → voice works → push notification when weekly workflow runs
13. Compare mode → one question → 3 models stream side by side → votes
14. Shares the best reply → opens in private window → renders correctly
15. Goes to Settings → exports all data → receives email with ZIP in < 10 min

If all 15 steps pass, print:

```
╔════════════════════════════════════════════════════════════╗
║  NEXUSAI ULTIMATE — FULL CHATGPT PARITY ACHIEVED           ║
║  Chat · Agents · Research · Canvas · Memory · Projects     ║
║  RAG · Image · Voice · Video · Workflows · Computer Use    ║
║  Every OS · Every device · Every SLA · Every phase done    ║
╚════════════════════════════════════════════════════════════╝
```
