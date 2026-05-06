# NexusAI — Project Memory

This document captures the architecture, all features built, every endpoint, every page, and the full build process. It lives in `raw/` so Graphify ingests it into the knowledge graph and any future Claude Code session has access to this context.

---

## What NexusAI is

An all-in-one AI platform that combines:
- **NexusChat** — multi-model chat (Claude / GPT / Gemini / Llama / Mistral / Cerebras / SambaNova)
- **NexusCode** — in-browser code playground
- **Multi-Agent Teams** — planner / 3 executors / critic flow
- **Image Studio** — text-to-image via free Pollinations.ai
- **Voice Mode** — full server-side STT (Whisper) + TTS (PlayAI)
- **Deep Research** — multi-step research with structured reports
- **Agents** — marketplace + custom agents
- **Codex** — background coding agent with diff output
- **Canvas** — long-form docs with AI rewrite
- **Knowledge Bases** — file upload + RAG with citations
- **Workflows** — chained LLM pipelines
- **Projects** — group conversations / files / instructions
- **Memory** — auto-persistent facts about the user

---

## Architecture

| Layer | Stack | Location |
|---|---|---|
| Frontend | Next.js 14 App Router · TypeScript · Tailwind · NextAuth | `nexusai/frontend/` |
| Backend | FastAPI · LiteLLM · PostgreSQL · Redis · OpenTelemetry · Sentry | `nexusai/backend/app/` |
| Database | PostgreSQL 15 + pgvector + Qdrant for RAG | local 5432 |
| Cache / Queue | Redis | local 6380 |
| Auth | JWT (HS256) + bcrypt + magic-link + OAuth (Google/GitHub) + guest | `nexusai/backend/app/api/v1/auth.py` |
| Streaming | Server-Sent Events (SSE) for chat, multi-agent, codex, research | every `*_router` |
| Storage | GCS in prod / local FS in dev | configurable |

**Frontend port: 3001 · Backend port: 8001 · Both already running**

---

## Backend endpoints (all under `/api/v1`)

| Router | File | Endpoints |
|---|---|---|
| Health | `health.py` | `GET /health`, `GET /health/ready` |
| Auth | `auth.py` | `register`, `login`, `guest`, `magic-link`, `magic-link/verify`, `verify-email`, `oauth/callback`, `me`, `logout` |
| Chat | `chat.py` | `models`, `stream`, `compare`, `{message_id}/feedback` |
| Conversations | `conversations.py` | full CRUD + `list/{session_id}`, `{id}/messages`, edit/branch |
| Memory | `memory.py` | full CRUD + `stats`, clear |
| Settings | `settings.py` | `byok` (GET/POST/DELETE), `password`, `export/request` |
| Billing | `billing.py` | `subscription`, `portal` |
| Projects | `projects.py` | full CRUD |
| Agents | `agents.py` | full CRUD + `mine`, marketplace search |
| Knowledge Base | `knowledge_base.py` | KB CRUD, file upload, search with citations |
| Image | `image.py` | generate, history, models, upscale, remove-bg |
| Voice | `voice.py` | `transcribe` (Whisper), `tts` (OpenAI/PlayAI) |
| Sandbox | `sandbox.py` | E2B sandbox start/stop/exec/files |
| Codex | `codex.py` | task CRUD + `{id}/stream` (SSE) |
| Multi-agent | `multi_agent.py` | `run` (SSE: planner → 3 executors → critic) |
| Research | `research.py` | reports CRUD + `{id}/stream` |
| Canvas | `canvas.py` | docs CRUD + `{id}/assist` (AI rewrite) |
| Workflows | `workflows.py` | workflow CRUD + run |
| Sharing | `sharing.py` | public share links |
| Search | `search.py` | full-text search |
| Export | `export.py` | data export ZIP |
| Computer Use | `computer_use.py` | Playwright in sandbox |
| Admin | `admin.py` | org / user / audit logs |
| Usage | `usage.py` | per-user usage tracking |
| **Waitlist** | `waitlist.py` (new) | `POST /waitlist` for ComingSoon pages |

---

## Frontend pages (route → file)

### Public
- `/` → `app/page.tsx` (home/landing)
- `/login` → `app/login/page.tsx`
- `/signup` → `app/signup/page.tsx`
- `/auth/magic` → `app/auth/magic/page.tsx`
- `/auth/verify-email` → `app/auth/verify-email/page.tsx`
- `/shared/[token]` → public share viewer

### Protected (redirect to login if signed-out)
| URL | File | Status |
|---|---|---|
| `/chat` | `app/chat/page.tsx` | Working — model picker, agents, BYOK, streaming, suggestions, mic |
| `/chat/[conversationId]` | `app/chat/[conversationId]/page.tsx` | Working — load msgs, memory toggle, edit/regen/copy/feedback |
| `/chat/compare` | `app/chat/compare/page.tsx` | Working — 2-3 models side-by-side, mic in composer |
| `/settings` | `app/settings/page.tsx` | Working — 8 tabs (Account, Appearance, Personalization, Data, Security, BYOK, Billing, Notifications), URL-synced, mobile responsive |
| `/settings/2fa` | `app/settings/2fa/page.tsx` | Working — coming-soon TOTP/WebAuthn UI |
| `/memory` | `app/memory/page.tsx` | Working |
| `/projects` | `app/projects/page.tsx` | Working — CRUD, modal create with icon picker |
| `/projects/[id]` | `app/projects/[id]/page.tsx` | Working — edit name/desc/system prompt, delete |
| `/agents` | `app/agents/page.tsx` | Working — marketplace with search + category filter |
| `/agents/mine` | `app/agents/mine/page.tsx` | Working |
| `/agents/create` | `app/agents/create/page.tsx` | Working — uses `AgentForm` |
| `/agents/[agentId]` | `app/agents/[agentId]/page.tsx` | Working — detail + starters |
| `/agents/[agentId]/edit` | `app/agents/[agentId]/edit/page.tsx` | Working |
| `/multi-agent` | `app/multi-agent/page.tsx` | Working — planner/executor/critic with SSE timeline |
| `/codex` | `app/codex/page.tsx` | Working — list + create with code/repo/language |
| `/codex/[taskId]` | `app/codex/[taskId]/page.tsx` | Working — SSE stream, copy/retry |
| `/canvas` | `app/canvas/page.tsx` | Working — list of docs |
| `/canvas/[id]` | `app/canvas/[id]/page.tsx` | Working — editor with selection toolbar (Rewrite/Shorten/Lengthen/Formal/Casual/Summarize), AI rewrite modal with diff view |
| `/kb` | `app/kb/page.tsx` | Working — list + create modal |
| `/kb/[id]` | `app/kb/[id]/page.tsx` | Working — drag-drop upload, document list, RAG search with citations |
| `/research` | `app/research/page.tsx` | Working — list + create |
| `/research/[reportId]` | `app/research/[reportId]/page.tsx` | Working — live SSE plan/subq_done/report events with timeline |
| `/workflows` | `app/workflows/page.tsx` | Working — chain LLM steps with `{{input}}` placeholders |
| `/image` | `app/image/page.tsx` | Working — Pollinations Flux (5 models, 4 ratios, 4 parallel variations, history, lightbox, download) |
| `/voice` | `app/voice/page.tsx` | Working — full voice mode (orb, MediaRecorder → Groq Whisper → chat stream → PlayAI TTS, barge-in) |
| `/code` | `app/code/page.tsx` | Working — project list with templates (HTML/React/Python/blank) |
| `/code/[projectId]` | `app/code/[projectId]/page.tsx` | Working — Monaco-style editor + file tree + iframe live preview |
| `/shared` | `app/shared/page.tsx` | Working — manage shared links |

---

## Critical components

| Component | Purpose | Used by |
|---|---|---|
| `components/coming-soon.tsx` | Production-grade "Coming Soon" page with notify-me waitlist | (no longer used — every page is real now) |
| `components/agents/AgentForm.tsx` | Shared create/edit form for agents | `/agents/create`, `/agents/[id]/edit` |
| `components/chat/Composer.tsx` | Message composer (text, mic, model picker, codex toggle) | `/chat`, `/chat/[id]` |
| `components/chat/CompareView.tsx` | Side-by-side model comparison | `/chat/compare` |
| `components/chat/ModelPicker.tsx` | Model dropdown (calls `/api/v1/chat/models`) | All chat surfaces |
| `components/chat/MessageBubble.tsx` | Message with copy/edit/regen/feedback (always-visible actions) | `MessageList` |
| `components/voice/VoiceButton.tsx` | Mic button with state machine (idle/recording/transcribing/speaking) | All chat composers |
| `hooks/use-voice.ts` | MediaRecorder → backend Whisper, no Google Speech dependency (works in Brave/Firefox) | `VoiceButton`, `/voice` |
| `hooks/use-chat.ts` | SSE chat streaming with optimistic UI | All chat surfaces |
| `lib/api.ts` | Auth-aware fetch wrapper + SSE helper | Everything |
| `middleware.ts` | Auth gate (public allow-list + token check) | All routes |

---

## Critical fixes shipped

1. **Mic in Brave / privacy browsers** — replaced Web Speech API with MediaRecorder + backend Whisper transcribe (no Google dependency).
2. **Model picker dropdown broken** — fixed endpoint mismatch (was `/api/v1/models`, now `/api/v1/chat/models`).
3. **Chat default model wrong** — removed hardcoded fake IDs, ModelPicker auto-selects first available real model from API.
4. **`<ComingSoon>` waitlist** — added `POST /api/v1/waitlist` to backend so notify-me forms work.
5. **Auth flows** — DB-backed register / login (bcrypt) / guest / magic-link / verify-email all wired end-to-end.
6. **Voice page rewritten** — production end-to-end: orb → MediaRecorder → Whisper → chat stream → PlayAI TTS → barge-in.
7. **Settings page** — 8 tabs, URL-synced, mobile-responsive sidebar collapses to top tab strip, no hanging spinners.
8. **40+ ComingSoon pages replaced with real working features** — agents marketplace, codex, canvas, KB, research, projects, multi-agent, workflows, NexusCode, voice, image studio.

---

## API contract reference

All endpoints require `Authorization: Bearer <jwt>` except:
- `POST /api/v1/auth/{register,login,guest,magic-link,magic-link/verify,verify-email,oauth/callback}` (anonymous)
- `GET /api/v1/health{,/ready}` (anonymous)
- `POST /api/v1/waitlist` (anonymous)
- `GET /api/v1/share/{token}` (public share)

Tokens are obtained from `/auth/{register,login,guest}` as `{access_token, user}` and stored client-side via NextAuth's CredentialsProvider.

---

## Operational

| What | Where |
|---|---|
| Start backend | `cd nexusai/backend && py -3 -m uvicorn app.main:app --port 8001 --reload` |
| Start frontend | `cd nexusai/frontend && npm run dev` |
| Smoke test | `cd nexusai/frontend && node scripts/smoke.mjs` (46 checks) |
| Refresh knowledge graph | `graphify update .` |
| Query knowledge graph | `graphify query "How does X work?"` |
| Visual graph | open `graphify-out/graph.html` |

### BYOK / API keys (configure in `/settings?tab=byok`)
| Provider | Where to get key | Free? |
|---|---|---|
| Groq (recommended) | https://console.groq.com/keys | ✅ Yes |
| Google Gemini | https://aistudio.google.com/app/apikey | ✅ Yes |
| Cerebras | https://cloud.cerebras.ai/platform/?modal=apikey | ✅ Yes |
| Mistral | https://console.mistral.ai/api-keys | ✅ Yes |
| SambaNova | https://cloud.sambanova.ai/apis | ✅ Yes |
| HuggingFace | https://huggingface.co/settings/tokens | ✅ Yes |
| Anthropic | https://console.anthropic.com/settings/keys | 💳 Paid |
| OpenAI | https://platform.openai.com/api-keys | 💳 Paid |

A single Groq key unlocks: chat (Llama 3.3 70B), mic transcription (Whisper), TTS (PlayAI), multi-agent, research, codex, canvas, workflows.

---

## What's intentionally NOT yet built

- Real Linux sandbox in NexusCode (UI works with HTML/CSS/JS preview; Linux/E2B integration deferred)
- Tavily-backed web search in Research (uses Llama-only synthesis for free tier)
- Stripe billing portal (UI shows "Pro plan coming soon")
- Tiptap rich-text Canvas (current uses textarea + AI assist; Tiptap deferred)
- ReactFlow visual workflow builder (current is linear step list — covers 80% of use cases)
- Yjs real-time collaboration on Canvas
- WebAuthn passkeys (`/settings/2fa` shows roadmap)

These are clearly explained in-product so users know what's coming next.

---

## Build verification

Last smoke test (run `node nexusai/frontend/scripts/smoke.mjs`):
- ✅ 46 / 46 checks pass
- All 32 frontend route families respond correctly
- All token-gated APIs work
- Auth flow (register → login → /auth/me) works end-to-end

Knowledge graph (last `graphify update .`):
- 2,303 nodes · 3,554 edges · 264 communities
- 346 source files indexed
- Both backends (legacy Pyxis + NexusAI) captured
