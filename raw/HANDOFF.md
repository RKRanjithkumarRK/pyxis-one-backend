# NexusAI — HANDOFF for the next Claude session / account

This file is the single source of truth for picking up where the last session left off. It is indexed by Graphify and lives in `raw/` so any AI assistant in this repo can find it.

---

## State as of last session

- **Frontend:** `http://localhost:3001` (Next.js 14 App Router) at `nexusai/frontend/`
- **Backend:** `http://localhost:8001` (FastAPI) at `nexusai/backend/app/`
- **Smoke test:** `node nexusai/frontend/scripts/smoke.mjs` → **46/46 pass**
- **Graphify graph:** 2,317 nodes · 3,593 edges · 257 communities · 348 files

## What works end-to-end (verified)

- Auth: register / login / guest / magic-link / verify-email / OAuth (Google, GitHub) / NextAuth integration
- Chat: streaming via SSE, model picker (`/api/v1/chat/models`), message edit/regen/copy/feedback, BYOK banner, multi-conversation
- All 32 frontend route families respond correctly (32/32)
- All token-gated APIs return 200 with a valid Bearer token
- Settings (8 tabs, URL-synced, mobile responsive)
- Voice: MediaRecorder → Groq Whisper STT → chat stream → Groq PlayAI TTS (Brave-aware error UI)
- Image Studio (free Pollinations Flux, 4 variations, lightbox, history)
- Multi-Agent (planner / 3 executors / critic, SSE timeline)
- Projects, Agents (built-in + custom), Codex, Canvas, KB (RAG), Research, Workflows, NexusCode

## Critical fixes in last turn (enterprise audit)

| # | Bug | File | Status |
|---|---|---|---|
| 1 | Chat messages lost on page refresh — no `persist` middleware | `lib/store/chat.ts` | ✅ Fixed: zustand `persist` + `onRehydrateStorage` resets transient stream state |
| 2 | `streamSSE` ignored `AbortSignal` after stream started | `lib/api.ts` | ✅ Fixed: signal-aborted check between every `reader.read()` + `reader.cancel()` on exit + one-shot listener |
| 3 | `request()` ignored caller's `AbortSignal` | `lib/api.ts` | ✅ Fixed: composes external + timeout via `AbortSignal.any()` |
| 4 | Default model was fake ID `groq-llama-70b` | `hooks/use-conversations.ts` | ✅ Fixed: `DEFAULT_MODEL = "llama-3.3-70b-versatile"` |
| 5 | `useChat.sendMessage` recreated on every token | `hooks/use-chat.ts` | ✅ Fixed: reads messages via `getState()`, no longer in deps |
| 6 | `groupConversations()` not memoized | `hooks/use-conversations.ts` | ✅ Fixed: `useMemo` + module-level inflight de-dupe |
| 7 | No 401 → re-auth flow | `lib/api.ts` + `hooks/use-auth.ts` | ✅ Fixed: dispatches `nexusai:auth-expired`; root listener routes to `/login?next=...` |
| 8 | Service worker cached stale shell | `public/sw.js` | ✅ Fixed: kill-switch SW that wipes caches + unregisters itself |
| 9 | Brave mic blocked silently | `hooks/use-voice.ts` + `components/voice/VoiceButton.tsx` | ✅ Fixed: detects Brave, shows shield-specific instructions, wider tooltip |
| 10 | Worktree backend was orphaned | `nexusai/backend/app/main.py` | ✅ Fixed: switched to real NexusAI backend, added `/waitlist` endpoint |

## Punch-list for next session (deferred — not in last turn)

These are real, scoped tasks. Pick any of them and execute — context is in the graph.

1. **OpenTelemetry first-token trace** on `/chat/stream`. Backend has telemetry middleware but no span on the LLM call. File: `nexusai/backend/app/api/v1/chat.py`.
2. **Tabs-sync via `BroadcastChannel`** so two open tabs see the same chat updates live. Hook into `useChatStore` `appendToken` / `appendMessage`.
3. **Refresh-token endpoint** — backend currently issues 30-day JWTs with no refresh path. Add `POST /api/v1/auth/refresh` that takes the current token and issues a fresh one if it's within 7 days of expiry; have `lib/api.ts` call it transparently on 401 instead of forcing re-login.
4. **Sentry on frontend** — DSN missing. Wire `@sentry/nextjs` (already in `package.json`?) at `app/layout.tsx`.
5. **Streaming edit/regenerate** — current edit-and-regen calls `/messages/{id}/edit` then triggers a new stream. The branch tree on the backend supports it; the frontend doesn't yet show parallel branches.
6. **Stripe billing portal** — `/settings?tab=billing` shows a "coming soon" card. Backend `billing.py` has stubs.
7. **WebAuthn passkeys** — `/settings/2fa` is currently a coming-soon card. Backend lacks implementation.
8. **Tavily-backed deep research** — current research uses Llama-only synthesis. Wire `TAVILY_API_KEY` (BYOK or server) for real web fetches in `nexusai/backend/app/api/v1/research.py`.
9. **E2B sandbox in NexusCode** — UI works for HTML/CSS/JS preview only. `nexusai/backend/app/api/v1/sandbox.py` has E2B integration; needs an `E2B_API_KEY` and frontend wiring at `app/code/[projectId]/page.tsx`.
10. **Reactflow visual workflows** — current `/workflows` is a linear step list; backend supports DAG. Replace the step list with `reactflow` (already in package.json deps as a candidate).

## Operational quick-start

```powershell
# Start backend
cd C:\Users\ranji\pyxis-one-backend\nexusai\backend
py -3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Start frontend (separate terminal)
cd C:\Users\ranji\pyxis-one-backend\nexusai\frontend
npm run dev

# Smoke test (after both up)
cd C:\Users\ranji\pyxis-one-backend\nexusai\frontend
node scripts/smoke.mjs   # expect 46/46 pass

# Refresh knowledge graph after changes
cd C:\Users\ranji\pyxis-one-backend
graphify update .

# Query the knowledge graph
graphify query "How does authentication work?" --budget 800
graphify explain "useChatStore"
graphify path "auth.py" "chat.py"
```

## Top-level files to read first

1. `raw/PROJECT_MEMORY.md` — full architecture, every endpoint, every page, BYOK URLs
2. `raw/HANDOFF.md` — this file
3. `nexusai/CLAUDE.md` — original product brief (21-phase build plan)
4. `nexusai/backend/app/api/v1/__init__.py` — every backend route registration
5. `nexusai/frontend/middleware.ts` — auth gate
6. `nexusai/frontend/lib/store/chat.ts` — chat state (persistent)
7. `nexusai/frontend/lib/api.ts` — API client + SSE helper + auth-expired event
8. `nexusai/frontend/hooks/use-chat.ts` — chat send / stream
9. `nexusai/frontend/scripts/smoke.mjs` — 46-check smoke test
10. `graphify-out/GRAPH_REPORT.md` — community-by-community navigation map
