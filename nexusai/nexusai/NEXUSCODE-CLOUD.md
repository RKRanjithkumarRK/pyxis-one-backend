# NEXUSCODE CLOUD — Full Codex Cloud Equivalent
# Upgrade NexusCode from Judge0 to real cloud Linux environments
# ADD TO EXISTING NEXUSAI-ULTIMATE BUILD

---

# IDENTITY

You are the engineering org at NexusAI. The previous build used Judge0 for
execution. That's not Codex Cloud. Codex Cloud means: every user gets a
real Linux container with a terminal, persistent files, package installation,
long-running processes, and live preview URLs. We build that. Exactly that.

Performance must match or exceed:
- ChatGPT: first token < 400ms, full streaming
- Claude Artifacts: render < 200ms after stream starts
- Cursor: inline completions < 150ms
- Codex/Replit: environment boot < 3 seconds, terminal < 100ms latency
- Perplexity: Deep Research 1-5 min end-to-end
- GitHub Copilot: ghost text appears < 200ms

Every interaction on every OS (Windows, macOS, Linux, iOS, Android, iPad,
Chromebook) must feel native.

---

# WHAT "CODEX CLOUD" ACTUALLY MEANS

When a user opens a project in NexusCode, they get:

1. Their own Linux container (Ubuntu 22.04, 2 vCPU, 2GB RAM, 5GB disk)
2. Real filesystem — edit, save, files persist across sessions
3. Real terminal — bash/zsh with internet access, sudo, apt, pip, npm
4. Pre-installed toolchains — Python, Node, Java, Go, Rust, .NET, Ruby, PHP
5. Install anything — pip install X, npm install Y, sudo apt install Z
6. Run long-running processes — npm run dev keeps running
7. Live preview URL — any exposed port gets https://preview-{id}.nexusai.dev
8. Real git — git clone, git push, GitHub auth
9. Share environment — send link, teammate joins the live session
10. AI has shell access — AI can run commands, install packages, test code itself
11. Persistent across sessions — close tab, come back tomorrow, everything still there
12. Fork/snapshot — branch your environment like git branches

---

# ARCHITECTURE

## Sandbox Provider: E2B (e2b.dev)
- Firecracker VMs, sub-second cold start
- Full Ubuntu environment
- Python + JS SDKs
- Process execution API
- File upload/download API
- Port exposure for previews
- Up to 24-hour sessions

## File Persistence: GCS
- Each project: gs://nexusai-projects/{user_id}/{project_id}/
- On sandbox start: download all files from GCS → sandbox filesystem
- On every file save: sync to GCS (debounced 2s)
- On sandbox stop: final full sync to GCS

## Real-time: WebSockets
- Terminal WS: bidirectional stdin/stdout between xterm.js and sandbox PTY
- File sync WS: push file changes to sandbox and other open clients
- Preview WS: notify client when sandbox exposes new port
- Powered by FastAPI WebSocket + Redis pub/sub

---

# EXECUTION ORDER

## Phase Code-Cloud-1: E2B integration
1. Sign up e2b.dev, get API key
2. Build custom E2B template (Dockerfile) + push with e2b template build
3. Write app/services/sandbox/e2b_service.py
4. Write app/services/storage/gcs_service.py
5. Write app/api/v1/sandbox.py REST endpoints
6. Add E2B_API_KEY to Secret Manager + .env

## Phase Code-Cloud-2: WebSockets
7. Write app/websocket/terminal.py
8. Write app/websocket/file_sync.py
9. Register WS routes in main.py
10. Test WebSocket connection

## Phase Code-Cloud-3: Frontend CloudIDE
11. Install xterm.js packages
12. Write components/code/CloudTerminal.tsx
13. Write components/code/PreviewPanel.tsx
14. Write hooks/useFileSync.ts
15. Write components/code/CloudIDE.tsx
16. Update app/(dashboard)/code/[projectId]/page.tsx

## Phase Code-Cloud-4: AI shell access
17. Update agent prompts with <shell> tags
18. Add extract_and_execute_shell in app/api/v1/projects.py
19. Stream shell execution results back as SSE
20. Test end-to-end with live preview

## Phase Code-Cloud-5: Cross-platform
21. Write components/code/MobileIDE.tsx
22. Add useIsMobile hook
23. Add PWA manifest.json + service worker
24. Test on all OS/devices

## Phase Code-Cloud-6: Performance
25. Redis response cache for LLM calls
26. E2B warm pool (5 sandboxes always ready)
27. OpenTelemetry spans for every SLA metric
28. Cloud CDN + HTTP/3

## Phase Code-Cloud-7: Testing parity
29. Side-by-side comparison vs Cursor, Replit
30. Fix any gap > 20% slower
31. Document in docs/performance.md

---

# PERFORMANCE SLAs

| Metric | Target |
|---|---|
| First token (chat streaming) | < 400ms |
| Inline autocomplete | < 150ms |
| Sandbox cold start | < 3 seconds |
| Sandbox warm start | < 500ms |
| Terminal keystroke | < 100ms RTT |
| File save synced | < 500ms |
| Preview URL ready | < 1 second |
| Deep Research | 1-5 min |

---

# RULES

1. No simulated environments. Real Linux containers only.
2. No Judge0 fallback. E2B is the only execution path.
3. Every feature works on every OS and device.
4. Performance SLAs are hard gates.
5. Zero TODOs. Zero placeholders. Every file complete.
6. After each phase, verify end-to-end before next.
7. Never stop until all 7 phases done and SLAs pass.

---

# EXECUTE NOW

Start with Phase Code-Cloud-1. When all 7 phases pass, run final test:
1. Open project
2. Ask AI: "Create a Next.js app with a /users page fetching from JSONPlaceholder"
3. AI writes all files, runs npm install, runs npm run dev
4. Preview panel auto-shows the app
5. Total time: < 60 seconds

If this works, print:
╔════════════════════════════════════════════════════════╗
║  NEXUSCODE CLOUD — CODEX EQUIVALENT READY              ║
║  Real Linux. Real Terminal. Real Preview.              ║
║  All OS. All Devices. SLAs met.                        ║
╚════════════════════════════════════════════════════════╝
