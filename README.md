# SFCC — Assistive Companion for Visually Impaired Users

A real-time, voice-first AI agent that continuously perceives its user's
surroundings (objects, people, text, distance, direction) and lets them ask
about it naturally, get help finding things, get help navigating indoors,
and manage daily tasks — all through one assistant, no per-feature screens.

This is being built **incrementally, segment by segment**. Nothing is marked
done until it is implemented, tested, and verified end-to-end. See
`ARCHITECTURE.md` for the full phased plan and current status.

## Status: Segments 1-6 — Foundation + core perception ✅

What exists right now, all implemented and tested (31 backend tests + 5
frontend tests passing — see `DEVELOPMENT.md` for exact commands):

- **Segment 1** — FastAPI backend (`/health`, `/ws`) + Astro health page.
- **Segment 2** — real-time camera pipeline: bounded, latest-frame-wins
  queue on the backend; `CameraSource`/`FrameSampler`/`FrameTransport` on
  the frontend. Proven not to accumulate an unbounded backlog (5,000
  flooded frames → 52 KB total memory growth).
- **Segment 3** — real YOLOv8n object detection (CPU, ~0.24s/frame once
  warmed up) + an IoU-based tracker with stable IDs and entry/exit/lost
  events.
- **Segment 4** — the world model that turns tracked objects into
  prioritized, capped entity summaries (never narrates everything it sees).
- **Segment 5** — a 7-bucket direction engine with hysteresis (no
  left/center/left flicker at boundaries).
- **Segment 6** — a confidence-gated qualitative distance estimator (never
  fake metric precision).

Still to come: voice, the Gemini-backed agent orchestrator (blocked on a
one-time local connectivity check — see `ARCHITECTURE.md`), memory,
navigation, and everything after. `ARCHITECTURE.md` has the full phased
plan and status table, updated every segment, not just at the end.

## Prerequisites

- Docker + Docker Compose (recommended), **or** Python 3.11+ and Node 18+
  installed directly.
- A Gemini API key (already configured in `backend/.env` for this machine).

## Running Segment 1 (the acceptance test)

### Option A — Docker (recommended)

```bash
docker compose up backend
```

Then in a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Astro prints (default `http://localhost:4321`). You should see
"Status: connected" and a log entry for the health message and the echoed
`hello-from-browser` message.

### Option B — without Docker

```bash
# Terminal 1 — backend
cd backend
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

### Automated check

```bash
cd backend
pip install -r requirements.txt
pytest
```

This runs the same assertions as the manual browser test (HTTP health +
WebSocket health/echo round trip) without needing a browser.

## Project layout

```
/frontend    Astro + TypeScript UI (voice-first; this is for demo/config/debug)
/backend     FastAPI + WebSockets + asyncio agent backend
/models      Local model weights / wrappers (YOLO, OCR, depth) — added per segment
/perception  Object detection, tracking, direction, depth, OCR
/agent       Orchestrator, planner, reference resolver, tool registry
/tools       Tool implementations the agent can call (find_object, read_text, ...)
/memory      Short-term / session / long-term memory
/navigation  Indoor navigation engine, world model extensions, replanning
/voice       STT / TTS integration, turn-taking, interruption handling
/safety      Confidence engine, safety policy, emotional-safety layer
/tests       Unit + integration + stress tests
/docs        Architecture notes, demo scripts, known limitations
```

## Key documents

- `ARCHITECTURE.md` — chosen stack, model choices and why, phased segment plan, current status
- `SECURITY.md` — credential handling, CORS, validation, no-arbitrary-execution rules
- `PRIVACY.md` — what is (and is never) stored, trusted-person consent model
- `DEVELOPMENT.md` — day-to-day dev workflow, testing, troubleshooting
