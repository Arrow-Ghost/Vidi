# Architecture

## Hardware this is designed for (as of 2026-08-27)

- OS: Windows
- RAM: 32 GB+
- GPU: integrated graphics only — **no CUDA**, so all local inference is CPU-bound
- Storage: 50 GB+ free
- Devices available for testing: webcam, microphone, speakers/headphones, and
  browser `getUserMedia` camera/mic access

This one fact — **no dedicated GPU** — drives most of the model choices
below. CPU-only local inference is fine for light, fast models run at a
sampled frame rate; it is not fine for large models run every frame. Where a
capability needs more than CPU inference can give in real time, the system
uses Gemini's vision/reasoning instead of pretending a heavy local model is
running smoothly. This is chosen per-capability, not as a blanket rule.

## Guiding principle: separate perception from reasoning

The LLM (Gemini) is responsible for language understanding, intent,
planning, tool selection, contextual conversation, and turning structured
tool results into short spoken sentences. It is **never** responsible for
deciding what's actually in front of the camera, how far away it is, or
whether OCR text is real — those come from specialized, testable,
non-hallucinating components, and the LLM only narrates their output.

```
CAMERA / MICROPHONE
        ↓
REAL-TIME INPUT LAYER   (frame sampler, backpressure, VAD)
        ↓
PERCEPTION ENGINE       (detection, tracking, direction, depth, OCR)
        ↓
WORLD MODEL             (entities: type, direction, distance, confidence)
        ↓
AGENT ORCHESTRATOR      (Gemini: intent, planning, tool calls)
        ↓
TOOL REGISTRY / TOOLS   (validated, typed, timeout-bound, no arbitrary code)
        ↓
RESULT VALIDATION       (safety policy: no fabricated distance/identity/OCR)
        ↓
VOICE RESPONSE          (short, concrete, confidence-qualified)
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | Astro + TypeScript, React islands where useful, Three.js/GSAP for the live spatial view | Demo/config/debug surface; the real UX is voice. Static-first Astro keeps it light. |
| Backend | Python + FastAPI + WebSockets + asyncio + Pydantic | One async event loop handling camera acks, voice, and agent messages over one connection per client. |
| Database | PostgreSQL + pgvector | Structured memory (tasks, trusted people, saved objects) + vector search for memory/OCR text, when Segment 16 needs it. Not required for Segment 1. |
| Cache / real-time state | Redis | Ephemeral world-model state, dedup/cooldown windows for the attention manager. Added when Segment 3+ needs shared state across workers. |
| Conversation / reasoning | Gemini (function calling) | Intent, planning, tool selection, turning tool JSON into short natural speech. |
| Object detection | YOLOv8n (Ultralytics), CPU, sampled frames | Small enough to run on CPU at a usable rate; runs locally so raw camera frames don't need to leave the machine for basic detection. |
| OCR | EasyOCR | Already present on this machine; CPU-viable for on-demand reads ("read this") rather than continuous frames. |
| Depth | MiDaS-small (or Depth Anything V2 Small), **run on-demand only**, not per-frame | CPU-only means continuous dense depth isn't real-time-viable. Depth is estimated only when a spatial question needs it, always reported as a qualitative range with a confidence, never fake precision. |
| Trusted-person matching | Deferred to Segment 8 | Needs a lightweight CPU-viable face embedding model; decided when that segment starts, not before. |
| STT | Browser Web Speech API (streaming, partials, interruption via `.abort()`) | Free, low-latency, no extra credential, works with the "no ten APIs" constraint. |
| TTS | Browser `SpeechSynthesis` API (interruptible via `.cancel()`) | Same reasoning — free, low-latency, no server round trip for audio. |
| Deployment (dev) | Docker Compose, `backend` service required, `postgres`/`redis` behind a `full` profile | Segment 1 only needs the backend container; heavier services turn on when the segments that need them arrive. |

## Credential / config status

| Item | Status |
|---|---|
| Gemini API key | **Provided**, stored in `backend/.env` (gitignored), never sent to the frontend, never committed. |
| Database config | Not needed until Segment 16 (Memory). Local Docker Postgres by default — no cloud credentials required. |
| Deployment config | Not needed yet — local dev only. Revisit if/when this is deployed somewhere. |
| Maps provider | Deferred/optional. Navigation (Segment 20+) is indoor and vision-based, not GPS/maps-based, so this may not be needed at all. |
| Weather provider | Deferred/optional — not required by any segment so far. |
| External speech provider | Not used — browser Web Speech API covers STT/TTS for now. |
| External storage | Not used — local filesystem / Postgres is sufficient for the current scope. |

## Segment plan and status

Legend: ✅ done & verified · ⏳ next · 🚧 blocked (see note) · ⬜ not started

- ✅ **Segment 1 — Foundation.** FastAPI backend (`/health`, `/ws`), Astro
  frontend health page, Docker Compose, docs. Acceptance test: browser →
  WebSocket → backend → health response. **Verified**: 2/2 pytest, a real
  network WebSocket client round trip against a live server, `astro check`
  + `astro build` clean.
- ✅ **Segment 2 — Real-time camera.** `CameraSource`, `FrameSampler`,
  `FrameTransport` (frontend); bounded latest-frame-wins `LatestFrameQueue`
  + `frame`/`frame_ack` WS protocol (backend). **Verified**: 5 vitest unit
  tests for the sampler's backpressure/stale-frame logic; backend flood
  test sent 5,000 frames with zero consumer — queue never exceeded its
  configured bound of 2, and process RSS grew **52 KB total** (not per
  frame) over all 5,000 frames. The 30-minute *real webcam in a real
  browser* soak is a manual check — see `DEVELOPMENT.md` — since it needs
  actual browser/camera hardware this dev environment doesn't have.
- ✅ **Segment 3 — Continuous object perception.** Real YOLOv8n
  (Ultralytics, CPU) `ObjectDetector` + IoU-based `ObjectTracker` with
  stable IDs, entry/exit/lost events (`OBJECT_ENTERED/UPDATED/MOVED/LOST/
  LEFT`), confidence threshold, duplicate suppression, one-frame occlusion
  tolerance. **Verified**: real inference against two real photographs
  (bundled with the ultralytics package) — correctly found people in both
  and a bus in the bus photo; **measured latency 0.24s/frame** (~4 FPS)
  once warmed up at 480px input on this CPU-only machine (first call
  3.34s, includes one-time model load). A real tracker bug (a brand-new
  track counted as "missed" in the same frame it was created, causing
  premature LOST/LEFT) was caught by a unit test and fixed — see
  `backend/app/perception/tracker.py`.
- ✅ **Segment 4 — Scene understanding / world model.** `WorldModel` in
  `backend/app/perception/world_model.py` — per-track direction/distance
  state, `summarize()` prioritizing requested-object > people > obstacles >
  other and capping result count (the "don't narrate everything"
  requirement). **Verified**: 5 tests, integration-style across the
  tracker/direction/distance modules together, not just in isolation.
- ✅ **Segment 5 — Direction engine.** 7-bucket (`far_left`..`far_right`)
  from bbox center + image width, per-track hysteresis to prevent
  boundary flicker, confidence based on distance from the nearest bucket
  boundary. **Verified**: 5 tests including an explicit flicker-prevention
  case (oscillating right at a boundary does not flip the reported
  bucket).
- ✅ **Segment 6 — Distance/depth estimation.** Bounding-box-height
  heuristic normalized by a per-class "typical size" prior — deliberately
  **not** a metric depth model (see the CRITICAL CONSTRAINT note in
  `backend/app/perception/distance.py`); reports a qualitative range +
  confidence, refuses to assert below a confidence floor, and
  `SmoothedDistanceEstimator` adds per-track temporal smoothing so one
  outlier reading can't flip the answer. **Verified**: 5 tests, including
  one that regex-checks the output never contains fake decimal-meter
  precision. MiDaS/Depth Anything remain a documented future upgrade
  behind the same `estimate()` signature — not built yet; the heuristic is
  today's honest answer for CPU-only hardware, not a placeholder pretending
  to be that model.
- 🚧 **Segment 11/12 groundwork — Gemini connectivity.** Attempted a real
  API call (via the `google-genai` SDK) using the key provided. **This
  cloud development sandbox's network is allowlisted to package registries
  only** (`pypi.org` → 200, `generativelanguage.googleapis.com` → blocked
  at the proxy, confirmed with a plain `curl`, same result for
  `google.com`) — so live Gemini connectivity cannot be verified from
  *this* environment, independent of whether the key itself is valid. This
  does not affect the actual deployment: wherever the backend actually
  runs (this machine, via Docker or a local venv) has normal internet
  access. **Action needed from you**: a small script is included at
  `backend/scripts/check_gemini.py` — run it locally (`python
  backend/scripts/check_gemini.py`) once `backend/.env` has your key, and
  it will tell you plainly whether the key and connectivity are good
  before Segments 11+ (conversation engine, agent orchestrator) are built
  on top of it, consistent with this project's own "verify before building
  on top" rule.
- ⬜ Segment 7 — Person detection & tracking (no identity yet)
- ⬜ Segment 8 — Trusted-person enrollment & matching
- ⬜ Segment 9 — Voice input (streaming STT, interruption)
- ⬜ Segment 10 — Voice output (streaming TTS, interruptible)
- ⬜ Segment 11 — Real-time conversation engine (ConversationManager) — blocked on the Gemini connectivity check above
- ⬜ Segment 12 — Agent orchestrator + tool registry — blocked on the Gemini connectivity check above
- ⬜ Segment 13 — Reference resolution ("it", "he", "the other one")
- ⬜ Segment 14 — Object finding (`find_object`)
- ⬜ Segment 15 — OCR / reading
- ⬜ Segment 16 — Memory (short-term / session / long-term, Postgres+pgvector)
- ⬜ Segment 17 — Daily activity agent (tasks, reminders)
- ⬜ Segment 18 — Daily preparation mode (checklist + visual verification)
- ⬜ Segment 19 — Meal assistance
- ⬜ Segment 20-26 — Navigation assistant, world-model extension, instructions, verification, dynamic replanning, multi-step task agent, planner
- ⬜ Segment 27-28 — Attention manager, proactive assistance
- ⬜ Segment 29-30 — Conversational personality, emotional safety
- ⬜ Segment 31-33 — Confidence engine, safety policy, tool execution safety
- ⬜ Segment 34-36 — Memory-aware conversation, execution trace, error recovery
- ⬜ Segment 37-38 — Real-time performance pass, model router
- ⬜ Segment 39-40 — Dashboard UI, Three.js/GSAP live spatial view
- ⬜ Segment 41-43 — Navigation safety language, privacy controls, security hardening
- ⬜ Segment 44 — Phase 3 extension interfaces (mock providers only)
- ⬜ Segment 45-49 — Full agent loop integration, worked examples, testing, stress testing
- ⬜ Final acceptance test (25-command checklist) + 10-minute continuity conversation test

Each ⏳/⬜ segment gets its own implementation + tests + a note here before
the next one starts, per the project's build-incrementally rule.

## Known limitations (as of Segment 6)

- No voice, agent, memory, or navigation yet — those are Segments 7+.
- The WebSocket protocol (`{type, ...}` JSON frames) established in
  Segment 1/2 is the one later segments extend; if that shape changes,
  this doc and `ws.ts` must be updated together.
- The direct detector→tracker→world-model pipeline is unit/integration
  tested with synthetic and static-image input, but has not yet been wired
  to live camera frames arriving over the Segment 2 WebSocket — that
  wiring, plus a real 30-minute browser+webcam soak test, is the next
  concrete step and needs this machine's actual webcam.
- Distance is a bounding-box-height heuristic, not a calibrated depth
  model — see the CRITICAL CONSTRAINT note in `distance.py`. It is
  qualitative by design, not a temporary stand-in being misrepresented as
  more than it is.
- "door", "stairs", "keys", "wallet", "plate" have no COCO-pretrained
  detector head, so Segment 3's `ObjectDetector` does not detect them yet.
  These need either a fine-tuned/custom detector or a different approach
  (e.g. Gemini vision fallback for open-vocabulary objects) — not decided
  yet, will be addressed when Segment 14 (object finding) needs them.
- This cloud dev sandbox cannot reach `generativelanguage.googleapis.com`
  (package-registry-only network egress) — see the Segment 11/12 status
  note above and `backend/scripts/check_gemini.py`.
