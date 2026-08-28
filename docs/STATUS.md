# Segment Status — honest accounting

The brief specifies 49 segments. This file states exactly which are actually
working, which are partial, and which have not been started. Nothing is
marked done unless it runs and is covered by a test.

**Legend:** ✅ done and tested · 🟡 partial (what's missing is stated) · ⬜ not started

## Working today — the vertical slice

| # | Segment | State | Evidence / gap |
|---|---|---|---|
| 1 | Project foundation | ✅ | FastAPI + Astro, strict env handling, docs. Docker compose file exists but is unused — Docker isn't installed and SQLite was chosen instead. |
| 2 | Real-time camera | ✅ | Permission/start/stop/pause/disconnect, FPS sampling, backpressure, latest-frame-wins. 250-frame soak: **−6.4 MB RSS**. 30-min *browser* soak is still a manual check. |
| 3 | Continuous object perception | ✅ | YOLOv8n via OpenVINO iGPU, IoU tracker, stable IDs, ENTERED/UPDATED/MOVED/LOST/LEFT, duplicate suppression, latency monitoring. |
| 4 | Scene understanding | ✅ | World model with prioritized, capped summaries (people → obstacles → other). |
| 5 | Direction engine | ✅ | 7-way buckets, per-track hysteresis, confidence per estimate. |
| 6 | Distance / depth | ✅ | Depth Anything V2 Small on the iGPU (47 ms), fused with the size prior. Agreement raises confidence; **gross disagreement reports UNKNOWN rather than picking a side**. Polarity and ordering asserted against real photo geometry. |
| 9 | Voice input | ✅ | Streaming mic → PCM16 worklet → Live API, with server-side VAD and barge-in. |
| 10 | Voice output | ✅ | Gapless scheduled playback, instant interrupt on barge-in. |
| 11 | Real-time conversation | ✅ | Gemini Live, ~390 ms to first audio. Pronoun follow-ups verified working. |
| 12 | Agent orchestrator | ✅ | 14 grounded tools; the model never executes code and never sees anything but structured tool results. |
| 14 | Object finding | ✅ | Candidate ranking, multi-candidate reporting, honest absence. |
| 16 | Memory | ✅ | SQLite + Gemini embeddings (768d), exact cosine. Semantic search finds memories sharing no keywords with the query; falls back to keyword search offline. Persists across restarts. |
| 15 | OCR / reading signboards | ✅ | Local RapidOCR (934 ms, offline) for continuous sign spotting with direction; Gemini for meaning, navigation intent and summaries. Verified 4/4 text and 4/4 positions on a known fixture, and Spanish signage on a real photo. |
| 17 | Daily tasks | ✅ | Create/list/complete, natural-language times converted by the model to ISO. An unparseable time stores the task with no timestamp rather than guessing a day. |
| 18 | Preparation mode | ✅ | Checklists cross-referenced against the live scene, split three ways: seen / not currently visible / cannot be checked at all — with ready-made phrasing so the categories are not blurred in speech. |
| 8 | Trusted person + belongings | ✅ | Explicit photo enrolment only. Faces via YuNet+SFace (different people separate at 0.04 vs 0.36 threshold); belongings via MobileNetV2 features with a hard object-class gate. Unenrolled faces are counted, never named or described. |
| 28 | Proactive assistance | ✅ | Four gates before any unprompted speech: urgency, novelty, cooldown, and silence while a turn is in progress. |
| 31 | Confidence engine | ✅ | Every perception carries confidence; hedging below threshold. |
| 32 | Safety policy | ✅ | Structural — the model can only speak values present in a tool result. |
| 33 | Tool execution safety | ✅ | Schema validation, timeouts, structured errors, bounded trace. |
| 4 | User profile | ✅ | Name, 18-trait personality vector, communication style and preferences, in their own table (Part 46). Editable by voice or from the dashboard. |
| 5 | Personality engine | ✅ | Traits derive real numbers — interruption threshold, cooldown, sentence budget, suggestion gates — not a prompt line. Measured: same friend event scores 0.81/0.43 for an extrovert and 0.53/0.73 for an introvert. |
| 15 | Personality mixing | ✅ | A vector, not a switch; opposing traits stay consistent. |
| 16 | Personality learning | ✅ | Explicit statements only. Clinical labels are refused, never stored. |
| 23 | Crowd detection | ✅ | Density, distribution, motion, flow, congestion, approach — not a head count. |
| 24 | Crowd classification | ✅ | 5 sizes, 3 densities; all thresholds configurable per deployment. |
| 25 | Static vs moving | ✅ | From smoothed per-track velocity added to ObjectTracker. |
| 26 | Crowd direction | ✅ | Relative only (left/right/ahead) — no compass claims. |
| 28 | Crowd flow | ✅ | Unidirectional / bidirectional / converging / diverging / chaotic, from velocity agreement. |
| 30 | Crowd approach | ✅ | Majority apparent-size growth; verified as `approaching=True`. |
| 32 | Crowd events | ✅ | 10 debounced transition events. |
| 34 | Memory-aware conversation | ✅ | A saved description is surfaced when several candidates are visible, explicitly without claiming a match. |
| 33 | Personality + crowd | ✅ | Same events, different profiles: friend SPOKE for an extrovert, held for an introvert; hazard spoke for both. |
| 36 | Proactive intelligence | ✅ | AttentionManager scores importance, confidence, novelty, personal and personality relevance against a personality-derived threshold. |
| 37 | Interruption priority | ✅ | 5 severity classes; SAFETY bypasses scoring, cooldown, quiet mode and the busy check. |
| 38 | Haptic abstraction | ✅ | Provider interface + Mock / Browser-vibrate / Gamepad, capability-declared. |
| 39 | Haptic events | ✅ | 14 event types, each with a pattern and priority. |
| 40 | Haptic language | ✅ | Consistent rhythm vocabulary; single-motor providers never claim direction. |
| 42 | Haptic cooldowns | ✅ | Per event instance; a persistent obstacle buzzes once, not every frame. |
| 43 | Personalized haptics | ✅ | enabled / safety-only / per-category / intensity. |
| 44 | Quiet mode | ✅ | Suppresses ambient + social; safety still speaks. |
| 48 | Event bus | ✅ | Typed WorldEvents, bounded history, isolated subscribers. |
| 52 | Friend circles | ✅ | Named groups; members without an enrolled face are flagged. |
| 55 | No hallucinated intent | ✅ | Movement is reported, destination and purpose never are. |
| 56 | Personality safety boundary | ✅ | Structural early-return before any personality value is read. |
| 57 | Identity confidence | ✅ | Bands at 0.90 / 0.75; below that no name is produced at all. |
| 58 | Anti-false-positive | ✅ | 3-frame confirmation. One spurious frame yields no name and no event. |
| 68 | Demo mode | ✅ | Deterministic scene injection, clearly flagged as demo. |
| 35 | Agent execution trace | ✅ | Live tool-call/result panel in the dashboard. |
| 37 | Real-time performance | ✅ | See [PERFORMANCE.md](PERFORMANCE.md) — all numbers measured. |

## Partial

| # | Segment | State | What is missing |
|---|---|---|---|
| 7 | Person detection | ✅ | Multi-person detection, tracking, direction, distance, velocity, and visibility states (candidate/confirmed/occluded/lost). |
| 13 | Context resolution | 🟡 | Pronoun/reference resolution works through Live session context + `entity_id` (verified). No explicit disambiguation prompt ("the backpack or the bottle?"). |
| 29 | Conversational personality | ✅ | 18 selectable traits plus verbosity, proactivity, humour and encouragement; style prompt derived from the vector. |
| 36 | Error recovery | 🟡 | Covered: frame decode, tool failure/timeout, WS reconnect with backoff, camera/mic permission failure, vision-unavailable degradation, depth/OCR/face/appearance model missing, OCR engine failure, embedding-API-down (keyword fallback), stale depth rejection, unreadable enrolment photo. |
| 61 | Model routing | 🟡 | Inference backends routed and measured (iGPU → ONNX CPU → OpenVINO CPU); YOLO / depth / OCR / face / appearance / Gemini each used for their own job, on staggered cadences (Part 60). No LLM-tier routing. |
| 39 | User interface | 🟡 | Live environment, conversation, world model, agent trace, memory & tasks, signs & text, enrolment, personality weights with quick personas, crowd & friends, and attention/haptic decisions. No navigation panel; no Three.js spatial view. |
| 42 | Privacy | 🟡 | No recording, no frame persistence. Enrolment stores a numeric embedding, never a photograph, and only for people/items the user deliberately added; strangers are never identified. Dashboard memory panel lists everything stored and can delete it; mass-deletion is a human gesture that is deliberately **not** a model-callable tool. Enrolled people/items have per-item Remove buttons; saved memories are deleted by voice or cleared in bulk. |
| 43 | Security | 🟡 | Key stays server-side, CORS, strict input validation, message size caps, no shell/code execution, bounded logs. Dashboard tool access is restricted to a read-only allowlist. No auth or rate limiting (single-user local app). |
| 49 | Testing | 🟡 | 202 backend + 5 frontend tests: unit, integration, soak, ONNX parity, depth-vs-ground-truth, OCR-vs-ground-truth, identity separation, enrolment round trip, live. Missing the rapid-movement / network-interruption stress matrix. |

## Not started

| # | Segment |
|---|---|
| 20–26 | Navigation engine, route verification, replanning, multi-step planner |
| 19 | Meal assistance |
| 30 | Emotional safety layer |
| 40 | Three.js / WebGL spatial visualization |
| 66 | Live 3D spatial view |
| 44 | Phase 3 device/emergency provider interfaces |

## Known limitations

1. **The camera has a narrow field of view.** The system says "not currently
   visible", never "not there". This is enforced in tool output, not just
   prompting.
2. **Distance is coarse and monocular.** A real depth model now backs it, but
   Depth Anything produces *relative* inverse depth with no metric scale, so no
   distance in metres is ever emitted. When depth and the size prior disagree
   badly, the system reports UNKNOWN instead of choosing one.
3. **COCO vocabulary only for objects.** Doors, stairs, keys and wallets have
   no detector class. Asking for them returns `supported: false` — "I cannot
   look for that" — rather than a misleading "I don't see it". Signs are now
   covered separately by OCR.
7. **Face detection degrades in poor light** before recognition does. At −40
   brightness YuNet missed a face entirely. The system then says it cannot see
   anyone's face, rather than guessing.
8. **Haptics are mock-only on this machine.** No controller, and desktop Chrome
   exposes no vibration API. Every haptic decision is real and visible in the
   dashboard; open the page on Android Chrome for actual vibration. A single
   motor cannot convey direction, and the system says so rather than implying it.
9. **No GPS and no maps provider.** Places are found from signage the camera can
   read, so the assistant can never say "50 metres ahead" — only a relative
   direction. Distances in metres would be fabricated.
10. **Crowd thresholds are uncalibrated for your camera.** The defaults suit a
   typical webcam field of view; `CrowdConfig` is meant to be tuned per setting.
11. **Two similar belongings cannot be told apart.** The measured margin between
   different objects is only ~0.11, and two same-class items differing only in
   colour are the weak case. The system reports "possible" and hedges rather
   than asserting.
4. **Browser camera untested in automation.** The in-app browser blocks camera
   access, so the live camera path has been verified by unit/integration tests
   and by graceful-failure behavior, but needs a manual run in real Chrome.
5. **The host machine is heavily loaded.** All benchmarks were taken at ~100%
   CPU from unrelated software; the iGPU path was chosen partly to route
   around this.
6. **Voice costs API credits** on every spoken turn.
