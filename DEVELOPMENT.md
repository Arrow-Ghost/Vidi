# Development

## Segment 1 — verified test run

Run from a clean checkout:

```bash
cd backend
pip install -r requirements.txt
pytest -v
```

Expected: `test_http_health` and `test_websocket_health_and_echo` both pass.
This is the automated equivalent of the manual "open the frontend, watch it
say connected" check — both exercise the exact same endpoint code.

Manual/browser version:

1. `docker compose up backend` (or `uvicorn app.main:app --reload` from
   `backend/` in a venv).
2. `cd frontend && npm install && npm run dev`.
3. Open the printed localhost URL. Expect "Status: connected", a "Received
   health: ..." log line, and a "Received echo: ..." log line for the
   `hello-from-browser` message the page sends automatically.

## Segments 2-6 — verified test run

```bash
cd backend
pip install -r requirements.txt   # now includes torch/ultralytics/opencv (~1GB) and google-genai
pytest -v
```

Expected: 31 tests pass, covering the frame queue, the camera flood/soak
test, real YOLOv8n detection against two bundled sample photos, the
tracker, the direction engine, the distance estimator, and the world
model. First run downloads `yolov8n.pt` (~6MB) automatically via
ultralytics into a cache — no manual step needed. Measured on this
project's dev machine: object detection ~0.24s/frame warmed up (~4 FPS) at
480px input size, CPU-only; the 5,000-frame camera flood test grew process
memory by 52 KB total (not per frame).

For the frontend camera modules:

```bash
cd frontend
npm install
npm test        # vitest — FrameSampler backpressure/stale-frame unit tests
```

### Manual checks that need real hardware (can't run in an automated CI-like environment)

- **30-minute camera soak, real browser + real webcam**: open the Segment 2
  frontend page with the camera wired in, leave it running 30 minutes, and
  watch the browser's task manager / devtools memory panel for growth.
  The backend half of this claim (bounded queue, flat memory under load) is
  already covered by the automated flood test above — this manual pass is
  specifically to catch anything the *browser* side does that a flood test
  using fake frames can't (canvas/blob GC behavior, MediaStream lifecycle
  over a long real session).
- **Gemini connectivity**: `python backend/scripts/check_gemini.py` — run
  this locally (this dev sandbox's network can't reach
  generativelanguage.googleapis.com to run it there; see ARCHITECTURE.md).
  It checks basic text generation and a function-calling round trip, and
  must pass before Segments 11+ (conversation engine, agent orchestrator)
  are built on top of it.

## Adding a new segment

1. Implement it in its own directory (`/perception`, `/agent`, etc.) — don't
   scatter one capability's code across unrelated folders.
2. Write unit tests for it under `/tests` (or colocated `tests/` next to the
   module, matching the backend's existing `backend/tests/` pattern).
3. Integrate it with the previous segment's verified interface — don't
   change an already-verified interface without updating its tests and the
   status table in `ARCHITECTURE.md`.
4. Run the relevant acceptance test from the project brief for that segment
   before marking it done.
5. Update `ARCHITECTURE.md`'s status table (⬜ → ⏳ → ✅) in the same change.

## Environment variables

Copy `.env.example` → `backend/.env` (and `frontend/.env` for `PUBLIC_*`
values) and fill in real values. Never commit either `.env` file — both are
gitignored.

## Known constraints for this environment

- No dedicated GPU — all local model inference (Segment 3 detector done;
  Segment 15 OCR still to come) is validated CPU-only, at a sampled frame
  rate, not per-frame. Segment 3's detector measured ~0.24s/frame (~4 FPS)
  once warmed up, at 480px input — see the Segments 2-6 section above.
- The machine this was originally scaffolded on (a cloud dev sandbox) can
  reach PyPI/npm but not arbitrary internet hosts (confirmed:
  `generativelanguage.googleapis.com` and `google.com` both blocked at the
  proxy, `pypi.org` fine). If you're developing somewhere with unrestricted
  internet (this machine, normally), this doesn't apply to you — it only
  means Gemini calls couldn't be *test-verified* from that original sandbox
  and had to wait for `check_gemini.py` to be run here instead.
