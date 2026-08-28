# Deploying Vidi — Vercel (frontend) + Render (backend)

Every field, with the values that actually work.

---

## Render — the backend

**New → Web Service → connect `Arrow-Ghost/Vidi`**

| Field | Value |
|---|---|
| **Name** | `vidi-backend` |
| **Language / Runtime** | `Python 3` |
| **Region** | pick the one nearest you — it is round-trip latency on *every camera frame* |
| **Branch** | `main` |
| **Root Directory** | `backend` |
| **Build Command** | `pip install -r requirements.txt && python ../scripts/fetch_models.py` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | **Standard (2 GB)** — see below |
| **Health Check Path** | `/health` |

### Environment variables

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | your key |
| `CORS_ORIGINS` | `https://<your-project>.vercel.app` |
| `PYTHON_VERSION` | `3.13.0` |

`CORS_ORIGINS` must be the **exact** origin — scheme included, **no trailing
slash**. `https://vidi.vercel.app/` with the slash will be rejected.

Leave `DATABASE_URL` and `REDIS_URL` empty. Storage is SQLite; those variables
exist for a future Postgres swap and nothing reads them today.

### Add a disk

**Settings → Disks → Add Disk**

| Field | Value |
|---|---|
| Name | `vidi-data` |
| Mount Path | `/opt/render/project/src/backend/data` |
| Size | 1 GB |

Without this, Render's filesystem is ephemeral and **every deploy wipes the
user's saved memories, tasks and enrolled people**.

### Why Standard and not Free

The process holds YOLO, Depth Anything, YuNet, SFace and MobileNet in memory
and measures **~460 MB resident**. Free and Starter are 512 MB, so the OOM
reaper kills it part-way through a session — which presents as the backend
randomly dying rather than as an obvious error.

Free also spins down after 15 minutes idle, and a cold start reloads every
model.

---

## Vercel — the frontend

**Add New → Project → import `Arrow-Ghost/Vidi`**

| Field | Value |
|---|---|
| **Framework Preset** | `Astro` |
| **Root Directory** | `frontend` |
| **Build Command** | `npm run build` *(leave default)* |
| **Output Directory** | `dist` *(leave default)* |
| **Install Command** | `npm install` *(leave default)* |
| **Node.js Version** | 20.x or later |

### Environment variables

| Key | Value |
|---|---|
| `PUBLIC_BACKEND_WS_URL` | `wss://vidi-backend.onrender.com/ws` |

**This one is not optional.** With it unset the page derives the backend from
its own address, which on Vercel means `wss://your-app.vercel.app:8000/ws` —
a port nothing is listening on. The page loads perfectly and hangs on
"connecting" forever.

Note `wss://` (not `ws://`) and **no port number**: Render terminates TLS on
443.

Do **not** set `PUBLIC_BACKEND_PORT`. It only applies to the local
same-host default.

---

## Order of operations

There is a circular dependency — each side needs the other's URL.

1. Deploy **Render** first. Set `GEMINI_API_KEY`; leave `CORS_ORIGINS` blank.
2. Copy the Render URL, e.g. `https://vidi-backend.onrender.com`.
3. Deploy **Vercel** with `PUBLIC_BACKEND_WS_URL` = that URL as `wss://…/ws`.
4. Copy the Vercel URL.
5. Back in Render, set `CORS_ORIGINS` to it and let it redeploy.

Verify: `https://vidi-backend.onrender.com/health` should return JSON with
`"vision_available": true`.

---

## What changed to make this deployable

Three things were broken for deployment and are now fixed:

**`requirements.txt` was missing four packages it actually imports** —
`openvino`, `onnxruntime`, `rapidocr`, `numpy`. The build would have installed
cleanly and then crashed on first frame.

**torch and ultralytics are gone from the runtime install.** They were only
ever used to convert `yolov8n.pt` into ONNX once; the hot path reads ONNX
directly. That ONNX file is now committed, so the deploy needs neither:

| | Install size |
|---|---|
| Before | **1,449 MB** |
| After | **443 MB** |

Verified with a clean venv containing no torch: detector picked its ONNX
backend, returned correct detections, and the app imported fine.
`requirements-dev.txt` still has them for re-exporting the model.

**`scripts/fetch_models.py` did not exist**, despite an error message telling
you to run it. It now downloads the ~150 MB of weights that are not in git,
and rejects a truncated file rather than letting it fail confusingly at load.

---

## Speed on cloud CPU

There is no Intel iGPU on Render, so it falls back to CPU. Measured here:

| Backend | Per frame | FPS |
|---|---:|---:|
| OpenVINO iGPU *(local only)* | 14 ms | 71 |
| OpenVINO CPU | 25 ms | 40 |
| ONNX Runtime CPU (2 threads) | 40 ms | 25 |

Comfortably fast enough. Render's shared vCPU is slower than this machine, so
expect roughly 6–15 FPS — still above the 6 FPS the browser actually sends.

(An older note in `PERFORMANCE.md` claimed 498 ms / 2 FPS for CPU. That figure
was stale; the numbers above are current.)

---

## Things to know before you rely on this

**Camera frames now cross the internet.** Locally the browser sends ~6 FPS of
JPEG to `localhost`. Deployed, that is roughly **300 KB/s upstream per user**
to Render, plus round-trip latency on every frame. Pick a nearby region.

**Cost is not just the instance.** Every spoken turn, deep read and door sweep
is a billed Gemini call, and the key sits on the server, so anyone who can
reach your Vercel URL spends your credits. There is **no authentication** —
this was built as a single-user local app. Do not publish the URL widely
without adding auth first.

**Privacy scope changes.** Enrolled face embeddings and saved memories now live
on Render's disk rather than the user's own machine. Worth saying so plainly if
anyone else uses it.

**Rotate the key** you have been developing with before it goes into a
deployment environment.
