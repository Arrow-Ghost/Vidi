# Measured Performance

Every number here was measured on the target machine. Nothing is estimated.
Re-run any of it yourself with the commands shown.

## Test machine

| | |
|---|---|
| CPU | Intel Core i7-1165G7 @ 2.80GHz — 4 cores / 8 threads |
| GPU | Intel Iris Xe Graphics (integrated, no CUDA) |
| RAM | 15.7 GB |
| OS | Windows 11 Home 10.0.26200 |
| Python | 3.13.0 |

**Important caveat:** during all of these measurements the machine was under
heavy external load — CPU pegged near **100%** and only **~1.8 GB RAM free**
(36.9 GB committed) from other software (language servers, browsers, editors).
A quiet machine will do better on the CPU rows. The GPU rows are largely
unaffected, which is a large part of why the iGPU path was chosen.

## The decisive finding: inference runtime

Same YOLOv8n weights, same 480×480 input, same machine:

| Runtime | p50 latency | Max FPS | Notes |
|---|---:|---:|---|
| PyTorch 2.13.0+cpu | 3614.5 ms | 0.3 | Unusable. MKL-DNN present and enabled. |
| OpenVINO CPU | 498.7 ms | 2.0 | CPU is saturated by other software. |
| ONNX Runtime CPU, 4 threads | 126.6 ms | 7.9 | More threads = worse, under contention. |
| ONNX Runtime CPU, 2 threads | 55.3 ms | 18.1 | **Portable fallback.** |
| **OpenVINO — Intel iGPU** | **8.9 ms** | **112.0** | **Default.** p95 = 9.7 ms. |

Torch's CPU convolution path is pathologically slow on this build — a raw
forward pass alone (no pre/postprocessing) took 3614 ms, while a pure
`matmul` benchmark on the same install reached 57.6 GFLOPS. Convolution, not
the machine, is the bottleneck. **Torch is therefore never used for
inference**; it is used only for the one-time ONNX export.

The iGPU wins twice over: it is fastest, *and* it runs on a processor that is
otherwise idle while the CPU is contended. That is what keeps the voice
stream smooth.

Reproduce:

```bash
python -c "from app.perception.runtime import select_backend, benchmark; from pathlib import Path; print(benchmark(select_backend(Path('../models/weights/yolov8n.onnx'))))"
```

## End-to-end perception

Full path — JPEG decode → letterbox → inference → NMS → tracker → world model,
on a 640×480 webcam-size frame:

| Metric | Value |
|---|---:|
| p50 | **35.6 ms** |
| p95 | 38.9 ms |
| Sustainable rate | **28 FPS** |
| Pure inference within that | 10.3–15.0 ms |

The browser samples at **6 FPS**, roughly 4.5× below capacity, deliberately
leaving headroom for the audio stream sharing the same WebSocket.

## Voice (Gemini Live)

Model comparison, measured over the real API:

| Model | Setup | First audio chunk | Tool calling |
|---|---:|---:|---|
| **gemini-3.1-flash-live-preview** | 320 ms | **391 ms** | yes |
| gemini-2.5-flash-native-audio-latest | 308 ms | 890 ms | yes (emits thinking text) |

`gemini-3.1-flash-live-preview` is the default. Note it is **AUDIO-output
only** — it rejects `responseModalities: ["TEXT"]` outright — so input and
output transcription are enabled explicitly to drive the UI and trace.

Full conversational turns including a tool round trip, measured end to end
(user text in → complete spoken answer out):

| Scenario | Total | Audio |
|---|---:|---:|
| Scene description | 5.7 s | 21 chunks |
| Count people | 7.5 s | 28 chunks |
| Pronoun follow-up + 3 tool calls | 4.9 s | 17 chunks |
| Object not visible | 4.4 s | 18 chunks |

These totals measure until the *last* audio chunk. Perceived latency is far
lower — the user hears speech beginning at ~390 ms and audio plays gaplessly
from there.

## Depth (Depth Anything V2 Small, ViT-S)

Measured on the Intel iGPU. The model's graph is dynamic; input is pinned to a
static shape because recompiling per frame costs more than the inference.

| Input | p50 latency | Max FPS |
|---|---:|---:|
| **224x224** | **47.3 ms** | **21.2** |
| 308x308 | 107.0 ms | 9.3 |
| 392x392 | 204.7 ms | 4.9 |

224x224 is the default (and a multiple of 14, the ViT patch size). Depth costs
roughly 3x what detection does and changes far more slowly than object
positions, so it runs on **every 3rd frame** and the most recent map is reused
in between. Maps older than 1.5 s are rejected outright, so a stale map
degrades to the size prior rather than producing confidently wrong distances.

With depth enabled the full pipeline still measures **33-59 ms p50** end to
end, comfortably above the 6 FPS the browser actually samples at.

### Does depth actually help?

Yes, measurably. On bus.jpg the bbox-height size prior alone puts all three
people in one identical bucket ("nearby"). With depth fused in, they separate
correctly:

| Entity | Size prior alone | With depth | Depth percentile |
|---|---|---|---:|
| person_1 | nearby | **close** | 0.817 |
| person_2 | nearby | **close** | 0.706 |
| person_3 | nearby | **far** | 0.270 |
| bus | far | nearby | 0.529 |

The ordering is verified against ground truth a human can check by opening the
photograph: the people stand in front of the bus, and the nearest person's
depth percentile (0.817) is correctly higher than the bus's (0.529).

Polarity is asserted by test rather than assumed — the ground at the bottom of
a frame must read nearer than the sky at the top (measured 3.96 vs 0.69).
Getting that backwards would invert every distance the system reports.

## Memory and embeddings

| Operation | Result |
|---|---|
| Embedding model | gemini-embedding-001 @ 768 dims |
| Storage | SQLite, brute-force exact cosine |
| Semantic hit with no shared keywords | "where do I have my lecture" -> classroom |
| Offline behaviour | falls back to keyword search, never errors |

Reduced-dimension embeddings are not unit-normalized by the API, so the client
normalizes them itself; skipping that would silently degrade recall.

## Reading text and signboards

Two engines, because they solve different problems:

| Engine | Latency | Cost | Used for |
|---|---:|---|---|
| **RapidOCR (PP-OCRv6), local** | **934 ms** | free, offline | continuous sign spotting |
| **Gemini 3.1 Flash Lite** | ~3500 ms | API credits | meaning, summaries, on demand |

The local engine runs every 3 s against the latest frame, on its own lock and
in an executor, so a ~1 s OCR pass never stalls the 35 ms detection loop or
the voice stream.

Both were measured on a signboard fixture with known ground truth (EXIT on the
left, Room 204 / Science Lab on the right, WET FLOOR in the centre):

| Engine | Text read | Positions correct |
|---|---|---|
| RapidOCR | 4/4 exact | 4/4 |
| Gemini | 4/4 exact | 4/4, plus meaning and navigational flags |

On a real photograph (a Madrid bus) Gemini also read Spanish signage
correctly, including the route and destination, and flagged it navigational.

### Announcing signs without being annoying

Unprompted speech passes four independent gates: urgency (only warnings and
navigation qualify), novelty (a sign is announced once), cooldown (20 s, or
8 s for warnings), and quiet (nothing while the user or assistant is
mid-turn). Everything else is silent.

## Recognizing enrolled people and belongings

### Faces — OpenCV YuNet (232 KB) + SFace (37 MB)

Measured on a real photograph containing two different people:

| Case | Similarity | Verdict |
|---|---:|---|
| Same person, brighter | 0.900 | match |
| Same person, half size | 0.919 | match |
| Same person, blurred | 0.861 | match |
| Same person, JPEG q30 | 0.657 | match |
| **Different person, same photo** | **0.038** | correctly rejected |
| **Different person (another photo)** | **0.119** | correctly rejected |

Threshold is SFace's published same-identity value, 0.363, with a higher bar
of 0.45 before the assistant may state a name without hedging. The margin
between same-person (0.66+) and different-person (0.12-) is very large, which
is what makes false identification unlikely.

**Known limitation:** at −40 brightness YuNet stopped *detecting* one of the
two faces entirely. Detection, not recognition, is the fragile part in poor
light — and the correct behaviour there is "I can't see anyone's face", which
is what the system says.

### Belongings — MobileNetV2 pooled features, 1280-d, on the iGPU

Two calibration findings, both of which changed the implementation:

1. **Class logits are useless as an appearance descriptor.** Reading the
   model's 1000-d output put a bus and a person at **0.708** cosine —
   indistinguishable from a match. The 1280-d pooled feature layer is used
   instead, found by walking the ONNX graph for the classifier's input.

2. **A colour histogram made it measurably worse**, at every weight tried:

   | colour weight | separation margin |
   |---:|---:|
   | 0.00 | **0.110** |
   | 0.15 | 0.086 |
   | 0.25 | 0.059 |
   | 0.35 | 0.032 |
   | 0.45 | 0.003 |

   So colour fusion is not used.

Calibrated over real crops against brightness / blur / scale / JPEG variants
of themselves:

| Measurement | Value |
|---|---:|
| Same object, worst case | **0.857** |
| Different objects, best case | **0.747** |
| Usable margin | 0.110 |

Thresholds follow from that: **possible ≥ 0.80**, **confident ≥ 0.86**. An
earlier guess of 0.62 would have reported two completely different objects as
a match.

End-to-end through the real enrolment path (detect object → crop → embed):

| Condition | Similarity | Band |
|---|---:|---|
| Same scene | 1.000 | confident |
| Brighter | 0.977 | confident |
| Blurred | 0.949 | confident |
| Resized | 0.997 | confident |
| **Different scene** | **no match** | correctly empty |

Object class is a hard gate: a chair is never reported as the user's backpack
no matter how similar the pixels are.

## Perception scheduler (Part 60)

Nothing runs everything on every frame. Cadences are staggered so the total
per-frame budget stays well under the browser's 6 FPS sampling rate:

| Stage | Cadence | Cost |
|---|---|---:|
| Object detection | every frame | ~9–16 ms |
| Crowd analysis | every frame (reads existing tracks) | negligible |
| Depth | every 3rd frame | ~47 ms |
| Face recognition | every 2nd frame, **only when people are detected** | ~15–25 ms |
| OCR | every 3s, own lock | ~934 ms |
| Gemini vision / Live | on demand only | 0.4–3.5 s |

The detector, depth model and recognition engines are process-wide singletons.
An earlier version rebuilt the depth model per WebSocket connection — visible
in the logs as a repeated "depth model ready" line, costing ~1s and a duplicate
GPU context per browser tab. Now compiled once at startup.

## Personality changes real behaviour

Identical injected events, two profiles, measured through the running server:

| Event | Extrovert+Explorer (bar 0.43) | Introvert+Calm (bar 0.67) |
|---|---|---|
| `CROWD_APPEARED` | held | held |
| `FRIEND_DETECTED_IN_CROWD` | **SPOKE** | held: below_threshold |
| `PLACE_DETECTED` | held: cooldown | held: suggestions disabled |
| `OBSTACLE_DETECTED` (safety) | **SPOKE** | **SPOKE** |

Haptics fired alongside: `friend_triple [70,60,70,60,140]` and
`urgent_burst [120,60,120,60,120,60,200]`.

## Memory stability

250 consecutive real JPEG frames through the full pipeline:

| Metric | Value |
|---|---:|
| RSS growth | **−6.4 MB** (no growth) |
| p50 latency | 35.7 ms |
| Tracks retained | bounded |

Plus 5000 frames through the bounded queue with no consumer at all: queue
never exceeded 2 items, 4998 dropped, and the survivors are the *newest*
frames (latest-frame-wins), because a stale frame is worthless to a user who
has moved.

```bash
pytest tests/test_camera_soak.py -s
```

## Test suite

| Suite | Tests | Result |
|---|---:|---|
| Backend (pytest) | 202 | all pass |
| Frontend (vitest) | 5 | all pass |
| Frontend build + typecheck | — | clean, 0 npm vulnerabilities |

Two further tests are opt-in because they make real Gemini calls (live voice
and real semantic search):

```bash
SFCC_LIVE_TESTS=1 pytest tests/test_integration.py -k live
```

## Detection accuracy

The hand-rolled ONNX path (letterbox + class-aware NMS) is asserted to match
ultralytics' own reference output on a real photograph: identical detection
count, identical classes, and every box overlapping the reference at **>0.9
IoU**. This test is what makes the 400× speedup safe to rely on — a fast
detector with subtly wrong box scaling would tell a blind user an obstacle is
on the wrong side.

```bash
pytest tests/test_postprocess.py
```
