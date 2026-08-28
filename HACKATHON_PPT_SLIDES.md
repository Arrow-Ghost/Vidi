# Hackathon Final Presentation — SFCC
## Real-Time Assistive AI Companion for Visually Impaired Users

> **Timing Budget:** 7 Minutes Presentation + 3 Minutes Q&A (Total 10 Mins)  
> **Target Slide Count:** 6–7 Slides + Live Demo

---

## Slide 1 (Slide A) | Problem & Solution

### **Header:**
**SFCC: Smart Focal Companion & Copilot**  
*Empowering Independence for Visually Impaired Individuals with Real-Time Multimodal AI*

### **Team Information:**
* **Team Name:** [Your Team Name]
* **Members:** [Member 1, Member 2, Member 3, Member 4]
* **Track / Category:** Agentic AI / Multimodal AI / Accessibility

---

### **1. Problem Statement:**
* **Continuous Visual Deficit:** 285M+ visually impaired people globally struggle with real-time navigation, locating misplaced personal items, reading dynamic signage, and social interactions in public spaces.
* **Flaws of Existing Solutions:**
  1. **High Latency (3–6s):** Traditional chained AI pipelines (STT $\rightarrow$ LLM $\rightarrow$ TTS) are dangerously slow when walking.
  2. **AI Hallucinations:** Generic Vision-LLMs fabricate object locations and invent fake metric distances (e.g. *"table is 2.1 meters away"*).
  3. **Cognitive Overload & Chatter:** Existing apps narrate *every* visible pixel, overwhelming the user with audio clutter.
  4. **Fragmented & Clunky:** Requiring complex touchscreen gestures across 5 different apps for OCR, object finding, and navigation.

### **2. Who Faces This Problem?**
* Blind and low-vision individuals, visually impaired students and professionals, elderly with vision loss, and orientation & mobility (O&M) specialists.

### **3. Our Solution (The Pitch):**
> **"SFCC is a real-time, voice-first multimodal AI companion that continuously perceives 3D surroundings, manages spatial memory, and communicates naturally with sub-400ms audio latency, zero hallucinations, and privacy-first local edge perception."**

---

## Slide 2 (Slide B.1) | Agentic AI — Autonomous Perception & World Model

### **Header:**
**How the AI Agent Thinks & Operates**

### **1. What Does the AI Agent Do?**
* Operates as an autonomous cognitive loop: continuously converts raw camera & mic streams into a structured **3D Spatial World Model**, prioritizes events through an **Attention Manager**, and reasons over the environment using **Gemini Live**.

### **2. The Autonomous Agentic Loop: Goal $\rightarrow$ Think $\rightarrow$ Decide $\rightarrow$ Act $\rightarrow$ Result**

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      THE AGENTIC COGNITIVE LOOP                        │
 ├───────────┬────────────────────────────────────────────────────────────┤
 │ 🎯 GOAL   │ User asks: "Where is my water bottle?" OR a hazard appears │
 ├───────────┼────────────────────────────────────────────────────────────┤
 │ 🧠 THINK  │ Agent evaluates 3D World Model state & resolves pronouns   │
 ├───────────┼────────────────────────────────────────────────────────────┤
 │ ⚡ DECIDE │ Selects optimal grounded tools (find_object, depth_check)  │
 ├───────────┼────────────────────────────────────────────────────────────┤
 │ 🛠️ ACT    │ Executes local CV engine & semantic vector search          │
 ├───────────┼────────────────────────────────────────────────────────────┤
 │ 📢 RESULT │ Speaks grounded answer: "On your right, 2 steps away."     │
 └───────────┴────────────────────────────────────────────────────────────┘
```

### **3. What Happens Completely Autonomously?**
* **Continuous 28 FPS Object Tracking:** Maintains persistent object IDs and state changes (`ENTERED`, `MOVED`, `LEFT`).
* **4-Gate Proactive Hazard Alerting:** Autonomous unprompted alerts filtered by *Urgency*, *Novelty*, *Cooldown*, and *Turn Quietness* (Safety hazards bypass all filters).
* **Background Sign Spotting:** Local RapidOCR autonomously identifies exit signs, room numbers, and hazard tags every 3 seconds.
* **Tactile Haptic Dispatch:** Automatically dispatches 14 unique vibration rhythms to connected devices based on scene events.

---

## Slide 3 (Slide B.2) | Agentic AI — Tools & Grounded Function Registry

### **Header:**
**Grounded Tool Ecosystem (Zero Hallucination Architecture)**

### **1. The 14 Grounded Tools Used by the Agent:**
* `find_object` — Pinpoints specific items and gives 7-bucket relative directions.
* `get_spatial_summary` — Summarizes prioritized entities (People $\rightarrow$ Obstacles $\rightarrow$ Objects).
* `estimate_distance` — Returns qualitative distance (`close`, `nearby`, `far`) using fused depth.
* `read_signboard` — Combines local RapidOCR spotting with cloud semantic text reasoning.
* `identify_person` — Matches enrolled faces via YuNet+SFace cosine similarity (0.04 vs 0.36 threshold).
* `identify_belonging` — Matches personal items using MobileNetV2 1280-d pooled features.
* `check_crowd` — Evaluates crowd density, flow patterns, and approach velocity.
* `query_memory` — Performs semantic search over past observations via 768-d vector embeddings.
* `set_daily_task` & `verify_checklist` — Manages tasks and checks pre-departure readiness.

### **2. Strict Anti-Hallucination Guardrail:**
* **Structural Safety Policy:** The Gemini LLM has **no arbitrary code execution**. It can **only** narrate the structured JSON returned by deterministic local tools. If an object is not in the camera frame, the tool returns `not_currently_visible`—preventing fabricated answers.

---

## Slide 4 (Slide C) | How We Built It — Architecture & Tech Stack

### **Header:**
**End-to-End System Architecture**

### **1. Architecture Diagram:**

```
 [ Webcam / Mic ]
        │
        ▼ (Streaming PCM16 Audio + Bounded Frame Queue)
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      BACKEND: FASTAPI + ASYNCIO                        │
 ├──────────────────────────────────┬─────────────────────────────────────┤
 │   LOCAL EDGE PERCEPTION (iGPU)   │     GEMINI LIVE AGENT ENGINE        │
 │ • YOLOv8n (8.9ms on iGPU)        │ • Gemini 3.1 Flash Live (391ms)     │
 │ • Depth Anything V2 (47ms)       │ • Native Bidirectional Audio        │
 │ • RapidOCR PP-OCRv6 (934ms)      │ • Pronoun & Spatial Resolution      │
 │ • YuNet + SFace (Face Embeddings)│ • 14 Grounded Tool Registry         │
 ├──────────────────────────────────┴─────────────────────────────────────┤
 │       STRUCTURED 3D WORLD MODEL + SQLITE 768-d SEMANTIC MEMORY         │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │ (Gapless Audio + Haptic Patterns)
                                    ▼
           [ User Earphone (Voice) + Haptic Controller (Vibration) ]
```

### **2. Key Technologies & Why We Chose Them:**

| Layer | Technology | Architectural Rationale ("Why") |
|---|---|---|
| **Voice & Reasoning** | **Google Gemini Live** (`gemini-3.1-flash-live-preview`) | **391ms latency to first audio chunk.** Native audio-to-audio streaming removes STT/TTS latency and supports real-time barge-in. |
| **Inference Runtime** | **OpenVINO (Intel Iris Xe iGPU)** | **400× speedup over PyTorch CPU (8.9ms vs 3614ms)**. Offloads vision models to iGPU, leaving CPU free for audio streams. |
| **Edge Vision** | **YOLOv8n + Depth Anything V2** | 28 FPS local object detection and relative monocular depth without cloud video streaming. |
| **Local OCR** | **RapidOCR (PP-OCRv6)** | Zero-cost, 934ms offline sign reading running in a background thread. |
| **Backend & WS** | **FastAPI + WebSockets + Pydantic** | Single asynchronous event loop handling audio, vision, and tool dispatch concurrently. |
| **Memory & Storage** | **SQLite + `gemini-embedding-001` (768-d)** | Embedded, persistent semantic memory with exact cosine matching and offline keyword fallback. |
| **Frontend UI** | **Astro + TypeScript + Web Audio Worklets** | Zero-bundle bloat, raw 16kHz PCM16 audio capture, and live spatial debug dashboard. |

---

## Slide 5 (Slide D) | Working Demo & Measured Impact

### **Header:**
**Live Demonstration & Empirical Performance**

### **1. Live Demo Flow (2.5 – 3 Minutes):**
1. **Scene Understanding & Pronoun Resolution:**
   * *Action:* User points camera at room $\rightarrow$ Asks *"What's in front of me?"* $\rightarrow$ Agent replies in ~390ms with 7-bucket spatial positions. Follow-up: *"How far is the chair on my left?"* $\rightarrow$ Correctly resolves context.
2. **Finding Misplaced Belongings:**
   * *Action:* User asks *"Where is my bag?"* $\rightarrow$ Agent scans scene, matches enrolled MobileNetV2 embedding, and guides user.
3. **Signboard Reading & Safety Barge-In:**
   * *Action:* Camera detects a hazard sign $\rightarrow$ Proactive haptic alert fires $\rightarrow$ User interrupts agent mid-speech (*barge-in*), and audio cuts off immediately (<50ms).

### **2. Measured Performance (Real Tested Benchmarks on Hardware):**

| Benchmark Metric | Measured Result | Impact / Significance |
|---|---|---|
| **YOLOv8n Detection Latency** | **8.9 ms** (OpenVINO iGPU) | Sustainable 28 FPS on standard laptop without discrete GPU |
| **Voice Time-to-First-Audio** | **391 ms** (Gemini Live) | Natural conversational flow without awkward multi-second pauses |
| **Depth Estimation Latency** | **47.3 ms** (Depth Anything V2) | Real-time relative distance ordering without fake metric claims |
| **Face Separation Margin** | **0.038 vs 0.900** (YuNet+SFace) | Zero false-positive matches for strangers (threshold: 0.36) |
| **Memory Leak Soak Test** | **0 MB Growth (5,000 frames)** | Bounded Latest-Frame Queue prevents device crashes |
| **Test Suite Coverage** | **207 Passed Tests** | 202 backend + 5 frontend automated unit/integration tests |

---

## Slide 6 (Slide E) | Why Our Solution? (Innovation & Impact)

### **Header:**
**Problem $\rightarrow$ Innovation $\rightarrow$ Impact**

### **1. What Makes Our Solution Radically Different?**
* **Separation of Perception from Reasoning:** Unlike generic Vision-LLMs that hallucinate images, SFCC's LLM *only narrates deterministic sensor data*.
* **No Fake Metric Claims:** SFCC never guesses "2.4 meters"; it computes qualitative depth ranges and honestly reports `UNKNOWN` when sensors disagree.
* **Privacy-by-Design:** No raw video or photos are saved or uploaded. Only irreversible mathematical embeddings (vectors) exist locally.
* **Zero-Lag Native Audio:** Sub-400ms response with instantaneous barge-in interruption.

### **2. Biggest Hackathon Achievements:**
* 🚀 **Engineered an 8.9ms Edge Vision Pipeline:** Converted YOLOv8n and Depth Anything V2 to OpenVINO iGPU, unlocking a 400× speedup over PyTorch CPU.
* 🎙️ **Full Gemini Live Bi-directional WebSocket Integration:** Combined streaming 16kHz PCM16 audio worklets with 14 function-calling tools.
* 🛡️ **207 Automated Tests:** Built with strict production engineering standards, including soak testing and ONNX mathematical parity checks.

### **3. Future Roadmap (What We Build Next):**
* **Wearable Smart Glasses Form Factor:** Porting client to ESP32-Cam / smart glasses hardware with bone-conduction audio.
* **Continuous Indoor Wayfinding:** Topological step-by-step navigation combining signboard OCR landmarks and step counting.
* **Fall & Emergency Caregiver Beaconing:** Automated emergency detection with geolocation dispatch.

---

## Slide 7 (Slide F) | Summary & Q&A

### **Header:**
**Empowering Independence Through Trustworthy AI**

```
┌────────────────────────────────────────────────────────────────────────┐
│                              KEY TAKEAWAY                              │
│                                                                        │
│   SFCC transforms AI from a slow, hallucinating gimmick into a safe,   │
│     real-time, voice-first spatial copilot for visually impaired       │
│                             individuals.                               │
└────────────────────────────────────────────────────────────────────────┘
```

* **Demo Video / GitHub:** [Repository Link]
* **Thank You!**
* **Open for Questions & Answers (3 Minutes)**
