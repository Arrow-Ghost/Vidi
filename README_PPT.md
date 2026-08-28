# SFCC (Smart Focal Companion & Copilot) — Presentation & Pitch Deck Input Guide

> **A real-time, voice-first multimodal AI companion for visually impaired individuals that continuously perceives their environment (objects, people, obstacles, signboards, depth, crowd dynamics) with sub-400ms latency, zero hallucinations, and strict privacy by design.**

---

## 1. Executive Summary & Pitch Hook

* **The Vision:** Visually impaired individuals navigate a world built for sight. Existing assistive tools are fragmented, clunky, slow, or dangerously prone to AI hallucinations.
* **The Product:** SFCC is an intelligent, voice-first wearable/mobile companion. It continuously perceives the visual world through a camera and microphone, maintains an active real-time 3D spatial world model, and interacts seamlessly through low-latency conversational audio.
* **Core Philosophy:** **Separation of Perception from Reasoning.** Perception runs through deterministic, calibrated local vision engines (YOLO, Depth Anything, YuNet, RapidOCR); reasoning runs through Google Gemini Live. The AI never guesses what is in front of the user—it only articulates verified, grounded sensor data.

---

## 2. The Problem Statement

### A. The Real-World Challenges for Visually Impaired Users
1. **Loss of Continuous Situational Awareness:** Navigating unfamiliar indoor/outdoor spaces, identifying obstacles at head or waist height, and tracking moving crowds requires constant mental effort.
2. **Finding Lost or Misplaced Belongings:** Everyday items (keys, backpack, water bottle, cane) are frequently misplaced, causing anxiety and loss of independence.
3. **Social Isolation in Public & Gatherings:** Inability to know who is nearby, whether a friend is waving or approaching, or if a room is crowded versus empty.
4. **Information Accessibility Barriers:** Dynamic signage, room numbers, exit signs, transit directions, and hazard warnings are inaccessible without asking strangers.
5. **Cognitive Overload from Bad Design:** Screen-based mobile apps require complex touch gestures, continuous tapping, and navigating multi-tiered menus that defeat the purpose of an eyes-free assistant.

### B. Why Existing AI & Assistive Solutions Fail
* **Generic Multimodal AI Hallucinations:** Asking standard Vision-LLMs "What is in front of me?" leads to fabricated descriptions, invented metric distances (e.g., claiming a table is "2.3 meters away" without calibration), or dangerous omissions.
* **Crippling Latency (2–6 seconds):** Traditional pipelines (Record Audio → STT → LLM Reasoning → Cloud TTS) take seconds to respond—unusable when walking towards an obstacle or holding a natural conversation.
* **Relentless "Chatterbox" Narration:** Basic assistive tools describe *everything* in the camera frame, creating overwhelming audio clutter rather than filtering for urgency and relevance.
* **Privacy Intrusions:** Cloud-based assistive apps stream raw video and faces to third-party servers, compromising both user privacy and the privacy of bystanders.
* **High Compute / Battery Demands:** Running heavy deep-learning models locally drains batteries and overheats devices without dedicated GPUs.

---

## 3. Core Features & Capabilities

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             SFCC CORE FEATURES                              │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│   SPATIAL PERCEPTION    │   VOICE CONVERSATION    │    PROACTIVE & SOCIAL   │
│ • Real-Time Detection   │ • Sub-400ms Gemini Live │ • Enrolled Friends/Items│
│ • 7-Bucket Direction    │ • Real-time Barge-In    │ • Crowd Flow & Velocity │
│ • Qualitative Depth     │ • Grounded 14 Tools     │ • 18-Trait Personality  │
│ • Local + Cloud OCR     │ • Pronoun Resolution    │ • Multi-modal Haptics   │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

### 1. Continuous Real-Time Object Perception & Tracking
* **28 FPS local inference** running on lightweight hardware using YOLOv8n and an IoU-based tracker with persistent entity IDs.
* Tracks object state transitions: `ENTERED`, `UPDATED`, `MOVED`, `LOST`, `LEFT`.
* **Bounded Latest-Frame Queue:** Zero memory backlog; drops stale frames so the user only receives fresh real-time information.

### 2. Spatial Direction Engine with Hysteresis
* Quantizes visual field into **7 directional buckets** (`far_left`, `left`, `center_left`, `center`, `center_right`, `right`, `far_right`).
* **Hysteresis Smoothing:** Prevents boundary flickering when an object hovers between two regions (e.g., oscillating between left and center).

### 3. Fused Qualitative Depth & Distance Estimation
* Uses **Depth Anything V2 Small** on the integrated GPU fused with physical bounding-box height priors.
* **Honest Safety Constraint:** Deliberately outputs qualitative ranges (`close`, `nearby`, `far`) and refuses to emit fake metric decimal precision.
* If depth prediction and geometric priors grossly disagree, it safely reports `UNKNOWN` rather than guessing.

### 4. Natural Voice-First Conversational Agent (Gemini Live)
* Native bidirectional audio streaming over WebSockets with **~390ms latency to first audio chunk**.
* **Instant Barge-In (Interruption):** Immediately stops talking when the user speaks or when a critical safety hazard appears.
* **Conversational Context & Pronoun Resolution:** Naturally handles follow-up queries like *"How far is it?"*, *"What color is the bag on my left?"*, or *"Is he looking at me?"*.

### 5. Dual-Tier OCR & Signboard Reading
* **Tier 1 (Fast & Offline):** Local RapidOCR (PP-OCRv6, 934ms) runs periodically in the background to detect room numbers, exits, and hazard signs without internet or API cost.
* **Tier 2 (Deep Semantic Cloud):** Gemini 3.1 Flash Lite handles complex text interpretation, summaries, translation, and navigation instructions on demand.

### 6. Enrolled Friends & Belongings Recognition (Privacy-First)
* **Face Matching:** OpenCV YuNet + SFace embeds enrolled faces into mathematical vectors. Strangers are counted but never named or described.
* **Belongings Matching:** MobileNetV2 1280-d pooled feature extraction with hard object-class gating (a chair is never matched to a backpack).
* Zero raw images are saved; only irreversible mathematical embeddings reside in local storage.

### 7. Crowd Dynamics & Flow Analysis
* Analyzes crowd density, relative flow direction (unidirectional, bidirectional, converging, chaotic), and approach velocity.
* Detects if people are approaching the user, helping navigate busy corridors, transit hubs, and events.

### 8. 18-Trait Personality Engine & Proactive Attention Manager
* Customizes the assistant's behavior via an 18-trait vector (introvert/extrovert, concise/detailed, proactive/passive).
* **4-Gate Proactive Filter:** Unprompted speech requires passing **Urgency**, **Novelty**, **Cooldown**, and **Quietness** gates to eliminate annoying audio clutter.
* Safety alerts always bypass cooldowns and quiet modes.

### 9. Multi-Modal Haptic Feedback System
* Translates visual events into **14 discrete haptic vibration rhythms** (hazard alerts, friend nearby, navigation pulses).
* Single-motor devices provide rhythm cues without misleading directional claims.

### 10. Long-Term Semantic Memory & Daily Preparation
* Built on SQLite with 768-dimensional vector embeddings for exact cosine semantic search.
* Helps recall where items were last seen (*"Where did I leave my keys?"*), manages daily tasks, and verifies checklists before leaving home.

---

## 4. Technology Stack & Architectural Rationale

| Layer / Component | Technology Chosen | Why We Chose It (Engineering Rationale) |
|---|---|---|
| **Voice & Multimodal Reasoning** | **Google Gemini Live** (`gemini-3.1-flash-live-preview`) | **Sub-400ms time-to-first-audio chunk.** Native audio-to-audio bidirectional streaming eliminates the latency of chaining separate STT, LLM, and TTS models. Native function calling enforces deterministic tool execution. |
| **Backend Framework** | **Python 3.11+ / FastAPI / WebSockets / asyncio** | Provides a single high-performance asynchronous event loop managing bidirectional PCM16 audio, camera frame acknowledgments, and tool orchestration over a single connection per client with strict Pydantic schema validation. |
| **Inference Acceleration** | **OpenVINO (Intel Iris Xe iGPU) + ONNX Runtime** | Delivers a **400× speedup over PyTorch CPU (8.9ms vs 3614ms for YOLO)**. Offloads vision models entirely to the integrated GPU, leaving 100% of CPU cores free for audio streaming, WebSocket handling, and logic. |
| **Object Detection & Tracking** | **YOLOv8n (ONNX) + Custom IoU Tracker** | Lightweight (3.2M params), runs at **28+ FPS locally**. Eliminates cloud latency and bandwidth costs while ensuring user camera feeds never leave the local device for basic perception. |
| **Depth Estimation** | **Depth Anything V2 Small (ViT-S)** | State-of-the-art monocular depth estimation running at 47ms on iGPU. Fused with bounding-box size priors to provide reliable qualitative spatial ordering. |
| **Local OCR** | **RapidOCR (PP-OCRv6)** | Runs locally in 934ms on CPU. Enables offline continuous signboard and hazard detection without consuming Gemini API tokens or requiring constant internet access. |
| **Face & Feature Embeddings** | **OpenCV YuNet + SFace / MobileNetV2 (1280-d)** | Ultra-lightweight (YuNet is 232 KB; SFace is 37 MB). Delivers massive identity separation margin (0.04 stranger vs 0.90 enrolled) without heavy CUDA dependencies. |
| **Frontend UI & Audio Worklets** | **Astro + TypeScript + Web Audio API** | Zero-bundle-overhead static delivery. Custom `AudioWorklet` captures streaming 16kHz PCM16 microphone audio and provides gapless audio playback with instant `.cancel()` barge-in. |
| **Storage & Semantic Memory** | **SQLite + `gemini-embedding-001` (768-d)** | Zero-maintenance embedded database. Exact cosine search enables instant semantic recall without needing heavy external vector databases, with automatic keyword fallback when offline. |
| **Haptics Abstraction** | **Navigator Vibration API / Gamepad API** | Provides a standardized tactile communication layer across mobile browsers and external wearable controllers without native app dependencies. |

---

## 5. Architectural Principles & Safety Guardrails

```
 ┌──────────────────────┐      ┌───────────────────────────────┐
 │ Camera & Microphone  │ ───► │ Local Edge Perception (iGPU) │
 └──────────────────────┘      │ (YOLO, Depth, OCR, YuNet)     │
                                └──────────────┬────────────────┘
                                               │ Structured World State
                                               ▼
 ┌──────────────────────┐      ┌───────────────────────────────┐
 │ Voice Response (TTS) │ ◄─── │ Gemini Live Orchestrator      │
 │ (Spoken Grounded Fact│      │ (14 Grounded Tools, No Code)  │
 └──────────────────────┘      └───────────────────────────────┘
```

1. **Separation of Perception from Reasoning:**
   * The LLM never invents visual facts. It can only narrate data returned by verified, deterministic local vision tools.
2. **Honest Field-of-View Handling:**
   * When an item is outside the camera angle, SFCC says *"not currently visible"*, never *"it is not in the room"*.
3. **No Fabricated Precision:**
   * Distances are given as qualitative zones (*"just within reach"*, *"a few steps ahead"*, *"far across the room"*), preventing dangerous missteps caused by pseudo-precise metrics.
4. **Privacy-by-Design:**
   * Continuous raw camera video is never saved to disk or uploaded to clouds. Only mathematical feature vectors for explicitly enrolled items/faces are stored locally.

---

## 6. Slide-by-Slide PPT Outline

### Slide 1: Title & Vision
* **Title:** SFCC — The Real-Time Assistive AI Companion for the Visually Impaired
* **Subtitle:** Bringing continuous spatial awareness, instant voice interaction, and independent mobility.
* **Presenter Info / Date**

### Slide 2: The Problem Space
* **Headline:** Visual Impairment & The Limitations of Modern Assistive Tech
* **Key Pain Points:** Fragmented single-purpose apps, high latency, hallucination risks in general-purpose AI, cognitive overload from relentless narration, and privacy vulnerabilities.

### Slide 3: The Solution — SFCC
* **Headline:** An Integrated, Voice-First Spatial Copilot
* **Core Pillars:** Continuous Local Perception + Grounded Multimodal LLM Reasoning + Tactile Haptics.
* **Hero Statistic:** Sub-400ms voice response, 28 FPS real-time local vision, 100% grounded facts.

### Slide 4: Real-Time Perception & World Modeling
* **Headline:** Seeing the World at 28 FPS on Integrated Hardware
* **Visuals / Architecture:** Camera → YOLOv8n + Depth Anything V2 → 7-Bucket Direction + World Model.
* **Key Highlight:** 400× speedup using OpenVINO iGPU acceleration (8.9ms latency).

### Slide 5: Natural Conversational Intelligence
* **Headline:** Zero-Lag Audio Interaction with Gemini Live
* **Key Features:** Streaming PCM16 WebSockets, real-time barge-in, pronoun & spatial reference resolution (*"What is that on my right?"*).

### Slide 6: Social Awareness & Personal Belongings
* **Headline:** Recognizing Friends, Finding Items, & Navigating Crowds
* **Key Features:** YuNet + SFace face recognition, MobileNetV2 belonging matching, crowd density and approach detection.
* **Privacy Promise:** Math embeddings only; zero photo storage.

### Slide 7: Intelligent Assistance without the Noise
* **Headline:** Adaptive Personality & Proactive Attention
* **Key Features:** 18-trait customizable vector, 4-gate proactive speech filter, urgent safety overrides, multi-modal haptic rhythms.

### Slide 8: Technology Stack & Why It Matters
* **Headline:** Engineered for Edge Performance & Reliability
* **Stack Grid:** Gemini Live (Voice/Reasoning), OpenVINO/iGPU (Vision), FastAPI/WebSockets (Async Core), Astro/TypeScript (Lightweight Client), SQLite + Vector Search (Memory).

### Slide 9: Measurable Performance & Benchmarks
* **Headline:** Tested, Calibrated, and Verified
* **Metrics Table:**
  * Object Detection: **8.9ms** (OpenVINO iGPU)
  * Voice Audio Latency: **391ms** (Gemini Live)
  * Memory Stability: **0 MB memory leak over 5,000 frames**
  * Depth Latency: **47ms** (Depth Anything V2)

### Slide 10: Summary & Next Steps
* **Headline:** Empowering Independence Through Trustworthy AI
* **Roadmap:** Mobile wearable integration, outdoor/indoor continuous wayfinding, expanded low-power edge accelerators.
