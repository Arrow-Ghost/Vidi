"""
Vidi backend — the single multiplexed real-time endpoint.

One WebSocket per client carries everything: sampled camera frames up,
microphone audio up, and perception updates + assistant audio down. Using a
single connection (rather than one per modality) is what keeps the camera and
the voice turn in the same ordering domain — the agent's answer is always
about the scene the user was actually looking at.

Message types
  client -> server : frame | audio | audio_end | text | start_voice | stop_voice
  server -> client : health | frame_ack | perception | tool_call | tool_result
                     user_transcript | agent_transcript | agent_audio
                     agent_turn_complete | voice_interrupted | voice_error
                     voice_disconnected | error
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.agent.live_session import LiveVoiceSession
from app.agent.attention import AttentionManager
from app.agent.proactive import ProactiveAnnouncer
from app.agent.recommendations import RecommendationEngine
from app.agent.tools import ToolRegistry
from app.agent.vision_llm import VisionReader
from app.core.config import get_settings
from app.memory.embeddings import EmbeddingClient
from app.memory.store import MemoryStore
from app.perception.object_detector import ObjectDetector
from app.perception.pipeline import FrameDecodeError, PerceptionPipeline, decode_frame
from app.events.bus import EventBus, EventType, Severity, WorldEvent
from app.haptics.engine import HapticEngine, HapticSettings, haptic_for_event
from app.perception.recognition import RecognitionEngine
from app.profile.personality import PersonalityEngine
from app.perception.runtime import benchmark
from app.perception.tracker import ObjectTracker
from app.perception.world_model import WorldModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vidi.backend")

settings = get_settings()

# The detector holds a compiled model on the iGPU and is expensive to build
# (~1s) but completely stateless across calls, so ONE instance is shared by
# every connection. Tracker and WorldModel are per-connection, because they
# carry that user's scene state.
_shared_detector: Optional[ObjectDetector] = None
_startup_benchmark: Dict[str, Any] = {}
# Memory is process-wide and persistent on disk: what the user asked to
# remember must survive both a reconnect and a restart.
_memory_store: Optional[MemoryStore] = None
_vision_reader: Optional[VisionReader] = None
# Face + appearance models are stateless and expensive to build, so one
# engine is shared; the enrolled-template cache lives inside it.
_recognition: Optional[RecognitionEngine] = None
# Like the detector, the depth model holds a compiled iGPU graph and is
# stateless across calls. Building one per connection cost ~1s of startup
# and a duplicate GPU context for every browser tab.
_shared_depth = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _shared_detector, _startup_benchmark, _memory_store, _vision_reader
    global _recognition, _shared_depth
    _vision_reader = VisionReader(settings.gemini_api_key)
    _memory_store = MemoryStore(
        embeddings=EmbeddingClient(settings.gemini_api_key)
    )
    logger.info("memory store ready at %s", _memory_store.db_path)
    try:
        _recognition = RecognitionEngine(store=_memory_store)
        logger.info(
            "recognition ready: faces=%s belongings=%s",
            _recognition.faces_available, _recognition.belongings_available,
        )
    except Exception:
        logger.exception("recognition failed to initialize")
        _recognition = None
    try:
        loop = asyncio.get_running_loop()
        _shared_detector = await loop.run_in_executor(None, ObjectDetector)
        _startup_benchmark = await loop.run_in_executor(
            None, benchmark, _shared_detector.backend
        )
        logger.info("perception ready: %s", _startup_benchmark)
        from app.perception.depth import DepthEstimator

        _shared_depth = await loop.run_in_executor(None, DepthEstimator)
    except Exception:
        # The app must still start so /health can REPORT the failure rather
        # than the process dying silently — degrade loudly, not fatally.
        logger.exception("perception failed to initialize; vision will be unavailable")
        _shared_detector = None
    yield
    _shared_detector = None
    if _memory_store is not None:
        _memory_store.close()
        # Clearing the reference matters: a connection opened after shutdown
        # would otherwise query a closed database and fail the whole session.
        _memory_store = None


app = FastAPI(title="Vidi Assistive Agent Backend", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tools that change how the assistant behaves; after any of these the live
# session's derived behaviour must be reloaded.
_PROFILE_TOOLS = frozenset({
    "set_name", "set_personality", "set_communication", "set_preference",
    "reset_profile",
})

FRAME_QUEUE_MAXSIZE = 2
MAX_MESSAGE_BYTES = 4 * 1024 * 1024  # a 4MB JPEG/audio chunk is already absurd


@app.get("/health")
async def health() -> dict:
    """Health + real measured perception numbers. Never reports a secret."""
    return {
        "status": "ok",
        "service": "vidi-backend",
        "gemini_configured": settings.gemini_configured,
        "vision_available": _shared_detector is not None,
        "vision_backend": _shared_detector.backend.name if _shared_detector else None,
        "vision_benchmark": _startup_benchmark or None,
        "memory_available": _memory_store is not None,
        "semantic_search": bool(_memory_store and _memory_store.embeddings.available),
        "deep_reading": bool(_vision_reader and _vision_reader.available),
        "face_recognition": bool(_recognition and _recognition.faces_available),
        "belonging_recognition": bool(_recognition and _recognition.belongings_available),
        "personality": True,
        "crowd_analysis": True,
        "haptics": True,
        "door_detection": settings.gemini_configured,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class ClientSession:
    """Per-connection state: this user's scene, tools, and voice session."""

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self.world = WorldModel()
        self.pipeline: Optional[PerceptionPipeline] = None
        # The profile is loaded per connection so a change made in one tab is
        # picked up by the next session without a restart.
        self.personality = PersonalityEngine(self._load_profile_safely())
        self.events = EventBus()
        self.attention = AttentionManager(self.personality)
        self.recommendations = RecommendationEngine(self.personality)
        self.haptics = HapticEngine(HapticSettings())
        self.haptics.apply_preferences(self.personality.profile.preferences)

        if _shared_detector is not None:
            self.pipeline = PerceptionPipeline(
                detector=_shared_detector,
                tracker=ObjectTracker(),
                world_model=self.world,
                recognition=_recognition,
                depth=_shared_depth,
                door_api_key=settings.gemini_api_key,
            )
        self.registry = ToolRegistry(
            self.world,
            store=_memory_store,
            pipeline=self.pipeline,
            vision=_vision_reader,
            recognition=_recognition,
            personality=self.personality,
            recommendations=self.recommendations,
        )
        self.announcer = ProactiveAnnouncer()
        self._ocr_task: Optional[asyncio.Task] = None
        self._ocr_ticks = 0
        self.agent_is_speaking = False
        self.voice: Optional[LiveVoiceSession] = None
        self.frames_received = 0
        self.frames_dropped = 0
        self._busy = False  # latest-frame-wins: drop while one is in flight
        self._send_lock = asyncio.Lock()

    @staticmethod
    def _load_profile_safely():
        """
        A profile that cannot be read must not take the connection down. The
        assistant still works with default behaviour; only personalisation is
        lost, and that is reported rather than hidden (Part 63).
        """
        if _memory_store is None:
            return None
        try:
            return _memory_store.load_profile()
        except Exception:  # noqa: BLE001
            logger.exception("could not load user profile; using defaults")
            return None

    async def send(self, payload: Dict[str, Any]) -> None:
        # Serialized because both the perception task and the voice receive
        # loop write to the same socket concurrently.
        async with self._send_lock:
            try:
                await self.ws.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                pass

    # ---------------- frames ----------------

    async def handle_frame(self, message: Dict[str, Any]) -> None:
        self.frames_received += 1
        seq = message.get("seq")

        if self.pipeline is None:
            await self.send({"type": "frame_ack", "seq": seq, "vision_available": False})
            return

        # Backpressure: if inference is still running, drop this frame rather
        # than queue it. A stale frame is worthless to a moving user.
        if self._busy:
            self.frames_dropped += 1
            await self.send(
                {"type": "frame_ack", "seq": seq, "dropped": True,
                 "frames_dropped": self.frames_dropped}
            )
            return

        self._busy = True
        try:
            result = await self.pipeline.process_payload(message.get("data", ""), seq)
        except FrameDecodeError as exc:
            await self.send({"type": "error", "scope": "frame", "message": str(exc), "seq": seq})
            return
        finally:
            self._busy = False

        # Crowd + friend transitions become world events, scored by the
        # attention manager rather than spoken directly.
        await self._publish_perception_events(result)

        await self.send(
            {
                "type": "perception",
                "seq": result.seq,
                "latency_ms": result.latency_ms,
                "depth_ms": result.depth_ms,
                "depth_age_s": result.depth_age_s,
                "image": {"width": result.image_width, "height": result.image_height},
                "entities": [
                    {
                        "id": e.id,
                        "type": e.type,
                        "direction": e.direction.bucket.value,
                        "direction_confidence": round(e.direction.confidence, 2),
                        "distance_range": e.distance.range.value,
                        "distance_confidence": round(e.distance.confidence, 2),
                        "confidence": round(e.confidence, 2),
                        "is_person": e.is_person,
                    }
                    for e in result.entities
                ],
                "boxes": [
                    {
                        "class": d.class_name,
                        "confidence": round(d.confidence, 2),
                        "x1": round(d.box.x1, 1), "y1": round(d.box.y1, 1),
                        "x2": round(d.box.x2, 1), "y2": round(d.box.y2, 1),
                    }
                    for d in result.detections
                ],
                "events": [
                    {"event": ev.event.value, "id": ev.track_id, "class": ev.class_name}
                    for ev in result.events
                ],
                "stats": self.pipeline.stats.snapshot(),
                "crowd": result.crowd.payload() if result.crowd else None,
                "obstacles": [o.payload() for o in getattr(result, "obstacle_alerts", [])],
                "doors": [d.payload() for d in self.pipeline.door_tracker.current()],
                "friends": [f.payload() for f in self.pipeline.friend_tracker.visible()],
                "faces_seen": result.faces_seen,
                "face_ms": result.face_ms,
                "frames_dropped": self.frames_dropped,
            }
        )

    # ---------------- signboard watching (Segments 15, 28) ----------------

    # OCR costs ~934ms, so it runs far slower than detection. 3s is frequent
    # enough to catch a sign while walking past it, without pinning a core.
    OCR_INTERVAL_S = 3.0
    # Doors cost a Gemini call each, so they sweep far more slowly than OCR
    # and only while the camera is actually delivering frames.
    DOOR_INTERVAL_S = 12.0

    async def _ocr_loop(self) -> None:
        """
        Periodically read text and offer genuinely new signage to the
        announcer. Runs for the life of the voice session.
        """
        try:
            while True:
                await asyncio.sleep(self.OCR_INTERVAL_S)
                if self.pipeline is None or not self.pipeline.ocr_available:
                    continue
                if self.pipeline.current_frame() is None:
                    continue

                try:
                    regions, new_signs = await self.pipeline.scan_text()
                except Exception:  # noqa: BLE001 - a bad pass must not kill the loop
                    logger.exception("OCR pass failed")
                    continue

                # Door sweep on its own slower cadence, sharing this loop.
                self._ocr_ticks += 1
                if (
                    self.pipeline.doors_available
                    and self._ocr_ticks % max(1, int(self.DOOR_INTERVAL_S / self.OCR_INTERVAL_S)) == 0
                ):
                    try:
                        _doors, door_events = await self.pipeline.scan_doors()
                        for _event_type, door, spoken in door_events:
                            await self.dispatch_event(WorldEvent(
                                type=EventType.OBJECT_DETECTED,
                                severity=Severity.NAVIGATION if door.is_exit
                                else Severity.ENVIRONMENT,
                                key="door:" + door.key + ":" + door.state.value,
                                summary=spoken,
                                confidence=door.confidence,
                                data={"door": door.payload()},
                            ))
                    except Exception:  # noqa: BLE001
                        logger.exception("door sweep failed")

                if regions:
                    await self.send(
                        {
                            "type": "signs",
                            "ocr_ms": self.pipeline.last_ocr_ms,
                            "regions": [r.payload() for r in regions],
                            "tracked": [s.payload() for s in self.pipeline.sign_tracker.current()],
                        }
                    )

                if not new_signs:
                    continue

                chosen = self.announcer.select(
                    new_signs, user_is_busy=self.agent_is_speaking
                )
                if not chosen:
                    continue

                self.pipeline.sign_tracker.mark_announced(chosen)
                self.announcer.mark_announced()
                await self.send(
                    {
                        "type": "proactive_announcement",
                        "signs": [s.payload() for s in chosen],
                    }
                )
                if self.voice is not None and self.voice.connected:
                    await self.voice.announce(self.announcer.compose(chosen))
        except asyncio.CancelledError:
            raise

    def start_sign_watch(self) -> None:
        if self._ocr_task is None or self._ocr_task.done():
            self._ocr_task = asyncio.create_task(self._ocr_loop())

    async def stop_sign_watch(self) -> None:
        if self._ocr_task is not None:
            self._ocr_task.cancel()
            try:
                await self._ocr_task
            except asyncio.CancelledError:
                pass
            self._ocr_task = None

    # ---------------- world events -> speech/haptics (Parts 36, 48, 49) ----------------

    async def _publish_perception_events(self, result) -> None:
        # Obstacles first, and at SAFETY severity: AttentionManager routes
        # these past every cooldown, preference and quiet-mode setting.
        for obstacle in getattr(result, "obstacle_alerts", []):
            await self.dispatch_event(WorldEvent(
                type=EventType.OBSTACLE_DETECTED,
                severity=Severity.SAFETY if obstacle.urgent else Severity.NAVIGATION,
                key="obstacle:" + obstacle.entity_id,
                summary=obstacle.describe(),
                confidence=obstacle.detection_confidence,
                data={"obstacle": obstacle.payload()},
            ))

        for event_type, summary in result.crowd_events:
            await self.dispatch_event(WorldEvent(
                type=event_type,
                severity=(
                    Severity.ENVIRONMENT
                    if event_type != EventType.CROWD_APPROACHING else Severity.NAVIGATION
                ),
                key="crowd:" + event_type.value,
                summary=summary,
                confidence=result.crowd.confidence if result.crowd else 0.6,
                data={"crowd": result.crowd.payload() if result.crowd else {}},
            ))

        for event_type, friend, summary in result.friend_events:
            await self.dispatch_event(WorldEvent(
                type=event_type,
                severity=Severity.SOCIAL,
                key="friend:" + friend.name + ":" + event_type.value,
                summary=summary,
                confidence=friend.identity_confidence,
                data={"friend": friend.payload(), "subject": friend.name},
            ))

    async def dispatch_event(self, event: WorldEvent) -> None:
        """
        One funnel for everything that might interrupt the user: score it, and
        only then speak and/or buzz. Deterministic by design (Part 49).
        """
        await self.events.publish(event)
        decision = self.attention.evaluate(event, agent_is_busy=self.agent_is_speaking)

        await self.send({
            "type": "attention_decision",
            **decision.payload(),
        })
        if not decision.speak:
            return

        if decision.haptic:
            haptic_event = haptic_for_event(event.type.value, event.severity.name)
            if haptic_event is not None:
                dispatch = self.haptics.trigger(haptic_event, key=event.key)
                if dispatch is not None:
                    await self.send({"type": "haptic", **dispatch.payload()})

        if self.voice is not None and self.voice.connected:
            behaviour = self.personality.behaviour()
            await self.voice.announce(
                "[PERCEPTION EVENT] " + event.summary + " "
                + "Tell the user this in at most "
                + str(behaviour.sentence_budget) + " short sentence(s). "
                + "Use the wording given; do not add anything perception did not "
                + "establish, and do not ask a follow-up question."
            )

    # ---------------- enrollment (Segment 8) ----------------

    async def handle_enrollment(self, message: Dict[str, Any]) -> None:
        """
        Enroll a person or belonging from a photo the user chose.

        This is the ONLY way a name ever becomes attachable to something the
        camera sees. It requires a deliberate human action — picking an image
        and typing a name — and it stores an embedding, never the photograph.
        """
        if _recognition is None or _memory_store is None:
            await self.send({"type": "enroll_result", "ok": False,
                             "message": "Recognition or storage is unavailable."})
            return

        kind = message.get("kind")
        name = (message.get("name") or "").strip()
        object_class = message.get("object_class") or None

        if kind not in ("person", "belonging"):
            await self.send({"type": "enroll_result", "ok": False,
                             "message": "kind must be 'person' or 'belonging'."})
            return
        if not name:
            await self.send({"type": "enroll_result", "ok": False,
                             "message": "A name is required."})
            return

        source = message.get("image")
        try:
            if source:
                image = decode_frame(source)
            else:
                # No photo supplied: enroll from what the camera sees now.
                image = self.pipeline.current_frame() if self.pipeline else None
                if image is None:
                    await self.send({"type": "enroll_result", "ok": False,
                                     "message": "No photo provided and no camera image available."})
                    return
        except FrameDecodeError as exc:
            await self.send({"type": "enroll_result", "ok": False,
                             "message": "That image couldn't be read: " + str(exc)})
            return

        loop = asyncio.get_running_loop()
        if kind == "person":
            if not _recognition.faces_available:
                await self.send({"type": "enroll_result", "ok": False,
                                 "message": "Face recognition models are not available."})
                return
            embedding = await loop.run_in_executor(
                None, _recognition.embed_face_from_photo, image
            )
            if embedding is None:
                await self.send({
                    "type": "enroll_result", "ok": False,
                    "message": (
                        "I couldn't find a clear face in that photo. Use one where the "
                        "person is facing the camera and reasonably close."
                    ),
                })
                return
        else:
            if not _recognition.belongings_available:
                await self.send({"type": "enroll_result", "ok": False,
                                 "message": "The appearance model is not available."})
                return
            # Crop to the object when we can find one of the stated class, so
            # the embedding describes the item rather than the whole room.
            crop = image
            if object_class and self.pipeline is not None:
                detections = await loop.run_in_executor(
                    None, self.pipeline.detector.detect, image
                )
                same_class = [d for d in detections if d.class_name == object_class]
                if same_class:
                    from app.perception.appearance import crop_from_box

                    best = max(same_class, key=lambda d: d.box.area)
                    cropped = crop_from_box(image, best.box)
                    if cropped is not None:
                        crop = cropped
            embedding = await loop.run_in_executor(
                None, _recognition.embed_object_from_photo, crop
            )
            if embedding is None:
                await self.send({"type": "enroll_result", "ok": False,
                                 "message": "I couldn't build an appearance profile from that photo."})
                return

        record = _memory_store.enroll(
            kind=kind, name=name, embedding=embedding.tolist(), object_class=object_class
        )
        _recognition.invalidate_cache()
        logger.info("enrolled %s %r (samples=%d)", kind, name, record.sample_count)
        await self.send({
            "type": "enroll_result",
            "ok": True,
            **record.payload(),
            "message": (
                "Enrolled " + name + "."
                + (" Add another photo from a different angle to improve accuracy."
                   if record.sample_count == 1 else
                   " Now using " + str(record.sample_count) + " photos.")
            ),
        })
        await self.send_enrollments()

    async def send_enrollments(self) -> None:
        if _memory_store is None:
            return
        await self.send({
            "type": "enrollments",
            "records": [r.payload() for r in _memory_store.list_enrollments()],
            "faces_available": bool(_recognition and _recognition.faces_available),
            "belongings_available": bool(_recognition and _recognition.belongings_available),
        })

    async def handle_delete_enrollment(self, message: Dict[str, Any]) -> None:
        if _memory_store is None:
            await self.send({"type": "error", "message": "storage unavailable"})
            return
        name = (message.get("name") or "").strip()
        if not name:
            await self.send({"type": "error", "message": "name is required"})
            return
        deleted = _memory_store.delete_enrollment(name, message.get("kind"))
        if _recognition is not None:
            _recognition.invalidate_cache()
        await self.send({"type": "enrollment_deleted", "name": name, "deleted_count": deleted})
        await self.send_enrollments()

    async def handle_set_profile(self, message: Dict[str, Any]) -> None:
        """
        Edit the profile from the dashboard.

        A DEDICATED message rather than an entry in the read-only tool
        allowlist: changing who the assistant thinks you are is a deliberate
        human action, exactly like enrolling a face or clearing memory, and
        must not be reachable by the model through the generic tool channel.
        """
        if _memory_store is None:
            await self.send({"type": "error", "message": "storage unavailable"})
            return

        from app.profile.personality import ALL_TRAITS, UserProfile

        profile = self.personality.profile
        if message.get("reset"):
            profile = UserProfile()
            self.personality.profile = profile

        if "name" in message:
            name = str(message.get("name") or "").strip()
            profile.name = name or None

        traits = message.get("traits")
        if isinstance(traits, dict):
            for trait, value in traits.items():
                if trait in ALL_TRAITS:
                    try:
                        profile.personality.set(trait, float(value))
                    except (TypeError, ValueError):
                        continue

        communication = message.get("communication")
        if isinstance(communication, dict):
            for field_name in ("verbosity", "proactive_level", "humor", "encouragement"):
                if field_name in communication:
                    setattr(profile.communication, field_name, str(communication[field_name]))

        preferences = message.get("preferences")
        if isinstance(preferences, dict):
            profile.preferences.update(preferences)

        _memory_store.save_profile(profile)
        self.haptics.apply_preferences(profile.preferences)
        logger.info("profile updated from dashboard (name=%s)", profile.name)
        await self.send_profile()

    async def send_profile(self) -> None:
        await self.send({
            "type": "profile",
            **self.personality.profile.payload(),
            "behaviour": self.personality.behaviour().payload(),
            "attention": self.attention.stats(),
            "haptics": self.haptics.status(),
            "recommendations": self.recommendations.stats(),
        })

    async def refresh_profile(self) -> None:
        """Re-read the profile after a tool changed it, so behaviour updates live."""
        if _memory_store is None:
            return
        self.personality.profile = _memory_store.load_profile()
        self.haptics.apply_preferences(self.personality.profile.preferences)
        await self.send_profile()

    async def handle_demo_event(self, message: Dict[str, Any]) -> None:
        """
        Inject a synthetic world event (Part 68). Clearly labelled as demo so
        nothing simulated can ever be mistaken for real perception.
        """
        try:
            event_type = EventType(str(message.get("event")))
            severity = Severity[str(message.get("severity", "ENVIRONMENT")).upper()]
        except (ValueError, KeyError):
            await self.send({"type": "error", "message": "unknown demo event"})
            return
        await self.dispatch_event(WorldEvent(
            type=event_type,
            severity=severity,
            key="demo:" + event_type.value,
            summary=str(message.get("summary", "")),
            confidence=float(message.get("confidence", 0.9)),
            data={"demo": True, "subject": str(message.get("subject", ""))},
        ))

    # ---------------- dashboard tool access ----------------

    # Read-only tools the UI may invoke directly. Everything that changes
    # state is excluded on purpose.
    READONLY_TOOLS = frozenset(
        {
            "list_saved_memories", "get_tasks", "describe_scene", "count_people",
            "recall", "list_enrolled", "find_signs", "who_is_here",
            "identify_belongings", "check_path", "describe_doors", "find_exit",
        }
    )

    async def invoke_readonly_tool(self, tool: str, args: Dict[str, Any]) -> None:
        if tool not in self.READONLY_TOOLS:
            await self.send(
                {"type": "error", "scope": "invoke_tool",
                 "message": "tool " + str(tool) + " is not available to the dashboard"}
            )
            return
        if tool not in self.registry.names():
            await self.send(
                {"type": "error", "scope": "invoke_tool",
                 "message": "tool " + str(tool) + " is not registered"}
            )
            return
        result = await self.registry.execute(tool, args)
        await self.send({"type": "tool_result", "tool": tool, "result": result})
        if tool in _PROFILE_TOOLS:
            await self.refresh_profile()

    # ---------------- voice ----------------

    async def start_voice(self) -> None:
        if self.voice is not None and self.voice.connected:
            return
        if not settings.gemini_api_key:
            await self.send(
                {"type": "voice_error",
                 "message": "GEMINI_API_KEY is not configured on the backend."}
            )
            return
        try:
            self.voice = LiveVoiceSession(
                api_key=settings.gemini_api_key,
                registry=self.registry,
                on_event=self._on_voice_event,
                style_prompt=self.personality.style_prompt(),
            )
            await self.voice.connect()
            self.start_sign_watch()
            await self.send({"type": "voice_ready", "model": self.voice._model})
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to start voice session")
            self.voice = None
            await self.send({"type": "voice_error", "message": str(exc)[:300]})

    async def _on_voice_event(self, event: Dict[str, Any]) -> None:
        """
        Mirror voice events to the browser, tracking whether the assistant is
        mid-turn so the announcer stays quiet while it is talking.
        """
        kind = event.get("type")
        if kind in ("agent_audio", "tool_call"):
            self.agent_is_speaking = True
        elif kind in ("agent_turn_complete", "voice_interrupted"):
            self.agent_is_speaking = False
        await self.send(event)

    async def stop_voice(self) -> None:
        await self.stop_sign_watch()
        if self.voice is not None:
            await self.voice.close()
            self.voice = None
        await self.send({"type": "voice_stopped"})

    async def close(self) -> None:
        await self.stop_sign_watch()
        await self.stop_voice()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session = ClientSession(websocket)
    logger.info("client connected (vision=%s)", session.pipeline is not None)

    try:
        await session.send(
            {
                "type": "health",
                "status": "ok",
                "service": "vidi-backend",
                "vision_available": session.pipeline is not None,
                "vision_backend": _shared_detector.backend.name if _shared_detector else None,
                "voice_available": settings.gemini_configured,
                "tools": session.registry.names(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        while True:
            raw = await websocket.receive_text()
            if len(raw) > MAX_MESSAGE_BYTES:
                await session.send(
                    {"type": "error", "message": "message exceeds size limit"}
                )
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await session.send({"type": "error", "message": "malformed JSON"})
                continue
            if not isinstance(message, dict):
                await session.send({"type": "error", "message": "message must be an object"})
                continue

            kind = message.get("type")
            if kind == "frame":
                await session.handle_frame(message)
            elif kind == "audio":
                if session.voice:
                    await session.voice.send_audio(message.get("data", ""))
            elif kind == "audio_end":
                if session.voice:
                    await session.voice.signal_audio_stream_end()
            elif kind == "text":
                if session.voice:
                    await session.voice.send_text(message.get("text", ""))
                else:
                    await session.send(
                        {"type": "voice_error", "message": "voice session not started"}
                    )
            elif kind == "start_voice":
                await session.start_voice()
            elif kind == "stop_voice":
                await session.stop_voice()
            elif kind == "invoke_tool":
                # Dashboard/caregiver access to READ-ONLY tools. Restricted to
                # an allowlist so the browser cannot drive arbitrary tools;
                # anything that mutates state stays with the agent or with the
                # explicit gestures below.
                await session.invoke_readonly_tool(
                    message.get("tool", ""), message.get("args") or {}
                )
            elif kind == "forget_all_memories":
                # Deliberately NOT a model-callable tool: erasing everything
                # must be an explicit human gesture, never something the
                # assistant can be talked into doing.
                if _memory_store is None:
                    await session.send({"type": "error", "message": "memory is unavailable"})
                else:
                    deleted = _memory_store.forget_all()
                    await session.send({"type": "memory_cleared", "deleted_count": deleted})
            elif kind == "enroll":
                await session.handle_enrollment(message)
            elif kind == "list_enrollments":
                await session.send_enrollments()
                await session.send_profile()
            elif kind == "delete_enrollment":
                await session.handle_delete_enrollment(message)
            elif kind == "set_profile":
                await session.handle_set_profile(message)
            elif kind == "get_profile":
                await session.send_profile()
            elif kind == "set_quiet_mode":
                session.attention.quiet_mode = bool(message.get("enabled"))
                await session.send({"type": "quiet_mode", "enabled": session.attention.quiet_mode})
            elif kind == "haptic_provider":
                # The browser reports what it can actually do, so the backend
                # never claims a capability the hardware lacks (Part 40).
                session.haptics.set_provider(
                    str(message.get("provider", "mock")),
                    bool(message.get("connected")),
                    bool(message.get("supports_direction")),
                )
                await session.send({"type": "haptic_status", **session.haptics.status()})
            elif kind == "test_haptic":
                from app.haptics.engine import HapticEvent

                try:
                    event = HapticEvent(str(message.get("event", "FRIEND_DETECTED")))
                except ValueError:
                    await session.send({"type": "error", "message": "unknown haptic event"})
                else:
                    dispatch = session.haptics.trigger(event, key="test", force=True)
                    if dispatch:
                        await session.send({"type": "haptic", **dispatch.payload()})
            elif kind == "demo_event":
                await session.handle_demo_event(message)
            elif kind == "ping":
                await session.send({"type": "pong"})
            else:
                await session.send(
                    {"type": "error", "message": "unknown message type: " + str(kind)}
                )

    except WebSocketDisconnect:
        logger.info(
            "client disconnected (frames=%d dropped=%d)",
            session.frames_received, session.frames_dropped,
        )
    finally:
        await session.close()
