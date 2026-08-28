"""
Tool registry — Segments 12, 33.

Every tool the model can call is declared here with a schema, validated
arguments, a timeout, and a structured result. The model NEVER executes code
and never sees anything but the structured result dict; it converts that into
speech. If a value is not present in a tool result, the model has no source
for it — which is the mechanism that stops fabricated distances, directions,
and identities (Segment 32 / SECURITY.md).

Tools read from the WorldModel. They never touch the detector directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.agent import vocabulary
from app.perception.distance import DistanceRange
from app.perception.world_model import Entity, WorldModel

logger = logging.getLogger("vidi.agent.tools")

DEFAULT_TOOL_TIMEOUT_S = 5.0

# Spoken phrasing for each direction bucket. The model is instructed to reuse
# the tool's own wording, so this is where "far_left" becomes something a
# person would actually say out loud.
DIRECTION_PHRASE = {
    "far_left": "far to your left",
    "left": "to your left",
    "center_left": "slightly to your left",
    "center": "directly ahead of you",
    "center_right": "slightly to your right",
    "right": "to your right",
    "far_right": "far to your right",
}

# Below this, the agent must not assert the object is what the user asked for.
LOW_CONFIDENCE_FLOOR = 0.55


class ToolError(Exception):
    """A tool failed in a way the agent should tell the user about honestly."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Awaitable[Dict[str, Any]]]
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S

    def declaration(self) -> Dict[str, Any]:
        """Gemini functionDeclaration form."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _entity_payload(entity: Entity) -> Dict[str, Any]:
    """
    The canonical structured view of one perceived entity.

    Distance is reported as a RANGE plus a confidence, never as a number: a
    monocular camera cannot produce metric distance, and emitting "2.4 meters"
    here would be this system's single most dangerous lie.
    """
    distance_known = entity.distance.range != DistanceRange.UNKNOWN
    return {
        "entity_id": entity.id,
        "type": entity.type,
        "direction": entity.direction.bucket.value,
        "direction_phrase": DIRECTION_PHRASE.get(
            entity.direction.bucket.value, entity.direction.bucket.value
        ),
        "direction_confidence": round(entity.direction.confidence, 2),
        "distance_range": entity.distance.range.value,
        "distance_phrase": entity.distance.label if distance_known else None,
        "distance_confidence": round(entity.distance.confidence, 2),
        "distance_is_reliable": distance_known,
        "detection_confidence": round(entity.confidence, 2),
        "identification_is_confident": entity.confidence >= LOW_CONFIDENCE_FLOOR,
    }


class ToolRegistry:
    def __init__(
        self,
        world: WorldModel,
        store: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        vision: Optional[Any] = None,
        recognition: Optional[Any] = None,
        personality: Optional[Any] = None,
        recommendations: Optional[Any] = None,
    ) -> None:
        self.world = world
        self.store = store
        self.pipeline = pipeline
        self.vision = vision
        self.recognition = recognition
        self.personality = personality
        self.recommendations = recommendations
        self._tools: Dict[str, ToolSpec] = {}
        self.call_log: List[Dict[str, Any]] = []  # developer trace (Segment 35)
        self._register_builtins()
        if store is not None:
            # Imported lazily so the perception-only path never needs the
            # memory subsystem (or a database) to exist at all.
            from app.agent.memory_tools import register_memory_tools

            register_memory_tools(self, store)
        if pipeline is not None and vision is not None:
            from app.agent.ocr_tools import register_ocr_tools

            register_ocr_tools(self, pipeline, vision)
        if recognition is not None:
            from app.agent.recognition_tools import register_recognition_tools

            register_recognition_tools(self, pipeline, recognition, store)
        if personality is not None and store is not None:
            from app.agent.profile_tools import register_profile_tools

            register_profile_tools(self, store, personality)
        if pipeline is not None and recommendations is not None:
            from app.agent.social_tools import register_social_tools

            register_social_tools(self, pipeline, recommendations, store)
        if pipeline is not None:
            from app.agent.navigation_tools import register_navigation_tools

            register_navigation_tools(self, pipeline)

    # ---------------- registry plumbing ----------------

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def declarations(self) -> List[Dict[str, Any]]:
        return [t.declaration() for t in self._tools.values()]

    def names(self) -> List[str]:
        return list(self._tools)

    async def execute(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate, run with a timeout, and always return a structured dict.

        A tool that raises must NOT propagate into the voice session — the
        agent needs a result it can speak honestly about ("I could not check
        that right now"), not a dropped conversational turn.
        """
        args = args or {}
        started = time.perf_counter()
        spec = self._tools.get(name)
        if spec is None:
            result = {"error": "unknown_tool", "message": "No tool named " + repr(name) + "."}
        else:
            try:
                self._validate(spec, args)
                result = await asyncio.wait_for(spec.handler(**args), timeout=spec.timeout_s)
            except asyncio.TimeoutError:
                logger.warning("tool %s timed out after %.1fs", name, spec.timeout_s)
                result = {"error": "timeout", "message": name + " took too long to respond."}
            except ToolError as exc:
                result = {"error": "tool_error", "message": str(exc)}
            except Exception as exc:  # noqa: BLE001 - must never kill the voice session
                logger.exception("tool %s failed", name)
                result = {"error": "internal_error", "message": name + " failed: " + str(exc)}

        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        self.call_log.append(
            {
                "tool": name,
                "args": args,
                "result": result,
                "latency_ms": elapsed_ms,
                "at": time.time(),
            }
        )
        del self.call_log[:-50]  # bounded developer trace
        return result

    @staticmethod
    def _validate(spec: ToolSpec, args: Dict[str, Any]) -> None:
        schema = spec.parameters or {}
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in args or args[required] in (None, ""):
                raise ToolError("missing required argument " + repr(required))
        for key in args:
            if key not in properties:
                raise ToolError("unexpected argument " + repr(key) + " for " + spec.name)

    # ---------------- the tools themselves ----------------

    def _register_builtins(self) -> None:
        self.register(
            ToolSpec(
                name="describe_scene",
                description=(
                    "Describe what the camera can currently see around the user. Use for "
                    "questions like what is around me, what can you see, what is in front "
                    "of me. Returns a prioritized, capped list with people and obstacles "
                    "first."
                ),
                parameters={"type": "OBJECT", "properties": {}},
                handler=self._describe_scene,
            )
        )
        self.register(
            ToolSpec(
                name="find_object",
                description=(
                    "Look for a specific object in the current camera view, such as a "
                    "backpack, bottle, or chair. Returns every matching candidate with "
                    "direction and distance. Absence means not currently visible, NOT that "
                    "the object is not present."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "object_name": {
                            "type": "STRING",
                            "description": "Everyday name of the object, e.g. backpack, my phone.",
                        }
                    },
                    "required": ["object_name"],
                },
                handler=self._find_object,
            )
        )
        self.register(
            ToolSpec(
                name="get_object_position",
                description=(
                    "Get the current direction and distance of one already-identified "
                    "entity, by the entity_id returned from a previous tool call. Use this "
                    "for follow-up questions like where is it now, or is it still there."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "entity_id": {
                            "type": "STRING",
                            "description": "entity_id from an earlier tool result.",
                        }
                    },
                    "required": ["entity_id"],
                },
                handler=self._get_object_position,
            )
        )
        self.register(
            ToolSpec(
                name="estimate_distance",
                description=(
                    "Estimate how far away an already-identified entity is. Returns a "
                    "coarse range and a confidence. If distance_is_reliable is false you "
                    "MUST tell the user you cannot reliably estimate the distance. Never "
                    "guess a number."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {"entity_id": {"type": "STRING"}},
                    "required": ["entity_id"],
                },
                handler=self._estimate_distance,
            )
        )
        self.register(
            ToolSpec(
                name="count_people",
                description=(
                    "Count the people currently visible and give each one's direction. Use "
                    "for questions like is anyone near me, or how many people are here. "
                    "This does NOT identify anyone by name."
                ),
                parameters={"type": "OBJECT", "properties": {}},
                handler=self._count_people,
            )
        )

    async def _describe_scene(self) -> Dict[str, Any]:
        entities = self.world.summarize(max_items=5)
        if not entities:
            return {
                "visible_entity_count": 0,
                "entities": [],
                "note": (
                    "Nothing recognizable is currently in view. This means the camera "
                    "cannot see anything it recognizes right now. It does NOT mean the "
                    "area is empty or safe."
                ),
            }
        return {
            "visible_entity_count": len(self.world.entities),
            "entities": [_entity_payload(e) for e in entities],
            "note": (
                "Only what the camera can currently see. Objects outside its field of "
                "view are unknown."
            ),
        }

    async def _find_object(self, object_name: str) -> Dict[str, Any]:
        classes = vocabulary.resolve(object_name)
        if classes is None:
            unsupported = vocabulary.is_known_unsupported(object_name)
            if unsupported:
                message = (
                    "I cannot reliably look for " + str(object_name) + " because my object "
                    "detector has no category for it. I genuinely do not know whether it "
                    "is there."
                )
            else:
                message = (
                    "I do not have a detector category matching " + str(object_name) + "."
                )
            return {
                "found": False,
                "searched_for": object_name,
                "supported": False,
                "message": message,
            }

        matches: List[Entity] = []
        for class_name in classes:
            matches.extend(self.world.find(class_name))
        if not matches:
            return {
                "found": False,
                "searched_for": object_name,
                "supported": True,
                "matched_classes": classes,
                "message": (
                    "No " + str(object_name) + " is visible in the camera's current view. "
                    "This does NOT mean it is absent — it may simply be outside the field "
                    "of view."
                ),
            }
        matches.sort(key=lambda e: e.confidence, reverse=True)
        result = {
            "found": True,
            "searched_for": object_name,
            "supported": True,
            "candidate_count": len(matches),
            "candidates": [_entity_payload(e) for e in matches[:4]],
        }

        # Segment 34: if the user previously told us something about this
        # object ("my backpack is blue"), surface it so the agent can use the
        # user's own description when several candidates are visible. This is
        # recall of an explicitly-saved fact — it never becomes a claim that
        # a particular candidate IS the user's object.
        if self.store is not None and len(matches) > 1:
            remembered = self.store.get_by_subject(object_name.strip().lower())
            if remembered is not None:
                result["remembered_description"] = remembered.content
                result["disambiguation_note"] = (
                    "The user previously described this object. You may mention that "
                    "description to help them identify which candidate is theirs, but do "
                    "NOT claim to have matched it — the detector cannot verify colour or "
                    "ownership."
                )
        return result

    async def _get_object_position(self, entity_id: str) -> Dict[str, Any]:
        entity = self.world.entities.get(entity_id)
        if entity is None:
            return {
                "found": False,
                "entity_id": entity_id,
                "message": (
                    "I have lost visual contact with that. It is no longer in the "
                    "camera's view."
                ),
            }
        return {"found": True, **_entity_payload(entity)}

    async def _estimate_distance(self, entity_id: str) -> Dict[str, Any]:
        entity = self.world.entities.get(entity_id)
        if entity is None:
            return {
                "found": False,
                "entity_id": entity_id,
                "message": "That is no longer visible, so I cannot estimate its distance.",
            }
        payload = _entity_payload(entity)
        if not payload["distance_is_reliable"]:
            payload["message"] = (
                "Depth confidence is below the reliability threshold. Tell the user you "
                "cannot reliably estimate this distance. Do not state a number."
            )
        return {"found": True, **payload}

    async def _count_people(self) -> Dict[str, Any]:
        people = [e for e in self.world.entities.values() if e.is_person]
        return {
            "person_count": len(people),
            "people": [_entity_payload(p) for p in people[:5]],
            "note": (
                "Detection only, no identification. Do not name or describe who these "
                "people are."
            ),
        }
