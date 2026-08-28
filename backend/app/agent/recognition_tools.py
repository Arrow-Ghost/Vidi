"""
Tools for recognizing enrolled people and belongings — Segment 8.

These are the only tools in the system that can attach a NAME to something the
camera sees, and they can only ever produce names the user deliberately
enrolled. Every result carries an explicit confidence band, and the tool
descriptions instruct the model on exactly how to hedge, because the failure
modes here are unusually costly:

  * Naming a stranger as the user's friend.
  * Sending a blind user to collect a bag that is not theirs.

Both are prevented structurally — no enrollment, no name — and then again by
the confidence bands surfaced in every payload.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.agent.tools import DIRECTION_PHRASE

logger = logging.getLogger("vidi.agent.recognition_tools")


def register_recognition_tools(registry, pipeline, recognition, store) -> None:
    from app.agent.tools import ToolSpec

    def _phrase(direction: str) -> str:
        return DIRECTION_PHRASE.get(direction, direction)

    async def who_is_here() -> Dict[str, Any]:
        """Identify any enrolled people currently visible."""
        if not recognition.faces_available:
            return {
                "available": False,
                "message": "Face recognition isn't available — the models failed to load.",
            }
        frame = pipeline.current_frame() if pipeline else None
        if frame is None:
            return {"available": True, "message": "There's no camera image yet."}

        people, face_count = recognition.identify_people(frame)
        enrolled_names = recognition.enrolled_names("person")

        if face_count == 0:
            return {
                "face_count": 0,
                "identified": [],
                "message": (
                    "I can't see anyone's face right now. That doesn't mean nobody is "
                    "there — they may be turned away or outside the camera's view."
                ),
            }

        identified = []
        for person in people:
            entry = person.payload()
            entry["direction_phrase"] = _phrase(person.direction.value)
            identified.append(entry)

        unknown = max(0, face_count - len(identified))
        return {
            "face_count": face_count,
            "identified": identified,
            "unrecognized_face_count": unknown,
            "enrolled_people": enrolled_names,
            "note": (
                "Only people the user has explicitly enrolled can be named. The "
                + str(unknown) + " unrecognized face(s) belong to people you must NOT "
                "identify, name, or describe — refer to them only as 'someone' or "
                "'a person'. If is_confident is false, hedge: say you think it might "
                "be them, not that it is."
            ),
        }

    async def find_person(name: str) -> Dict[str, Any]:
        """Look for one specific enrolled person."""
        if not recognition.faces_available:
            return {"found": False, "available": False,
                    "message": "Face recognition isn't available right now."}

        enrolled = recognition.enrolled_names("person")
        matched_name = next((n for n in enrolled if n.lower() == name.strip().lower()), None)
        if matched_name is None:
            return {
                "found": False,
                "enrolled": False,
                "searched_for": name,
                "enrolled_people": enrolled,
                "message": (
                    name + " hasn't been enrolled, so I have no way to recognize them. "
                    "The user can add them from the dashboard by choosing a photo."
                ),
            }

        frame = pipeline.current_frame() if pipeline else None
        if frame is None:
            return {"found": False, "enrolled": True,
                    "message": "There's no camera image yet."}

        people, face_count = recognition.identify_people(frame)
        hit = next((p for p in people if p.name.lower() == matched_name.lower()), None)
        if hit is None:
            return {
                "found": False,
                "enrolled": True,
                "searched_for": matched_name,
                "faces_seen": face_count,
                "message": (
                    "I can't see " + matched_name + " in the camera's current view. That "
                    "does NOT mean they aren't nearby — they could be outside the view or "
                    "facing away."
                ),
            }

        payload = hit.payload()
        payload["direction_phrase"] = _phrase(hit.direction.value)
        return {
            "found": True,
            "enrolled": True,
            **payload,
            "note": (
                "If is_confident is false you MUST hedge — say you think it might be "
                "them and suggest they check, rather than stating it as fact."
            ),
        }

    async def identify_belongings() -> Dict[str, Any]:
        """Check whether any visible objects are the user's enrolled items."""
        if not recognition.belongings_available:
            return {
                "available": False,
                "message": "Belonging recognition isn't available — the model failed to load.",
            }
        frame = pipeline.current_frame() if pipeline else None
        if frame is None:
            return {"available": True, "message": "There's no camera image yet."}

        detections = list(pipeline.detector.detect(frame)) if pipeline else []
        matches = recognition.identify_belongings(frame, detections)
        enrolled = recognition.enrolled_names("belonging")

        if not enrolled:
            return {
                "match_count": 0,
                "enrolled_belongings": [],
                "message": (
                    "The user hasn't enrolled any belongings yet, so I can't tell their "
                    "items apart from anyone else's. They can add one from the dashboard."
                ),
            }
        if not matches:
            return {
                "match_count": 0,
                "enrolled_belongings": enrolled,
                "message": (
                    "None of the objects I can see look like the user's enrolled items. "
                    "They may simply be out of view."
                ),
            }

        payload = []
        for match in matches:
            entry = match.payload()
            entry["direction_phrase"] = _phrase(match.direction.value)
            payload.append(entry)
        return {
            "match_count": len(payload),
            "matches": payload,
            "enrolled_belongings": enrolled,
            "note": (
                "match_band 'confident' may be stated plainly. 'possible' MUST be "
                "hedged — say it looks like theirs but you can't be certain. Appearance "
                "matching cannot separate two identical items."
            ),
        }

    async def list_enrolled() -> Dict[str, Any]:
        """What the user has enrolled — for 'who do you know', 'what do you recognize'."""
        if store is None:
            return {"people": [], "belongings": [], "message": "Storage isn't available."}
        records = store.list_enrollments()
        return {
            "people": [r.name for r in records if r.kind == "person"],
            "belongings": [
                {"name": r.name, "object_class": r.object_class}
                for r in records if r.kind == "belonging"
            ],
            "total": len(records),
        }

    async def forget_enrolled(name: str) -> Dict[str, Any]:
        """Delete an enrolled person or belonging."""
        if store is None:
            return {"deleted": False, "message": "Storage isn't available."}
        deleted = store.delete_enrollment(name)
        recognition.invalidate_cache()
        return {
            "deleted": deleted > 0,
            "deleted_count": deleted,
            "name": name,
            "message": (
                "I've deleted everything I had stored for " + name + "."
                if deleted else
                "There was nothing enrolled under " + name + ", so nothing was deleted."
            ),
        }

    registry.register(ToolSpec(
        name="who_is_here",
        description=(
            "Identify any enrolled people whose faces are currently visible. Use for "
            "'who is that', 'is anyone I know here', 'who's in the room'. Only people "
            "the user explicitly enrolled can be named; everyone else is reported only "
            "as an unrecognized face and must never be described or identified."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=who_is_here,
        timeout_s=15.0,
    ))
    registry.register(ToolSpec(
        name="find_person",
        description=(
            "Look for one specific enrolled person by name, e.g. 'find Rahul'. Returns "
            "their direction if visible. Not finding them means they are not in the "
            "camera's view — never that they are absent."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {"name": {"type": "STRING"}},
            "required": ["name"],
        },
        handler=find_person,
        timeout_s=15.0,
    ))
    registry.register(ToolSpec(
        name="identify_belongings",
        description=(
            "Check whether any object in view is one of the user's own enrolled "
            "belongings, e.g. telling their backpack apart from someone else's. Use for "
            "'is that my bag', 'is my backpack here', 'which one is mine'."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=identify_belongings,
        timeout_s=20.0,
    ))
    registry.register(ToolSpec(
        name="list_enrolled",
        description=(
            "List the people and belongings the user has enrolled. Use for 'who do you "
            "recognize', 'what of mine do you know', 'who have I added'."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=list_enrolled,
    ))
    registry.register(ToolSpec(
        name="forget_enrolled",
        description=(
            "Delete an enrolled person or belonging by name. Use when the user says to "
            "forget someone or stop recognizing something."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {"name": {"type": "STRING"}},
            "required": ["name"],
        },
        handler=forget_enrolled,
    ))
