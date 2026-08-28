"""
Crowd, friend and recommendation tools — Parts 19-21, 25-34, 52, 53.

These answer the Part 78 questions: "Is Ravi in the crowd?", "Is the crowd
moving?", "Which direction?", "Is it getting closer?", "How dense is it?",
"Is there somewhere interesting nearby?".

Every result is read from state that perception already computed and debounced,
so asking repeatedly is cheap and gives a stable answer rather than a
frame-by-frame flicker.

THE HONESTY RULES, encoded in the payloads rather than left to the prompt:
  * counts carry count_is_approximate — detection misses occluded people
  * directions are relative (left/right/ahead), never compass bearings
  * a friend below the confirmation threshold is simply not named
  * movement is reported; intent never is (Part 55)
  * "not visible" never becomes "not there"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vidi.agent.social_tools")


def register_social_tools(registry, pipeline, recommendations, store) -> None:
    from app.agent.tools import ToolSpec

    # ---------------- crowds ----------------

    async def describe_crowd() -> Dict[str, Any]:
        if pipeline is None:
            return {"available": False, "message": "Perception is unavailable."}
        state = pipeline.crowd_engine.state
        if not state.is_crowd:
            return {
                "is_crowd": False,
                "person_count": state.person_count,
                "message": (
                    "I can't see a crowd in the camera's current view. That does NOT mean "
                    "the area is empty or clear — only that I don't currently detect one."
                ),
            }
        return {
            "is_crowd": True,
            **state.payload(),
            "spoken_description": state.describe(),
            "note": (
                "The count is approximate — people behind others are not detected. Say "
                "'roughly' or 'about'. Directions are relative to the user, never compass "
                "points. Never say the crowd is safe or that a path is clear."
            ),
        }

    async def get_crowd_movement() -> Dict[str, Any]:
        """'Is the crowd moving?', 'which way?', 'is it getting closer?'"""
        if pipeline is None:
            return {"available": False}
        state = pipeline.crowd_engine.state
        if not state.is_crowd:
            return {"is_crowd": False,
                    "message": "I don't currently detect a crowd to describe."}
        return {
            "is_crowd": True,
            "motion": state.motion.value,
            "is_moving": state.motion.value != "static",
            "flow": state.flow.value,
            "flow_direction": state.flow_direction,
            "approaching": state.approaching,
            "departing": state.departing,
            "congested": state.congested,
            "confidence": round(state.confidence, 2),
            "note": (
                "Report the movement only. Do NOT say where the crowd is going or why — "
                "you cannot know their destination or intent."
            ),
        }

    # ---------------- friends ----------------

    async def who_is_nearby() -> Dict[str, Any]:
        """Enrolled people currently confirmed in view."""
        if pipeline is None:
            return {"available": False}
        tracker = pipeline.friend_tracker
        visible = tracker.visible()
        crowd = pipeline.crowd_engine.state

        return {
            "friend_count": len(visible),
            "friends": [f.payload() for f in visible],
            "spoken": [tracker.describe(f) for f in visible],
            "in_crowd": crowd.is_crowd,
            "enrolled_people": (
                [r.name for r in store.list_enrollments("person")] if store else []
            ),
            "note": (
                "Only enrolled people appear here. Anyone else present must be referred "
                "to as 'someone' and never described or identified. If is_confident is "
                "false, hedge rather than assert."
                if visible else
                "Nobody the user has enrolled is currently confirmed in view. That does "
                "NOT mean they are absent — they may be out of view or facing away."
            ),
        }

    async def locate_friend(name: str) -> Dict[str, Any]:
        """'Where is Ravi?', 'Is Ravi in the crowd?'"""
        if pipeline is None:
            return {"found": False, "available": False}
        enrolled = [r.name for r in store.list_enrollments("person")] if store else []
        matched = next((n for n in enrolled if n.lower() == name.strip().lower()), None)
        if matched is None:
            return {
                "found": False, "enrolled": False, "searched_for": name,
                "enrolled_people": enrolled,
                "message": (
                    name + " has not been enrolled, so I have no way to recognize them. "
                    "They can be added from the dashboard with a photo."
                ),
            }

        friend = pipeline.friend_tracker.get(matched)
        crowd = pipeline.crowd_engine.state
        if friend is None or friend.state.value in ("candidate", "lost"):
            return {
                "found": False, "enrolled": True, "searched_for": matched,
                "crowd_present": crowd.is_crowd,
                "message": (
                    "I can't see " + matched + " in the camera's view right now. That does "
                    "NOT mean they aren't nearby — they may be outside the view, facing "
                    "away, or hidden behind other people."
                ),
            }
        if friend.state.value == "occluded":
            return {
                "found": False, "enrolled": True, "searched_for": matched,
                "occluded": True, "last_direction": friend.direction.value,
                "message": (
                    "The crowd is blocking my view of " + matched + " right now."
                    if friend.in_crowd else
                    "I've lost sight of " + matched + " for the moment."
                ),
            }
        return {
            "found": True, "enrolled": True,
            **friend.payload(),
            "spoken_description": pipeline.friend_tracker.describe(friend),
            "crowd_present": crowd.is_crowd,
            "note": (
                "Report their direction and movement only. Never say where they are "
                "heading or why."
            ),
        }

    async def find_friend_group(group: str = "") -> Dict[str, Any]:
        """Part 52: 'is anyone from my college friends nearby?'"""
        if store is None or pipeline is None:
            return {"available": False}
        groups = store.list_friend_groups()
        if group:
            members = store.get_friend_group(group)
            if members is None:
                return {"found": False, "group": group,
                        "available_groups": sorted(groups),
                        "message": "No friend group called " + repr(group) + "."}
            wanted = {m.lower() for m in members}
        else:
            wanted = {m.lower() for members in groups.values() for m in members}
            if not wanted:
                return {"found": False,
                        "message": "No friend groups have been set up yet."}

        present = [
            f for f in pipeline.friend_tracker.visible() if f.name.lower() in wanted
        ]
        return {
            "found": bool(present),
            "group": group or "all groups",
            "present_count": len(present),
            "present": [f.payload() for f in present],
            "spoken": [pipeline.friend_tracker.describe(f) for f in present],
            "message": (
                "" if present else
                "I can't see anyone from that group right now, though they may be out "
                "of view."
            ),
        }

    # ---------------- recommendations ----------------

    async def suggest_place() -> Dict[str, Any]:
        """'Is there somewhere interesting nearby?', 'recommend somewhere'."""
        if pipeline is None or recommendations is None:
            return {"available": False}
        signs = pipeline.sign_tracker.current() if pipeline.ocr_available else []
        entities = list(pipeline.world.entities.values())
        # The user asked, so answer regardless of their proactive-suggestion setting.
        found = recommendations.recommend(signs, entities, limit=2, proactive=False)
        if not found:
            return {
                "found": False,
                "message": (
                    "I haven't detected anywhere notable from here. I can only recognise "
                    "places from signs I can read or obvious surroundings — there may well "
                    "be somewhere nearby that I simply can't see."
                ),
            }
        for item in found:
            recommendations.mark_mentioned(item.place)
        return {
            "found": True,
            "count": len(found),
            "suggestions": [r.payload() for r in found],
            "spoken": [recommendations.describe(r) for r in found],
            "note": (
                "Attribute the evidence — say you can see a sign for it. Never describe a "
                "place as nice, quiet, safe or busy unless perception established that. "
                "This system has no GPS or maps, so NEVER give a distance in metres."
            ),
        }

    # ---------------- registration ----------------

    registry.register(ToolSpec(
        name="describe_crowd",
        description=(
            "Describe the crowd currently in view: how many people roughly, how dense, "
            "and where. Use for 'is it crowded', 'how many people are here', 'how dense "
            "is the crowd'."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=describe_crowd,
    ))
    registry.register(ToolSpec(
        name="get_crowd_movement",
        description=(
            "Report how the crowd is moving: static or moving, which relative direction, "
            "whether it is approaching, dispersing or congested. Use for 'is the crowd "
            "moving', 'which way is it moving', 'is it getting closer'."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=get_crowd_movement,
    ))
    registry.register(ToolSpec(
        name="who_is_nearby",
        description=(
            "List enrolled people confirmed in view right now, with direction and "
            "movement. Use for 'who's nearby', 'are any of my friends here', 'is anyone I "
            "know around'."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=who_is_nearby,
    ))
    registry.register(ToolSpec(
        name="locate_friend",
        description=(
            "Find one enrolled person by name and report where they are, including "
            "whether they are inside a crowd or currently blocked from view. Use for "
            "'find Ravi', 'where is Ravi', 'is Ravi in the crowd'."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {"name": {"type": "STRING"}},
            "required": ["name"],
        },
        handler=locate_friend,
    ))
    registry.register(ToolSpec(
        name="find_friend_group",
        description=(
            "Check whether anyone from a named friend group is nearby, e.g. 'is anyone "
            "from my college friends here'. Omit group to check every group."
        ),
        parameters={"type": "OBJECT", "properties": {"group": {"type": "STRING"}}},
        handler=find_friend_group,
    ))
    registry.register(ToolSpec(
        name="suggest_place",
        description=(
            "Suggest somewhere nearby that matches the user's stated interests, based on "
            "signs the camera can read and the surroundings. Use for 'is there anywhere "
            "interesting nearby', 'is there a park', 'recommend somewhere'. ALWAYS call "
            "this when the user asks such a question, even if they previously asked you "
            "not to suggest places unprompted — that setting stops you volunteering "
            "suggestions, it never stops you answering a direct question."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=suggest_place,
    ))
