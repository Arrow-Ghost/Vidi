"""
Doors, exits and what is in the user's way.

Four tools, split by cost and by question:

  check_path       what is close and in front of me?     local, instant, free
  describe_doors   what doors can you see?               cached sweep, instant
  find_exit        where is the way out?                 cached sweep, instant
  scan_for_doors   look again, properly                  Gemini, ~3.4s

The three instant ones read state the pipeline already gathered, so the user
can ask repeatedly while walking without paying for it. scan_for_doors forces
a fresh look, for when they have turned around or want to be sure.

THE RULE THAT MATTERS MOST HERE: none of these ever reports that the way ahead
is clear. Detecting nothing means nothing was recognised — glass, steps down,
low obstacles and anything outside the camera's narrow view are all invisible.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vidi.agent.navigation_tools")

NOT_CLEAR = (
    "IMPORTANT: this does NOT mean the way ahead is clear or safe. It means "
    "nothing recognised was detected in a narrow camera view. Never tell the "
    "user a path is clear, safe, or free of obstacles - only ever say what you "
    "do or do not currently detect."
)


def register_navigation_tools(registry, pipeline) -> None:
    from app.agent.tools import ToolSpec

    # ---------------- what is in the way ----------------

    async def check_path() -> Dict[str, Any]:
        """'Is anything in front of me?', 'can I walk forward?'"""
        if pipeline is None or not hasattr(pipeline, "obstacle_monitor"):
            return {"available": False, "message": "Perception is unavailable."}

        entities = list(pipeline.world.entities.values())
        obstacles = pipeline.obstacle_monitor.scan(entities)

        if not obstacles:
            return {
                "obstacle_count": 0,
                "message": (
                    "I don't detect anything close in front of you right now."
                ),
                "note": NOT_CLEAR,
            }

        return {
            "obstacle_count": len(obstacles),
            "obstacles": [o.payload() for o in obstacles],
            "closest": obstacles[0].payload(),
            "note": (
                "Lead with the closest one and use its own wording. " + NOT_CLEAR
            ),
        }

    # ---------------- doors ----------------

    async def describe_doors() -> Dict[str, Any]:
        """'Are there any doors?', 'is the door open?'"""
        if pipeline is None or not hasattr(pipeline, "door_tracker"):
            return {"available": False}

        doors = pipeline.door_tracker.current()
        if not doors:
            return {
                "door_count": 0,
                "message": (
                    "I haven't detected any doors in the camera's recent view. They "
                    "may simply be out of view - try turning slowly, or ask me to "
                    "scan again."
                ),
            }
        return {
            "door_count": len(doors),
            "doors": [d.payload() for d in doors],
            "note": (
                "Report each door's direction and whether it is open or closed. If "
                "is_confident is false, hedge - say there may be a door. Never say "
                "a door is an exit unless is_exit is true."
            ),
        }

    async def find_exit() -> Dict[str, Any]:
        """'Where's the exit?', 'how do I get out?'"""
        if pipeline is None or not hasattr(pipeline, "door_tracker"):
            return {"available": False}

        tracker = pipeline.door_tracker
        exits = tracker.exits()
        if exits:
            best = exits[0]
            return {
                "found": True,
                "exit_count": len(exits),
                "exits": [d.payload() for d in exits],
                "best": best.payload(),
                "evidence": best.evidence,
                "note": (
                    "Say where it is and what makes it an exit (the evidence field, "
                    "usually an EXIT sign). Do not promise it is the nearest or the "
                    "safest way out - you only know what this camera has seen."
                ),
            }

        # No confirmed exit, but signage may still point the way.
        sign_hint = []
        if getattr(pipeline, "ocr_available", False):
            for sign in pipeline.sign_tracker.current():
                if sign.navigational and "exit" in sign.text.lower():
                    sign_hint.append(sign.payload())

        doors = tracker.current()
        return {
            "found": False,
            "exit_count": 0,
            "doors_seen": len(doors),
            "doors": [d.payload() for d in doors[:3]],
            "exit_signs": sign_hint,
            "message": (
                "I haven't confirmed an exit in what the camera has seen. "
                + ("There are exit signs I can read, though. "
                   if sign_hint else "")
                + ("I can see other doors, but nothing marks them as an exit. "
                   if doors else "")
                + "Ask me to scan again after turning, and I'll take another look."
            ),
            "note": (
                "Do NOT guess which door is the exit. Report the doors you can see "
                "and any exit signage, and say plainly that none is confirmed."
            ),
        }

    async def scan_for_doors() -> Dict[str, Any]:
        """Force a fresh look — 'scan the room', 'look again for a way out'."""
        if pipeline is None or not hasattr(pipeline, "scan_doors"):
            return {"available": False}
        if not pipeline.doors_available:
            return {
                "available": False,
                "message": (
                    "Door scanning needs the online vision service, which isn't "
                    "available right now. I can still tell you about obstacles and "
                    "read signs."
                ),
            }

        # The user explicitly asked for a fresh look, so bypass the
        # scene-change gate.
        doors, _events = await pipeline.scan_doors(force=True)
        if not doors:
            return {
                "door_count": 0,
                "message": (
                    "I looked, and I can't see any doors from here. Try turning "
                    "slowly so I can see more of the room."
                ),
            }
        exits = [d for d in doors if d.is_exit]
        return {
            "door_count": len(doors),
            "doors": [d.payload() for d in doors],
            "exit_count": len(exits),
            "latency_ms": pipeline.door_detector.last_latency_ms,
            "note": (
                "Summarise briefly: how many doors, where, and which is an exit if "
                "any. Do not list every detail unless asked."
            ),
        }

    # ---------------- registration ----------------

    registry.register(ToolSpec(
        name="check_path",
        description=(
            "Report anything close and directly in the user's way - a chair, a "
            "table, a person. Use for 'is anything in front of me', 'can I walk "
            "forward', 'what's ahead'. Instant and free, so use it freely. It can "
            "never confirm a path is clear, only report what it detects."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=check_path,
    ))
    registry.register(ToolSpec(
        name="describe_doors",
        description=(
            "List the doors the camera has recently seen, with direction and "
            "whether each is open or closed. Use for 'are there any doors', 'is "
            "that door open', 'what's on my left'."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=describe_doors,
    ))
    registry.register(ToolSpec(
        name="find_exit",
        description=(
            "Find the way out. Use for 'where's the exit', 'how do I get out', "
            "'find the entrance'. Only reports a door as an exit when there is "
            "visible evidence such as an EXIT sign."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=find_exit,
        timeout_s=15.0,
    ))
    registry.register(ToolSpec(
        name="scan_for_doors",
        description=(
            "Take a fresh, careful look for doors and exits right now. Use for "
            "'scan the room', 'look again', or after the user has turned around. "
            "Takes a few seconds, so prefer describe_doors or find_exit unless a "
            "fresh look is genuinely needed."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=scan_for_doors,
        timeout_s=50.0,
    ))
