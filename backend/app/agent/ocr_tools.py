"""
Reading tools — Segment 15.

Three distinct jobs, deliberately separate tools rather than one overloaded
"read" call, because they have very different costs and failure modes:

  find_signs   — what signage is around me right now?   local, fast, free
  read_text    — read and explain what is in front of me  Gemini, ~3.5s
  summarize_document — long text, spoken-length summary   Gemini, ~3.5s

find_signs answers from the local OCR pass that has already run, so it is
instant and can be asked repeatedly while walking. read_text does a fresh
deep read when the user actually points the camera at something.

HONESTY RULES ENCODED HERE:
- Text that could not be read confidently is never invented. Low-confidence
  reads are reported as unreadable, with advice to move closer.
- "No signs visible" never means "there are no signs" — only that none were
  readable in the camera's current view.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.agent.tools import DIRECTION_PHRASE

logger = logging.getLogger("vidi.agent.ocr_tools")

# Below this a local read is reported as "there is text but I can't read it"
# rather than spoken as fact.
READABLE_FLOOR = 0.70


def register_ocr_tools(registry, pipeline, vision) -> None:
    """
    Attach reading tools. `pipeline` supplies frames and the local OCR pass;
    `vision` is the Gemini reader (may be unavailable — tools degrade).
    """
    from app.agent.tools import ToolSpec

    async def find_signs() -> Dict[str, Any]:
        """Signage currently in view, from the continuous local OCR pass."""
        if not pipeline.ocr_available:
            return {
                "available": False,
                "message": "Text reading is not available — the OCR engine failed to load.",
            }

        # Refresh first so the answer reflects now, not the last tick.
        await pipeline.scan_text()
        signs = pipeline.sign_tracker.current()

        if not signs:
            return {
                "found": False,
                "sign_count": 0,
                "message": (
                    "I can't read any signs in the camera's current view. That does NOT "
                    "mean there are no signs — they may be outside the view, too far, or "
                    "at an angle I can't read."
                ),
            }

        payload = []
        for sign in signs:
            entry = sign.payload()
            entry["direction_phrase"] = DIRECTION_PHRASE.get(
                sign.direction.value, sign.direction.value
            )
            entry["readable"] = sign.confidence >= READABLE_FLOOR
            payload.append(entry)

        return {
            "found": True,
            "sign_count": len(payload),
            "signs": payload,
            "note": (
                "Signs are listed with warnings first, then navigational ones. Read the "
                "text verbatim — do not paraphrase a sign's wording. If readable is "
                "false, say you can see text there but cannot read it clearly."
            ),
        }

    async def read_text(question: str = "") -> Dict[str, Any]:
        """Deep read of whatever the camera is pointed at, with meaning."""
        frame = pipeline.current_frame()
        if frame is None:
            return {
                "ok": False,
                "message": "There's no camera image yet, so there is nothing to read.",
            }
        if not vision.available:
            # Fall back to the local engine — worse, but real.
            regions, _ = await pipeline.scan_text()
            if not regions:
                return {
                    "ok": False,
                    "message": (
                        "I can't read any text there. Try moving the camera closer or "
                        "holding it steady."
                    ),
                }
            return {
                "ok": True,
                "degraded": True,
                "readings": [r.payload() for r in regions],
                "note": (
                    "Read using the offline engine only, so the exact words are "
                    "available but not their meaning."
                ),
            }

        result = await vision.read_text(frame)
        if not result.get("ok"):
            return result

        readings = result.get("readings", [])
        if not readings:
            return {
                "ok": True,
                "reading_count": 0,
                "message": (
                    "I couldn't find any readable text in view. It may be out of frame, "
                    "too small, or too blurry."
                ),
            }

        confident = [r for r in readings if r["confidence"] >= READABLE_FLOOR]
        unclear = [r for r in readings if r["confidence"] < READABLE_FLOOR]
        for reading in readings:
            reading["direction_phrase"] = DIRECTION_PHRASE.get(
                reading["direction"], reading["direction"]
            )

        return {
            "ok": True,
            "reading_count": len(readings),
            "readings": readings,
            "confident_count": len(confident),
            "unclear_count": len(unclear),
            "latency_ms": result.get("latency_ms"),
            "note": (
                "Quote the text verbatim. Anything with confidence below 0.7 must be "
                "described as unclear — tell the user to move closer rather than "
                "guessing what it says."
            ),
        }

    async def summarize_document(question: str = "") -> Dict[str, Any]:
        """Spoken-length summary of a page, menu, or notice."""
        frame = pipeline.current_frame()
        if frame is None:
            return {"ok": False, "message": "There's no camera image yet."}
        if not vision.available:
            return {
                "ok": False,
                "message": (
                    "I can read individual words offline, but summarising a document "
                    "needs the online reader, which isn't available right now."
                ),
            }
        return await vision.summarize(frame, question)

    registry.register(ToolSpec(
        name="find_signs",
        description=(
            "List the signboards and signage the camera can currently read, each with "
            "its direction. Use for 'are there any signs', 'what does that sign say', "
            "'is there a sign for the exit', and whenever the user is trying to find "
            "their way. Fast and free — safe to call often while walking."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=find_signs,
        timeout_s=12.0,
    ))

    registry.register(ToolSpec(
        name="read_text",
        description=(
            "Carefully read and explain the text the camera is pointed at, including "
            "what it means. Use for 'read this', 'what does this say', labels, "
            "packaging, notices, and signs the user is holding the camera up to. "
            "Slower than find_signs, so use it when the user is deliberately reading "
            "something rather than just walking past."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "question": {
                    "type": "STRING",
                    "description": "What the user specifically wants to know, if anything.",
                }
            },
        },
        handler=read_text,
        timeout_s=50.0,
    ))

    registry.register(ToolSpec(
        name="summarize_document",
        description=(
            "Read a longer document, menu, letter or notice and give a short spoken "
            "summary. Use for 'summarize this', 'what's important here', 'what's on "
            "the menu'."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "question": {"type": "STRING", "description": "Any specific question."}
            },
        },
        handler=summarize_document,
        timeout_s=50.0,
    ))
