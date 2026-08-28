"""
Gemini vision client — deep reading and interpretation (Segment 15).

The local OCR engine reads text. This reads *meaning*: what a sign is telling
the user, whether it matters for navigation, and what a document actually
says when it is too long or too messy for PP-OCR.

Measured on the target machine (docs/PERFORMANCE.md):
  gemini-3.1-flash-lite   ~3.5 s per image, returned all 4 signs on the test
                          fixture with correct text, positions and
                          navigational classification.

Called ON DEMAND ONLY. Running this per frame would be both slow and
expensive; the local engine covers the continuous case.

The prompt is explicit that unreadable text must be omitted rather than
guessed at, and every reading carries its own confidence so the agent can
hedge. Structured output (responseSchema) is used so nothing has to be parsed
out of prose.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np

from app.perception.direction import DirectionBucket, _raw_bucket

logger = logging.getLogger("vidi.agent.vision_llm")

VISION_MODEL = "gemini-3.1-flash-lite"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
REQUEST_TIMEOUT_S = 45.0

# Downscale before upload: a 640px-wide JPEG is plenty for sign reading and
# keeps the request small on a slow connection.
MAX_UPLOAD_WIDTH = 1024
JPEG_QUALITY = 85

READ_PROMPT = """You are the reading system for a blind user's assistive device.
List EVERY piece of readable text visible in this image.

For each one return:
- text: the exact text, verbatim
- meaning: what it tells the user, in one short plain sentence
- kind: one of sign, label, document, screen, other
- navigational: true if it helps someone find their way (exits, room numbers,
  directions, platforms, floors, entrances), false otherwise
- box_2d: [ymin, xmin, ymax, xmax] normalized to 0-1000
- confidence: 0.0-1.0, how certain you are you read it correctly

CRITICAL: report ONLY text you can actually read. If something is blurry,
cut off, or too small, either omit it or give it a low confidence. NEVER guess
at illegible text — a wrong reading of a sign can send a blind user the wrong
way or into danger."""

SUMMARY_PROMPT = """You are reading a document aloud to a blind user.
Read the text in this image and give a SHORT spoken summary of what it says —
the parts a person would actually need. Lead with the most important thing.

If the image is too blurry or the text too small to read reliably, say so
plainly instead of guessing. Never invent content that is not visible."""

READ_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "readings": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "meaning": {"type": "STRING"},
                    "kind": {"type": "STRING"},
                    "navigational": {"type": "BOOLEAN"},
                    "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["text", "meaning", "kind", "navigational", "box_2d", "confidence"],
            },
        },
        "overall_note": {"type": "STRING"},
    },
    "required": ["readings"],
}


@dataclass
class Reading:
    text: str
    meaning: str
    kind: str
    navigational: bool
    confidence: float
    direction: DirectionBucket

    def payload(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "meaning": self.meaning,
            "kind": self.kind,
            "navigational": self.navigational,
            "confidence": round(self.confidence, 2),
            "direction": self.direction.value,
        }


def encode_frame(frame: np.ndarray, max_width: int = MAX_UPLOAD_WIDTH) -> str:
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise ValueError("could not encode frame as JPEG")
    return base64.b64encode(buffer.tobytes()).decode()


class VisionReader:
    """
    Gemini-backed reader. Every method returns a structured dict and never
    raises — a network failure must surface as "I couldn't read that right
    now", not as a dropped conversational turn.
    """

    def __init__(self, api_key: Optional[str], model: str = VISION_MODEL) -> None:
        self._api_key = api_key
        self.model = model
        self.last_latency_ms: Optional[float] = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def _post(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = API_BASE + self.model + ":generateContent"
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
                response = await client.post(
                    url, headers={"x-goog-api-key": self._api_key}, json=body
                )
            self.last_latency_ms = (time.monotonic() - started) * 1000
            if response.status_code != 200:
                logger.warning("vision call failed: %s %s", response.status_code,
                               response.text[:200])
                return None
            return response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("vision call errored: %s", exc)
            return None

    async def read_text(self, frame: np.ndarray) -> Dict[str, Any]:
        """Read and interpret every piece of text in the frame."""
        if not self.available:
            return {"ok": False, "reason": "no_api_key",
                    "message": "Deep reading is unavailable because no API key is configured."}
        try:
            encoded = encode_frame(frame)
        except ValueError as exc:
            return {"ok": False, "reason": "encode_failed", "message": str(exc)}

        body = {
            "contents": [{"parts": [
                {"text": READ_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
            ]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": READ_SCHEMA,
                # Deterministic: the same sign must not read differently twice.
                "temperature": 0.0,
            },
        }
        payload = await self._post(body)
        if payload is None:
            return {"ok": False, "reason": "unavailable",
                    "message": "I couldn't read that right now — the reading service didn't respond."}

        try:
            raw = json.loads(payload["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.warning("could not parse vision response: %s", exc)
            return {"ok": False, "reason": "bad_response",
                    "message": "I couldn't make sense of what I read there."}

        readings = [
            Reading(
                text=item.get("text", ""),
                meaning=item.get("meaning", ""),
                kind=item.get("kind", "other"),
                navigational=bool(item.get("navigational")),
                confidence=float(item.get("confidence", 0.0)),
                direction=_direction_from_box(item.get("box_2d")),
            )
            for item in raw.get("readings", [])
            if item.get("text")
        ]
        return {
            "ok": True,
            "reading_count": len(readings),
            "readings": [r.payload() for r in readings],
            "latency_ms": round(self.last_latency_ms or 0, 0),
            "note": raw.get("overall_note", ""),
        }

    async def summarize(self, frame: np.ndarray, question: str = "") -> Dict[str, Any]:
        """Spoken-length summary of a document/menu/notice."""
        if not self.available:
            return {"ok": False, "reason": "no_api_key",
                    "message": "Deep reading is unavailable because no API key is configured."}
        try:
            encoded = encode_frame(frame)
        except ValueError as exc:
            return {"ok": False, "reason": "encode_failed", "message": str(exc)}

        prompt = SUMMARY_PROMPT
        if question:
            prompt += "\n\nThe user specifically asked: " + question

        body = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
            ]}],
            "generationConfig": {"temperature": 0.2},
        }
        payload = await self._post(body)
        if payload is None:
            return {"ok": False, "reason": "unavailable",
                    "message": "I couldn't read that right now."}
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            return {"ok": False, "reason": "bad_response",
                    "message": "I couldn't make sense of that page."}
        return {"ok": True, "summary": text, "latency_ms": round(self.last_latency_ms or 0, 0)}


def _direction_from_box(box: Optional[List[int]]) -> DirectionBucket:
    """
    Gemini returns [ymin, xmin, ymax, xmax] normalized to 0-1000. Convert the
    horizontal centre into the same direction buckets the rest of the system
    uses, so spoken directions are consistent no matter which reader produced
    them.
    """
    if not box or len(box) != 4:
        return DirectionBucket.CENTER
    try:
        center = (float(box[1]) + float(box[3])) / 2 / 1000.0
    except (TypeError, ValueError):
        return DirectionBucket.CENTER
    return _raw_bucket(min(max(center, 0.0), 1.0))
