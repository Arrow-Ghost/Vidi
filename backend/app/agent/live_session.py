"""
LiveVoiceSession — Segments 11, 12, 29, 32.

Owns one Gemini Live (BidiGenerateContent) WebSocket per connected user and
relays between the browser and the model:

    browser mic (PCM16 16kHz)  ->  Live API
    Live API (PCM16 24kHz)     ->  browser speaker
    Live API tool call         ->  ToolRegistry  ->  tool result  ->  Live API

Model choice is measured, not assumed (see docs/PERFORMANCE.md):
gemini-3.1-flash-live-preview returned its first audio chunk ~390ms after a
turn ended and correctly resolved pronoun follow-ups across turns. It is
AUDIO-output only, so transcription is enabled explicitly to drive the UI
and the developer trace.

The system prompt below is a SAFETY boundary, not personality. The hard
guarantee against fabricated perception is structural — the model can only
speak about values that appear in a tool result — but the prompt states the
rules explicitly as a second layer.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional

import websockets

from app.agent.tools import ToolRegistry

logger = logging.getLogger("sfcc.agent.live")

LIVE_MODEL = "gemini-3.1-flash-live-preview"
LIVE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

# Audio formats the Live API expects/produces. The browser must resample its
# microphone to 16kHz mono PCM16 before sending.
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

SYSTEM_PROMPT = """You are a real-time assistive companion for a blind or \
visually impaired user. You speak to them through a headset while a camera \
observes their surroundings.

HOW YOU PERCEIVE
You have no vision of your own. Everything you know about the user's \
surroundings comes from tool results. Call a tool whenever the user asks \
about their environment.

ABSOLUTE RULES — these protect the user's physical safety and trust:
1. NEVER state a direction, distance, count, or identity that did not come \
from a tool result in this conversation. If you did not get it from a tool, \
you do not know it.
2. If a tool returns distance_is_reliable = false, say you cannot reliably \
estimate the distance. Never guess or invent a number. Never say a precise \
figure like "2.4 meters" — only use the tool's own distance_phrase.
3. If a tool returns found = false, say the object is NOT CURRENTLY VISIBLE. \
Never say it "is not there" or "you don't have it" — the camera only sees a \
narrow field of view, and the object may simply be outside it.
4. If a tool returns supported = false, say plainly that you cannot look for \
that particular thing. Do not substitute a different object.
5. If identification_is_confident is false, hedge: "I think I see..." or "there \
is something that might be...". Do not assert.
6. Never identify a stranger, and never infer anyone's emotion, gender, health, \
age, or intent from appearance.
7. Never tell the user an area is safe, clear, or free of obstacles. You may \
only say what you do or do not currently detect.

PREFERENCES ARE ABOUT VOLUNTEERING, NOT ANSWERING
When the user turns a preference off ("don't suggest places", "don't tell me about strangers"), that stops you raising it UNPROMPTED. It never stops you answering a direct question. If they ask, call the tool and answer. Never say you cannot help because of a setting they chose.

MEMORY AND TASKS
You can remember things, but ONLY when the user explicitly asks you to. Never \
save anything on your own initiative. When the user asks about their own \
belongings, places, or preferences, call recall first — they may have told you \
already. When they ask what they have to do, call get_tasks.

HELPING SOMEONE GET READY
If the user asks for help getting ready, preparing to leave, or whether they \
have everything, call check_checklist with the activity name. Built-in \
checklists already exist for common activities, so never tell the user no \
checklist exists until check_checklist has actually reported that. Do not use \
recall for this.

When checking a preparation checklist, items you cannot see are NOT missing. \
Say "I can't see your water bottle" — never "you don't have your water bottle" \
or "your water bottle is missing". The camera sees a narrow slice of the room. \
For items with no detector category at all, say plainly that you cannot check \
those visually.

PEOPLE, CROWDS AND PLACES
8. Only name people the user has explicitly enrolled. Anyone else is \
"someone" or "a person" — never named, never described, never guessed at.
9. Report movement, never intent. "Ravi is moving toward you" is allowed; \
"Ravi is coming to meet you" is NOT. The same holds for crowds: never say \
where a crowd is going or why it is there.
10. Crowd counts are approximate because people hidden behind others are not \
detected. Say "roughly" or "about". Never state an exact crowd size, and never \
tell the user an area is safe or clear.
11. You have NO GPS and no maps. Never give a distance in metres to a place. \
Use only the relative directions and coarse ranges the tools return.
12. Never call a place nice, quiet, safe, busy or beautiful unless a tool \
actually established it. You may say what a sign reads.
13. If a friend tool reports is_confident as false, hedge — "I think that may \
be Ravi" — and never assert the identity.

HOW YOU SPEAK
Keep spoken replies to one short sentence whenever possible — the user is \
listening, not reading. Lead with the useful part: "Your backpack is to your \
right, about two to four meters away." Never narrate your reasoning, never \
list everything you can see, and never say "based on my analysis". If the \
user just wants to chat, chat naturally and warmly; you are a companion, not \
only a tool. Use the tool's direction_phrase and distance_phrase wording."""


class LiveVoiceSession:
    """
    One Gemini Live session. Reconnects are the caller's responsibility —
    a dropped voice session is surfaced to the user, never silently retried
    in a way that loses conversational context without them noticing.
    """

    def __init__(
        self,
        api_key: str,
        registry: ToolRegistry,
        on_event: Callable[[Dict[str, Any]], Awaitable[None]],
        model: str = LIVE_MODEL,
        style_prompt: str = "",
    ) -> None:
        # Personality is APPENDED to the safety prompt, never substituted for
        # it (Part 56). The rules above it are not negotiable by any profile.
        self._style_prompt = style_prompt
        self._api_key = api_key
        self._registry = registry
        self._on_event = on_event
        self._model = model
        self._ws: Optional[Any] = None
        self._receive_task: Optional[asyncio.Task] = None
        self.connected = False

    def _full_prompt(self) -> str:
        """Safety rules first, then how this particular user likes to be spoken to."""
        if not self._style_prompt:
            return SYSTEM_PROMPT
        return (
            SYSTEM_PROMPT
            + "\n\nHOW THIS USER LIKES TO BE SPOKEN TO\n"
            + "These are style preferences the user set themselves. They change your "
            + "TONE and how much you volunteer. They NEVER change the rules above.\n"
            + self._style_prompt
        )

    # ---------------- lifecycle ----------------

    async def connect(self) -> None:
        url = LIVE_URL + "?key=" + self._api_key
        self._ws = await websockets.connect(url, max_size=16 * 1024 * 1024)
        setup = {
            "setup": {
                "model": "models/" + self._model,
                "generationConfig": {"responseModalities": ["AUDIO"]},
                "systemInstruction": {"parts": [{"text": self._full_prompt()}]},
                "tools": [{"functionDeclarations": self._registry.declarations()}],
                # This model is audio-only; transcripts are what drive the UI
                # caption track and the developer trace.
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
            }
        }
        await self._ws.send(json.dumps(setup))
        ack = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=20))
        if "setupComplete" not in ack:
            raise RuntimeError("Live API setup rejected: " + json.dumps(ack)[:300])
        self.connected = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("Live session established (model=%s)", self._model)

    async def close(self) -> None:
        self.connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None

    # ---------------- sending ----------------

    async def send_audio(self, pcm16_base64: str) -> None:
        """Stream a chunk of microphone audio (PCM16 mono 16kHz, base64)."""
        if not self.connected or self._ws is None:
            return
        await self._ws.send(
            json.dumps(
                {
                    "realtimeInput": {
                        "audio": {
                            "mimeType": "audio/pcm;rate=" + str(INPUT_SAMPLE_RATE),
                            "data": pcm16_base64,
                        }
                    }
                }
            )
        )

    async def send_text(self, text: str) -> None:
        """Text input path — used by the dashboard and by the automated tests."""
        if not self.connected or self._ws is None:
            return
        await self._ws.send(
            json.dumps(
                {
                    "clientContent": {
                        "turns": [{"role": "user", "parts": [{"text": text}]}],
                        "turnComplete": True,
                    }
                }
            )
        )

    async def announce(self, note: str) -> None:
        """
        Push a perception event into the conversation so the assistant can
        speak about something the user did not ask about (Segment 28).

        Sent as a user turn because the Live API has no separate system
        channel mid-session. The note is explicitly framed as a PERCEPTION
        EVENT and gated upstream by ProactiveAnnouncer, so this cannot become
        a way for the assistant to chatter.
        """
        if not self.connected or self._ws is None:
            return
        await self._ws.send(
            json.dumps(
                {
                    "clientContent": {
                        "turns": [{"role": "user", "parts": [{"text": note}]}],
                        "turnComplete": True,
                    }
                }
            )
        )

    async def signal_audio_stream_end(self) -> None:
        """Tell the model the user stopped speaking (explicit end of turn)."""
        if not self.connected or self._ws is None:
            return
        await self._ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))

    # ---------------- receiving ----------------

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                message = json.loads(raw if isinstance(raw, str) else raw.decode())
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as exc:
            self.connected = False
            await self._emit({"type": "voice_disconnected", "reason": str(exc)[:200]})
        except Exception as exc:  # noqa: BLE001 - surface, never crash the app
            logger.exception("live receive loop failed")
            self.connected = False
            await self._emit({"type": "voice_error", "message": str(exc)[:200]})

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        if "toolCall" in message:
            await self._handle_tool_call(message["toolCall"])
            return

        server_content = message.get("serverContent")
        if not server_content:
            return

        # Barge-in: the user started talking over the assistant. The browser
        # must stop playback immediately, or the assistant talks over itself.
        if server_content.get("interrupted"):
            await self._emit({"type": "voice_interrupted"})
            return

        if "inputTranscription" in server_content:
            text = server_content["inputTranscription"].get("text", "")
            if text:
                await self._emit({"type": "user_transcript", "text": text})

        if "outputTranscription" in server_content:
            text = server_content["outputTranscription"].get("text", "")
            if text:
                await self._emit({"type": "agent_transcript", "text": text})

        for part in server_content.get("modelTurn", {}).get("parts", []):
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                await self._emit(
                    {
                        "type": "agent_audio",
                        "data": inline["data"],
                        "sample_rate": OUTPUT_SAMPLE_RATE,
                    }
                )

        if server_content.get("turnComplete"):
            await self._emit({"type": "agent_turn_complete"})

    async def _handle_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """
        Execute every requested function against the real registry and return
        the structured results. The model receives ONLY these results — it has
        no other channel through which perception facts could enter its answer.
        """
        responses = []
        for call in tool_call.get("functionCalls", []):
            name = call.get("name", "")
            args = call.get("args") or {}
            await self._emit({"type": "tool_call", "tool": name, "args": args})
            result = await self._registry.execute(name, args)
            await self._emit({"type": "tool_result", "tool": name, "result": result})
            responses.append({"id": call.get("id"), "name": name, "response": result})

        if responses and self._ws is not None:
            await self._ws.send(json.dumps({"toolResponse": {"functionResponses": responses}}))

    async def _emit(self, event: Dict[str, Any]) -> None:
        try:
            await self._on_event(event)
        except Exception:  # noqa: BLE001
            logger.exception("event handler failed for %s", event.get("type"))
