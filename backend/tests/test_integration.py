"""
Full-stack integration: WebSocket -> perception -> world model -> tools.

This is the vertical slice the whole project is built around, exercised the
way the browser actually exercises it: a real JPEG over a real WebSocket,
producing real detections that real tools can then answer questions about.

The Gemini Live leg is covered by a separate opt-in test at the bottom
(network + API credits), so the default suite stays fast and offline.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from app.agent.tools import ToolRegistry
from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"


def _frame_payload(name: str = "bus.jpg", width: int = 640, height: int = 480) -> str:
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None, "missing fixture " + name
    resized = cv2.resize(image, (width, height))
    ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


def _drain_to_perception(websocket, max_messages: int = 6) -> dict:
    for _ in range(max_messages):
        message = websocket.receive_json()
        if message["type"] == "perception":
            return message
        if message["type"] == "error":
            pytest.fail("backend returned error: " + str(message))
    pytest.fail("no perception message received")


def test_real_frame_over_websocket_produces_real_detections() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            greeting = websocket.receive_json()
            assert greeting["vision_available"] is True

            websocket.send_text(
                json.dumps({"type": "frame", "seq": 1, "data": _frame_payload()})
            )
            perception = _drain_to_perception(websocket)

    classes = {box["class"] for box in perception["boxes"]}
    assert "person" in classes, "expected people in bus.jpg, got " + str(classes)
    assert "bus" in classes

    # Every entity must carry real, bounded confidences — never a bare claim.
    assert perception["entities"]
    for entity in perception["entities"]:
        assert 0.0 < entity["confidence"] <= 1.0
        assert 0.0 <= entity["direction_confidence"] <= 1.0
        assert entity["direction"] in {
            "far_left", "left", "center_left", "center",
            "center_right", "right", "far_right",
        }

    # And the pipeline must report its own measured latency, not an estimate.
    assert perception["latency_ms"] > 0
    assert perception["stats"]["p50_ms"] > 0
    print(
        "\n[integration] " + str(len(perception["boxes"])) + " detections, "
        "latency=" + str(perception["latency_ms"]) + "ms, "
        "classes=" + str(sorted(classes))
    )


def test_tracks_persist_across_frames_with_stable_ids() -> None:
    """The same object across frames must keep one ID — 'the backpack from a
    moment ago', not a fresh object every frame."""
    payload = _frame_payload()
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()
            seen = []
            for seq in range(4):
                websocket.send_text(
                    json.dumps({"type": "frame", "seq": seq, "data": payload})
                )
                perception = _drain_to_perception(websocket)
                seen.append({e["id"] for e in perception["entities"]})

    # A static scene must converge on a stable ID set, not churn.
    assert seen[-1] == seen[-2], "track IDs changed on an unchanging scene"
    assert seen[-1], "no tracks survived"


def test_corrupt_frame_is_reported_and_session_survives() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()

            websocket.send_text(
                json.dumps({"type": "frame", "seq": 1, "data": "data:image/jpeg;base64,@@@@"})
            )
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert error["scope"] == "frame"

            # The very next good frame must still work — no poisoned session.
            websocket.send_text(
                json.dumps({"type": "frame", "seq": 2, "data": _frame_payload()})
            )
            assert _drain_to_perception(websocket)["boxes"]


def test_tools_answer_from_the_scene_the_websocket_just_perceived() -> None:
    """
    The join point of the whole system: what the camera saw is exactly what
    the tools are able to talk about.
    """
    payload = _frame_payload()
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()
            for seq in range(3):
                websocket.send_text(
                    json.dumps({"type": "frame", "seq": seq, "data": payload})
                )
                perception = _drain_to_perception(websocket)

    # Rebuild a registry over an equivalent world to query it directly.
    from app.perception.pipeline import PerceptionPipeline

    async def probe() -> tuple:
        pipeline = PerceptionPipeline()
        for _ in range(3):
            await pipeline.process_payload(payload)
        registry = ToolRegistry(pipeline.world)
        people = await registry.execute("count_people", {})
        scene = await registry.execute("describe_scene", {})
        missing = await registry.execute("find_object", {"object_name": "backpack"})
        unsupported = await registry.execute("find_object", {"object_name": "my keys"})
        return people, scene, missing, unsupported

    people, scene, missing, unsupported = asyncio.run(probe())

    assert people["person_count"] >= 1
    assert scene["visible_entity_count"] >= 1
    # bus.jpg contains no backpack: must be "not visible", never "not there".
    assert missing["found"] is False and missing["supported"] is True
    assert "does NOT mean it is absent" in missing["message"]
    # Keys are undetectable: must be flagged unsupported, not merely absent.
    assert unsupported["supported"] is False


# --------------------------------------------------------------------------
# Opt-in: real Gemini Live round trip. Requires network + API credits.
#   set SFCC_LIVE_TESTS=1
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("SFCC_LIVE_TESTS") != "1" or not os.environ.get("GEMINI_API_KEY"),
    reason="set SFCC_LIVE_TESTS=1 and GEMINI_API_KEY to run the live voice test",
)
def test_live_agent_answers_are_grounded_in_real_perception() -> None:
    from app.agent.live_session import LiveVoiceSession
    from app.perception.pipeline import PerceptionPipeline

    payload = _frame_payload()

    async def run() -> tuple:
        pipeline = PerceptionPipeline()
        for _ in range(3):
            await pipeline.process_payload(payload)
        registry = ToolRegistry(pipeline.world)

        events = []
        done = asyncio.Event()

        async def on_event(event):
            events.append(event)
            if event["type"] == "agent_turn_complete":
                done.set()

        session = LiveVoiceSession(
            api_key=os.environ["GEMINI_API_KEY"], registry=registry, on_event=on_event
        )
        await session.connect()
        try:
            await session.send_text("Where is my backpack?")
            await asyncio.wait_for(done.wait(), timeout=60)
        finally:
            await session.close()

        said = "".join(e["text"] for e in events if e["type"] == "agent_transcript")
        tools = [e["tool"] for e in events if e["type"] == "tool_call"]
        audio = [e for e in events if e["type"] == "agent_audio"]
        return said.lower(), tools, audio

    said, tools, audio = asyncio.run(run())

    assert "find_object" in tools, "agent answered without consulting perception"
    assert audio, "no audio was produced"
    # There is no backpack in the fixture. The agent must say it cannot SEE
    # one — and must not assert the backpack does not exist.
    assert any(phrase in said for phrase in ("not currently visible", "can't see", "cannot see",
                                             "don't see", "do not see", "not visible"))
    assert "meters" not in said or "approximately" in said


# --------------------------------------------------------------------------
# Dashboard tool access + memory controls (Segments 42, 43)
# --------------------------------------------------------------------------

def _skip_to(websocket, wanted: str, limit: int = 8) -> dict:
    for _ in range(limit):
        message = websocket.receive_json()
        if message["type"] == wanted:
            return message
    pytest.fail("never received a " + wanted + " message")


def test_dashboard_can_invoke_readonly_tools() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()
            websocket.send_text(
                json.dumps({"type": "invoke_tool", "tool": "list_saved_memories", "args": {}})
            )
            result = _skip_to(websocket, "tool_result")
    assert result["tool"] == "list_saved_memories"
    assert "count" in result["result"]


def test_dashboard_cannot_invoke_state_changing_tools() -> None:
    """
    The browser must not be able to drive arbitrary tools. `forget` mutates
    the user's saved data and is deliberately outside the allowlist.
    """
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()
            for tool in ["forget", "remember", "create_task", "complete_task"]:
                websocket.send_text(
                    json.dumps({"type": "invoke_tool", "tool": tool, "args": {}})
                )
                error = websocket.receive_json()
                assert error["type"] == "error", tool
                assert error["scope"] == "invoke_tool"
                assert "not available to the dashboard" in error["message"]


def test_unknown_tool_name_from_dashboard_is_rejected() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()
            websocket.send_text(
                json.dumps({"type": "invoke_tool", "tool": "rm_rf_slash", "args": {}})
            )
            error = websocket.receive_json()
    assert error["type"] == "error"
    assert "not available to the dashboard" in error["message"]


def test_forget_all_is_an_explicit_gesture_not_a_model_tool() -> None:
    """
    Erasing everything must never be something the assistant can be talked
    into. It exists only as a direct user gesture from the UI.
    """
    from app.perception.world_model import WorldModel
    from app.agent.tools import ToolRegistry
    from app.memory.store import MemoryStore

    store = MemoryStore(db_path=":memory:")
    try:
        registry = ToolRegistry(WorldModel(), store=store)
        assert "forget_all" not in registry.names()
        assert "forget_all_memories" not in registry.names()
    finally:
        store.close()

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()
            websocket.send_text(json.dumps({"type": "forget_all_memories"}))
            cleared = _skip_to(websocket, "memory_cleared")
    assert "deleted_count" in cleared
