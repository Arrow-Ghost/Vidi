"""
Segment 1 acceptance test: browser -> WebSocket -> backend -> response.

The plain-text echo from the original scaffold is deliberately gone: the
protocol is now strictly JSON objects with a `type`, so anything else is a
client bug and must be reported as an error rather than silently echoed back.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_reports_status_without_leaking_secrets() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "vidi-backend"
    # gemini_configured is a derived boolean; the key itself must never appear.
    assert isinstance(body["gemini_configured"], bool)
    serialized = json.dumps(body)
    assert "AQ." not in serialized and "api_key" not in serialized.lower()


def test_health_reports_real_vision_backend_and_benchmark() -> None:
    """Performance numbers must be measured and surfaced, never guessed."""
    with TestClient(app) as started:  # triggers lifespan/startup
        body = started.get("/health").json()
        assert body["vision_available"] is True
        assert body["vision_backend"]
        bench = body["vision_benchmark"]
        assert bench and bench["p50_ms"] > 0 and bench["max_fps"] > 0


def test_websocket_greets_with_capabilities() -> None:
    with client.websocket_connect("/ws") as websocket:
        greeting = websocket.receive_json()
        assert greeting["type"] == "health"
        assert greeting["status"] == "ok"
        # The client needs to know what it can actually use before it offers
        # those affordances to the user.
        assert "vision_available" in greeting
        assert "voice_available" in greeting
        assert "find_object" in greeting["tools"]


def test_ping_pong_round_trip() -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.send_text(json.dumps({"type": "ping"}))
        assert websocket.receive_json()["type"] == "pong"


def test_malformed_and_unknown_messages_are_rejected_not_ignored() -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()

        websocket.send_text("not json at all")
        assert websocket.receive_json()["type"] == "error"

        websocket.send_text(json.dumps(["a", "list", "not", "an", "object"]))
        assert websocket.receive_json()["type"] == "error"

        websocket.send_text(json.dumps({"type": "nonsense"}))
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert "nonsense" in error["message"]


def test_text_without_voice_session_reports_error_rather_than_pretending() -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.send_text(json.dumps({"type": "text", "text": "what do you see?"}))
        response = websocket.receive_json()
        assert response["type"] == "voice_error"
