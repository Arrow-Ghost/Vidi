#!/usr/bin/env python3
"""
Standalone Gemini connectivity + function-calling check.

Why this exists: the cloud sandbox this project was originally scaffolded
in cannot reach generativelanguage.googleapis.com (its network is
allowlisted to package registries only), so Gemini connectivity could not
be verified there — see ARCHITECTURE.md's Segment 11/12 status note. This
script is meant to be run on a machine with normal internet access (this
one) to close that gap before Segments 11+ (conversation engine, agent
orchestrator) get built on top of it.

Usage:
    cd backend
    python scripts/check_gemini.py

Requires backend/.env to contain a real GEMINI_API_KEY (copy
backend/.env.example if you haven't yet) and `pip install -r
requirements.txt` to have been run in this environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    if not settings.gemini_api_key:
        print("FAIL: GEMINI_API_KEY is not set in backend/.env")
        return 1

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("FAIL: google-genai is not installed. Run: pip install google-genai")
        return 1

    client = genai.Client(api_key=settings.gemini_api_key)

    # 1. Plain text round trip.
    print("Checking basic text generation...")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Reply with exactly the word: PONG",
        )
        print(f"  OK: model replied {response.text!r}")
    except Exception as exc:  # noqa: BLE001 — this is a diagnostic script
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        return 1

    # 2. Function calling — this is what the agent orchestrator (Segment 12)
    #    depends on: Gemini choosing a tool and structured arguments, never
    #    inventing a tool result itself.
    print("Checking function calling...")
    find_object_tool = types.FunctionDeclaration(
        name="find_object",
        description="Search the current camera view for a named object.",
        parameters={
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "description": "The object to search for, e.g. 'backpack'"},
            },
            "required": ["object_name"],
        },
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Where is my backpack?",
            config=types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=[find_object_tool])],
            ),
        )
        call = response.candidates[0].content.parts[0].function_call
        if call is None:
            print(f"  FAIL: model did not call a tool. Raw response: {response.text!r}")
            return 1
        print(f"  OK: model called {call.name}({dict(call.args)})")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        return 1

    print("\nAll checks passed — Gemini connectivity and function calling both work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
