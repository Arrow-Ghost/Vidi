# Security

## Secrets

- All secrets (Gemini API key, future DB credentials) live only in
  `backend/.env`, which is gitignored. `backend/.env.example` documents the
  shape with empty values.
- The frontend never receives a secret. Only `PUBLIC_*` prefixed, non-secret
  values (e.g. the backend WebSocket URL) are readable in browser code —
  this is enforced by Astro/Vite convention (`import.meta.env.PUBLIC_*`) and
  must not be worked around.
- `Settings` (`backend/app/core/config.py`) is the only place `os.environ` /
  env files are read. No other backend module should read env vars directly
  — this keeps every secret access auditable in one file.
- API responses may expose a *derived boolean* (e.g. `gemini_configured`)
  but never the underlying value.

## Network

- CORS origins are explicit and configured via `CORS_ORIGINS`, not `*`, once
  a real frontend origin is deployed. The Segment 1 default
  (`http://localhost:4321`) matches the Astro dev server.
- WebSocket connections will validate origin and (once auth exists, Segment
  12+) a session/auth token before accepting anything beyond the health
  handshake.

## Tool execution (from Segment 12 onward)

- The LLM never executes arbitrary code. Every tool is a typed, named,
  schema-validated function the backend implements and executes; the LLM
  only selects which tool to call and with what (validated) arguments.
- Every tool has: name, description, JSON schema, timeout, error handling,
  and a structured result schema. Malformed or out-of-schema tool calls are
  rejected before execution, not "best-effort" interpreted.
- Tool results are the only source of truth the agent may speak from — the
  agent must not invent distances, identities, or OCR text not present in a
  tool result (see `PRIVACY.md` and the safety policy in `ARCHITECTURE.md`).

## Data handling

- No permanent camera or microphone recording by default (see `PRIVACY.md`).
- Trusted-person biometric data (Segment 8) is stored only for explicitly
  enrolled people and is deletable/revocable by the user at any time.

## Dependencies

- Backend and frontend dependency versions are pinned
  (`requirements.txt`, `package.json`) rather than left floating, so a
  `pip install` / `npm install` six months from now doesn't silently change
  behavior.

## Reporting gaps

This file is updated every segment that touches auth, storage, or network
boundaries — it is not a one-time checklist filled in at the end.
