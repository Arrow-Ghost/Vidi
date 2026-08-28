"""
Text embeddings for semantic memory search (Segment 16).

Uses Gemini's gemini-embedding-001 over plain REST — no SDK — because the
rest of this codebase already talks to the API that way and it keeps the
dependency surface small.

DEGRADATION IS A FEATURE HERE: if the network is down or the key is missing,
embedding returns None and the store falls back to keyword matching rather
than raising. A blind user asking "where did I say I put my charger?" should
get a keyword hit instead of an error, and should never be told the memory
does not exist just because embedding failed.

Reduced-dimension outputs (anything below the model's native 3072) are NOT
unit-normalized by the API, so this module normalizes them itself. Skipping
that would make cosine similarity subtly wrong and silently degrade recall.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import httpx

logger = logging.getLogger("sfcc.memory.embeddings")

EMBED_MODEL = "gemini-embedding-001"
EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    + EMBED_MODEL
    + ":embedContent"
)
# 768 keeps vectors small enough that brute-force cosine over a personal
# memory store (hundreds of entries) is instant, with no vector database.
EMBED_DIMS = 768
REQUEST_TIMEOUT_S = 15.0


def _normalize(values: List[float]) -> List[float]:
    magnitude = math.sqrt(sum(v * v for v in values))
    if magnitude == 0:
        return values
    return [v / magnitude for v in values]


class EmbeddingClient:
    def __init__(self, api_key: Optional[str], dims: int = EMBED_DIMS) -> None:
        self._api_key = api_key
        self._dims = dims
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def embed(self, text: str, task_type: str = "SEMANTIC_SIMILARITY") -> Optional[List[float]]:
        """
        Returns a unit-length vector, or None if embedding is unavailable.
        Never raises — callers fall back to keyword search.
        """
        if not self._api_key or not text.strip():
            return None
        payload = {
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self._dims,
        }
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
                response = await client.post(
                    EMBED_URL,
                    headers={"x-goog-api-key": self._api_key},
                    json=payload,
                )
            if response.status_code != 200:
                self.last_error = "HTTP " + str(response.status_code)
                logger.warning("embedding failed: %s %s", response.status_code, response.text[:200])
                return None
            values = response.json()["embedding"]["values"]
            self.last_error = None
            return _normalize(values)
        except Exception as exc:  # noqa: BLE001 - degradation, not failure
            self.last_error = str(exc)[:200]
            logger.warning("embedding unavailable: %s", exc)
            return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Both vectors are stored unit-length, so this is just a dot product."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
