"""
Long-term memory + tasks — Segments 16, 17, 34.

SQLite, chosen deliberately over Postgres/pgvector for this deployment: no
daemon, no Docker, no RAM overhead on a machine that has ~1.8GB free, and it
survives reboots. Semantic search is brute-force cosine over stored vectors,
which is exact (not approximate like ANN indexes) and instant at personal
scale — a user will have hundreds of memories, not millions.

Everything goes through the MemoryStore interface, so swapping in Postgres
later touches this file only.

PRIVACY (see PRIVACY.md): only what the user explicitly asks to remember is
stored. No frames, no audio, no automatic transcript capture. Every record is
individually deletable, and `forget_all` really does delete.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.memory.embeddings import EmbeddingClient, cosine_similarity

logger = logging.getLogger("sfcc.memory.store")

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sfcc.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    subject      TEXT NOT NULL,
    content      TEXT NOT NULL,
    embedding    TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject);
CREATE INDEX IF NOT EXISTS idx_memories_kind    ON memories(kind);

CREATE TABLE IF NOT EXISTS tasks (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    due_at       REAL,
    due_text     TEXT,
    priority     TEXT NOT NULL DEFAULT 'normal',
    recurrence   TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    created_at   REAL NOT NULL,
    completed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

CREATE TABLE IF NOT EXISTS enrollments (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,          -- 'person' | 'belonging'
    name         TEXT NOT NULL,
    object_class TEXT,                   -- detector class, belongings only
    embedding    TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 1,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_enrollments_name ON enrollments(kind, name);

CREATE TABLE IF NOT EXISTS profile (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS friend_groups (
    name       TEXT PRIMARY KEY,
    members    TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS checklists (
    name       TEXT PRIMARY KEY,
    items      TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

# What a memory is ABOUT. Kept explicit so the UI and the delete tools can
# reason about categories without parsing free text.
MEMORY_KINDS = {"object", "person", "place", "preference", "fact"}


@dataclass
class Memory:
    id: str
    kind: str
    subject: str
    content: str
    created_at: float
    updated_at: float
    embedding: Optional[List[float]] = field(default=None, repr=False)
    score: Optional[float] = None

    def payload(self) -> Dict[str, Any]:
        data = {
            "memory_id": self.id,
            "kind": self.kind,
            "subject": self.subject,
            "content": self.content,
            "created_at": self.created_at,
        }
        if self.score is not None:
            data["match_score"] = round(self.score, 3)
        return data


@dataclass
class Enrollment:
    """
    A person or belonging the user DELIBERATELY enrolled.

    Stores an embedding, never a photograph — there is no way to reconstruct
    someone's face from this row, and deleting it removes the only record.
    """

    id: str
    kind: str            # 'person' | 'belonging'
    name: str
    object_class: Optional[str]
    embedding: List[float] = field(default_factory=list, repr=False)
    sample_count: int = 1
    created_at: float = 0.0

    def vector(self) -> "np.ndarray":
        import numpy as np

        return np.asarray(self.embedding, dtype=np.float32)

    def payload(self) -> Dict[str, Any]:
        return {
            "enrollment_id": self.id,
            "kind": self.kind,
            "name": self.name,
            "object_class": self.object_class,
            "sample_count": self.sample_count,
        }


@dataclass
class Task:
    id: str
    title: str
    due_at: Optional[float]
    due_text: Optional[str]
    priority: str
    recurrence: Optional[str]
    status: str
    created_at: float
    completed_at: Optional[float] = None

    def payload(self) -> Dict[str, Any]:
        return {
            "task_id": self.id,
            "title": self.title,
            "due": self.due_text,
            "priority": self.priority,
            "recurrence": self.recurrence,
            "status": self.status,
        }


class MemoryStore:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        embeddings: Optional[EmbeddingClient] = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because tool handlers run on the event loop
        # while inference runs in an executor thread; writes are serialized by
        # SQLite's own locking.
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()
        self.embeddings = embeddings or EmbeddingClient(None)

    def close(self) -> None:
        self._db.close()

    # ---------------- memories ----------------

    async def remember(self, kind: str, subject: str, content: str) -> Memory:
        """Store (or update) one explicitly-requested memory."""
        kind = kind if kind in MEMORY_KINDS else "fact"
        subject = subject.strip().lower()
        now = time.time()

        # Embedding text includes the subject so "backpack" retrieves a memory
        # whose content is only "it is blue".
        vector = await self.embeddings.embed(subject + ": " + content)
        encoded = json.dumps(vector) if vector else None

        existing = self._db.execute(
            "SELECT id, created_at FROM memories WHERE subject = ? AND kind = ?",
            (subject, kind),
        ).fetchone()

        if existing:
            self._db.execute(
                "UPDATE memories SET content = ?, embedding = ?, updated_at = ? WHERE id = ?",
                (content, encoded, now, existing["id"]),
            )
            self._db.commit()
            return Memory(existing["id"], kind, subject, content,
                          existing["created_at"], now, vector)

        memory_id = "mem_" + uuid.uuid4().hex[:12]
        self._db.execute(
            "INSERT INTO memories (id, kind, subject, content, embedding, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory_id, kind, subject, content, encoded, now, now),
        )
        self._db.commit()
        return Memory(memory_id, kind, subject, content, now, now, vector)

    async def search(self, query: str, limit: int = 5, min_score: float = 0.55) -> List[Memory]:
        """
        Semantic search, with keyword fallback whenever embeddings are
        unavailable (offline, no key, API error) or nothing clears min_score.
        """
        rows = self._db.execute("SELECT * FROM memories").fetchall()
        if not rows:
            return []

        query_vector = await self.embeddings.embed(query)
        if query_vector:
            scored: List[Memory] = []
            for row in rows:
                if not row["embedding"]:
                    continue
                memory = self._row_to_memory(row)
                memory.score = cosine_similarity(query_vector, json.loads(row["embedding"]))
                scored.append(memory)
            hits = [m for m in scored if m.score is not None and m.score >= min_score]
            if hits:
                hits.sort(key=lambda m: m.score or 0, reverse=True)
                return hits[:limit]

        return self._keyword_search(query, rows, limit)

    def _keyword_search(self, query: str, rows, limit: int) -> List[Memory]:
        terms = [t for t in query.lower().split() if len(t) > 2]
        matches: List[Memory] = []
        for row in rows:
            haystack = (row["subject"] + " " + row["content"]).lower()
            overlap = sum(1 for term in terms if term in haystack)
            if overlap:
                memory = self._row_to_memory(row)
                memory.score = overlap / max(len(terms), 1)
                matches.append(memory)
        matches.sort(key=lambda m: m.score or 0, reverse=True)
        return matches[:limit]

    def get_by_subject(self, subject: str) -> Optional[Memory]:
        row = self._db.execute(
            "SELECT * FROM memories WHERE subject = ? ORDER BY updated_at DESC LIMIT 1",
            (subject.strip().lower(),),
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def list_memories(self, kind: Optional[str] = None) -> List[Memory]:
        if kind:
            rows = self._db.execute(
                "SELECT * FROM memories WHERE kind = ? ORDER BY updated_at DESC", (kind,)
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def forget(self, subject: Optional[str] = None, memory_id: Optional[str] = None) -> int:
        """Returns how many records were actually deleted — never claims more."""
        if memory_id:
            cursor = self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        elif subject:
            cursor = self._db.execute(
                "DELETE FROM memories WHERE subject = ?", (subject.strip().lower(),)
            )
        else:
            return 0
        self._db.commit()
        return cursor.rowcount

    def forget_all(self) -> int:
        cursor = self._db.execute("DELETE FROM memories")
        self._db.commit()
        return cursor.rowcount

    @staticmethod
    def _row_to_memory(row) -> Memory:
        return Memory(
            id=row["id"], kind=row["kind"], subject=row["subject"],
            content=row["content"], created_at=row["created_at"],
            updated_at=row["updated_at"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        )

    # ---------------- tasks (Segment 17) ----------------

    def add_task(
        self,
        title: str,
        due_at: Optional[float] = None,
        due_text: Optional[str] = None,
        priority: str = "normal",
        recurrence: Optional[str] = None,
    ) -> Task:
        task_id = "task_" + uuid.uuid4().hex[:12]
        now = time.time()
        self._db.execute(
            "INSERT INTO tasks (id, title, due_at, due_text, priority, recurrence, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (task_id, title, due_at, due_text, priority, recurrence, now),
        )
        self._db.commit()
        return Task(task_id, title, due_at, due_text, priority, recurrence, "pending", now)

    def list_tasks(self, status: Optional[str] = "pending") -> List[Task]:
        if status:
            rows = self._db.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY COALESCE(due_at, 9e18), created_at",
                (status,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM tasks ORDER BY COALESCE(due_at, 9e18), created_at"
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def complete_task(self, task_id: str) -> bool:
        cursor = self._db.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ? AND status != 'done'",
            (time.time(), task_id),
        )
        self._db.commit()
        return cursor.rowcount > 0

    def find_task_by_title(self, text: str) -> Optional[Task]:
        row = self._db.execute(
            "SELECT * FROM tasks WHERE status = 'pending' AND LOWER(title) LIKE ?"
            " ORDER BY created_at DESC LIMIT 1",
            ("%" + text.strip().lower() + "%",),
        ).fetchone()
        return self._row_to_task(row) if row else None

    def delete_task(self, task_id: str) -> bool:
        cursor = self._db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self._db.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_task(row) -> Task:
        return Task(
            id=row["id"], title=row["title"], due_at=row["due_at"],
            due_text=row["due_text"], priority=row["priority"],
            recurrence=row["recurrence"], status=row["status"],
            created_at=row["created_at"], completed_at=row["completed_at"],
        )

    # ---------------- enrollments (Segment 8) ----------------

    def enroll(
        self,
        kind: str,
        name: str,
        embedding: List[float],
        object_class: Optional[str] = None,
    ) -> Enrollment:
        """
        Store or refine an enrollment.

        Re-enrolling an existing name AVERAGES the new embedding into the old
        one rather than replacing it, so a second photo from another angle
        makes recognition better instead of discarding what was learnt.
        """
        kind = "person" if kind == "person" else "belonging"
        name = name.strip()
        key = name.lower()
        now = time.time()

        existing = self._db.execute(
            "SELECT * FROM enrollments WHERE kind = ? AND LOWER(name) = ?", (kind, key)
        ).fetchone()

        if existing:
            import numpy as np

            previous = np.asarray(json.loads(existing["embedding"]), dtype=np.float32)
            fresh = np.asarray(embedding, dtype=np.float32)
            count = existing["sample_count"]
            merged = (previous * count + fresh) / (count + 1)
            norm = float(np.linalg.norm(merged))
            if norm > 0:
                merged = merged / norm
            self._db.execute(
                "UPDATE enrollments SET embedding = ?, sample_count = ?, updated_at = ?,"
                " object_class = COALESCE(?, object_class) WHERE id = ?",
                (json.dumps(merged.tolist()), count + 1, now, object_class, existing["id"]),
            )
            self._db.commit()
            return Enrollment(
                existing["id"], kind, existing["name"],
                object_class or existing["object_class"],
                merged.tolist(), count + 1, existing["created_at"],
            )

        enrollment_id = "enr_" + uuid.uuid4().hex[:12]
        self._db.execute(
            "INSERT INTO enrollments (id, kind, name, object_class, embedding,"
            " sample_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (enrollment_id, kind, name, object_class, json.dumps(list(embedding)), now, now),
        )
        self._db.commit()
        return Enrollment(enrollment_id, kind, name, object_class, list(embedding), 1, now)

    def list_enrollments(self, kind: Optional[str] = None) -> List[Enrollment]:
        if kind:
            rows = self._db.execute(
                "SELECT * FROM enrollments WHERE kind = ? ORDER BY name", (kind,)
            ).fetchall()
        else:
            rows = self._db.execute("SELECT * FROM enrollments ORDER BY kind, name").fetchall()
        return [self._row_to_enrollment(r) for r in rows]

    def get_enrollment(self, kind: str, name: str) -> Optional[Enrollment]:
        row = self._db.execute(
            "SELECT * FROM enrollments WHERE kind = ? AND LOWER(name) = ?",
            (kind, name.strip().lower()),
        ).fetchone()
        return self._row_to_enrollment(row) if row else None

    def delete_enrollment(self, name: str, kind: Optional[str] = None) -> int:
        """Removes the only stored record of that person or item."""
        if kind:
            cursor = self._db.execute(
                "DELETE FROM enrollments WHERE kind = ? AND LOWER(name) = ?",
                (kind, name.strip().lower()),
            )
        else:
            cursor = self._db.execute(
                "DELETE FROM enrollments WHERE LOWER(name) = ?", (name.strip().lower(),)
            )
        self._db.commit()
        return cursor.rowcount

    def delete_all_enrollments(self) -> int:
        cursor = self._db.execute("DELETE FROM enrollments")
        self._db.commit()
        return cursor.rowcount

    @staticmethod
    def _row_to_enrollment(row) -> Enrollment:
        return Enrollment(
            id=row["id"], kind=row["kind"], name=row["name"],
            object_class=row["object_class"],
            embedding=json.loads(row["embedding"]),
            sample_count=row["sample_count"], created_at=row["created_at"],
        )

    # ---------------- user profile + personality (Parts 4, 5, 46) ----------------
    #
    # Kept in its own table, separate from `memories`, because they are
    # different things: a memory is a fact the user told us ("my classroom is
    # Room 204"); the profile is how the assistant should BEHAVE. Part 46 asks
    # for exactly this separation.

    def save_profile(self, profile) -> None:
        payload = {
            "name": profile.name,
            "personality": profile.personality.weights,
            "communication": profile.communication.payload(),
            "preferences": profile.preferences,
        }
        now = time.time()
        for key, value in payload.items():
            self._db.execute(
                "INSERT INTO profile (key, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
                " updated_at = excluded.updated_at",
                (key, json.dumps(value), now),
            )
        self._db.commit()

    def load_profile(self):
        from app.profile.personality import (
            CommunicationStyle, PersonalityVector, UserProfile,
        )

        rows = {r["key"]: json.loads(r["value"]) for r in
                self._db.execute("SELECT key, value FROM profile").fetchall()}
        communication = rows.get("communication") or {}
        return UserProfile(
            name=rows.get("name"),
            personality=PersonalityVector(weights=dict(rows.get("personality") or {})),
            communication=CommunicationStyle(
                verbosity=communication.get("verbosity", "medium"),
                proactive_level=communication.get("proactive_level", "medium"),
                humor=communication.get("humor", "light"),
                encouragement=communication.get("encouragement", "medium"),
            ),
            preferences=dict(rows.get("preferences") or {}),
        )

    def reset_profile(self) -> int:
        cursor = self._db.execute("DELETE FROM profile")
        self._db.commit()
        return cursor.rowcount

    # ---------------- friend groups (Part 52) ----------------

    def save_friend_group(self, name: str, members: List[str]) -> None:
        self._db.execute(
            "INSERT INTO friend_groups (name, members, created_at) VALUES (?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET members = excluded.members",
            (name.strip().lower(), json.dumps([m.strip() for m in members]), time.time()),
        )
        self._db.commit()

    def get_friend_group(self, name: str) -> Optional[List[str]]:
        row = self._db.execute(
            "SELECT members FROM friend_groups WHERE name = ?", (name.strip().lower(),)
        ).fetchone()
        return json.loads(row["members"]) if row else None

    def list_friend_groups(self) -> Dict[str, List[str]]:
        rows = self._db.execute("SELECT name, members FROM friend_groups").fetchall()
        return {r["name"]: json.loads(r["members"]) for r in rows}

    def delete_friend_group(self, name: str) -> int:
        cursor = self._db.execute(
            "DELETE FROM friend_groups WHERE name = ?", (name.strip().lower(),)
        )
        self._db.commit()
        return cursor.rowcount

    # ---------------- checklists (Segment 18) ----------------

    def save_checklist(self, name: str, items: List[str]) -> None:
        self._db.execute(
            "INSERT INTO checklists (name, items, created_at) VALUES (?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET items = excluded.items",
            (name.strip().lower(), json.dumps(items), time.time()),
        )
        self._db.commit()

    def get_checklist(self, name: str) -> Optional[List[str]]:
        row = self._db.execute(
            "SELECT items FROM checklists WHERE name = ?", (name.strip().lower(),)
        ).fetchone()
        return json.loads(row["items"]) if row else None

    def list_checklists(self) -> Dict[str, List[str]]:
        rows = self._db.execute("SELECT name, items FROM checklists").fetchall()
        return {r["name"]: json.loads(r["items"]) for r in rows}
