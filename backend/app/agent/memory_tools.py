"""
Memory, task, and preparation tools — Segments 16, 17, 18, 34.

Registered onto the ToolRegistry only when a MemoryStore is configured, so
the perception-only path keeps working with no database at all.

DESIGN NOTE ON TIME: natural language like "college at 10 tomorrow" is
converted to a concrete timestamp by the MODEL, which is genuinely good at
that, and passed here as ISO 8601. This layer does not attempt its own date
parsing — a half-working date parser silently scheduling a reminder for the
wrong day is worse than requiring the model to be explicit. `due_text` keeps
the user's own phrasing so the assistant can repeat it back naturally.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agent import vocabulary
from app.memory.store import MEMORY_KINDS, MemoryStore

logger = logging.getLogger("sfcc.agent.memory_tools")

# Segment 18: sensible starting checklist, overridable per user.
DEFAULT_CHECKLISTS: Dict[str, List[str]] = {
    "college": ["backpack", "phone", "wallet", "keys", "water bottle", "id card"],
    "work": ["laptop", "phone", "wallet", "keys", "backpack"],
    "gym": ["water bottle", "phone", "keys"],
}


def _join(items: List[str]) -> str:
    """Natural spoken list: 'a', 'a and b', 'a, b and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _parse_iso(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        text = value.replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except (ValueError, TypeError):
        logger.warning("could not parse due_iso=%r; storing without a timestamp", value)
        return None


def register_memory_tools(registry, store: MemoryStore) -> None:
    """Attach memory/task/checklist tools to an existing ToolRegistry."""
    from app.agent.tools import ToolSpec, _entity_payload

    # ---------------- memory (Segment 16 / 34) ----------------

    async def remember(subject: str, content: str, kind: str = "fact") -> Dict[str, Any]:
        if kind not in MEMORY_KINDS:
            kind = "fact"
        memory = await store.remember(kind, subject, content)
        return {
            "saved": True,
            "memory_id": memory.id,
            "subject": memory.subject,
            "kind": memory.kind,
            "content": memory.content,
        }

    async def recall(query: str) -> Dict[str, Any]:
        results = await store.search(query, limit=5)
        if not results:
            return {
                "found": False,
                "query": query,
                "message": (
                    "Nothing has been saved about that. Only things the user explicitly "
                    "asked to remember are stored."
                ),
            }
        return {
            "found": True,
            "query": query,
            "match_count": len(results),
            "memories": [m.payload() for m in results],
        }

    async def forget(subject: str) -> Dict[str, Any]:
        deleted = store.forget(subject=subject)
        return {
            "deleted_count": deleted,
            "subject": subject,
            # Report the truth: claiming success on a no-op would leave the
            # user believing private data was erased when it was not.
            "message": (
                "Deleted " + str(deleted) + " saved memory about " + subject + "."
                if deleted
                else "There was nothing saved about " + subject + ", so nothing was deleted."
            ),
        }

    async def list_saved_memories(kind: str = "") -> Dict[str, Any]:
        memories = store.list_memories(kind or None)
        return {
            "count": len(memories),
            "memories": [m.payload() for m in memories[:25]],
        }

    # ---------------- tasks (Segment 17) ----------------

    async def create_task(
        title: str,
        due_iso: str = "",
        due_text: str = "",
        priority: str = "normal",
        recurrence: str = "",
    ) -> Dict[str, Any]:
        task = store.add_task(
            title=title,
            due_at=_parse_iso(due_iso),
            due_text=due_text or None,
            priority=priority if priority in {"low", "normal", "high"} else "normal",
            recurrence=recurrence or None,
        )
        return {"created": True, **task.payload()}

    async def get_tasks(status: str = "pending") -> Dict[str, Any]:
        tasks = store.list_tasks(status or None)
        if not tasks:
            return {"count": 0, "tasks": [], "message": "There are no tasks saved."}
        return {"count": len(tasks), "tasks": [t.payload() for t in tasks]}

    async def complete_task(task_id: str = "", title: str = "") -> Dict[str, Any]:
        target_id = task_id
        if not target_id and title:
            match = store.find_task_by_title(title)
            if match is None:
                return {
                    "completed": False,
                    "message": "No pending task matching " + repr(title) + " was found.",
                }
            target_id = match.id
        if not target_id:
            return {"completed": False, "message": "Provide either task_id or title."}
        done = store.complete_task(target_id)
        return {
            "completed": done,
            "task_id": target_id,
            "message": "Marked complete." if done else "That task was not pending.",
        }

    # ---------------- preparation mode (Segment 18) ----------------

    async def check_checklist(name: str) -> Dict[str, Any]:
        """
        Cross-references a checklist against what the camera can currently see.

        The wording of the result matters enormously: an item that is not
        visible is reported as NOT SEEN, never as missing. The camera has a
        narrow field of view, and telling a blind user their wallet "is
        missing" when it is simply behind them would be actively harmful.
        """
        items = store.get_checklist(name) or DEFAULT_CHECKLISTS.get(name.strip().lower())
        if not items:
            return {
                "found": False,
                "checklist": name,
                "available": sorted(set(list(store.list_checklists()) + list(DEFAULT_CHECKLISTS))),
                "message": "No checklist named " + repr(name) + " exists.",
            }

        seen, not_seen, cannot_check = [], [], []
        for item in items:
            classes = vocabulary.resolve(item)
            if classes is None:
                cannot_check.append(item)
                continue
            matches = []
            for class_name in classes:
                matches.extend(registry.world.find(class_name))
            if matches:
                best = max(matches, key=lambda e: e.confidence)
                seen.append({"item": item, **_entity_payload(best)})
            else:
                not_seen.append(item)

        # The three categories mean genuinely different things and were being
        # blurred together in speech ("I can't check your water bottle" when it
        # was merely out of view). Handing the model ready-made phrasing for
        # each category, rather than only a rule, keeps them distinct.
        say = []
        if seen:
            say.append("I can see your " + _join([s["item"] for s in seen]) + ".")
        if not_seen:
            say.append("I can't see your " + _join(not_seen) + " right now.")
        if cannot_check:
            say.append("I have no way to check your " + _join(cannot_check) + ".")

        return {
            "found": True,
            "checklist": name,
            "items": items,
            "seen": seen,
            "not_currently_visible": not_seen,
            "cannot_be_checked_visually": cannot_check,
            "suggested_phrasing": " ".join(say),
            "note": (
                "These three groups mean DIFFERENT things and must not be mixed up. "
                "not_currently_visible = in the checklist but not in the camera's view "
                "right now; say 'I can't see your X right now'. NEVER say it is missing "
                "or that the user does not have it. "
                "cannot_be_checked_visually = there is no detector for it at all; say "
                "'I have no way to check your X'. "
                "Prefer speaking suggested_phrasing, which already states each group "
                "correctly."
            ),
        }

    async def save_checklist(name: str, items: str) -> Dict[str, Any]:
        parsed = [i.strip() for i in items.split(",") if i.strip()]
        if not parsed:
            return {"saved": False, "message": "No items provided."}
        store.save_checklist(name, parsed)
        return {"saved": True, "checklist": name, "items": parsed}

    # ---------------- registration ----------------

    registry.register(ToolSpec(
        name="remember",
        description=(
            "Save something the user explicitly asked you to remember, such as "
            "'my backpack is blue' or 'my classroom is Room 204'. Only call this when "
            "the user actually asks you to remember something."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "subject": {"type": "STRING", "description": "Short key, e.g. 'backpack', 'classroom'."},
                "content": {"type": "STRING", "description": "What to remember about it."},
                "kind": {"type": "STRING",
                         "description": "One of: object, person, place, preference, fact."},
            },
            "required": ["subject", "content"],
        },
        handler=remember,
    ))
    registry.register(ToolSpec(
        name="recall",
        description=(
            "Search everything the user has previously asked you to remember. Use this "
            "before answering questions about the user's own belongings, places, or "
            "preferences."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING"}},
            "required": ["query"],
        },
        handler=recall,
    ))
    registry.register(ToolSpec(
        name="forget",
        description="Delete a saved memory by its subject. Use when the user says to forget something.",
        parameters={
            "type": "OBJECT",
            "properties": {"subject": {"type": "STRING"}},
            "required": ["subject"],
        },
        handler=forget,
    ))
    registry.register(ToolSpec(
        name="list_saved_memories",
        description="List everything currently saved. Use when the user asks what you remember.",
        parameters={"type": "OBJECT", "properties": {"kind": {"type": "STRING"}}},
        handler=list_saved_memories,
    ))
    registry.register(ToolSpec(
        name="create_task",
        description=(
            "Create a task or reminder. Convert any natural date/time the user gives "
            "('at 10 tomorrow') into due_iso as ISO 8601, and keep their own wording in "
            "due_text so it can be repeated back naturally."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING"},
                "due_iso": {"type": "STRING", "description": "ISO 8601, e.g. 2026-08-29T10:00:00."},
                "due_text": {"type": "STRING", "description": "The user's own phrasing."},
                "priority": {"type": "STRING", "description": "low, normal, or high."},
                "recurrence": {"type": "STRING", "description": "e.g. daily, weekly."},
            },
            "required": ["title"],
        },
        handler=create_task,
    ))
    registry.register(ToolSpec(
        name="get_tasks",
        description="List the user's tasks. Use for 'what do I have today', 'what do I need to do'.",
        parameters={"type": "OBJECT", "properties": {"status": {"type": "STRING"}}},
        handler=get_tasks,
    ))
    registry.register(ToolSpec(
        name="complete_task",
        description="Mark a task done, by task_id or by matching its title.",
        parameters={
            "type": "OBJECT",
            "properties": {"task_id": {"type": "STRING"}, "title": {"type": "STRING"}},
        },
        handler=complete_task,
    ))
    registry.register(ToolSpec(
        name="check_checklist",
        description=(
            "USE THIS FIRST whenever the user asks for help getting ready, preparing to "
            "leave, or checking they have everything — for example 'help me get ready for "
            "college', 'am I ready to go', 'do I have everything'. Pass the activity name "
            "(e.g. 'college', 'work', 'gym'). Built-in checklists already exist for common "
            "activities, so call this BEFORE concluding that no checklist exists; do NOT "
            "use recall for this. It checks the list against what the camera can currently "
            "see. Items not seen are NOT missing — they are simply outside the camera's view."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {"name": {"type": "STRING"}},
            "required": ["name"],
        },
        handler=check_checklist,
    ))
    registry.register(ToolSpec(
        name="save_checklist",
        description="Create or replace a named checklist. items is a comma-separated list.",
        parameters={
            "type": "OBJECT",
            "properties": {"name": {"type": "STRING"}, "items": {"type": "STRING"}},
            "required": ["name", "items"],
        },
        handler=save_checklist,
    ))
