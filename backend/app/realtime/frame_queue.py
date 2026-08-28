"""
Latest-frame-wins queue for the real-time input layer (Segment 2).

The system must never accumulate frames faster than perception can consume
them. This queue holds at most `maxsize` items; once full, the OLDEST item
is dropped to make room for the newest. That guarantees O(maxsize) memory
regardless of how long the system runs or how far capture outpaces
consumption — there is no path to a growing backlog.

Segment 2 wires this up with no consumer yet (Segment 3's detector is the
first real consumer), so today it simply proves frames arrive, get bounded,
and old ones get dropped — exactly the behavior required before anything is
built on top of it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class QueueStats:
    received: int = 0
    dropped: int = 0
    consumed: int = 0


class LatestFrameQueue(Generic[T]):
    def __init__(self, maxsize: int = 2) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = maxsize
        self._items: list[T] = []
        self._condition = asyncio.Condition()
        self.stats = QueueStats()

    async def put(self, item: T) -> None:
        async with self._condition:
            self.stats.received += 1
            self._items.append(item)
            while len(self._items) > self._maxsize:
                self._items.pop(0)
                self.stats.dropped += 1
            self._condition.notify_all()

    async def get(self) -> T:
        async with self._condition:
            while not self._items:
                await self._condition.wait()
            item = self._items.pop(0)
            self.stats.consumed += 1
            return item

    def qsize(self) -> int:
        return len(self._items)
