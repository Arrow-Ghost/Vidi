import asyncio

import pytest

from app.realtime.frame_queue import LatestFrameQueue


def test_bounded_size_drops_oldest() -> None:
    async def run() -> None:
        q: LatestFrameQueue[int] = LatestFrameQueue(maxsize=2)
        for i in range(10):
            await q.put(i)
        assert q.qsize() == 2
        assert q.stats.received == 10
        assert q.stats.dropped == 8
        # The two survivors must be the two most recent items (8, 9).
        first = await q.get()
        second = await q.get()
        assert (first, second) == (8, 9)

    asyncio.run(run())


def test_rejects_non_positive_maxsize() -> None:
    with pytest.raises(ValueError):
        LatestFrameQueue(maxsize=0)


def test_get_waits_for_item() -> None:
    async def run() -> None:
        q: LatestFrameQueue[str] = LatestFrameQueue(maxsize=1)

        async def producer() -> None:
            await asyncio.sleep(0.05)
            await q.put("frame")

        asyncio.create_task(producer())
        item = await asyncio.wait_for(q.get(), timeout=1.0)
        assert item == "frame"

    asyncio.run(run())
