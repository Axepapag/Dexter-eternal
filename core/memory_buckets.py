from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_name(name: str) -> str:
    name = (name or "unknown").strip().lower()
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (" ", ".", ":", "/"):
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "unknown"


@dataclass
class BucketEvent:
    id: str
    ts: float
    source: str
    type: str
    payload: Any
    metadata: Dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "ts": self.ts,
                "source": self.source,
                "type": self.type,
                "payload": self.payload,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
            default=str,
        ) + "\n"


class BucketWriter:
    """
    Append-only bucket writer.

    Producers enqueue events quickly; a background task flushes to disk.
    Each bucket is a single file: data/buckets/<bucket>.jsonl
    """

    def __init__(self, base_dir: str, bucket_name: str, flush_every: float = 0.25, max_batch: int = 64):
        self.base_dir = Path(base_dir)
        self.bucket_name = _safe_name(bucket_name)
        self.flush_every = float(flush_every)
        self.max_batch = int(max_batch)
        self.path = self.base_dir / f"{self.bucket_name}.jsonl"

        self._q: asyncio.Queue[BucketEvent] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Best-effort final flush
        await self._flush_once()

    def enqueue(self, event_type: str, payload: Any, source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        evt = BucketEvent(
            id=uuid.uuid4().hex,
            ts=time.time(),
            source=source or self.bucket_name,
            type=str(event_type or "event"),
            payload=payload,
            metadata=metadata or {},
        )
        try:
            self._q.put_nowait(evt)
        except Exception:
            # Drop only if queue is misbehaving; keep runtime non-blocking.
            pass
        return evt.id

    async def _flush_once(self) -> None:
        items = []
        while len(items) < self.max_batch:
            try:
                items.append(self._q.get_nowait())
            except asyncio.QueueEmpty:
                break
            except Exception:
                break
        if not items:
            return

        text = "".join(e.to_jsonl() for e in items)
        # Write in a thread to avoid blocking the loop on slow disks.
        await asyncio.to_thread(self._append_text, text)

    def _append_text(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(text)

    async def _worker(self) -> None:
        try:
            while self._running:
                try:
                    await asyncio.sleep(self.flush_every)
                    await self._flush_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Keep running; bucket IO issues should not crash Dexter.
                    await asyncio.sleep(0.5)
        finally:
            try:
                await self._flush_once()
            except Exception:
                pass


class BucketManager:
    """Manages per-source bucket writers."""

    def __init__(self, base_dir: str, flush_every: float = 0.25):
        self.base_dir = base_dir
        self.flush_every = float(flush_every)
        self._writers: Dict[str, BucketWriter] = {}
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

    async def stop(self) -> None:
        self._running = False
        writers = list(self._writers.values())
        self._writers.clear()
        for w in writers:
            try:
                await w.stop()
            except Exception:
                pass

    def _get(self, bucket_name: str) -> BucketWriter:
        key = _safe_name(bucket_name)
        w = self._writers.get(key)
        if w is None:
            w = BucketWriter(self.base_dir, key, flush_every=self.flush_every)
            self._writers[key] = w
        return w

    async def ensure_started(self, bucket_name: str) -> None:
        w = self._get(bucket_name)
        await w.start()

    def enqueue(self, bucket_name: str, event_type: str, payload: Any, source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        w = self._get(bucket_name)
        # Fire-and-forget start if needed.
        if self._running and (w._task is None or not w._running):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(w.start())
            except Exception:
                pass
        return w.enqueue(event_type=event_type, payload=payload, source=source, metadata=metadata)

