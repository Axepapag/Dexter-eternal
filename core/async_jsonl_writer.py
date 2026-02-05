"""
async_jsonl_writer.py - Non-blocking JSONL writes for Dexter.

Why this exists:
- Many parts of the runtime are async and must not block the event loop on disk I/O.
- We want "fire-and-forget" logging with bounded overhead and backpressure.

Implementation:
- A single background thread serializes all JSONL appends (per process).
- Producers enqueue (path, json_line) into a bounded Queue.
- If the queue is full, we drop (best-effort) rather than block the event loop.

This is not a durability system; it's telemetry.
"""

from __future__ import annotations

import json
import os
import threading
import time
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class _WriteItem:
    path: str
    line: str


class AsyncJSONLWriter:
    def __init__(self, max_queue: int = 5000):
        self._q: "queue.Queue[Optional[_WriteItem]]" = queue.Queue(maxsize=max(100, int(max_queue)))
        self._t: Optional[threading.Thread] = None
        self._running = False
        self._metrics: Dict[str, int] = {
            "enqueued": 0,
            "written": 0,
            "dropped": 0,
            "errors": 0,
        }

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._t = threading.Thread(target=self._run, name="async-jsonl-writer", daemon=True)
        self._t.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._running = False
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        t = self._t
        if t and t.is_alive():
            try:
                t.join(timeout=timeout)
            except Exception:
                pass
        self._t = None

    def enqueue(self, path: str, payload: Dict[str, Any]) -> bool:
        if not path:
            return False
        if not self._running:
            self.start()
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Directory creation may still fail; attempt write anyway.
            pass

        rec = dict(payload)
        rec.setdefault("ts", time.time())
        try:
            line = json.dumps(rec, ensure_ascii=False, default=str)
        except Exception:
            # Last resort: store as string.
            line = json.dumps({"ts": time.time(), "raw": str(payload)}, ensure_ascii=False, default=str)

        item = _WriteItem(path=str(path), line=line)
        try:
            self._q.put_nowait(item)
            self._metrics["enqueued"] += 1
            return True
        except queue.Full:
            self._metrics["dropped"] += 1
            return False
        except Exception:
            self._metrics["errors"] += 1
            return False

    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    def _run(self) -> None:
        while self._running:
            try:
                item = self._q.get()
            except Exception:
                continue
            if item is None:
                break
            try:
                with open(item.path, "a", encoding="utf-8") as fh:
                    fh.write(item.line + "\n")
                self._metrics["written"] += 1
            except Exception:
                self._metrics["errors"] += 1


_GLOBAL: Optional[AsyncJSONLWriter] = None


def get_jsonl_writer() -> AsyncJSONLWriter:
    global _GLOBAL
    if _GLOBAL is None:
        # Allow env override for higher throughput / more buffering.
        max_queue = int(os.getenv("DEXTER_JSONL_LOG_QUEUE", "5000") or "5000")
        _GLOBAL = AsyncJSONLWriter(max_queue=max_queue)
    return _GLOBAL


def enqueue_jsonl(path: str, payload: Dict[str, Any]) -> bool:
    """
    Best-effort enqueue of a JSONL record.

    Enablement:
    - env `DEXTER_ASYNC_JSONL_LOG=1` enables async writes
    - env `DEXTER_ASYNC_JSONL_LOG=0` disables (falls back to sync at call site)
    """
    if not _truthy_env("DEXTER_ASYNC_JSONL_LOG"):
        return False
    return get_jsonl_writer().enqueue(path, payload)

