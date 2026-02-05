import asyncio
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict, deque
import json


@dataclass
class Message:
    timestamp: float
    source: str
    content: Dict[str, Any]
    priority: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "content": self.content,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", "unknown"),
            content=data.get("content", {}),
            priority=data.get("priority", "medium")
        )


class ResponseTank:
    def __init__(
        self,
        max_history: int = 1000,
        ingest_maxsize: int = 2000,
        high_timeout_sec: float = 0.2,
        medium_timeout_sec: float = 0.05,
        low_timeout_sec: float = 0.0,
    ):
        self._messages: deque[Message] = deque(maxlen=max_history)
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._subscribers: Dict[str, asyncio.Queue[Message]] = defaultdict(
            lambda: asyncio.Queue(maxsize=100)
        )
        self._lock = asyncio.Lock()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._ingest_queue: "queue.Queue[Optional[Message]]" = queue.Queue(maxsize=int(ingest_maxsize))
        self._worker_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._enqueue_timeouts = {
            "high": float(high_timeout_sec),
            "medium": float(medium_timeout_sec),
            "low": float(low_timeout_sec),
        }
        self._metrics: Dict[str, int] = {
            "published": 0,
            "dropped": 0,
            "blocked": 0,
            "enqueued": 0,
            "subscriber_dropped": 0,
        }
    
    async def start(self):
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._processor_task = asyncio.create_task(self._process_messages())
        self._worker_thread = threading.Thread(target=self._ingest_worker, daemon=True)
        self._worker_thread.start()
    
    async def stop(self):
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        try:
            self._ingest_queue.put(None, timeout=0.5)
        except Exception:
            try:
                _ = self._ingest_queue.get_nowait()
                self._ingest_queue.put_nowait(None)
            except Exception:
                pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
    
    async def _process_messages(self):
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                async with self._lock:
                    self._messages.append(message)
                
                await self._distribute(message)
                
            except asyncio.TimeoutError:
                continue

    def _ingest_worker(self):
        while self._running:
            try:
                msg = self._ingest_queue.get()
            except Exception:
                continue
            if msg is None:
                break
            self._metrics["enqueued"] += 1
            if not self._loop or self._loop.is_closed():
                continue
            def _push():
                try:
                    self._message_queue.put_nowait(msg)
                except Exception:
                    pass
            self._loop.call_soon_threadsafe(_push)
    
    async def _distribute(self, message: Message):
        for pattern, subscribers in self._subscriptions.items():
            if self._matches_pattern(message.source, pattern):
                for subscriber_id in subscribers:
                    queue = self._subscribers.get(subscriber_id)
                    if queue:
                        try:
                            queue.put_nowait(message)
                        except asyncio.QueueFull:
                            self._metrics["subscriber_dropped"] += 1
    
    def _matches_pattern(self, source: str, pattern: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return source.startswith(prefix)
        return source == pattern
    
    async def publish(
        self,
        source: str,
        content: Dict[str, Any],
        priority: str = "medium"
    ) -> None:
        if not self._running:
            await self.start()
        message = Message(
            timestamp=time.time(),
            source=source,
            content=content,
            priority=priority
        )
        timeout = self._enqueue_timeouts.get(priority, self._enqueue_timeouts["medium"])
        try:
            if timeout <= 0:
                self._ingest_queue.put_nowait(message)
            else:
                if self._ingest_queue.full():
                    self._metrics["blocked"] += 1
                await asyncio.to_thread(self._ingest_queue.put, message, True, timeout)
            self._metrics["published"] += 1
        except queue.Full:
            self._metrics["dropped"] += 1
    
    async def subscribe(
        self,
        subscriber_id: str,
        source_patterns: List[str]
    ) -> "Subscriber":
        async with self._lock:
            for pattern in source_patterns:
                self._subscriptions[pattern].add(subscriber_id)
        
        return Subscriber(
            tank=self,
            subscriber_id=subscriber_id,
            source_patterns=source_patterns
        )
    
    async def unsubscribe(self, subscriber_id: str):
        async with self._lock:
            for pattern, subscribers in self._subscriptions.items():
                subscribers.discard(subscriber_id)
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
    
    async def get_recent(
        self,
        source: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 10,
        since: Optional[float] = None
    ) -> List[Message]:
        async with self._lock:
            messages = list(self._messages)
        
        filtered = []
        for msg in reversed(messages):
            if since and msg.timestamp <= since:
                continue
            if source and msg.source != source and not source.endswith("*"):
                if not (source.endswith("*") and msg.source.startswith(source[:-1])):
                    continue
            if priority and msg.priority != priority:
                continue
            filtered.append(msg)
            if len(filtered) >= limit:
                break
        
        return filtered
    
    async def get_queue(self, subscriber_id: str) -> asyncio.Queue:
        return self._subscribers[subscriber_id]

    def get_metrics(self) -> Dict[str, int]:
        return dict(self._metrics)


class Subscriber:
    def __init__(
        self,
        tank: ResponseTank,
        subscriber_id: str,
        source_patterns: List[str]
    ):
        self._tank = tank
        self._subscriber_id = subscriber_id
        self._source_patterns = source_patterns
        self._queue: Optional[asyncio.Queue] = None
    
    async def __aenter__(self):
        self._queue = await self._tank.get_queue(self._subscriber_id)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._tank.unsubscribe(self._subscriber_id)
    
    async def receive(
        self,
        timeout: float = 1.0
    ) -> Optional[Message]:
        if not self._queue:
            raise RuntimeError("Subscriber not entered via async context manager")
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def receive_all(
        self,
        timeout: float = 0.1,
        max_messages: int = 10
    ) -> List[Message]:
        if not self._queue:
            raise RuntimeError("Subscriber not entered via async context manager")
        
        messages = []
        first = await self.receive(timeout)
        if first:
            messages.append(first)
        
        while len(messages) < max_messages:
            try:
                msg = self._queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        
        return messages
    
    async def wait_for(
        self,
        predicate: callable,
        timeout: float = 5.0
    ) -> Optional[Message]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = await self.receive(0.5)
            if msg and predicate(msg):
                return msg
        return None


_global_tank: Optional[ResponseTank] = None


def get_global_tank() -> ResponseTank:
    global _global_tank
    if _global_tank is None:
        _global_tank = ResponseTank()
    return _global_tank
