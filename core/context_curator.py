import asyncio
import hashlib
import json
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple

from core import embedding_client


class ContextCurator:
    """Curates staged artifacts before they hit Dexter's context bundle."""

    def __init__(self, staged_context, config: Dict[str, Any]):
        self._staged_context = staged_context
        self._lock = asyncio.Lock()
        self.config = config.get("context_curator", {}) if config else {}
        self.enabled = bool(self.config.get("enabled", True))
        self.max_history = int(self.config.get("max_history_per_source", 16))
        self.duplicate_window = float(self.config.get("duplicate_window_seconds", 20.0))
        self.prune_seconds = float(self.config.get("prune_seconds", 900.0))
        self.embedding_cfg = self.config.get("embeddings", {})
        self._history: Dict[str, deque[Tuple[str, float]]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

    async def stage_structured_artifact(
        self,
        source: str,
        artifact_type: str,
        payload: Any,
        confidence: float,
        priority: int,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.enabled:
            await self._staged_context.stage_artifact(
                source=source,
                artifact_type=artifact_type,
                payload=payload,
                confidence=confidence,
                priority=priority,
                timestamp=timestamp,
                metadata=metadata,
            )
            return

        ts = float(timestamp if timestamp is not None else time.time())
        summary = self._summarize(payload)
        digest = self._digest(summary)

        async with self._lock:
            if self._is_duplicate(source, digest, ts):
                return
            self._record(source, digest, ts)

        enriched_metadata = (metadata.copy() if metadata else {})
        enriched_metadata["curator_summary"] = summary
        enriched_metadata["curator_digest"] = digest
        embedding = await self._compute_embedding(summary)
        if embedding is not None:
            enriched_metadata["curator_embedding_model"] = self.embedding_cfg.get(
                "model", "qwen3-embedding:0.6b-fp16"
            )
            enriched_metadata["curator_embedding_len"] = len(embedding)

        await self._staged_context.stage_artifact(
            source=source,
            artifact_type=artifact_type,
            payload=payload,
            confidence=confidence,
            priority=priority,
            timestamp=ts,
            metadata=enriched_metadata,
        )

    def _summarize(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload.strip()
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(payload)

    def _digest(self, summary: str) -> str:
        return hashlib.sha256(summary.encode("utf-8")).hexdigest()

    def _is_duplicate(self, source: str, digest: str, timestamp: float) -> bool:
        bucket = self._history[source]
        for existing_digest, ts in bucket:
            if existing_digest == digest and (timestamp - ts) <= self.duplicate_window:
                return True
        return False

    def _record(self, source: str, digest: str, timestamp: float):
        bucket = self._history[source]
        bucket.append((digest, timestamp))
        cutoff = timestamp - self.prune_seconds
        while bucket and bucket[0][1] < cutoff:
            bucket.popleft()

    async def _compute_embedding(self, summary: str) -> Optional[list]:
        if not self.embedding_cfg.get("enabled", False):
            return None
        return await embedding_client.embed_text_async(summary, config={"embeddings": self.embedding_cfg})
