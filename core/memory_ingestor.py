import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.brain_schema import ensure_brain_schema
from core.memory_buckets import BucketManager
from core.tool_agent_provider import AsyncAIProvider, ChatMessage
from core.utils import extract_json
from core.vector_store import store_embedding


@dataclass
class MemoryEvent:
    source: str
    content: str
    metadata: Dict[str, Any]
    timestamp: float


class MemoryIngestor:
    """
    Asynchronous memory ingestion pipeline:
    - Accepts events into a queue
    - Uses LLM to extract structured memory objects
    - Emits extraction results as bucket events for a single DB writer
    """

    def __init__(self, config: Dict[str, Any], bucket_manager: Optional[BucketManager] = None):
        self.config = config or {}
        self.bucket_manager = bucket_manager
        self.queue: asyncio.Queue[MemoryEvent] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def _db_path(self) -> str:
        db_path = self.config.get("database_path", "brain.db")
        base_dir = Path(__file__).resolve().parents[1]
        if not Path(db_path).is_absolute():
            return str(base_dir / db_path)
        return db_path

    def _memory_llm(self) -> tuple[AsyncAIProvider, str]:
        mem_cfg = self.config.get("memory_extraction", {}) or {}
        slot = mem_cfg.get("slot", "memory")
        from core.llm_slots import resolve_llm_slot

        provider_name, provider_cfg, model = resolve_llm_slot(self.config, slot)
        return AsyncAIProvider(provider_name, provider_cfg), model

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._worker())
        print("[MemoryIngestor] Online", flush=True)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def enqueue(self, source: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        if not content:
            return
        evt = MemoryEvent(
            source=source,
            content=str(content),
            metadata=metadata or {},
            timestamp=time.time(),
        )
        try:
            self.queue.put_nowait(evt)
        except Exception:
            pass

    async def _worker(self):
        while self._running:
            try:
                evt = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                await self._process(evt)
            except Exception as exc:
                print(f"[MemoryIngestor] Error: {exc}", flush=True)

    async def _process(self, evt: MemoryEvent):
        provider, model = self._memory_llm()
        system_prompt = (
            "You are a memory extraction engine. Output ONLY valid JSON.\n"
            "Extract structured memory from the input.\n"
            "Schema:\n"
            "{\n"
            "  \"facts\": [\"...\"],\n"
            "  \"triples\": [{\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\", \"confidence\": 0.0}],\n"
            "  \"entities\": [{\"name\": \"...\", \"type\": \"person|org|place|object|concept|event\", \"description\": \"...\"}],\n"
            "  \"patterns\": [{\"trigger\": \"...\", \"steps\": [\"...\"], \"confidence\": 0.0}],\n"
            "  \"episodes\": [{\"summary\": \"...\", \"tags\": [\"...\"], \"confidence\": 0.0}]\n"
            "}\n"
            "Keep items short. Return empty lists if nothing applies."
        )
        user_prompt = (
            f"SOURCE: {evt.source}\n"
            f"TIMESTAMP: {evt.timestamp}\n"
            f"METADATA: {json.dumps(evt.metadata, ensure_ascii=False, default=str)}\n\n"
            f"CONTENT:\n{evt.content}"
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        try:
            response = await provider.chat(messages, model, temperature=0.2, max_tokens=2000)
        finally:
            try:
                await provider.close()
            except Exception:
                pass

        if not response.success:
            return

        payload = extract_json(response.content) or {}

        # Preferred path: single-writer via bucket pipeline (non-blocking for runtime).
        if self.bucket_manager is not None:
            try:
                self.bucket_manager.enqueue(
                    bucket_name="memory_extractions",
                    event_type="memory_extraction",
                    source=evt.source,
                    payload={
                        "raw_source": evt.source,
                        "raw_metadata": evt.metadata or {},
                        "raw_content": evt.content,
                        "extracted": payload,
                    },
                    metadata={"pipeline": "MemoryIngestor"},
                )
            except Exception:
                pass
            return

        # Fallback: legacy direct DB writes (kept for backward compatibility).
        await asyncio.to_thread(self._write_memory, evt, payload)

    def _write_memory(self, evt: MemoryEvent, payload: Dict[str, Any]):
        db_path = self._db_path()
        ensure_brain_schema(db_path)
        conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
        cfg = self.config
        try:
            self._insert_facts(conn, payload.get("facts") or [], evt)
            self._insert_triples(conn, payload.get("triples") or [], evt)
            self._insert_entities(conn, payload.get("entities") or [], evt)
            self._insert_patterns(conn, payload.get("patterns") or [], evt)
            self._insert_episodes(conn, payload.get("episodes") or [], evt)
            # Always store raw fragment for traceability
            self._insert_fragment(conn, evt)
            conn.commit()
        finally:
            conn.close()

    def _insert_facts(self, conn, facts: List[Any], evt: MemoryEvent):
        cur = conn.cursor()
        for fact in facts:
            text = str(fact).strip()
            if not text:
                continue
            cur.execute(
                "INSERT INTO facts (text, session, ts, source, context) VALUES (?, ?, ?, ?, ?)",
                (text, "core", evt.timestamp, evt.source, json.dumps(evt.metadata, ensure_ascii=False, default=str)),
            )
            fact_id = cur.lastrowid
            store_embedding(conn, "fact", fact_id, text, config=self.config, source=evt.source)

    def _insert_triples(self, conn, triples: List[Dict[str, Any]], evt: MemoryEvent):
        cur = conn.cursor()
        for triple in triples:
            subj = str(triple.get("subject", "")).strip()
            pred = str(triple.get("predicate", "")).strip()
            obj = str(triple.get("object", "")).strip()
            if not (subj and pred and obj):
                continue
            confidence = float(triple.get("confidence", 0.7))
            cur.execute(
                "INSERT INTO triples (session, subject, predicate, object, ts, confidence, source, context) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "core",
                    subj,
                    pred,
                    obj,
                    evt.timestamp,
                    confidence,
                    evt.source,
                    json.dumps(evt.metadata, ensure_ascii=False, default=str),
                ),
            )
            triple_id = cur.lastrowid
            store_embedding(
                conn,
                "triple",
                triple_id,
                f"{subj} {pred} {obj}",
                config=self.config,
                source=evt.source,
            )

    def _insert_entities(self, conn, entities: List[Dict[str, Any]], evt: MemoryEvent):
        cur = conn.cursor()
        for ent in entities:
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            etype = str(ent.get("type", "")).strip()
            desc = str(ent.get("description", "")).strip()
            cur.execute(
                "INSERT OR REPLACE INTO entities (name, type, description, ts, source, context) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    name,
                    etype,
                    desc,
                    evt.timestamp,
                    evt.source,
                    json.dumps(evt.metadata, ensure_ascii=False, default=str),
                ),
            )
            store_embedding(conn, "entity", name, name + " " + desc, config=self.config, source=evt.source)

    def _insert_patterns(self, conn, patterns: List[Dict[str, Any]], evt: MemoryEvent):
        cur = conn.cursor()
        for pat in patterns:
            trigger = str(pat.get("trigger", "")).strip()
            steps = pat.get("steps") or []
            if not trigger:
                continue
            cur.execute(
                "INSERT INTO patterns (trigger_intent, steps_json, ts, source, context) VALUES (?, ?, ?, ?, ?)",
                (
                    trigger,
                    json.dumps(steps, ensure_ascii=False, default=str),
                    evt.timestamp,
                    evt.source,
                    json.dumps(evt.metadata, ensure_ascii=False, default=str),
                ),
            )
            pat_id = cur.lastrowid
            store_embedding(conn, "pattern", pat_id, trigger, config=self.config, source=evt.source)

    def _insert_episodes(self, conn, episodes: List[Dict[str, Any]], evt: MemoryEvent):
        cur = conn.cursor()
        for ep in episodes:
            summary = str(ep.get("summary", "")).strip()
            if not summary:
                continue
            tags = ep.get("tags") or []
            context = dict(evt.metadata)
            context["tags"] = tags
            cur.execute(
                "INSERT INTO fragments (parent_type, parent_id, text, ts, source, context) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "episode",
                    evt.source,
                    summary,
                    evt.timestamp,
                    evt.source,
                    json.dumps(context, ensure_ascii=False, default=str),
                ),
            )
            frag_id = cur.lastrowid
            store_embedding(conn, "fragment", frag_id, summary, config=self.config, source=evt.source)

    def _insert_fragment(self, conn, evt: MemoryEvent):
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO fragments (parent_type, parent_id, text, ts, source, context) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "event",
                evt.source,
                evt.content,
                evt.timestamp,
                evt.source,
                json.dumps(evt.metadata, ensure_ascii=False, default=str),
            ),
        )
        frag_id = cur.lastrowid
        store_embedding(conn, "fragment", frag_id, evt.content, config=self.config, source=evt.source)
