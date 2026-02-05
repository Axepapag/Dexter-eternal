from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.brain_schema import ensure_brain_schema
from core import embedding_client


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)


@dataclass
class BucketCursor:
    path: str
    offset: int = 0


class MemoryDBWriter:
    """
    Single-writer that drains bucket JSONL files and commits to brain.db.

    - Raw events -> fragments (always)
    - memory_extractions events -> facts/triples/entities/patterns/fragments + vectors
    - history events -> history
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._cursors: Dict[str, BucketCursor] = {}

        mb = self.config.get("memory_buckets", {}) or {}
        self.buckets_dir = str(_repo_root() / (mb.get("dir") or "data/buckets"))
        self.state_path = str(_repo_root() / (mb.get("state_path") or "data/buckets/state.json"))
        self.poll_seconds = float(mb.get("poll_seconds", 0.5))
        self.max_lines_per_tick = int(mb.get("max_lines_per_tick", 200))

        self.db_path = self._resolve_db_path()

    def _resolve_db_path(self) -> str:
        db_path = self.config.get("database_path", "brain.db")
        p = Path(db_path)
        if not p.is_absolute():
            return str(_repo_root() / p)
        return str(p)

    async def start(self) -> None:
        if self._running:
            return
        Path(self.buckets_dir).mkdir(parents=True, exist_ok=True)
        ensure_brain_schema(self.db_path)
        self._load_state()
        self._running = True
        self._task = asyncio.create_task(self._worker())
        print("[MemoryDBWriter] Online", flush=True)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._save_state()
        print("[MemoryDBWriter] Offline", flush=True)

    def _load_state(self) -> None:
        try:
            p = Path(self.state_path)
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            for path, off in data.items():
                try:
                    self._cursors[str(path)] = BucketCursor(path=str(path), offset=int(off or 0))
                except Exception:
                    pass
        except Exception:
            return

    def _save_state(self) -> None:
        try:
            p = Path(self.state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {c.path: c.offset for c in self._cursors.values()}
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _list_bucket_files(self) -> List[str]:
        try:
            paths = []
            for entry in os.listdir(self.buckets_dir):
                if not entry.endswith(".jsonl"):
                    continue
                paths.append(str(Path(self.buckets_dir) / entry))
            return sorted(paths)
        except Exception:
            return []

    async def _worker(self) -> None:
        while self._running:
            try:
                await self._drain_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[MemoryDBWriter] Error: {exc}", flush=True)
            await asyncio.sleep(self.poll_seconds)

    async def _drain_once(self) -> None:
        bucket_files = self._list_bucket_files()
        if not bucket_files:
            return

        lines_processed = 0
        for path in bucket_files:
            cursor = self._cursors.get(path)
            if cursor is None:
                cursor = BucketCursor(path=path, offset=0)
                self._cursors[path] = cursor

            new_lines, new_offset = await asyncio.to_thread(self._read_new_lines, path, cursor.offset, self.max_lines_per_tick - lines_processed)
            if not new_lines:
                cursor.offset = new_offset
                continue

            await self._process_lines(new_lines)
            cursor.offset = new_offset
            lines_processed += len(new_lines)
            if lines_processed >= self.max_lines_per_tick:
                break

        if lines_processed:
            self._save_state()

    def _read_new_lines(self, path: str, offset: int, max_lines: int) -> Tuple[List[str], int]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                fh.seek(max(0, int(offset or 0)))
                lines = []
                for _ in range(max_lines):
                    line = fh.readline()
                    if not line:
                        break
                    lines.append(line)
                new_offset = fh.tell()
            return lines, new_offset
        except Exception:
            return [], int(offset or 0)

    async def _process_lines(self, lines: List[str]) -> None:
        # Offload blocking DB + embedding work to a thread.
        await asyncio.to_thread(self._process_lines_sync, lines)

    def _process_lines_sync(self, lines: List[str]) -> None:
        events = []
        for line in lines:
            line = (line or "").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue
            if isinstance(evt, dict):
                events.append(evt)
        if not events:
            return

        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass

        try:
            ensure_brain_schema(self.db_path)
            for evt in events:
                self._apply_event_sync(conn, evt)
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _apply_event_sync(self, conn: sqlite3.Connection, evt: Dict[str, Any]) -> None:
        ev_id = str(evt.get("id") or "")
        ev_ts = float(evt.get("ts") or time.time())
        ev_source = str(evt.get("source") or "unknown")
        ev_type = str(evt.get("type") or "event")
        payload = evt.get("payload")
        metadata = evt.get("metadata") or {}

        # 1) Always store as raw fragment (forever).
        self._insert_fragment_sync(
            conn=conn,
            parent_type=f"bucket:{ev_source}",
            parent_id=ev_id or ev_source,
            text=_as_text(payload),
            ts=ev_ts,
            source=ev_source,
            context={"event_type": ev_type, "metadata": metadata},
            embed=True,
            item_type="fragment",
        )

        # 2) History event (tool execution trace)
        if ev_type == "history" and isinstance(payload, dict):
            self._insert_history(conn, payload, ev_ts)
            return

        # 3) Memory extraction event: write structured objects.
        if ev_type == "memory_extraction" and isinstance(payload, dict):
            extracted = payload.get("extracted") or {}
            raw_source = str(payload.get("raw_source") or ev_source)
            raw_meta = payload.get("raw_metadata") or {}
            self._apply_extraction_sync(conn, extracted, raw_source, raw_meta, ev_ts)

    def _insert_history(self, conn: sqlite3.Connection, payload: Dict[str, Any], ts: float) -> None:
        try:
            c = conn.cursor()
            c.execute(
                "INSERT INTO history (intent, step_index, task, skill_id, tool_call, result, decision, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    payload.get("intent"),
                    payload.get("step_index"),
                    payload.get("task"),
                    payload.get("skill_id"),
                    json.dumps(payload.get("tool_call") or {}, ensure_ascii=False, default=str),
                    json.dumps(payload.get("result") or {}, ensure_ascii=False, default=str),
                    payload.get("decision"),
                    ts,
                ),
            )
        except Exception:
            return

    def _apply_extraction_sync(self, conn: sqlite3.Connection, extracted: Dict[str, Any], source: str, meta: Dict[str, Any], ts: float) -> None:
        # facts
        for fact in (extracted.get("facts") or []):
            text = str(fact).strip()
            if not text:
                continue
            fact_id = self._insert_fact(conn, text, source, meta, ts)
            if fact_id is not None:
                self._store_vector_sync(conn, "fact", str(fact_id), text, source)

        # triples
        for triple in (extracted.get("triples") or []):
            if not isinstance(triple, dict):
                continue
            subj = str(triple.get("subject") or "").strip()
            pred = str(triple.get("predicate") or "").strip()
            obj = str(triple.get("object") or "").strip()
            if not (subj and pred and obj):
                continue
            conf = float(triple.get("confidence", 0.7) or 0.7)
            triple_id = self._insert_triple(conn, subj, pred, obj, conf, source, meta, ts)
            if triple_id is not None:
                self._store_vector_sync(conn, "triple", str(triple_id), f"{subj} {pred} {obj}", source)

        # entities
        for ent in (extracted.get("entities") or []):
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name") or "").strip()
            if not name:
                continue
            etype = str(ent.get("type") or "").strip()
            desc = str(ent.get("description") or "").strip()
            self._upsert_entity(conn, name, etype, desc, source, meta, ts)
            self._store_vector_sync(conn, "entity", name, f"{name} {desc}".strip(), source)

        # patterns
        for pat in (extracted.get("patterns") or []):
            if not isinstance(pat, dict):
                continue
            trigger = str(pat.get("trigger") or "").strip()
            steps = pat.get("steps") or []
            if not trigger:
                continue
            pat_id = self._insert_pattern(conn, trigger, steps, source, meta, ts)
            if pat_id is not None:
                self._store_vector_sync(conn, "pattern", str(pat_id), f"{trigger} {json.dumps(steps, ensure_ascii=False, default=str)}", source)

        # episodes -> fragments (parent_type episode)
        for ep in (extracted.get("episodes") or []):
            if not isinstance(ep, dict):
                continue
            summary = str(ep.get("summary") or "").strip()
            if not summary:
                continue
            tags = ep.get("tags") or []
            ctx = dict(meta or {})
            ctx["tags"] = tags
            self._insert_fragment_sync(
                conn=conn,
                parent_type="episode",
                parent_id=source,
                text=summary,
                ts=ts,
                source=source,
                context=ctx,
                embed=True,
                item_type="fragment",
            )

    def _insert_fact(self, conn: sqlite3.Connection, text: str, source: str, meta: Dict[str, Any], ts: float) -> Optional[int]:
        try:
            c = conn.cursor()
            c.execute(
                "INSERT INTO facts (text, session, ts, source, context) VALUES (?, ?, ?, ?, ?)",
                (text, "core", ts, source, json.dumps(meta or {}, ensure_ascii=False, default=str)),
            )
            return int(c.lastrowid)
        except Exception:
            return None

    def _insert_triple(self, conn: sqlite3.Connection, subj: str, pred: str, obj: str, conf: float, source: str, meta: Dict[str, Any], ts: float) -> Optional[int]:
        try:
            c = conn.cursor()
            c.execute(
                "INSERT INTO triples (session, subject, predicate, object, ts, confidence, source, context) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("core", subj, pred, obj, ts, conf, source, json.dumps(meta or {}, ensure_ascii=False, default=str)),
            )
            return int(c.lastrowid)
        except Exception:
            return None

    def _upsert_entity(self, conn: sqlite3.Connection, name: str, etype: str, desc: str, source: str, meta: Dict[str, Any], ts: float) -> None:
        try:
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO entities (name, type, description, ts, source, context) VALUES (?, ?, ?, ?, ?, ?)",
                (name, etype, desc, ts, source, json.dumps(meta or {}, ensure_ascii=False, default=str)),
            )
        except Exception:
            return

    def _insert_pattern(self, conn: sqlite3.Connection, trigger: str, steps: Any, source: str, meta: Dict[str, Any], ts: float) -> Optional[int]:
        try:
            c = conn.cursor()
            c.execute(
                "INSERT INTO patterns (trigger_intent, steps_json, ts, source, context) VALUES (?, ?, ?, ?, ?)",
                (trigger, json.dumps(steps or [], ensure_ascii=False, default=str), ts, source, json.dumps(meta or {}, ensure_ascii=False, default=str)),
            )
            return int(c.lastrowid)
        except Exception:
            return None

    def _insert_fragment_sync(
        self,
        conn: sqlite3.Connection,
        parent_type: str,
        parent_id: str,
        text: str,
        ts: float,
        source: str,
        context: Dict[str, Any],
        embed: bool,
        item_type: str,
    ) -> None:
        if not text:
            return
        try:
            c = conn.cursor()
            c.execute(
                "INSERT INTO fragments (parent_type, parent_id, text, ts, source, context) VALUES (?, ?, ?, ?, ?, ?)",
                (parent_type, str(parent_id), text, ts, source, json.dumps(context or {}, ensure_ascii=False, default=str)),
            )
            frag_id = str(c.lastrowid)
        except Exception:
            return
        if embed:
            self._store_vector_sync(conn, item_type, frag_id, text, source)

    def _store_vector_sync(self, conn: sqlite3.Connection, item_type: str, item_id: str, text: str, source: str) -> None:
        cfg = (self.config or {}).get("embeddings", {}) if self.config else {}
        if not cfg.get("enabled", False):
            return
        vec = embedding_client.embed_text(text, self.config)
        if not vec:
            return
        model = cfg.get("model") or "qwen3-embedding:0.6b-fp16"
        try:
            c = conn.cursor()
            c.execute(
                "INSERT INTO vectors (item_type, item_id, embedding, model, dimensions, ts, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (item_type, str(item_id), json.dumps(vec), model, len(vec), time.time(), source),
            )
        except Exception:
            return
