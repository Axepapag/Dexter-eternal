import json
import os
import sqlite3
import time
from typing import Any, Dict, Optional

from core.brain_schema import ensure_brain_schema
from core.importance import record_access, upsert_importance
from core.vector_store import store_embedding

__tool_prefix__ = "kg"


def _db_path() -> str:
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tools_dir)
    config_path = os.path.join(repo_root, "configs", "core_config.json")
    
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            cfg = json.load(f)
            db_path = cfg.get("database_path", os.path.join(repo_root, "brain.db"))
            if not os.path.isabs(db_path):
                db_path = os.path.join(repo_root, db_path)
            return db_path

    return os.path.join(repo_root, "brain.db")


def _connect() -> sqlite3.Connection:
    db = _db_path()
    ensure_brain_schema(db)
    return sqlite3.connect(db, timeout=10, check_same_thread=False)


def _load_config() -> Dict[str, Any]:
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tools_dir)
    config_path = os.path.join(repo_root, "configs", "core_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_triple(
    subject: str,
    predicate: str,
    object: str,
    session: Optional[str] = None,
    confidence: float = 1.0,
    source: str = "",
    context: Optional[Dict[str, Any]] = None,
    ts: Optional[float] = None,
    pinned: bool = False,
) -> Dict[str, Any]:
    """Add a knowledge graph triple."""
    if not subject or not predicate or not object:
        return {"success": False, "error": "Missing subject, predicate, or object"}
    session = session or os.getenv("BRAIN_SESSION", "core")
    ts = ts if ts is not None else time.time()
    ctx_json = json.dumps(context or {}, default=str)
    try:
        conn = _connect()
        c = conn.cursor()
        c.execute(
            "INSERT INTO triples (session, subject, predicate, object, ts, confidence, source, context, pinned) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session, subject, predicate, object, ts, confidence, source, ctx_json, int(bool(pinned))),
        )
        conn.commit()
        triple_id = c.lastrowid
        cfg = _load_config()
        upsert_importance(
            conn=conn,
            table="triples",
            key_col="id",
            key_val=triple_id,
            created_ts=ts,
            source=source,
            pinned=bool(pinned),
            config=cfg,
        )
        store_embedding(
            conn=conn,
            item_type="triple",
            item_id=triple_id,
            text=f"{subject} {predicate} {object}",
            config=cfg,
            source=source,
        )
        conn.close()
        return {
            "success": True,
            "id": triple_id,
            "session": session,
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "ts": ts,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def query(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object: Optional[str] = None,
    session: Optional[str] = None,
    limit: int = 50,
    fuzzy: bool = True,
) -> Dict[str, Any]:
    """Query triples by subject/predicate/object."""
    session = session or os.getenv("BRAIN_SESSION", "core")
    clauses = []
    params = []
    if subject:
        clauses.append("subject LIKE ?" if fuzzy else "subject = ?")
        params.append(f"%{subject}%" if fuzzy else subject)
    if predicate:
        clauses.append("predicate LIKE ?" if fuzzy else "predicate = ?")
        params.append(f"%{predicate}%" if fuzzy else predicate)
    if object:
        clauses.append("object LIKE ?" if fuzzy else "object = ?")
        params.append(f"%{object}%" if fuzzy else object)
    if session and session != "*":
        clauses.append("session = ?")
        params.append(session)
    where = " AND ".join(clauses) if clauses else "1=1"
    try:
        conn = _connect()
        c = conn.cursor()
        c.execute(
            f"SELECT id, subject, predicate, object, ts, confidence, source, context, session, importance, access_count, last_accessed "
            f"FROM triples WHERE {where} ORDER BY id DESC LIMIT ?",
            params + [limit],
        )
        rows = c.fetchall()
        results = []
        cfg = _load_config()
        for row in rows:
            triple_id = row[0]
            record_access(conn, "triples", "id", triple_id, config=cfg)
            try:
                ctx = json.loads(row[7]) if row[7] else {}
            except json.JSONDecodeError:
                ctx = {}
            results.append({
                "id": triple_id,
                "subject": row[1],
                "predicate": row[2],
                "object": row[3],
                "ts": row[4],
                "confidence": row[5],
                "source": row[6],
                "context": ctx,
                "session": row[8],
                "importance": row[9],
                "access_count": row[10],
                "last_accessed": row[11],
            })
        conn.close()
        return {"success": True, "count": len(results), "results": results}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
