import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from core.brain_schema import ensure_brain_schema
from core.importance import record_access, upsert_importance
from core.vector_store import store_embedding

__tool_prefix__ = "memory"


def _db_path() -> str:
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tools_dir)
    config_path = os.path.join(repo_root, "configs", "core_config.json")

    env_path = os.getenv("DEXTER_DB_PATH")
    if env_path:
        return env_path
    
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
    conn = sqlite3.connect(db, timeout=10, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return conn


def add_fact(
    text: str,
    session: Optional[str] = None,
    ts: Optional[float] = None,
    source: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    pinned: bool = False,
) -> Dict[str, Any]:
    """Add a fact to Dexter's memory (facts table)."""
    if not text:
        return {"success": False, "error": "Missing text"}
    session = session or os.getenv("BRAIN_SESSION", "core")
    ts = ts if ts is not None else time.time()
    try:
        conn = _connect()
        c = conn.cursor()
        ctx_json = json.dumps(context or {}, ensure_ascii=False, default=str)
        src = source or "User"
        c.execute(
            "INSERT INTO facts (text, session, ts, source, context, pinned) VALUES (?, ?, ?, ?, ?, ?)",
            (text, session, ts, src, ctx_json, int(bool(pinned))),
        )
        conn.commit()
        fact_id = c.lastrowid
        upsert_importance(
            conn=conn,
            table="facts",
            key_col="id",
            key_val=fact_id,
            created_ts=ts,
            source=src,
            pinned=bool(pinned),
            config=_load_config(),
        )
        store_embedding(
            conn=conn,
            item_type="fact",
            item_id=fact_id,
            text=text,
            config=_load_config(),
            source=src,
        )
        conn.close()
        return {"success": True, "id": fact_id, "text": text, "session": session, "ts": ts}
    except Exception as exc:
        return {"success": False, "error": str(exc), "session": session}


def _load_config() -> Dict[str, Any]:
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tools_dir)
    config_path = os.path.join(repo_root, "configs", "core_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def search(query: str, session: Optional[str] = None, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
    """Search facts by substring."""
    if not query:
        return {"success": False, "error": "Missing query"}
    session = session or os.getenv("BRAIN_SESSION", "core")
    try:
        conn = _connect()
        c = conn.cursor()
        if session == "*" or session is None:
            c.execute(
                "SELECT id, text, session, ts, importance, access_count, last_accessed, source FROM facts "
                "WHERE text LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (f"%{query}%", limit, offset),
            )
        else:
            c.execute(
                "SELECT id, text, session, ts, importance, access_count, last_accessed, source FROM facts "
                "WHERE session=? AND text LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (session, f"%{query}%", limit, offset),
            )
        rows = c.fetchall()
        results = []
        cfg = _load_config()
        for r in rows:
            fact_id = r[0]
            record_access(conn, "facts", "id", fact_id, config=cfg)
            results.append({
                "id": fact_id,
                "text": r[1],
                "session": r[2],
                "ts": r[3],
                "importance": r[4],
                "access_count": r[5],
                "last_accessed": r[6],
                "source": r[7],
            })
        conn.close()
        return {"success": True, "query": query, "count": len(results), "results": results}
    except Exception as exc:
        return {"success": False, "error": str(exc), "query": query}
