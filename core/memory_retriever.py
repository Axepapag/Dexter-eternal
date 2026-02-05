import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.vector_store import search_similar
from skills.memory_ops import _db_path as _memory_db_path


def _fetch_rows(
    conn: sqlite3.Connection,
    table: str,
    columns: Iterable[str],
    ids: List[Any],
) -> List[Dict[str, Any]]:
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    cols = ", ".join(columns)
    cur = conn.cursor()
    cur.execute(f"SELECT {cols} FROM {table} WHERE id IN ({placeholders})", ids)
    rows = cur.fetchall()
    results = []
    for row in rows:
        record = {}
        for idx, col in enumerate(columns):
            record[col] = row[idx]
        results.append(record)
    return results


def retrieve_context(
    query: str,
    config: Optional[Dict[str, Any]] = None,
    limit: int = 12,
) -> Dict[str, List[Dict[str, Any]]]:
    conn = sqlite3.connect(_memory_db_path(), timeout=10, check_same_thread=False)
    results = {"facts": [], "triples": [], "entities": [], "patterns": [], "fragments": []}
    try:
        if query:
            cur = conn.cursor()
            like = f"%{query}%"
            cur.execute("SELECT id, text, ts, source FROM facts WHERE text LIKE ? ORDER BY ts DESC LIMIT ?", (like, limit))
            results["facts"] = [
                {"id": r[0], "text": r[1], "ts": r[2], "source": r[3]}
                for r in cur.fetchall()
            ]
            cur.execute(
                "SELECT id, subject, predicate, object, ts, source FROM triples "
                "WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ? "
                "ORDER BY ts DESC LIMIT ?",
                (like, like, like, limit),
            )
            results["triples"] = [
                {
                    "id": r[0],
                    "subject": r[1],
                    "predicate": r[2],
                    "object": r[3],
                    "ts": r[4],
                    "source": r[5],
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                "SELECT name, type, description, ts, source FROM entities WHERE name LIKE ? OR description LIKE ? "
                "ORDER BY ts DESC LIMIT ?",
                (like, like, limit),
            )
            results["entities"] = [
                {
                    "name": r[0],
                    "type": r[1],
                    "description": r[2],
                    "ts": r[3],
                    "source": r[4],
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                "SELECT id, trigger_intent, steps_json, ts, source FROM patterns WHERE trigger_intent LIKE ? "
                "ORDER BY ts DESC LIMIT ?",
                (like, limit),
            )
            results["patterns"] = [
                {
                    "id": r[0],
                    "trigger": r[1],
                    "steps": r[2],
                    "ts": r[3],
                    "source": r[4],
                }
                for r in cur.fetchall()
            ]

        # Vector search for deeper recall
        vector_hits = search_similar(
            conn=conn,
            query_text=query,
            config=config,
            limit=limit,
            scan_limit=max(200, limit * 30),
            item_types=["fact", "triple", "entity", "pattern", "fragment"],
        )
        if vector_hits:
            fact_ids = [h["item_id"] for h in vector_hits if h["item_type"] == "fact"]
            triple_ids = [h["item_id"] for h in vector_hits if h["item_type"] == "triple"]
            pattern_ids = [h["item_id"] for h in vector_hits if h["item_type"] == "pattern"]
            fragment_ids = [h["item_id"] for h in vector_hits if h["item_type"] == "fragment"]
            entity_ids = [h["item_id"] for h in vector_hits if h["item_type"] == "entity"]

            results["facts"].extend(_fetch_rows(conn, "facts", ["id", "text", "ts", "source"], fact_ids))
            results["triples"].extend(
                _fetch_rows(conn, "triples", ["id", "subject", "predicate", "object", "ts", "source"], triple_ids)
            )
            results["patterns"].extend(
                _fetch_rows(conn, "patterns", ["id", "trigger_intent", "steps_json", "ts", "source"], pattern_ids)
            )
            results["fragments"].extend(
                _fetch_rows(conn, "fragments", ["id", "text", "ts", "source"], fragment_ids)
            )
            # Entities use name as ID, so we fetch with LIKE if needed
            if entity_ids:
                cur = conn.cursor()
                placeholders = ", ".join("?" for _ in entity_ids)
                cur.execute(
                    f"SELECT name, type, description, ts, source FROM entities WHERE name IN ({placeholders})",
                    entity_ids,
                )
                results["entities"].extend(
                    {
                        "name": r[0],
                        "type": r[1],
                        "description": r[2],
                        "ts": r[3],
                        "source": r[4],
                    }
                    for r in cur.fetchall()
                )

        return results
    finally:
        conn.close()
