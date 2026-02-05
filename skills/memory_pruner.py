#!/usr/bin/env python3
"""
MemoryPruner - Keeps Dexter's local brain lean and efficient.
Prunes only by importance (time only influences importance via last access).
"""
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.brain_schema import ensure_brain_schema
from core.importance import recompute_table


def _load_config() -> Dict[str, Any]:
    repo_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
    config_path = repo_root / "configs" / "core_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _prune_table(
    conn: sqlite3.Connection,
    table: str,
    key_col: str,
    max_rows: Optional[int],
    importance_floor: float,
    delete_enabled: bool,
) -> Tuple[int, int]:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = int(cur.fetchone()[0] or 0)
    if not max_rows or count <= max_rows:
        return (0, count)

    to_delete = count - max_rows
    cur.execute(
        f"""
        SELECT {key_col}
        FROM {table}
        WHERE (pinned IS NULL OR pinned = 0) AND importance <= ?
        ORDER BY importance ASC
        LIMIT ?
        """,
        (importance_floor, to_delete),
    )
    ids = [row[0] for row in cur.fetchall()]
    if delete_enabled and ids:
        placeholders = ", ".join("?" for _ in ids)
        cur.execute(
            f"DELETE FROM {table} WHERE {key_col} IN ({placeholders})",
            ids,
        )
        return (len(ids), count - len(ids))
    return (0, count)


def prune_local_memory(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Prunes low-importance memory once local tables exceed configured caps.

    NOTE: Dexter-Eternal policy (as requested): never delete canonical memory from disk.
    This skill currently supports deletions only when explicitly enabled, but we hard-disable
    deletes to ensure the local brain grows forever.
    """
    cfg = config or _load_config()
    prune_cfg = cfg.get("memory_pruning", {}) or {}
    if not prune_cfg.get("enabled", False):
        return {"success": False, "error": "memory_pruning_disabled"}

    from skills.memory_ops import _db_path
    db = _db_path()
    ensure_brain_schema(db)

    max_facts = prune_cfg.get("max_facts")
    max_triples = prune_cfg.get("max_triples")
    max_entities = prune_cfg.get("max_entities")
    max_patterns = prune_cfg.get("max_patterns")
    max_fragments = prune_cfg.get("max_fragments")
    importance_floor = float(prune_cfg.get("importance_floor", 0.15))
    # Hard-disable deletes; implement "export to archive DB" before any deletion is permitted.
    delete_enabled = False

    conn = sqlite3.connect(db, timeout=10, check_same_thread=False)
    recompute_table(conn, "facts", config=cfg)
    recompute_table(conn, "triples", config=cfg)
    recompute_table(conn, "entities", key_col="name", config=cfg)
    recompute_table(conn, "patterns", config=cfg)
    recompute_table(conn, "fragments", config=cfg)

    results = {}
    pruned, remaining = _prune_table(conn, "facts", "id", max_facts, importance_floor, delete_enabled)
    results["facts"] = {"pruned": pruned, "remaining": remaining}
    pruned, remaining = _prune_table(conn, "triples", "id", max_triples, importance_floor, delete_enabled)
    results["triples"] = {"pruned": pruned, "remaining": remaining}
    pruned, remaining = _prune_table(conn, "entities", "name", max_entities, importance_floor, delete_enabled)
    results["entities"] = {"pruned": pruned, "remaining": remaining}
    pruned, remaining = _prune_table(conn, "patterns", "id", max_patterns, importance_floor, delete_enabled)
    results["patterns"] = {"pruned": pruned, "remaining": remaining}
    pruned, remaining = _prune_table(conn, "fragments", "id", max_fragments, importance_floor, delete_enabled)
    results["fragments"] = {"pruned": pruned, "remaining": remaining}

    conn.commit()
    conn.close()

    total_pruned = sum(item["pruned"] for item in results.values())
    return {
        "success": True,
        "pruned": total_pruned,
        "delete_enabled": delete_enabled,
        "tables": results,
    }


def run_test():
    return prune_local_memory()
