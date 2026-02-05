from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional


def _score_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = (config or {}).get("importance_scoring", {}) if config else {}
    return {
        "recency_weight": float(cfg.get("recency_weight", 0.55)),
        "usage_weight": float(cfg.get("usage_weight", 0.25)),
        "source_weight": float(cfg.get("source_weight", 0.15)),
        "manual_weight": float(cfg.get("manual_weight", 0.05)),
        "half_life_days": float(cfg.get("half_life_days", 21.0)),
        "max_access": int(cfg.get("max_access", 50)),
        "source_weights": dict(cfg.get("source_weights", {}) or {}),
        "default_source_weight": float(cfg.get("default_source_weight", 0.5)),
    }


def compute_importance(
    created_ts: Optional[float],
    last_accessed: Optional[float],
    access_count: int,
    source: Optional[str],
    pinned: bool,
    config: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> float:
    if pinned:
        return 1.0
    cfg = _score_config(config)
    now_ts = now if now is not None else time.time()
    ref_ts = last_accessed or created_ts or now_ts
    age_days = max(0.0, (now_ts - ref_ts) / 86400.0)
    half_life = max(cfg["half_life_days"], 1.0)
    recency = math.exp(-age_days / half_life)
    max_access = max(cfg["max_access"], 1)
    usage = math.log1p(max(0, access_count)) / math.log1p(max_access)
    source_weight_map = cfg["source_weights"]
    src_weight = source_weight_map.get(source or "", cfg["default_source_weight"])
    manual = 0.0
    score = (
        cfg["recency_weight"] * recency
        + cfg["usage_weight"] * usage
        + cfg["source_weight"] * src_weight
        + cfg["manual_weight"] * manual
    )
    return max(0.0, min(1.0, score))


def record_access(
    conn,
    table: str,
    key_col: str,
    key_val: Any,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    now_ts = time.time()
    cur = conn.cursor()
    cur.execute(
        f"SELECT ts, last_accessed, access_count, source, pinned FROM {table} WHERE {key_col} = ?",
        (key_val,),
    )
    row = cur.fetchone()
    if not row:
        return
    created_ts, last_accessed, access_count, source, pinned = row
    access_count = int(access_count or 0) + 1
    importance = compute_importance(
        created_ts=created_ts,
        last_accessed=now_ts,
        access_count=access_count,
        source=source,
        pinned=bool(pinned),
        config=config,
        now=now_ts,
    )
    try:
        cur.execute(
            f"UPDATE {table} SET access_count = ?, last_accessed = ?, importance = ? WHERE {key_col} = ?",
            (access_count, now_ts, importance, key_val),
        )
        conn.commit()
    except Exception:
        # Importance bookkeeping is best-effort; ignore transient sqlite locks.
        try:
            conn.rollback()
        except Exception:
            pass


def upsert_importance(
    conn,
    table: str,
    key_col: str,
    key_val: Any,
    created_ts: Optional[float],
    source: Optional[str],
    pinned: bool,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    now_ts = time.time()
    importance = compute_importance(
        created_ts=created_ts,
        last_accessed=created_ts,
        access_count=1,
        source=source,
        pinned=pinned,
        config=config,
        now=now_ts,
    )
    cur = conn.cursor()
    cur.execute(
        f"UPDATE {table} SET access_count = 1, last_accessed = ?, importance = ? WHERE {key_col} = ?",
        (created_ts or now_ts, importance, key_val),
    )
    conn.commit()


def recompute_table(
    conn,
    table: str,
    key_col: str = "id",
    config: Optional[Dict[str, Any]] = None,
) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT {key_col}, ts, last_accessed, access_count, source, pinned FROM {table}")
    rows = cur.fetchall()
    now_ts = time.time()
    updated = 0
    for key_val, created_ts, last_accessed, access_count, source, pinned in rows:
        importance = compute_importance(
            created_ts=created_ts,
            last_accessed=last_accessed,
            access_count=int(access_count or 0),
            source=source,
            pinned=bool(pinned),
            config=config,
            now=now_ts,
        )
        cur.execute(
            f"UPDATE {table} SET importance = ? WHERE {key_col} = ?",
            (importance, key_val),
        )
        updated += 1
    conn.commit()
    return updated
