from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from core import embedding_client


def store_embedding(
    conn,
    item_type: str,
    item_id: Any,
    text: str,
    config: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> Optional[int]:
    # Legacy sync path (may block the event loop if called from async code).
    vector = embedding_client.embed_text(text, config=config)
    if vector is None:
        return None
    cfg = (config or {}).get("embeddings", {}) if config else {}
    model = cfg.get("model") or "qwen3-embedding:0.6b-fp16"
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO vectors (item_type, item_id, embedding, model, dimensions, ts, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            item_type,
            str(item_id),
            json.dumps(vector),
            model,
            len(vector),
            time.time(),
            source or "",
        ),
    )
    conn.commit()
    return cur.lastrowid


async def store_embedding_async(
    conn,
    item_type: str,
    item_id: Any,
    text: str,
    config: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> Optional[int]:
    vector = await embedding_client.embed_text_async(text, config=config)
    if vector is None:
        return None
    cfg = (config or {}).get("embeddings", {}) if config else {}
    model = cfg.get("model") or "qwen3-embedding:0.6b-fp16"
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO vectors (item_type, item_id, embedding, model, dimensions, ts, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            item_type,
            str(item_id),
            json.dumps(vector),
            model,
            len(vector),
            time.time(),
            source or "",
        ),
    )
    conn.commit()
    return cur.lastrowid


def search_similar(
    conn,
    query_text: str,
    config: Optional[Dict[str, Any]] = None,
    limit: int = 5,
    scan_limit: int = 500,
    item_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    vector = embedding_client.embed_text(query_text, config=config)
    if vector is None:
        return []
    cfg = (config or {}).get("embeddings", {}) if config else {}
    model = cfg.get("model") or "qwen3-embedding:0.6b-fp16"

    params: List[Any] = [model]
    where = "model = ?"
    if item_types:
        placeholders = ", ".join("?" for _ in item_types)
        where += f" AND item_type IN ({placeholders})"
        params.extend(list(item_types))

    cur = conn.cursor()
    cur.execute(
        f"SELECT item_type, item_id, embedding, ts FROM vectors WHERE {where} ORDER BY ts DESC LIMIT ?",
        params + [scan_limit],
    )
    rows = cur.fetchall()
    if not rows:
        return []

    query_vec = np.array(vector, dtype=np.float32)
    q_norm = np.linalg.norm(query_vec)
    results = []
    for item_type, item_id, emb_json, ts in rows:
        try:
            emb = np.array(json.loads(emb_json), dtype=np.float32)
        except Exception:
            continue
        denom = (np.linalg.norm(emb) * q_norm) or 1.0
        score = float(np.dot(query_vec, emb) / denom)
        results.append({
            "item_type": item_type,
            "item_id": item_id,
            "score": score,
            "ts": ts,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]
