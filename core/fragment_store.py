from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from core.importance import upsert_importance
from core.vector_store import store_embedding


def store_fragment(
    conn,
    text: str,
    parent_type: str = "",
    parent_id: Optional[str] = None,
    source: str = "",
    context: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    if not text:
        return None
    ts = time.time()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO fragments (parent_type, parent_id, text, ts, source, context) VALUES (?, ?, ?, ?, ?, ?)",
        (
            parent_type,
            str(parent_id) if parent_id is not None else "",
            text,
            ts,
            source,
            json.dumps(context or {}, ensure_ascii=False, default=str),
        ),
    )
    fragment_id = cur.lastrowid
    upsert_importance(
        conn=conn,
        table="fragments",
        key_col="id",
        key_val=fragment_id,
        created_ts=ts,
        source=source,
        pinned=False,
        config=config,
    )
    store_embedding(
        conn=conn,
        item_type="fragment",
        item_id=fragment_id,
        text=text,
        config=config,
        source=source,
    )
    return fragment_id
