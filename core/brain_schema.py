from __future__ import annotations

import sqlite3
from typing import Iterable


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str]]) -> None:
    existing = _table_columns(conn, table)
    for name, ddl in columns:
        if name in existing:
            continue
        cur = conn.cursor()
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def ensure_brain_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            session TEXT,
            ts REAL,
            importance REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed REAL,
            source TEXT,
            context TEXT,
            pinned INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session TEXT,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            ts REAL,
            confidence REAL,
            source TEXT,
            context TEXT,
            importance REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed REAL,
            pinned INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
            name TEXT PRIMARY KEY,
            type TEXT,
            description TEXT,
            ts REAL,
            importance REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed REAL,
            source TEXT,
            context TEXT,
            pinned INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_intent TEXT,
            steps_json TEXT,
            ts REAL,
            importance REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed REAL,
            source TEXT,
            context TEXT,
            pinned INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fragments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_type TEXT,
            parent_id TEXT,
            text TEXT NOT NULL,
            ts REAL,
            importance REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed REAL,
            source TEXT,
            context TEXT,
            pinned INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL,
            item_id TEXT NOT NULL,
            embedding TEXT NOT NULL,
            model TEXT,
            dimensions INTEGER,
            ts REAL,
            source TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent TEXT,
            step_index INTEGER,
            task TEXT,
            skill_id TEXT,
            tool_call TEXT,
            result TEXT,
            decision TEXT,
            ts REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL,
            item_id TEXT NOT NULL,
            last_synced REAL,
            status TEXT,
            checksum TEXT,
            target TEXT,
            UNIQUE(item_type, item_id, target)
        )
        """
    )

    _ensure_columns(
        conn,
        "facts",
        [
            ("ts", "ts REAL"),
            ("importance", "importance REAL DEFAULT 0.5"),
            ("access_count", "access_count INTEGER DEFAULT 0"),
            ("last_accessed", "last_accessed REAL"),
            ("source", "source TEXT"),
            ("context", "context TEXT"),
            ("pinned", "pinned INTEGER DEFAULT 0"),
        ],
    )
    _ensure_columns(
        conn,
        "triples",
        [
            ("ts", "ts REAL"),
            ("importance", "importance REAL DEFAULT 0.5"),
            ("access_count", "access_count INTEGER DEFAULT 0"),
            ("last_accessed", "last_accessed REAL"),
            ("pinned", "pinned INTEGER DEFAULT 0"),
        ],
    )
    _ensure_columns(
        conn,
        "entities",
        [
            ("ts", "ts REAL"),
            ("importance", "importance REAL DEFAULT 0.5"),
            ("access_count", "access_count INTEGER DEFAULT 0"),
            ("last_accessed", "last_accessed REAL"),
            ("source", "source TEXT"),
            ("context", "context TEXT"),
            ("pinned", "pinned INTEGER DEFAULT 0"),
        ],
    )
    _ensure_columns(
        conn,
        "patterns",
        [
            ("ts", "ts REAL"),
            ("importance", "importance REAL DEFAULT 0.5"),
            ("access_count", "access_count INTEGER DEFAULT 0"),
            ("last_accessed", "last_accessed REAL"),
            ("source", "source TEXT"),
            ("context", "context TEXT"),
            ("pinned", "pinned INTEGER DEFAULT 0"),
        ],
    )
    _ensure_columns(
        conn,
        "fragments",
        [
            ("ts", "ts REAL"),
            ("importance", "importance REAL DEFAULT 0.5"),
            ("access_count", "access_count INTEGER DEFAULT 0"),
            ("last_accessed", "last_accessed REAL"),
            ("source", "source TEXT"),
            ("context", "context TEXT"),
            ("pinned", "pinned INTEGER DEFAULT 0"),
        ],
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_facts_importance ON facts(importance)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_triples_importance ON triples(importance)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_vectors_item ON vectors(item_type, item_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_history_ts ON history(ts)")
    conn.commit()
    conn.close()
