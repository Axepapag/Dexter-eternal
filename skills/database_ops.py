#!/usr/bin/env python3
"""
Database Operations for Dexter
SQLite with retry logic, connection pooling, and business schemas
"""

import sqlite3
import os
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta

__tool_prefix__ = "db"

# Default database location
DEFAULT_DB_DIR = Path(__file__).parent.parent / "data"
DEFAULT_DB_DIR.mkdir(exist_ok=True)
DEFAULT_DB = DEFAULT_DB_DIR / "dexter.db"


class DatabaseError(Exception):
    """Custom database error."""
    pass


def _db_retry(func):
    """Decorator to retry database operations on lock/busy errors."""
    def wrapper(*args, **kwargs):
        retries = 3
        last_exc = None
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    time.sleep(0.5 * (attempt + 1))
                    last_exc = e
                    continue
                raise DatabaseError(f"Database error: {e}")
        raise DatabaseError(f"Database busy/locked after {retries} retries: {last_exc}")
    return wrapper


def _get_db_path(db_path: Optional[str] = None) -> str:
    """Get database path, defaulting to standard location."""
    if db_path is None:
        return str(DEFAULT_DB)
    return db_path


@_db_retry
def db_query(db_path: Optional[str] = None, query: str = "", params: tuple = ()) -> Dict[str, Any]:
    """
    Execute a read-only SQL query.
    
    Args:
        db_path: Database path (default: dexter.db)
        query: SQL query string
        params: Query parameters
    """
    db_path = _get_db_path(db_path)
    
    if not os.path.exists(db_path):
        return {"success": False, "error": f"Database not found: {db_path}"}
    
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = [dict(row) for row in rows]
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    finally:
        conn.close()


@_db_retry
def db_execute(db_path: Optional[str] = None, query: str = "", params: tuple = ()) -> Dict[str, Any]:
    """
    Execute a write SQL command (INSERT, UPDATE, DELETE).
    
    Args:
        db_path: Database path
        query: SQL command
        params: Query parameters
    """
    db_path = _get_db_path(db_path)
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        
        return {
            "success": True,
            "affected_rows": cursor.rowcount,
            "last_insert_id": cursor.lastrowid
        }
    finally:
        conn.close()


@_db_retry
def db_init_schema(db_path: Optional[str] = None, schema_name: str = "default") -> Dict[str, Any]:
    """
    Initialize database with a predefined schema.
    
    Args:
        db_path: Database path
        schema_name: Schema to create (default, business, agent_memory)
    """
    db_path = _get_db_path(db_path)
    
    schemas = {
        "default": """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                type TEXT,
                data TEXT
            );
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                level TEXT,
                message TEXT
            );
        """,
        "business": """
            CREATE TABLE IF NOT EXISTS revenue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                amount REAL,
                currency TEXT DEFAULT 'USD',
                notes TEXT
            );
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                category TEXT,
                amount REAL,
                description TEXT
            );
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                target_amount REAL,
                deadline TEXT,
                status TEXT DEFAULT 'active'
            );
        """,
        "agent_memory": """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                key TEXT UNIQUE,
                value TEXT,
                category TEXT
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                agent TEXT,
                role TEXT,
                content TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
            CREATE INDEX IF NOT EXISTS idx_conversations_agent ON conversations(agent);
        """
    }
    
    if schema_name not in schemas:
        return {
            "success": False,
            "error": f"Unknown schema: {schema_name}",
            "available_schemas": list(schemas.keys())
        }
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.executescript(schemas[schema_name])
        conn.commit()
        
        return {
            "success": True,
            "message": f"Schema '{schema_name}' initialized",
            "database": db_path
        }
    finally:
        conn.close()


@_db_retry
def db_save_fact(key: str, value: str, category: str = "general",
                db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Save a fact to the agent memory database.
    
    Args:
        key: Fact identifier
        value: Fact content
        category: Fact category
        db_path: Database path
    """
    db_path = _get_db_path(db_path)
    
    # Ensure schema exists
    db_init_schema(db_path, "agent_memory")
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO facts (key, value, category, timestamp)
            VALUES (?, ?, ?, ?)
        """, (key, value, category, datetime.now().isoformat()))
        conn.commit()
        
        return {
            "success": True,
            "key": key,
            "category": category
        }
    finally:
        conn.close()


@_db_retry
def db_get_fact(key: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve a fact from the agent memory database.
    
    Args:
        key: Fact identifier
        db_path: Database path
    """
    result = db_query(db_path, "SELECT * FROM facts WHERE key = ?", (key,))
    
    if result.get("count", 0) > 0:
        fact = result["results"][0]
        return {
            "success": True,
            "found": True,
            "key": key,
            "value": fact.get("value"),
            "category": fact.get("category"),
            "timestamp": fact.get("timestamp")
        }
    
    return {
        "success": True,
        "found": False,
        "key": key
    }


@_db_retry
def db_search_facts(query: str, category: Optional[str] = None,
                   db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Search facts by key or value.
    
    Args:
        query: Search query
        category: Optional category filter
        db_path: Database path
    """
    db_path = _get_db_path(db_path)
    
    sql = "SELECT * FROM facts WHERE key LIKE ? OR value LIKE ?"
    params = [f"%{query}%", f"%{query}%"]
    
    if category:
        sql += " AND category = ?"
        params.append(category)
    
    sql += " ORDER BY timestamp DESC"
    
    return db_query(db_path, sql, tuple(params))


@_db_retry
def db_list_tables(db_path: Optional[str] = None) -> Dict[str, Any]:
    """List all tables in the database."""
    db_path = _get_db_path(db_path)
    
    result = db_query(db_path, "SELECT name FROM sqlite_master WHERE type='table'")
    
    if result.get("success"):
        tables = [row["name"] for row in result.get("results", [])]
        return {
            "success": True,
            "tables": tables,
            "count": len(tables)
        }
    
    return result


@_db_retry
def db_table_info(table_name: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get schema information for a table.
    
    Args:
        table_name: Table to inspect
        db_path: Database path
    """
    db_path = _get_db_path(db_path)
    
    # Get column info
    result = db_query(db_path, f"PRAGMA table_info({table_name})")
    
    if result.get("success"):
        columns = [
            {
                "name": row["name"],
                "type": row["type"],
                "notnull": bool(row["notnull"]),
                "pk": bool(row["pk"])
            }
            for row in result.get("results", [])
        ]
        
        # Get row count
        count_result = db_query(db_path, f"SELECT COUNT(*) as count FROM {table_name}")
        row_count = count_result.get("results", [{}])[0].get("count", 0) if count_result.get("success") else 0
        
        return {
            "success": True,
            "table": table_name,
            "columns": columns,
            "row_count": row_count
        }
    
    return {"success": False, "error": f"Could not get info for table {table_name}"}


# Backwards compatibility
query = db_query
execute = db_execute
save_fact = db_save_fact
get_fact = db_get_fact
