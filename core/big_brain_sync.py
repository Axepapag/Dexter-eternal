from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.brain_schema import ensure_brain_schema
from core.importance import recompute_table
from skills.memory_ops import _db_path as local_db_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _repo_root() / candidate


def _load_big_brain_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("big_brain", {}) or {}

def _sync_mode(cfg: Dict[str, Any]) -> str:
    mode = (cfg.get("mode") or cfg.get("sync_mode") or "drive").strip().lower()
    return mode or "drive"


def _sync_state_path(cfg: Dict[str, Any]) -> Path:
    return _resolve_path(cfg.get("sync_state_path", "memory/big_brain_sync_state.json"))


def _load_sync_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"processed_files": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"processed_files": []}


def _save_sync_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _resolve_drive_file_id(cfg: Dict[str, Any]) -> Optional[str]:
    file_id = cfg.get("drive_file_id")
    if file_id:
        return file_id
    url = cfg.get("drive_file_url") or ""
    if "/file/d/" in url:
        return url.split("/file/d/")[1].split("/")[0]
    if "id=" in url:
        return url.split("id=")[-1].split("&")[0]
    return None


def _get_drive_service(cfg: Dict[str, Any]):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    oauth_cfg = cfg.get("oauth", {}) or {}
    scopes = oauth_cfg.get("scopes") or ["https://www.googleapis.com/auth/drive.file"]
    client_secret_path = _resolve_path(oauth_cfg.get("client_secret_path", "secrets/google_client.json"))
    token_path = _resolve_path(oauth_cfg.get("token_path", "secrets/google_token.json"))
    use_console = bool(oauth_cfg.get("use_console", True))

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), scopes)
            creds = flow.run_console() if use_console else flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as fh:
            fh.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def _ensure_sync_folder(service, parent_id: Optional[str], folder_name: str) -> Optional[str]:
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    resp = service.files().list(q=query, fields="files(id, name)").execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]
    body = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        body["parents"] = [parent_id]
    folder = service.files().create(body=body, fields="id").execute()
    return folder.get("id")


def _get_parent_folder_id(service, file_id: str) -> Optional[str]:
    meta = service.files().get(fileId=file_id, fields="parents").execute()
    parents = meta.get("parents") or []
    return parents[0] if parents else None


def _upload_file(service, local_path: Path, folder_id: Optional[str]) -> str:
    from googleapiclient.http import MediaFileUpload

    body = {"name": local_path.name}
    if folder_id:
        body["parents"] = [folder_id]
    media = MediaFileUpload(str(local_path), mimetype="application/json", resumable=True)
    file = service.files().create(body=body, media_body=media, fields="id").execute()
    return file.get("id")


def _download_file(service, file_id: str, dest_path: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _hash_record(record: Dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _export_candidates(conn, table: str, threshold: float, batch_size: int) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT id, ts, importance, access_count, last_accessed, source, pinned
        FROM {table}
        WHERE importance <= ? AND (pinned IS NULL OR pinned = 0)
        ORDER BY importance ASC
        LIMIT ?
        """,
        (threshold, batch_size),
    )
    rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "ts": row[1],
            "importance": row[2],
            "access_count": row[3],
            "last_accessed": row[4],
            "source": row[5],
            "pinned": row[6],
        }
        for row in rows
    ]


def _fetch_record(conn, table: str, record_id: int) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    if table == "facts":
        cur.execute("SELECT id, text, session, ts, importance, access_count, last_accessed, source, context, pinned FROM facts WHERE id = ?", (record_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "text": row[1],
            "session": row[2],
            "ts": row[3],
            "importance": row[4],
            "access_count": row[5],
            "last_accessed": row[6],
            "source": row[7],
            "context": row[8],
            "pinned": row[9],
        }
    if table == "triples":
        cur.execute(
            "SELECT id, session, subject, predicate, object, ts, confidence, source, context, importance, access_count, last_accessed, pinned "
            "FROM triples WHERE id = ?",
            (record_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "session": row[1],
            "subject": row[2],
            "predicate": row[3],
            "object": row[4],
            "ts": row[5],
            "confidence": row[6],
            "source": row[7],
            "context": row[8],
            "importance": row[9],
            "access_count": row[10],
            "last_accessed": row[11],
            "pinned": row[12],
        }
    if table == "entities":
        cur.execute(
            "SELECT name, type, description, ts, source, context, importance, access_count, last_accessed, pinned FROM entities WHERE name = ?",
            (record_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "name": row[0],
            "type": row[1],
            "description": row[2],
            "ts": row[3],
            "source": row[4],
            "context": row[5],
            "importance": row[6],
            "access_count": row[7],
            "last_accessed": row[8],
            "pinned": row[9],
        }
    if table == "patterns":
        cur.execute(
            "SELECT id, trigger_intent, steps_json, ts, source, context, importance, access_count, last_accessed, pinned FROM patterns WHERE id = ?",
            (record_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "trigger_intent": row[1],
            "steps_json": row[2],
            "ts": row[3],
            "source": row[4],
            "context": row[5],
            "importance": row[6],
            "access_count": row[7],
            "last_accessed": row[8],
            "pinned": row[9],
        }
    if table == "fragments":
        cur.execute(
            "SELECT id, parent_type, parent_id, text, ts, source, context, importance, access_count, last_accessed, pinned FROM fragments WHERE id = ?",
            (record_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "parent_type": row[1],
            "parent_id": row[2],
            "text": row[3],
            "ts": row[4],
            "source": row[5],
            "context": row[6],
            "importance": row[7],
            "access_count": row[8],
            "last_accessed": row[9],
            "pinned": row[10],
        }
    return None


def export_low_importance(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _load_big_brain_cfg(config)
    if not cfg.get("enabled", False):
        return {"success": False, "error": "big_brain_disabled"}
    if not cfg.get("export_enabled", True):
        return {"success": False, "error": "export_disabled"}

    mode = _sync_mode(cfg)
    file_id = None
    if mode == "drive":
        file_id = _resolve_drive_file_id(cfg)
        if not file_id:
            return {"success": False, "error": "missing_drive_file_id"}

    db_path = local_db_path()
    ensure_brain_schema(db_path)
    import sqlite3

    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    recompute_table(conn, "facts", config=config)
    recompute_table(conn, "triples", config=config)
    recompute_table(conn, "entities", key_col="name", config=config)
    recompute_table(conn, "patterns", config=config)
    recompute_table(conn, "fragments", config=config)

    threshold = float(cfg.get("export_importance_threshold", 0.25))
    batch_size = int(cfg.get("export_batch_size", 200))

    export_items = []
    for table in ("facts", "triples", "entities", "patterns", "fragments"):
        candidates = _export_candidates(conn, table, threshold, batch_size)
        for candidate in candidates:
            record = _fetch_record(conn, table, candidate["id"])
            if not record:
                continue
            export_items.append({"type": table[:-1], "record": record})

    if not export_items:
        conn.close()
        return {"success": True, "exported": 0}

    sync_dir = _resolve_path(cfg.get("export_dir", "artifacts/big_brain_sync"))
    sync_dir.mkdir(parents=True, exist_ok=True)
    export_path = sync_dir / f"dexter_export_{int(time.time())}.jsonl"
    with open(export_path, "w", encoding="utf-8") as fh:
        for item in export_items:
            item["checksum"] = _hash_record(item["record"])
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    target_ref = None
    if mode == "local":
        sync_dir = _resolve_path(cfg.get("sync_folder", "dexter_sync"))
        sync_dir.mkdir(parents=True, exist_ok=True)
        target_path = sync_dir / export_path.name
        if target_path.resolve() != export_path.resolve():
            shutil.copy2(export_path, target_path)
        target_ref = str(target_path)
    else:
        service = _get_drive_service(cfg)
        parent_id = _get_parent_folder_id(service, file_id)
        folder_id = _ensure_sync_folder(service, parent_id, cfg.get("sync_folder", "dexter_sync"))
        file_id = _upload_file(service, export_path, folder_id)
        target_ref = file_id

    cur = conn.cursor()
    now_ts = time.time()
    for item in export_items:
        record = item["record"]
        cur.execute(
            "INSERT OR REPLACE INTO sync_state (item_type, item_id, last_synced, status, checksum, target) VALUES (?, ?, ?, ?, ?, ?)",
            (item["type"], str(record.get("id") or record.get("name")), now_ts, "exported", item["checksum"], "big_brain"),
        )
    conn.commit()
    conn.close()

    response = {"success": True, "exported": len(export_items)}
    if mode == "local":
        response["file_path"] = target_ref
        response["mode"] = "local"
    else:
        response["file_id"] = file_id
        response["mode"] = "drive"
    return response


def import_updates(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _load_big_brain_cfg(config)
    if not cfg.get("enabled", False):
        return {"success": False, "error": "big_brain_disabled"}
    if not cfg.get("import_enabled", True):
        return {"success": False, "error": "import_disabled"}

    mode = _sync_mode(cfg)
    files: List[Dict[str, Any]] = []
    service = None

    if mode == "local":
        sync_dir = _resolve_path(cfg.get("sync_folder", "dexter_sync"))
        if not sync_dir.exists():
            return {"success": True, "imported": 0, "mode": "local"}
        for path in sorted(sync_dir.glob("dexter_export_*.jsonl")):
            files.append({"id": path.name, "name": path.name, "path": path})
    else:
        file_id = _resolve_drive_file_id(cfg)
        if not file_id:
            return {"success": False, "error": "missing_drive_file_id"}
        service = _get_drive_service(cfg)
        parent_id = _get_parent_folder_id(service, file_id)
        folder_id = _ensure_sync_folder(service, parent_id, cfg.get("sync_folder", "dexter_sync"))
        query = f"'{folder_id}' in parents and trashed=false and name contains 'dexter_export_'"
        files = service.files().list(q=query, fields="files(id, name, createdTime)").execute().get("files", [])
    if not files:
        return {"success": True, "imported": 0, "mode": mode}

    sync_state_path = _sync_state_path(cfg)
    state = _load_sync_state(sync_state_path)
    processed = set(state.get("processed_files", []))

    db_path = local_db_path()
    ensure_brain_schema(db_path)
    import sqlite3

    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    imported = 0
    for file in files:
        if file["id"] in processed:
            continue
        if mode == "local":
            local_path = Path(file["path"])
        else:
            local_path = _resolve_path(cfg.get("import_dir", "memory/big_brain_cache")) / file["name"]
            _download_file(service, file["id"], local_path)
        with open(local_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                item_type = payload.get("type")
                record = payload.get("record") or {}
                if not item_type or not record:
                    continue
                if _insert_record(conn, item_type, record):
                    imported += 1
        processed.add(file["id"])

    conn.commit()
    conn.close()
    state["processed_files"] = sorted(processed)
    _save_sync_state(sync_state_path, state)
    return {"success": True, "imported": imported, "mode": mode}


def _insert_record(conn, item_type: str, record: Dict[str, Any]) -> bool:
    cur = conn.cursor()
    if item_type == "fact":
        cur.execute("SELECT id FROM facts WHERE text = ? LIMIT 1", (record.get("text"),))
        if cur.fetchone():
            return False
        cur.execute(
            "INSERT INTO facts (text, session, ts, importance, access_count, last_accessed, source, context, pinned) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.get("text"),
                record.get("session"),
                record.get("ts"),
                record.get("importance"),
                record.get("access_count"),
                record.get("last_accessed"),
                record.get("source"),
                record.get("context"),
                record.get("pinned", 0),
            ),
        )
        return True
    if item_type == "triple":
        cur.execute(
            "SELECT id FROM triples WHERE subject = ? AND predicate = ? AND object = ? LIMIT 1",
            (record.get("subject"), record.get("predicate"), record.get("object")),
        )
        if cur.fetchone():
            return False
        cur.execute(
            "INSERT INTO triples (session, subject, predicate, object, ts, confidence, source, context, importance, access_count, last_accessed, pinned) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.get("session"),
                record.get("subject"),
                record.get("predicate"),
                record.get("object"),
                record.get("ts"),
                record.get("confidence"),
                record.get("source"),
                record.get("context"),
                record.get("importance"),
                record.get("access_count"),
                record.get("last_accessed"),
                record.get("pinned", 0),
            ),
        )
        return True
    if item_type == "entity":
        name = record.get("name")
        if not name:
            return False
        cur.execute(
            "INSERT OR REPLACE INTO entities (name, type, description, ts, source, context, importance, access_count, last_accessed, pinned) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                record.get("type"),
                record.get("description"),
                record.get("ts"),
                record.get("source"),
                record.get("context"),
                record.get("importance"),
                record.get("access_count"),
                record.get("last_accessed"),
                record.get("pinned", 0),
            ),
        )
        return True
    if item_type == "pattern":
        cur.execute(
            "INSERT INTO patterns (trigger_intent, steps_json, ts, source, context, importance, access_count, last_accessed, pinned) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.get("trigger_intent"),
                record.get("steps_json"),
                record.get("ts"),
                record.get("source"),
                record.get("context"),
                record.get("importance"),
                record.get("access_count"),
                record.get("last_accessed"),
                record.get("pinned", 0),
            ),
        )
        return True
    if item_type == "fragment":
        cur.execute(
            "INSERT INTO fragments (parent_type, parent_id, text, ts, source, context, importance, access_count, last_accessed, pinned) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.get("parent_type"),
                record.get("parent_id"),
                record.get("text"),
                record.get("ts"),
                record.get("source"),
                record.get("context"),
                record.get("importance"),
                record.get("access_count"),
                record.get("last_accessed"),
                record.get("pinned", 0),
            ),
        )
        return True
    return False


def sync_all(config: Dict[str, Any]) -> Dict[str, Any]:
    export_res = export_low_importance(config)
    import_res = import_updates(config)
    return {
        "success": bool(export_res.get("success")) and bool(import_res.get("success")),
        "export": export_res,
        "import": import_res,
    }
