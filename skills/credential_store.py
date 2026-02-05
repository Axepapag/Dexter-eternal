import os
import json
import time
from typing import Dict, Any, Optional

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "credentials.json")
KEY_ENV = "TOOLS_CRED_KEY"


def _require_crypto():
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography not installed. Install requirements.txt")


def _load_key() -> bytes:
    key = os.getenv(KEY_ENV)
    if not key:
        raise RuntimeError(f"Missing encryption key. Set {KEY_ENV} to a Fernet key (base64 32 bytes).")
    try:
        return key.encode()
    except Exception as exc:
        raise RuntimeError(f"Invalid {KEY_ENV}: {exc}")


def _cipher():
    _require_crypto()
    return Fernet(_load_key())


def _ensure_store_dir():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)


def _load_store() -> Dict[str, Any]:
    if not os.path.exists(DATA_PATH):
        return {}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def _save_store(data: Dict[str, Any]) -> None:
    _ensure_store_dir()
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_credential(service: str, username: str, password: str, notes: Optional[str] = None) -> Dict[str, Any]:
    """
    Save credentials for a service using Fernet encryption. Key is read from env TOOLS_CRED_KEY.
    """
    c = _cipher()
    store = _load_store()
    payload = {
        "username": c.encrypt(username.encode()).decode(),
        "password": c.encrypt(password.encode()).decode(),
        "notes": c.encrypt(notes.encode()).decode() if notes else None,
        "updated_at": time.time(),
    }
    store[service] = payload
    _save_store(store)
    return {"success": True, "service": service, "stored": True}


def get_credential(service: str) -> Dict[str, Any]:
    """
    Retrieve credentials for a service. Returns decrypted fields.
    """
    c = _cipher()
    store = _load_store()
    if service not in store:
        return {"success": False, "error": f"Service '{service}' not found"}
    entry = store[service]
    try:
        return {
            "success": True,
            "service": service,
            "username": c.decrypt(entry["username"].encode()).decode(),
            "password": c.decrypt(entry["password"].encode()).decode(),
            "notes": c.decrypt(entry["notes"].encode()).decode() if entry.get("notes") else None,
            "updated_at": entry.get("updated_at"),
        }
    except Exception as exc:
        return {"success": False, "error": f"Decrypt failed: {exc}"}


def list_credentials() -> Dict[str, Any]:
    """
    List stored services (names only, not secrets).
    """
    store = _load_store()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_path = os.path.relpath(DATA_PATH, repo_root)
    return {"success": True, "services": list(store.keys()), "path": rel_path}
