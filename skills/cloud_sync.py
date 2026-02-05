#!/usr/bin/env python3
"""
CloudSync - Synchronizes local summaries with the Google Drive Big Brain.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict


def _load_config() -> Dict[str, Any]:
    repo_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
    config_path = repo_root / "configs" / "core_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sync_to_big_brain() -> Dict[str, Any]:
    """
    Pushes low-importance memory to the Big Brain and pulls any new updates.
    """
    from core.big_brain_sync import sync_all

    config = _load_config()
    result = sync_all(config)
    export = result.get("export") or {}
    imported = result.get("import") or {}
    return {
        "success": bool(result.get("success")),
        "exported": export.get("exported", 0),
        "imported": imported.get("imported", 0),
        "export": export,
        "import": imported,
    }


def run_test():
    return sync_to_big_brain()
