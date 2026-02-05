from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

import aiohttp


def embed_text(text: str, config: Optional[Dict[str, Any]] = None) -> Optional[List[float]]:
    if not text:
        return None
    # Tests should not depend on a running local embedding server.
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None
    cfg = (config or {}).get("embeddings", {}) if config else {}
    if not cfg.get("enabled", False):
        return None
    base_url = (cfg.get("base_url") or "http://localhost:11434").rstrip("/")
    model = cfg.get("model") or "qwen3-embedding:0.6b-fp16"
    timeout = float(cfg.get("timeout_sec", 30))
    payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        return data.get("embedding")
    except Exception:
        return None


async def embed_text_async(text: str, config: Optional[Dict[str, Any]] = None) -> Optional[List[float]]:
    """
    Async embedding call for use inside Dexter's asyncio runtime.

    This avoids blocking the event loop (which can delay timeouts and stall all other
    async tasks) when the embedding provider is slow or unavailable.
    """
    if not text:
        return None
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None
    cfg = (config or {}).get("embeddings", {}) if config else {}
    if not cfg.get("enabled", False):
        return None

    base_url = (cfg.get("base_url") or "http://localhost:11434").rstrip("/")
    model = cfg.get("model") or "qwen3-embedding:0.6b-fp16"
    timeout_sec = float(cfg.get("timeout_sec", 30))
    endpoint = f"{base_url}/api/embeddings"
    payload = {"model": model, "prompt": text}

    timeout = aiohttp.ClientTimeout(total=timeout_sec, connect=timeout_sec, sock_connect=timeout_sec, sock_read=timeout_sec)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=payload) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                emb = data.get("embedding")
                return emb if isinstance(emb, list) else None
    except asyncio.CancelledError:
        raise
    except Exception:
        return None
