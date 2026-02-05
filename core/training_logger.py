import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()

try:
    from core.async_jsonl_writer import enqueue_jsonl as _enqueue_jsonl
except Exception:  # pragma: no cover
    _enqueue_jsonl = None


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: str, payload: Dict[str, Any]) -> bool:
    if not path:
        return False
    _ensure_parent(path)
    payload = dict(payload)
    payload.setdefault("ts", time.time())
    line = json.dumps(payload, ensure_ascii=False)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return True


def enqueue_jsonl(path: str, payload: Dict[str, Any]) -> bool:
    """
    Best-effort, non-blocking JSONL append.

    This uses the background JSONL writer if enabled via env `DEXTER_ASYNC_JSONL_LOG=1`.
    If not enabled (or unavailable), returns False and the caller can fall back to sync.
    """
    if _enqueue_jsonl is None:
        return False
    try:
        return bool(_enqueue_jsonl(path, payload))
    except Exception:
        return False


def log_tool_call(
    path: str,
    intent: Optional[str],
    task: str,
    skill_id: Optional[str],
    tool_name: Optional[str],
    arguments: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    call_source: str,
    skill_confidence: Optional[float] = None,
    tool_confidence: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    payload = {
        "intent": intent,
        "task": task,
        "skill_id": skill_id,
        "skill_confidence": skill_confidence,
        "tool_name": tool_name,
        "arguments": arguments or {},
        "result": result or {},
        "call_source": call_source,
        "tool_confidence": tool_confidence,
    }
    if extra:
        payload.update(extra)
    return append_jsonl(path, payload)


def enqueue_tool_call(
    path: str,
    intent: Optional[str],
    task: str,
    skill_id: Optional[str],
    tool_name: Optional[str],
    arguments: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    call_source: str,
    skill_confidence: Optional[float] = None,
    tool_confidence: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    payload = {
        "intent": intent,
        "task": task,
        "skill_id": skill_id,
        "skill_confidence": skill_confidence,
        "tool_name": tool_name,
        "arguments": arguments or {},
        "result": result or {},
        "call_source": call_source,
        "tool_confidence": tool_confidence,
    }
    if extra:
        payload.update(extra)
    return enqueue_jsonl(path, payload)


def log_plan_event(
    path: str,
    goal: Optional[str],
    plan: Optional[Dict[str, Any]],
    source: str,
    template_id: Optional[str] = None,
    confidence: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    payload = {
        "event": "plan_created",
        "goal": goal,
        "plan": plan or {},
        "source": source,
        "template_id": template_id,
        "confidence": confidence,
    }
    if extra:
        payload.update(extra)
    return append_jsonl(path, payload)


def enqueue_plan_event(
    path: str,
    goal: Optional[str],
    plan: Optional[Dict[str, Any]],
    source: str,
    template_id: Optional[str] = None,
    confidence: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    payload = {
        "event": "plan_created",
        "goal": goal,
        "plan": plan or {},
        "source": source,
        "template_id": template_id,
        "confidence": confidence,
    }
    if extra:
        payload.update(extra)
    return enqueue_jsonl(path, payload)


def log_experience(
    path: str,
    intent: Optional[str],
    step_index: int,
    task: str,
    skill_id: Optional[str],
    tool_name: Optional[str],
    arguments: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    decision: Optional[str],
    plan: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    payload = {
        "event": "step_result",
        "intent": intent,
        "step_index": step_index,
        "task": task,
        "skill_id": skill_id,
        "tool_name": tool_name,
        "arguments": arguments or {},
        "result": result or {},
        "decision": decision,
        "plan": plan or {},
    }
    if extra:
        payload.update(extra)
    return append_jsonl(path, payload)


def enqueue_experience(
    path: str,
    intent: Optional[str],
    step_index: int,
    task: str,
    skill_id: Optional[str],
    tool_name: Optional[str],
    arguments: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    decision: Optional[str],
    plan: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    payload = {
        "event": "step_result",
        "intent": intent,
        "step_index": step_index,
        "task": task,
        "skill_id": skill_id,
        "tool_name": tool_name,
        "arguments": arguments or {},
        "result": result or {},
        "decision": decision,
        "plan": plan or {},
    }
    if extra:
        payload.update(extra)
    return enqueue_jsonl(path, payload)
