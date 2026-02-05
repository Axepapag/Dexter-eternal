"""
tracing.py - Lightweight tracing/profiling utilities (stdlib-only).

Design goals:
- Near-zero overhead when disabled (default).
- Safe to import anywhere (no heavy deps, no side effects).
- Works for sync and async spans.

This is intentionally minimal: it is not a full OpenTelemetry implementation.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Optional, Tuple


_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "dexter_trace_id", default=None
)
_span_stack_var: contextvars.ContextVar[Tuple[int, ...]] = contextvars.ContextVar(
    "dexter_span_stack", default=()
)


def _now_ns() -> int:
    # perf_counter_ns is monotonic and high-resolution; good for durations.
    return time.perf_counter_ns()


def _new_trace_id() -> str:
    # Avoid uuid import; good-enough unique trace IDs for local profiling.
    return f"t{os.getpid()}-{_now_ns()}"


@dataclass(frozen=True)
class SpanRecord:
    name: str
    start_ns: int
    end_ns: int
    duration_ms: float
    trace_id: str
    span_id: int
    parent_span_id: Optional[int] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": self.duration_ms,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "attrs": self.attrs,
            "error": self.error,
        }


class _NullSpan:
    __slots__ = ()

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    async def __aenter__(self) -> "_NullSpan":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def set_attr(self, key: str, value: Any) -> None:
        return

    def record_error(self, err: str) -> None:
        return


class NullTracer:
    __slots__ = ()

    enabled = False

    def span(self, name: str, **attrs: Any) -> _NullSpan:
        return _NullSpan()

    def event(self, name: str, **attrs: Any) -> None:
        return

    def recent(self, limit: int = 200) -> list[SpanRecord]:
        return []

    def summarize(self) -> Dict[str, Any]:
        return {"enabled": False}

    def export_jsonl(self, path: str, limit: int = 0) -> int:
        return 0


class Span:
    __slots__ = ("_tracer", "_name", "_attrs", "_start_ns", "_span_id", "_parent", "_trace_id", "_token_stack")

    def __init__(self, tracer: "Tracer", name: str, attrs: Dict[str, Any]):
        self._tracer = tracer
        self._name = name
        self._attrs = attrs
        self._start_ns = 0
        self._span_id = 0
        self._parent: Optional[int] = None
        self._trace_id = ""
        self._token_stack: Optional[contextvars.Token] = None

    def set_attr(self, key: str, value: Any) -> None:
        self._attrs[key] = value

    def record_error(self, err: str) -> None:
        self._attrs.setdefault("error", err)

    def __enter__(self) -> "Span":
        self._start_ns = _now_ns()
        self._trace_id = _trace_id_var.get() or _new_trace_id()
        _trace_id_var.set(self._trace_id)

        stack = _span_stack_var.get()
        self._parent = stack[-1] if stack else None
        self._span_id = self._tracer._next_span_id()
        self._token_stack = _span_stack_var.set(stack + (self._span_id,))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            end_ns = _now_ns()
            dur_ms = (end_ns - self._start_ns) / 1_000_000.0
            err = None
            if exc is not None:
                err = f"{exc_type.__name__}: {exc}"
            self._tracer._record(
                SpanRecord(
                    name=self._name,
                    start_ns=self._start_ns,
                    end_ns=end_ns,
                    duration_ms=dur_ms,
                    trace_id=self._trace_id,
                    span_id=self._span_id,
                    parent_span_id=self._parent,
                    attrs=dict(self._attrs),
                    error=err,
                )
            )
        finally:
            # Restore previous stack.
            if self._token_stack is not None:
                try:
                    _span_stack_var.reset(self._token_stack)
                except Exception:
                    pass
        return False

    async def __aenter__(self) -> "Span":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)


class Tracer:
    """
    In-memory tracer with a bounded ring buffer of completed spans.

    Notes:
    - This only stores completed spans, not in-flight.
    - Export is best-effort; failures are swallowed to keep instrumentation safe.
    """

    def __init__(self, *, enabled: bool, max_spans: int = 2000):
        self.enabled = bool(enabled)
        self._spans: Deque[SpanRecord] = deque(maxlen=max(100, int(max_spans)))
        self._lock = threading.Lock()
        self._span_id = 0

    def _next_span_id(self) -> int:
        with self._lock:
            self._span_id += 1
            return self._span_id

    def span(self, name: str, **attrs: Any) -> Span | _NullSpan:
        if not self.enabled:
            return _NullSpan()
        return Span(self, name, dict(attrs))

    def event(self, name: str, **attrs: Any) -> None:
        if not self.enabled:
            return
        # Represent events as zero-duration spans for a uniform record format.
        now = _now_ns()
        trace_id = _trace_id_var.get() or _new_trace_id()
        _trace_id_var.set(trace_id)
        stack = _span_stack_var.get()
        parent = stack[-1] if stack else None
        span_id = self._next_span_id()
        self._record(
            SpanRecord(
                name=name,
                start_ns=now,
                end_ns=now,
                duration_ms=0.0,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent,
                attrs=dict(attrs),
                error=None,
            )
        )

    def _record(self, rec: SpanRecord) -> None:
        try:
            with self._lock:
                self._spans.append(rec)
        except Exception:
            return

    def recent(self, limit: int = 200) -> list[SpanRecord]:
        if not self.enabled:
            return []
        n = max(0, int(limit))
        with self._lock:
            if n <= 0:
                return list(self._spans)
            return list(self._spans)[-n:]

    def summarize(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        with self._lock:
            spans = list(self._spans)
        by_name: Dict[str, Dict[str, Any]] = {}
        for s in spans:
            bucket = by_name.setdefault(
                s.name, {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
            )
            bucket["count"] += 1
            bucket["total_ms"] += float(s.duration_ms)
            if s.duration_ms > bucket["max_ms"]:
                bucket["max_ms"] = float(s.duration_ms)
        # Compute avg in a second pass.
        for v in by_name.values():
            v["avg_ms"] = (v["total_ms"] / v["count"]) if v["count"] else 0.0
        return {"enabled": True, "spans": len(spans), "by_name": by_name}

    def export_jsonl(self, path: str, limit: int = 0) -> int:
        """
        Export recent spans to JSONL.
        Returns number of lines written (best-effort).
        """
        if not self.enabled:
            return 0
        try:
            if limit and limit > 0:
                recs = self.recent(limit=limit)
            else:
                recs = self.recent(limit=0)
            if not recs:
                return 0
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")
            return len(recs)
        except Exception:
            return 0


_GLOBAL_TRACER: Tracer | NullTracer = NullTracer()


def init_tracer(config: Optional[Dict[str, Any]] = None) -> Tracer | NullTracer:
    """
    Initialize the global tracer once per process.

    Enablement:
    - env `DEXTER_TRACE=1` always enables tracing
    - config `instrumentation.tracing.enabled=true` enables tracing
    """
    global _GLOBAL_TRACER
    cfg = config or {}
    inst = (cfg.get("instrumentation") or {}) if isinstance(cfg, dict) else {}
    tcfg = (inst.get("tracing") or {}) if isinstance(inst, dict) else {}

    env_enabled = str(os.getenv("DEXTER_TRACE", "")).strip().lower() in ("1", "true", "yes", "on")
    enabled = env_enabled or bool(tcfg.get("enabled", False))
    max_spans = int(tcfg.get("max_spans", 2000))
    _GLOBAL_TRACER = Tracer(enabled=enabled, max_spans=max_spans) if enabled else NullTracer()
    return _GLOBAL_TRACER


def get_tracer() -> Tracer | NullTracer:
    return _GLOBAL_TRACER


async def trace_await(
    awaitable,
    name: str,
    *,
    warn_ms: Optional[float] = None,
    **attrs: Any,
):
    """
    Await a coroutine/future while recording a span. If warn_ms is set and the
    await is slow, emit a one-line TRACE log (stream-only).
    """
    tracer = get_tracer()
    if not getattr(tracer, "enabled", False):
        return await awaitable

    started = time.monotonic()
    with tracer.span(name, **attrs):
        out = await awaitable
    if warn_ms is not None:
        dur_ms = (time.monotonic() - started) * 1000.0
        if dur_ms >= float(warn_ms):
            try:
                # Keep payload short; logs should be safe in always-on stream.
                print(f"TRACE: slow await {name} took {dur_ms:.0f}ms attrs={attrs}", flush=True)
            except Exception:
                pass
    return out

