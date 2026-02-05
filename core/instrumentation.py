"""
instrumentation.py - Runtime instrumentation for blocking behavior.

Currently provides:
- Event loop lag monitor (detects when the asyncio loop is starved / blocked).

All features are best-effort and safe to leave enabled in production:
failures are swallowed and overhead is bounded.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.tracing import get_tracer


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur.get(part)
    return default if cur is None else cur


@dataclass
class InstrumentationSettings:
    enabled: bool = True

    # Event loop lag
    loop_lag_enabled: bool = True
    loop_lag_interval_ms: float = 100.0
    loop_lag_warn_ms: float = 250.0
    loop_lag_critical_ms: float = 1000.0
    loop_lag_min_log_interval_sec: float = 5.0

    # Slow operations
    slow_await_warn_ms: float = 2000.0
    slow_tool_warn_ms: float = 2000.0
    slow_llm_warn_ms: float = 8000.0

    @staticmethod
    def from_config(config: Optional[Dict[str, Any]] = None) -> "InstrumentationSettings":
        cfg = config or {}

        # Global enable can be overridden by env for quick debugging.
        enabled = bool(_cfg_get(cfg, "instrumentation.enabled", True))
        if _truthy_env("DEXTER_INSTRUMENTATION"):
            enabled = True
        if _truthy_env("DEXTER_NO_INSTRUMENTATION"):
            enabled = False

        s = InstrumentationSettings(enabled=enabled)
        s.loop_lag_enabled = bool(_cfg_get(cfg, "instrumentation.event_loop_lag.enabled", True))
        s.loop_lag_interval_ms = float(_cfg_get(cfg, "instrumentation.event_loop_lag.interval_ms", s.loop_lag_interval_ms))
        s.loop_lag_warn_ms = float(_cfg_get(cfg, "instrumentation.event_loop_lag.warn_ms", s.loop_lag_warn_ms))
        s.loop_lag_critical_ms = float(_cfg_get(cfg, "instrumentation.event_loop_lag.critical_ms", s.loop_lag_critical_ms))
        s.loop_lag_min_log_interval_sec = float(
            _cfg_get(cfg, "instrumentation.event_loop_lag.min_log_interval_sec", s.loop_lag_min_log_interval_sec)
        )

        s.slow_await_warn_ms = float(_cfg_get(cfg, "instrumentation.slow_await_warn_ms", s.slow_await_warn_ms))
        s.slow_tool_warn_ms = float(_cfg_get(cfg, "instrumentation.slow_tool_warn_ms", s.slow_tool_warn_ms))
        s.slow_llm_warn_ms = float(_cfg_get(cfg, "instrumentation.slow_llm_warn_ms", s.slow_llm_warn_ms))
        return s


class EventLoopLagMonitor:
    """
    Periodically measures event-loop lag as:
      actual_wakeup_time - expected_wakeup_time

    Large lag indicates blocking code on the event loop or scheduler starvation.
    """

    def __init__(self, settings: InstrumentationSettings):
        self._s = settings
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_log = 0.0

    def start(self) -> None:
        if not self._s.enabled or not self._s.loop_lag_enabled:
            return
        if self._running:
            return
        self._running = True
        try:
            self._task = asyncio.create_task(self._run(), name="event-loop-lag-monitor")
        except Exception:
            self._task = None
            self._running = False

    async def stop(self) -> None:
        self._running = False
        t = self._task
        self._task = None
        if not t:
            return
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _run(self) -> None:
        tracer = get_tracer()
        loop = asyncio.get_running_loop()
        interval_s = max(0.01, float(self._s.loop_lag_interval_ms) / 1000.0)
        expected = loop.time() + interval_s

        while self._running:
            try:
                await asyncio.sleep(interval_s)
                now = loop.time()
                lag_s = max(0.0, now - expected)
                expected = now + interval_s
                lag_ms = lag_s * 1000.0

                if lag_ms < float(self._s.loop_lag_warn_ms):
                    continue

                # Trace event for offline analysis.
                try:
                    if getattr(tracer, "enabled", False):
                        tracer.event(
                            "event_loop_lag",
                            lag_ms=float(lag_ms),
                            warn_ms=float(self._s.loop_lag_warn_ms),
                            critical_ms=float(self._s.loop_lag_critical_ms),
                            interval_ms=float(self._s.loop_lag_interval_ms),
                        )
                except Exception:
                    pass

                # Rate-limited log for humans.
                now_wall = time.monotonic()
                critical = lag_ms >= float(self._s.loop_lag_critical_ms)
                if critical or (now_wall - self._last_log) >= float(self._s.loop_lag_min_log_interval_sec):
                    self._last_log = now_wall
                    try:
                        level = "CRITICAL" if critical else "WARN"
                        print(f"TRACE: {level} event loop lag {lag_ms:.0f}ms", flush=True)
                    except Exception:
                        pass
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never crash the host loop.
                await asyncio.sleep(interval_s)


def maybe_enable_asyncio_debug(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Optional: enable asyncio debug hooks if requested.

    This can be helpful for catching slow callbacks, but it is intentionally opt-in.
    """
    cfg = config or {}
    if not bool(_cfg_get(cfg, "instrumentation.asyncio_debug", False)) and not _truthy_env("DEXTER_ASYNCIO_DEBUG"):
        return
    try:
        loop = asyncio.get_running_loop()
        loop.set_debug(True)
        # Lower value => more sensitive warnings.
        loop.slow_callback_duration = float(_cfg_get(cfg, "instrumentation.slow_callback_duration_sec", 0.2))
        print("[Instrumentation] asyncio debug enabled", flush=True)
    except Exception:
        return

