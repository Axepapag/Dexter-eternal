#!/usr/bin/env python3
"""
Dexter.py - The Unified Brain Entry Point
Handles multi-TRM orchestration and tiered reasoning.
"""

import os
import sys
import asyncio
import json
import time
import socket
import subprocess
import argparse
import queue
import signal
import errno
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix module loading for subdirectories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(BASE_DIR / "core"))
sys.path.append(str(BASE_DIR / "skills"))

# Imports at top level to ensure global access
from core.skill_librarian import SkillLibrarian
from core.reasoning_engine_unified import ReasoningEngine
from core.tool_agent import ToolAgent
import core.tool_executor as tool_executor
from core.async_executor import AsyncToolExecutor
from core.context_bundle import ContextBundler
from core.evolution_engine import EvolutionEngine
from core.brain_schema import ensure_brain_schema
from skills.memory_ops import _db_path as memory_db_path
from core.bootstrap import run_trm_tool_pipeline
from core.trm_tool_policy import TRMToolPolicy
from core.rolling_metrics import RollingAccuracy
from core.training_logger import (
    enqueue_experience,
    enqueue_tool_call,
    log_experience,
    log_tool_call,
)
from core.toolbook_utils import build_tool_template
from core.speaker import speak_out_loud
from core.api import start_api_server, broadcast_thought
from core.response_tank import get_global_tank
from core.rolling_context_bundle import get_global_bundle, get_staged_bundle, StagedContextBundle
from core.llm_think_tank import LLMThinkTank
from core.memory_trm import create_memory_trm
from core.memory_ingestor import MemoryIngestor
from core.memory_retriever import retrieve_context
from core.memory_buckets import BucketManager
from core.memory_db_writer import MemoryDBWriter
from core.persistent_bundle import PersistentArtifactBundle
from core.utils import extract_json
from core.trained_trm_loader import load_all_trms, get_tool_trm, get_memory_trm
from core.tool_trm_gateway import ToolTRMGateway, ToolRequest, ContextChannel, get_gateway, ToolResult, ExecutionMetadata
from core.trm_online_trainer import init_online_trainers, get_online_trainer
from core.tracing import init_tracer
from core.instrumentation import EventLoopLagMonitor, InstrumentationSettings, maybe_enable_asyncio_debug
import threading

# ═══════════════════════════════════════════════════════════════════════════════
# DUAL TERMINAL SYSTEM - Stream terminal for logs, Conversation terminal for chat
# ═══════════════════════════════════════════════════════════════════════════════

STREAM_PORT = 19847  # Port for stream terminal connection
_original_print = print


class _StreamHub:
    """
    Non-blocking stream broadcaster.

    The main runtime must never do blocking socket sends. We enqueue outbound
    text and a dedicated thread pushes it to connected clients.
    """

    def __init__(self, max_queue: int = 5000):
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._q: "queue.Queue[str | None]" = queue.Queue(maxsize=max(100, int(max_queue)))
        self._running = True
        self._t = threading.Thread(target=self._writer, name="dexter-stream-writer", daemon=True)
        self._t.start()

    def add_client(self, client: socket.socket) -> None:
        try:
            client.setblocking(True)
        except Exception:
            pass
        with self._lock:
            self._clients.append(client)

    def broadcast(self, text: str) -> None:
        if not text or not str(text).strip():
            return
        try:
            self._q.put_nowait(str(text))
        except queue.Full:
            # Best-effort: drop rather than block the runtime.
            return
        except Exception:
            return

    def _writer(self) -> None:
        while self._running:
            try:
                msg = self._q.get()
            except Exception:
                continue
            if msg is None:
                break
            data = msg.encode("utf-8", errors="replace")
            with self._lock:
                clients = list(self._clients)
            dead: list[socket.socket] = []
            for client in clients:
                try:
                    client.sendall(data)
                except Exception:
                    dead.append(client)
            if dead:
                with self._lock:
                    for c in dead:
                        try:
                            c.close()
                        except Exception:
                            pass
                        try:
                            self._clients.remove(c)
                        except Exception:
                            pass


_STREAM_HUB = _StreamHub(max_queue=int(os.getenv("DEXTER_STREAM_QUEUE", "5000") or "5000"))

# Prefixes that indicate system/internal messages (go to stream only)
_STREAM_ONLY_PREFIXES = [
    "[Brain]", "[Gateway]", "[Tool TRM", "[Memory TRM", "[Reasoning TRM",
    "[Forge]", "[ContextChannel", "[StagedBundle", "[Trainer]",
    "[Online Trainer]", "[Executor]", "[API]", "[Forge]", "[Skill",
    "[TRM", "[Stream]", "[Config]", "[Evolution]", "[Librarian]",
    "DEBUG:", "TRACE:", "INFO:", "WARNING:", "ERROR:",
]

# --- Minimal terminal UI helpers (for runtime model picking) ---
def _clear_screen() -> None:
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        return


def _interactive_select(title: str, items: List[str], start_index: int = 0) -> Optional[int]:
    """
    Very small TUI: arrow up/down + Enter to select, Esc to cancel.

    This is intentionally synchronous; call it via `asyncio.to_thread(...)` so it
    never blocks the event loop.
    """
    if not items:
        return None
    idx = max(0, min(int(start_index or 0), len(items) - 1))

    def _render() -> None:
        _clear_screen()
        print(title)
        print("-" * min(80, max(10, len(title))))
        print("Up/Down to move, Enter to select, Esc to cancel\n")
        for i, line in enumerate(items):
            prefix = " > " if i == idx else "   "
            print(prefix + line)

    def _get_key() -> str:
        if os.name == "nt":
            try:
                import msvcrt  # type: ignore

                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # special key prefix
                    ch2 = msvcrt.getwch()
                    return f"{ch}{ch2}"
                return ch
            except Exception:
                return ""
        # POSIX fallback: raw mode for arrows (best-effort)
        try:
            import termios  # type: ignore
            import tty  # type: ignore

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    # Possibly an escape sequence (arrow)
                    ch += sys.stdin.read(2)
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            return ""

    _render()
    while True:
        k = _get_key()
        if not k:
            continue
        if k in ("\r", "\n"):
            return idx
        if k in ("\x1b", "\x1b\x1b", "\x1b\x1b\x1b"):  # Esc
            return None

        # Windows arrow keys: \xe0H up, \xe0P down
        if k in ("\x00H", "\xe0H", "\x1b[A"):
            idx = (idx - 1) % len(items)
            _render()
            continue
        if k in ("\x00P", "\xe0P", "\x1b[B"):
            idx = (idx + 1) % len(items)
            _render()
            continue

# Prefixes that should show in conversation terminal
_CONVERSATION_PREFIXES = [
    "[Dexter][Jeffrey]", "[Jeffrey]", ">>> ",
]

# System messages to filter out of conversation (even if they have [Dexter])
_CONVERSATION_FILTER = [
    "[Dexter][System] Working:", "[Dexter][System] Calling:", "[Dexter][System] Step",
    "[Dexter][System] Selected Skill:", "[Dexter][System] Task received:",
    "[Dexter][System] Preset Intent:", "[Dexter][System] Starting", "[Dexter][System] Waking",
    "[Dexter][System] Rolling", "[Dexter][System] Tool Result:", "[Dexter][System] Staged context",
    "[Dexter][System] Executed:", "[Dexter][System] Goal achieved early.",
    "[Dexter][System] Online TRM Trainer started",
    "[Dexter][System] ✨ New skill", "[Dexter][System] Received from",
]

def _compact_text(text: str, max_words: int = 40, max_chars: int = 240) -> str:
    """Compact text for TRM training targets."""
    if not text:
        return ""
    cleaned = " ".join(str(text).replace("\n", " ").replace("\r", " ").split())
    words = cleaned.split()
    if max_words and len(words) > max_words:
        words = words[:max_words]
    compact = " ".join(words)
    if max_chars and len(compact) > max_chars:
        compact = compact[:max_chars].rstrip()
    return compact


def _compact_meta(metadata: dict, keys: List[str]) -> str:
    if not metadata:
        return ""
    parts = []
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if value is None:
            continue
        value_text = _compact_text(value, max_words=12, max_chars=80)
        if value_text:
            parts.append(f"{key}={value_text}")
    return ";".join(parts)


def _memory_target(sender: str, msg_type: str, message: str, metadata: dict | None) -> str:
    summary = _compact_text(message, max_words=40, max_chars=240)
    meta_bits = _compact_meta(metadata or {}, ["intent", "tool", "tool_used", "success", "error", "task"])
    meta_str = f" <META:{meta_bits}>" if meta_bits else ""
    return f"<MEMORY> [{sender}|{msg_type}] {summary}{meta_str}"


def _reasoning_target(sender: str, msg_type: str, message: str, metadata: dict | None) -> str:
    summary = _compact_text(message, max_words=32, max_chars=200)
    outcome = ""
    if msg_type == "result" and metadata:
        success = metadata.get("success", metadata.get("ok", False))
        outcome = "SUCCESS" if success else "FAIL"
    outcome_str = f" OUTCOME={outcome}" if outcome else ""
    return f"<STEP> OBSERVE {sender}/{msg_type}{outcome_str} <STEP> {summary}"


def _find_pids_on_port(port: int):
    """Return PIDs currently listening on the given port."""
    pids = set()
    try:
        if sys.platform == "win32":
            result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                parts = [part for part in line.split() if part]
                if len(parts) < 5:
                    continue
                local_addr = parts[1]
                state = parts[3] if len(parts) > 3 else ""
                if local_addr.endswith(f":{port}") and state.upper() == "LISTENING":
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        continue
        else:
            result = subprocess.run(["lsof", "-i", f":{port}", "-Pn"], capture_output=True, text=True)
            for line in result.stdout.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pids.add(int(parts[1]))
                    except ValueError:
                        continue
    except Exception:
        pass
    return pids


def _terminate_process_on_port(port: int) -> bool:
    """Terminate any process bound to the given port."""
    pids = _find_pids_on_port(port)
    if not pids:
        return False
    for pid in pids:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    return True


class StreamWriter:
    """Redirects stdout to both console and stream terminal."""
    def __init__(self, original_stdout, is_conversation_terminal=False):
        self.original = original_stdout
        self.is_conversation = is_conversation_terminal
        
    def write(self, text):
        if self.is_conversation:
            # Conversation terminal: clean output for user interaction
            should_show = False
            
            # Check if it's a conversation message
            for prefix in _CONVERSATION_PREFIXES:
                if prefix in text:
                    should_show = True
                    break
            
            # Filter out system messages that have conversation prefixes
            for filter_text in _CONVERSATION_FILTER:
                if filter_text in text:
                    should_show = False
                    break
            
            if should_show:
                self.original.write(text)
        else:
            self.original.write(text)
        
        # Always send to stream clients (they get EVERYTHING)
        _broadcast_to_stream(text)
        
    def flush(self):
        self.original.flush()
    
    def isatty(self):
        """Required for uvicorn/logging compatibility."""
        return hasattr(self.original, 'isatty') and self.original.isatty()
    
    @property
    def encoding(self):
        return getattr(self.original, 'encoding', 'utf-8')
    
    def fileno(self):
        return self.original.fileno() if hasattr(self.original, 'fileno') else -1


def _broadcast_to_stream(text):
    """Send text to all connected stream terminals."""
    try:
        _STREAM_HUB.broadcast(text)
    except Exception:
        return


def _stream_server():
    """Background server accepting stream terminal connections."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bind_attempt = 0
    while True:
        try:
            server.bind(('127.0.0.1', STREAM_PORT))
            break
        except OSError as exc:
            bind_attempt += 1
            _original_print(f"[Stream] Port {STREAM_PORT} bind failed (attempt {bind_attempt}): {exc}", flush=True)
            if exc.errno in (errno.EADDRINUSE, 10048):
                terminated = _terminate_process_on_port(STREAM_PORT)
                if terminated and bind_attempt < 3:
                    time.sleep(1)
                    continue
            server.close()
            raise
    server.listen(5)
    _original_print(f"[Stream] Listening on port {STREAM_PORT}", flush=True)
    while True:
        client, addr = server.accept()
        try:
            _STREAM_HUB.add_client(client)
        except Exception:
            try:
                client.close()
            except Exception:
                pass


def _run_stream_terminal():
    """Run as the stream terminal - receives and displays ALL system output."""
    print("═" * 70)
    print("  DEXTER ACTIVITY STREAM - Full System Logs")
    print("═" * 70)
    print()
    print("  This terminal shows EVERYTHING:")
    print("  • TRM activity (Tool, Memory, Reasoning)")
    print("  • Context bundle triggers and injections")
    print("  • Tool calls, results, errors")
    print("  • Channel messages between Dexter ↔ Forge")
    print("  • Training events and weight updates")
    print("  • Skill creation and forge events")
    print()
    print("═" * 70)
    print()
    
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', STREAM_PORT))
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                sys.stdout.write(data.decode('utf-8', errors='replace'))
                sys.stdout.flush()
        except ConnectionRefusedError:
            print("[Stream] Waiting for main Dexter process...")
            time.sleep(1)
        except Exception as e:
            print(f"[Stream] Connection lost: {e}")
            time.sleep(1)


def _spawn_stream_terminal():
    """Spawn a new terminal window for streaming output."""
    script_path = os.path.abspath(__file__)
    
    if sys.platform == 'win32':
        # Windows: use 'start' to open new cmd window
        subprocess.Popen(
            f'start "Dexter Stream" cmd /k python "{script_path}" --stream-mode',
            shell=True
        )
    elif sys.platform == 'darwin':
        # macOS: use osascript to open Terminal
        subprocess.Popen([
            'osascript', '-e',
            f'tell application "Terminal" to do script "python3 \\"{script_path}\\" --stream-mode"'
        ])
    else:
        # Linux: try common terminals
        for term in ['gnome-terminal', 'xterm', 'konsole']:
            try:
                if term == 'gnome-terminal':
                    subprocess.Popen([term, '--', 'python3', script_path, '--stream-mode'])
                else:
                    subprocess.Popen([term, '-e', f'python3 {script_path} --stream-mode'])
                break
            except FileNotFoundError:
                continue


class Dexter:
    def __init__(self):
        # Paths
        self.repo_root = BASE_DIR
        self.models_dir = BASE_DIR / "models"
        self._config_path = BASE_DIR / "configs" / "core_config.json"
        
        # Initialize Core Systems
        self.config = self._load_json(self._config_path)
        # Lightweight tracing/instrumentation (safe defaults; enabled via config/env).
        init_tracer(self.config)
        self._instrumentation = InstrumentationSettings.from_config(self.config)
        self._loop_lag_monitor = EventLoopLagMonitor(self._instrumentation)
        self._apply_max_training()
        self.conversation_cfg = self._load_conversation_cfg(self.config)
        self.bootstrap_cfg = self.config.get("bootstrap", {}) or {}
        self._ensure_brain_db()

        self.response_tank = get_global_tank()
        self.rolling_context = get_global_bundle()
        self.staged_context = get_staged_bundle()  # Accumulates background updates
        self.orchestrator_bundle = PersistentArtifactBundle("orchestrator")
        self.think_tank_bundle = PersistentArtifactBundle("think_tank")
        self.llm_think_tank = LLMThinkTank(self.config)

        # Tool result reactions:
        # Tool execution happens asynchronously via Forge; without a reactor, tool results
        # only update context and never trigger the orchestrator LLM to "notice" and respond.
        self._tool_result_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=int(os.getenv("DEXTER_TOOL_RESULT_QUEUE", "500") or "500"))
        self._tool_result_reactor_task: Optional[asyncio.Task] = None
        self._think_tank_reload_task: Optional[asyncio.Task] = None

        # Bucket-based ingestion + single DB writer (append-only, non-blocking producers).
        mb_cfg = self.config.get("memory_buckets", {}) or {}
        mb_dir = mb_cfg.get("dir") or "data/buckets"
        mb_path = Path(mb_dir)
        if not mb_path.is_absolute():
            mb_path = self.repo_root / mb_path
        self.bucket_manager = BucketManager(
            base_dir=str(mb_path),
            flush_every=float(mb_cfg.get("flush_every", 0.25)),
        )
        self.memory_db_writer = MemoryDBWriter(self.config)
        self.memory_ingestor = MemoryIngestor(self.config, bucket_manager=self.bucket_manager)

        self.memory_trm = create_memory_trm(
            self.config,
            self.response_tank,
            self.rolling_context,
        )
        # IntentReasonerTRM is deprecated: raw user input must broadcast directly to all modules.
        self.intent_trm_enabled = False
        self.intent_trm_primary = False
        self.intent_trm_min_confidence = 1.0
        mem_cfg = self.config.get("memory_trm", {}) or {}
        self.memory_trm_enabled = bool(mem_cfg.get("enabled", True))
        wm_cfg = self.config.get("orchestrator_working_memory", {}) or {}
        self.working_memory_enabled = bool(wm_cfg.get("enabled", True))
        self.working_memory_path = self._resolve_repo_path(
            wm_cfg.get("path", "data/orchestrator_working_memory.jsonl")
        )
        self._working_memory_lock = threading.Lock()
        self._last_injection_payload: Optional[Dict[str, Any]] = None
        
        self.librarian = SkillLibrarian(self.repo_root)
        self.reasoning = ReasoningEngine(self.config)
        self.tool_agent = ToolAgent(self.config)
        self.tool_executor = tool_executor
        self.evolution = EvolutionEngine(self.repo_root, self.config)
        self.context_bundler = ContextBundler(self.repo_root, self.config)
        self.context_bundles = []
        self._plan_task: Optional[asyncio.Task] = None
        exec_cfg = self.config.get("executor", {}) or {}
        self.async_executor = AsyncToolExecutor(
            max_workers=int(exec_cfg.get("max_workers", 20)),
            result_handler=self._handle_tool_result,
        )
        self.trm_tool_policy = TRMToolPolicy(self.config)
        self.training_log_cfg = self.config.get("training_log", {}) or {}
        log_path = self.training_log_cfg.get("path", str(self.repo_root / "dexter_TRMs" / "datasets" / "runtime" / "tool_calls.jsonl"))
        if not os.path.isabs(log_path):
            log_path = str(self.repo_root / log_path)
        self.training_log_path = log_path
        self.experience_log_cfg = self.config.get("experience_log", {}) or {}
        exp_path = self.experience_log_cfg.get("path", str(self.repo_root / "dexter_TRMs" / "datasets" / "runtime" / "experience.jsonl"))
        if not os.path.isabs(exp_path):
            exp_path = str(self.repo_root / exp_path)
        self.experience_log_path = exp_path
        trm_cfg = self.config.get("trm_tool_policy", {}) or {}
        self.trm_accuracy = RollingAccuracy(
            window=trm_cfg.get("rolling_window", 200),
            min_samples=trm_cfg.get("min_samples", 20),
        )
        self.trm_accuracy_gate = bool(trm_cfg.get("use_accuracy_gate", True))
        self.trm_accuracy_threshold = float(trm_cfg.get("accuracy_threshold", 0.8))
        self.trm_tune_cfg = {
            "enabled": bool(trm_cfg.get("auto_tune_enabled", True)),
            "target": float(trm_cfg.get("target_accuracy", 0.85)),
            "hysteresis": float(trm_cfg.get("target_hysteresis", 0.05)),
            "step": float(trm_cfg.get("tune_step", 0.02)),
            "min_execute": float(trm_cfg.get("min_execute_threshold", 0.7)),
            "max_execute": float(trm_cfg.get("max_execute_threshold", 0.98)),
            "min_shadow": float(trm_cfg.get("min_shadow_threshold", 0.4)),
            "max_shadow": float(trm_cfg.get("max_shadow_threshold", 0.95)),
            "every_n": int(trm_cfg.get("tune_every_n_updates", 20)),
        }
        self.trm_tune_counter = 0
        
        # Load Reflex Core (Vision-to-Action)
        self.reflex_active = self._load_trm("reflex_core.pt")
        
        # Load Trained TRMs from TinyRecursiveModels (GPU-trained on GCloud)
        self._init_trained_trms()
        
        # Tool TRM Gateway - Central execution layer
        self.tool_gateway = ToolTRMGateway(self.config)
        
        # Dexter's Context Channel - receives messages from Forge
        # When messages arrive, they trigger injection into Dexter's LLM context
        self.context_channel = ContextChannel(
            owner="dexter",
            on_trigger=self._on_context_channel_trigger,
        )

        # Context curator ensures staged artifacts remain curated and deduplicated
        from core.context_curator import ContextCurator
        self.context_curator = ContextCurator(self.staged_context, self.config)
        
        # Connect Forge Gateway to Dexter's channel for bidirectional communication
        self.tool_gateway.connect_to_dexter(self.context_channel)
        
        # Subscribe TRMs to communication channels (they learn from all traffic)
        self._subscribe_trms_to_channels()

        # Stage ResponseTank artifacts (memory/reasoning/think-tank/user/tool) for deterministic injection.
        self._artifact_task = None

        # Shared Mental State (The "Blackboard")
        self.state = {
            "intent": "None",
            "plan": {},
            "active_step": 0,
            "human_presence": True,
            "last_action_result": None,
            "last_tool_call": None,
            "last_tool_error": None,
        }
        self.state_lock = asyncio.Lock()
        self.running = True
        self.chat_history = []
        self.pending_goal = None
        self.user_last_input = 0.0
        self.user_tasks = set()
        self.user_msg_lock = asyncio.Lock()
        self._message_seq = 0
        self.user_focus_window = int(
            self.conversation_cfg.get("user_focus_window_sec", 30)
        )

        # Autonomy loop state (self-directed objectives)
        self.autonomy_cfg = self.config.get("autonomy_loop", {}) or {}
        self._autonomy_backlog = []
        self._autonomy_last_tick = 0.0
        self._autonomy_last_generation = 0.0
        # Runtime override: when set (True/False), it takes precedence over config flags.
        self._autonomy_runtime_enabled: bool | None = None
        self._autonomy_backlog_path = self._resolve_repo_path(
            self.autonomy_cfg.get("persist_path", "data/autonomy_backlog.json")
        )
        self._load_autonomy_backlog()

    def _strip_json_comments(self, text: str) -> str:
        """Remove // and /* */ comments outside of string literals."""
        out = []
        i = 0
        in_str = False
        escape = False
        length = len(text)
        while i < length:
            ch = text[i]
            if in_str:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
                i += 1
                continue
            if ch == "\"":
                in_str = True
                out.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < length:
                nxt = text[i + 1]
                if nxt == "/":
                    i += 2
                    while i < length and text[i] not in "\r\n":
                        i += 1
                    continue
                if nxt == "*":
                    i += 2
                    while i + 1 < length and not (text[i] == "*" and text[i + 1] == "/"):
                        i += 1
                    i += 2 if i + 1 < length else 1
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _strip_trailing_commas(self, text: str) -> str:
        """Remove trailing commas before } or ] outside of string literals."""
        out = []
        i = 0
        in_str = False
        escape = False
        length = len(text)
        while i < length:
            ch = text[i]
            if in_str:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
                i += 1
                continue
            if ch == "\"":
                in_str = True
                out.append(ch)
                i += 1
                continue
            if ch == ",":
                j = i + 1
                while j < length and text[j] in " \t\r\n":
                    j += 1
                if j < length and text[j] in "}]":
                    i += 1
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _clean_json_text(self, text: str) -> str:
        text = text.lstrip("\ufeff")
        text = self._strip_json_comments(text)
        text = self._strip_trailing_commas(text)
        return text

    def _load_json(self, path: Path):
        if not path.exists():
            return {}
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[Config] Read failed ({path}): {exc}", flush=True)
            return {}
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            cleaned = self._clean_json_text(raw)
            if cleaned != raw:
                try:
                    parsed = json.loads(cleaned)
                    print(f"[Config] Parsed {path} with lenient cleanup ({exc})", flush=True)
                    return parsed
                except json.JSONDecodeError as exc2:
                    print(f"[Config] JSON parse failed ({path}): {exc2}", flush=True)
                    return {}
            print(f"[Config] JSON parse failed ({path}): {exc}", flush=True)
            return {}

    def _resolve_repo_path(self, maybe_relative: str) -> str:
        if not maybe_relative:
            return str(self.repo_root / "data" / "autonomy_backlog.json")
        p = Path(maybe_relative)
        if p.is_absolute():
            return str(p)
        return str(self.repo_root / p)

    def _load_autonomy_backlog(self) -> None:
        """Best-effort load of persisted backlog."""
        try:
            path = Path(self._autonomy_backlog_path)
            if not path.exists():
                self._autonomy_backlog = []
                return
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                self._autonomy_backlog = []
                return
            data = json.loads(raw)
            if isinstance(data, list):
                self._autonomy_backlog = [str(x) for x in data if str(x).strip()]
            else:
                self._autonomy_backlog = []
        except Exception:
            self._autonomy_backlog = []

    def _save_autonomy_backlog(self) -> None:
        try:
            path = Path(self._autonomy_backlog_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._autonomy_backlog, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _format_config_summary(self) -> str:
        llm_slots = self.config.get("llm_slots", {}) or {}
        slot_summary = {}
        for name, cfg in llm_slots.items():
            if not isinstance(cfg, dict):
                continue
            slot_summary[name] = {
                "provider": cfg.get("provider_name"),
                "model": cfg.get("model"),
                "fallbacks": cfg.get("fallback_slots") or [],
            }
        summary = {
            "providers": list((self.config.get("providers", {}) or {}).keys()),
            "llm_slots": slot_summary,
            "trained_trms": self.config.get("trained_trms", {}) or {},
            "tool_gateway": self.config.get("tool_trm_gateway", {}) or {},
            "conversation": {
                "chat_slot": self.conversation_cfg.get("chat_slot"),
                "auto_plan": self.conversation_cfg.get("auto_plan"),
                "require_explicit_plan": self.conversation_cfg.get("require_explicit_plan"),
                "immediate_ack": self.conversation_cfg.get("immediate_ack"),
            },
        }
        return json.dumps(summary, ensure_ascii=False, default=str)

    def _build_runtime_status(self) -> Dict[str, Any]:
        return {
            "trms_loaded": list(self.trained_trms.keys()) if isinstance(self.trained_trms, dict) else [],
            "memory_trm_enabled": self.memory_trm_enabled,
            "tool_gateway_ready": True,
            "think_tank_slots": self.llm_think_tank.get_enabled_slots(),
            "conversation": {
                "chat_slot": self.conversation_cfg.get("chat_slot"),
                "auto_plan": self.conversation_cfg.get("auto_plan"),
                "require_explicit_plan": self.conversation_cfg.get("require_explicit_plan"),
            },
            "config_summary": self._format_config_summary(),
        }

    def _get_config_value(self, path: str) -> Any:
        if not path:
            return None
        parts = [p for p in path.strip().split(".") if p]
        node: Any = self.config
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node.get(part)
        return node

    def _set_config_value(self, path: str, value: Any) -> bool:
        if not path:
            return False
        parts = [p for p in path.strip().split(".") if p]
        if not parts:
            return False
        node: Any = self.config
        for part in parts[:-1]:
            if part not in node or not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
        return True

    def _parse_config_value(self, raw: str) -> Any:
        if raw is None:
            return None
        text = raw.strip()
        if not text:
            return ""
        # Try JSON first for numbers, booleans, arrays, objects.
        try:
            return json.loads(text)
        except Exception:
            return text

    def _build_carry_state(
        self,
        user_msg: str,
        user_response: str,
        internal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        carry = {
            "timestamp": time.time(),
            "user_input": _compact_text(user_msg, max_words=60, max_chars=400),
            "assistant_response": _compact_text(user_response, max_words=80, max_chars=500),
        }
        if isinstance(internal, dict) and internal:
            carry["internal"] = {
                "tasks": internal.get("tasks") or [],
                "questions_for_think_tank": internal.get("questions_for_think_tank") or [],
                "memory_updates": internal.get("memory_updates") or [],
                "reasoning_requests": internal.get("reasoning_requests") or [],
            }
        return carry

    def _persist_config(self) -> bool:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:
            print(f"[Config] Save failed: {exc}", flush=True)
            return False

    def _refresh_runtime_config(self) -> None:
        """Refresh in-memory views that read from config. Some changes still require restart."""
        self.conversation_cfg = self._load_conversation_cfg(self.config)
        self.bootstrap_cfg = self.config.get("bootstrap", {}) or {}
        mem_cfg = self.config.get("memory_trm", {}) or {}
        self.memory_trm_enabled = bool(mem_cfg.get("enabled", True))

    def _provider_model_pairs(self) -> List[tuple[str, str]]:
        pairs: List[tuple[str, str]] = []
        providers = self.config.get("providers", {}) or {}
        if not isinstance(providers, dict):
            return pairs
        for pname, pcfg in providers.items():
            if not isinstance(pcfg, dict):
                continue
            models: List[str] = []
            m0 = pcfg.get("model")
            if m0:
                models.append(str(m0))
            extra = pcfg.get("models") or []
            if isinstance(extra, list):
                for m in extra:
                    if m is None:
                        continue
                    ms = str(m)
                    if ms and ms not in models:
                        models.append(ms)
            for m in models:
                pairs.append((str(pname), str(m)))
        return pairs

    def _set_llm_slot(self, slot_name: str, provider_name: str, model: str) -> None:
        slot = (slot_name or "").strip() or "orchestrator"
        pn = (provider_name or "").strip()
        mm = (model or "").strip()
        if not pn or not mm:
            return
        self.config.setdefault("llm_slots", {})
        if not isinstance(self.config["llm_slots"], dict):
            self.config["llm_slots"] = {}
        slot_cfg = self.config["llm_slots"].get(slot, {}) or {}
        if not isinstance(slot_cfg, dict):
            slot_cfg = {}
        slot_cfg["provider_name"] = pn
        slot_cfg["model"] = mm
        self.config["llm_slots"][slot] = slot_cfg

    def _ui_pick_orchestrator_model(self) -> None:
        pairs = self._provider_model_pairs()
        if not pairs:
            print("[ModelPicker] No providers/models found in config['providers'].", flush=True)
            return
        chat_slot = str(self.conversation_cfg.get("chat_slot", "orchestrator") or "orchestrator")
        slot_cfg = (self.config.get("llm_slots", {}) or {}).get(chat_slot, {}) or {}
        cur_p = str(slot_cfg.get("provider_name") or "")
        cur_m = str(slot_cfg.get("model") or "")

        labels: List[str] = []
        start_idx = 0
        for i, (p, m) in enumerate(pairs):
            labels.append(f"{p}: {m}")
            if cur_p and cur_m and p == cur_p and m == cur_m:
                start_idx = i

        sel = _interactive_select(f"Select Orchestrator Model (slot={chat_slot})", labels, start_index=start_idx)
        if sel is None:
            return
        provider_name, model = pairs[int(sel)]
        self._set_llm_slot(chat_slot, provider_name, model)
        self._persist_config()
        self._refresh_runtime_config()
        print(f"[ModelPicker] Orchestrator slot '{chat_slot}' -> {provider_name}/{model}", flush=True)

    def _think_tank_slots_from_config(self) -> List[Dict[str, Any]]:
        cfg = self.config.get("llm_think_tank", {}) or {}
        slots = cfg.get("slots") or []
        if isinstance(slots, list):
            out = []
            for s in slots:
                if isinstance(s, dict):
                    out.append(s)
            return out
        return []

    def _ui_pick_think_tank_model(self) -> Optional[tuple[str, str]]:
        slots = self._think_tank_slots_from_config()
        if not slots:
            print("[ModelPicker] No think tank slots found in config['llm_think_tank']['slots'].", flush=True)
            return None

        slot_labels: List[str] = []
        for s in slots:
            name = str(s.get("name") or "unnamed")
            enabled = bool(s.get("enabled", False))
            provider = str(s.get("provider_name") or "")
            model = str(s.get("model") or "")
            tag = "ENABLED" if enabled else "disabled"
            if provider:
                slot_labels.append(f"{name} [{tag}] ({provider}/{model})")
            else:
                slot_labels.append(f"{name} [{tag}] (model={model})")

        sel_slot = _interactive_select("Select Think Tank Slot", slot_labels, start_index=0)
        if sel_slot is None:
            return None
        chosen = slots[int(sel_slot)]
        name = str(chosen.get("name") or "unnamed")
        provider = str(chosen.get("provider_name") or "")
        cur_model = str(chosen.get("model") or "")

        # If the slot uses a provider_name, pick from that provider's model list (mirrors config['providers']).
        # Otherwise (raw URL mode), only allow selecting the configured model string.
        if not provider:
            one = [cur_model or "(empty model)"]
            _ = _interactive_select(f"Select Model For Think Tank Slot '{name}'", one, start_index=0)
            return (name, cur_model)

        pairs = [pm for pm in self._provider_model_pairs() if pm[0] == provider]
        if not pairs:
            one = [cur_model or "(empty model)"]
            _ = _interactive_select(f"Select Model For Think Tank Slot '{name}' ({provider})", one, start_index=0)
            return (name, cur_model)

        labels = [f"{p}: {m}" for (p, m) in pairs]
        start_idx = 0
        for i, (_, m) in enumerate(pairs):
            if cur_model and m == cur_model:
                start_idx = i
                break
        sel_model = _interactive_select(f"Select Model For Think Tank Slot '{name}' ({provider})", labels, start_index=start_idx)
        if sel_model is None:
            return None
        _, model = pairs[int(sel_model)]
        return (name, model)

    def _apply_think_tank_model(self, slot_name: str, model: str) -> bool:
        slots = self._think_tank_slots_from_config()
        changed = False
        for s in slots:
            if str(s.get("name") or "") == str(slot_name or ""):
                s["model"] = str(model or "")
                changed = True
                break
        if not changed:
            return False
        # Write back into config (preserving order and other fields).
        self.config.setdefault("llm_think_tank", {})
        if not isinstance(self.config["llm_think_tank"], dict):
            self.config["llm_think_tank"] = {}
        self.config["llm_think_tank"]["slots"] = slots
        self._persist_config()
        return True

    async def _restart_think_tank(self) -> None:
        try:
            await self.llm_think_tank.stop()
        except Exception:
            pass
        try:
            self.llm_think_tank = LLMThinkTank(self.config)
            await self.llm_think_tank.start()
        except Exception:
            pass

    async def _append_working_memory(self, payload: Dict[str, Any]) -> None:
        if not self.working_memory_enabled:
            return
        path = Path(self.working_memory_path)
        line = json.dumps(payload, ensure_ascii=False, default=str)

        def _write():
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._working_memory_lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:
            print(f"[WorkingMemory] Write failed: {exc}", flush=True)

    def _autonomy_status(self) -> Dict[str, Any]:
        cfg = self.autonomy_cfg or {}
        enabled = self._autonomy_runtime_enabled
        if enabled is None:
            enabled = bool(cfg.get("enabled", False)) or bool((self.config.get("self_evolution", {}) or {}).get("autonomy_enabled", False))
        return {
            "enabled": bool(enabled),
            "backlog_len": len(self._autonomy_backlog),
            "backlog_preview": self._autonomy_backlog[:3],
            "idle_seconds": float(cfg.get("idle_seconds", 20)),
            "tick_seconds": float(cfg.get("tick_seconds", 15)),
            "generate_when_empty": bool(cfg.get("generate_when_empty", True)),
            "slot": (cfg.get("generation", {}) or {}).get("slot", "orchestrator"),
        }

    def _apply_max_training(self) -> None:
        cfg = self.config.get("max_training", {}) or {}
        env_flag = os.getenv("DEXTER_MAX_TRAINING", "").strip().lower()
        enabled = bool(cfg.get("enabled")) or env_flag in ("1", "true", "yes", "on")
        if not enabled:
            return
        self.config.setdefault("training_log", {})["enabled"] = True
        self.config.setdefault("experience_log", {})["enabled"] = True
        self.config.setdefault("plan_log", {})["enabled"] = True
        trm_cfg = self.config.setdefault("trm_tool_policy", {})
        if cfg.get("force_tool_trm", True):
            trm_cfg["enabled"] = True
            trm_cfg.setdefault("use_for_execution", True)

    def _run_bootstrap(self) -> None:
        cfg = self.bootstrap_cfg or {}
        if not cfg.get("enabled", False):
            return
        print("[Bootstrap] Running TRM tool pipeline...", flush=True)
        run_trm_tool_pipeline(
            repo_root=self.repo_root,
            include_success=bool(cfg.get("include_success", True)),
            train_model=bool(cfg.get("train_model", True)),
            seq_len=int(cfg.get("seq_len", 128)),
            augmentations=int(cfg.get("augmentations", 5)),
            epochs=int(cfg.get("epochs", 50)),
            batch_size=int(cfg.get("batch_size", 16)),
            lr=float(cfg.get("lr", 1e-4)),
        )

    def _load_trm(self, filename: str):
        path = self.models_dir / filename
        if path.exists():
            print(f"[Brain] {filename} online.")
            return True 
        return False

    def _init_trained_trms(self):
        """Load GPU-trained TRMs from TinyRecursiveModels."""
        trained_cfg = self.config.get("trained_trms", {}) or {}
        self.use_trained_tool_trm = bool(trained_cfg.get("tool_enabled", True))
        self.use_trained_memory_trm = bool(trained_cfg.get("memory_enabled", True))
        self.trained_trm_confidence_threshold = float(trained_cfg.get("confidence_threshold", 0.7))
        
        self.trained_trms = {}
        try:
            self.trained_trms = load_all_trms(device="cpu")
            if self.trained_trms:
                print(f"[Brain] Trained TRMs loaded: {list(self.trained_trms.keys())}", flush=True)
        except Exception as e:
            print(f"[Brain] Trained TRM loading failed: {e}", flush=True)
        
        # References to specific TRMs
        self.trained_tool_trm = self.trained_trms.get("tool")
        self.trained_memory_trm = self.trained_trms.get("memory")
        self.trained_reasoning_trm = self.trained_trms.get("reasoning")
    
    def _subscribe_trms_to_channels(self):
        """
        Subscribe TRMs to communication channels.
        
        TRMs learn from ALL traffic flowing through the system:
        - Memory TRM: Stores everything, builds episodic memory
        - Reasoning TRM: Observes patterns, learns strategies
        - Tool TRM: Already learning via gateway, but also sees Dexter's perspective
        
        Channel messages also get staged for LLM context injection.
        """
        # Create staging wrapper that both learns AND stages
        async def stage_and_learn_memory(sender, message, msg_type, metadata):
            # Stage for LLM context
            await self.staged_context.stage(
                source="channel_messages",
                content=f"[{sender}|{msg_type}]: {message[:300]}",
                priority=7 if msg_type == "result" else 5,
                metadata=metadata,
            )
            # Also feed to Memory TRM
            if self.trained_memory_trm and self.trained_memory_trm.is_ready():
                self.trained_memory_trm.ingest_message(sender, message, msg_type, metadata)
            # Also record for online memory TRM training
            trainer = get_online_trainer("memory")
            if trainer:
                prompt = f"<ENCODE> [{sender}|{msg_type}] {message}"
                if metadata:
                    try:
                        prompt += f" <META> {json.dumps(metadata)}"
                    except Exception:
                        pass
                target = _memory_target(sender, msg_type, message, metadata or {})
                success = metadata.get("success", metadata.get("ok", True)) if metadata else True
                trainer.record_text_example(prompt, target, success=bool(success), context=metadata or {})
        
        async def stage_and_learn_reasoning(sender, message, msg_type, metadata):
            # Stage for LLM context (reasoning insights)
            if msg_type in ("result", "response"):
                await self.staged_context.stage(
                    source="reasoning_trm",
                    content=f"[Observed]: {message}",
                    priority=6,
                    metadata=metadata,
                )
            # Also feed to Reasoning TRM
            if self.trained_reasoning_trm and self.trained_reasoning_trm.is_ready():
                self.trained_reasoning_trm.ingest_message(sender, message, msg_type, metadata)
            # Also record for online reasoning TRM training
            trainer = get_online_trainer("reasoning")
            if trainer:
                prompt = f"<OBSERVE> [{sender}|{msg_type}] {message[:500]}"
                if msg_type == "result" and metadata:
                    success = metadata.get("success", metadata.get("ok", False))
                    prompt += f" <OUTCOME:{'SUCCESS' if success else 'FAIL'}>"
                target = _reasoning_target(sender, msg_type, message, metadata or {})
                success = metadata.get("success", metadata.get("ok", True)) if metadata else True
                trainer.record_text_example(prompt, target, success=bool(success), context=metadata or {})
        
        # Subscribe wrappers to Dexter's channel (sees Forge responses)
        self.context_channel.subscribe(stage_and_learn_memory)
        print("[Brain] Memory TRM + Staged subscribed to Dexter channel", flush=True)
        
        self.context_channel.subscribe(stage_and_learn_reasoning)
        print("[Brain] Reasoning TRM + Staged subscribed to Dexter channel", flush=True)
        
        # Also subscribe to Forge's inbound channel (sees Dexter's requests)
        self.tool_gateway.inbound_channel.subscribe(stage_and_learn_memory)
        print("[Brain] Memory TRM + Staged subscribed to Forge channel", flush=True)
        
        self.tool_gateway.inbound_channel.subscribe(stage_and_learn_reasoning)
        print("[Brain] Reasoning TRM + Staged subscribed to Forge channel", flush=True)

    def _load_conversation_cfg(self, config: dict) -> dict:
        cfg = dict(config.get("conversation", {}) or {})
        cfg.setdefault("require_explicit_plan", False)
        cfg.setdefault("task_prefixes", ["task:", "plan:", "do:", "run:", "execute:", "build:", "make:", "run ", "execute "])
        cfg.setdefault("auto_plan", True)
        cfg.setdefault("action_keywords", ["build", "make", "create", "fix", "analyze", "review", "summarize", "write", "install", "set up", "update", "configure", "run", "test", "refactor", "optimize", "deploy", "generate", "code"])
        cfg.setdefault("action_on_questions", False)
        cfg.setdefault("question_action_mode", "smart")
        # Never ask for confirmation (forced by user request).
        cfg["question_requires_confirmation"] = False
        # When auto_plan triggers for a natural-language action request, skip confirmation entirely.
        # Values: "never" | "always" | "destructive_only"
        cfg["action_confirmation_policy"] = "never"
        cfg.setdefault(
            "destructive_keywords",
            [
                "delete", "remove", "erase", "wipe", "destroy",
                "format", "truncate", "drop", "reset",
                "kill", "terminate", "taskkill", "stop process",
                "uninstall",
                "overwrite", "replace",
                "rm ", "del ", "rmdir", "rd ",
            ],
        )
        cfg.setdefault("question_default_skill", "research_ops")
        cfg.setdefault("question_force_action_prefixes", ["lookup:", "search:", "find:"])
        cfg.setdefault("question_force_chat_prefixes", ["chat:", "opinion:", "brainstorm:"])
        cfg.setdefault("opinion_keywords", [
            "opinion", "your take", "your thoughts", "you think", "do you think",
            "do you believe", "in your view", "do you feel", "your perspective",
            "why do you think", "philosophy", "debate", "argue", "op-ed"
        ])
        cfg.setdefault("confirm_keywords", ["yes", "yep", "yeah", "ok", "okay", "do it", "go ahead", "please do", "start", "proceed", "begin"])
        cfg.setdefault("decline_keywords", ["no", "nope", "not now", "don't", "stop", "cancel", "never mind"])
        cfg.setdefault("chat_slot", "orchestrator")
        cfg.setdefault("announce_planning", False)
        cfg.setdefault("speak_system_messages", False)
        cfg.setdefault("speak_chat", False)
        cfg.setdefault("max_history", 12)
        cfg.setdefault("immediate_ack", True)
        cfg.setdefault("user_focus_window_sec", 30)
        return cfg

    async def _seed_rolling_context(self) -> None:
        try:
            await self.rolling_context.set_working_directory(os.getcwd())
        except Exception:
            pass
        try:
            registry = tool_executor._build_registry()
            for tool_name in registry:
                await self.rolling_context.add_capability(tool_name, "tool")
        except Exception as exc:
            self._print_internal("System", f"Rolling context tool seed failed: {exc}")
        try:
            skill_registry = self.librarian.skill_registry or {}
            for skill_id in skill_registry.keys():
                await self.rolling_context.add_capability(skill_id, "skill")
        except Exception as exc:
            self._print_internal("System", f"Rolling context skill seed failed: {exc}")
        try:
            await self.rolling_context.set_enabled_llms(self.llm_think_tank.get_enabled_slots())
        except Exception:
            pass

    async def _stage_startup_status(self) -> None:
        """Stage startup diagnostics into the orchestrator staged bundle."""
        try:
            payload = {
                "trms_loaded": list(self.trained_trms.keys()),
                "memory_trm_enabled": self.memory_trm_enabled,
                "think_tank_slots": self.llm_think_tank.get_enabled_slots(),
                "tool_gateway_ready": True,
                "config_summary": self._format_config_summary(),
            }
            await self.staged_context.stage_artifact(
                source="system",
                artifact_type="startup_status",
                payload=payload,
                confidence=0.9,
                priority=8,
                metadata={"event": "startup"},
            )
        except Exception:
            pass

    async def _start_trm_stack(self) -> None:
        # Start bucket manager + DB writer before other pipelines emit memory.
        try:
            await self.bucket_manager.start()
        except Exception as exc:
            print(f"[Buckets] Startup failed: {exc}", flush=True)
        try:
            await self.memory_db_writer.start()
        except Exception as exc:
            print(f"[MemoryDBWriter] Startup failed: {exc}", flush=True)

        await self.response_tank.start()
        await self.llm_think_tank.start()
        if self.memory_trm_enabled:
            try:
                await self.memory_trm.start()
            except Exception as exc:
                print(f"[Memory TRM] Startup failed: {exc}", flush=True)
        try:
            await self.memory_ingestor.start()
        except Exception as exc:
            print(f"[MemoryIngestor] Startup failed: {exc}", flush=True)
        await self._seed_rolling_context()

    def _should_plan(self, user_msg: str) -> bool:
        if not user_msg:
            return False
        if not self.conversation_cfg.get("require_explicit_plan", True):
            return False
        lowered = user_msg.strip().lower()
        for prefix in self.conversation_cfg.get("task_prefixes", []):
            if lowered.startswith(prefix):
                return True
        return False

    def _strip_task_prefix(self, user_msg: str) -> str:
        lowered = user_msg.strip()
        lower_lowered = lowered.lower()
        for prefix in self.conversation_cfg.get("task_prefixes", []):
            if lower_lowered.startswith(prefix):
                # Preserve natural-language command prefixes like "run " so the gateway heuristics can extract the command.
                if prefix.endswith(":"):
                    return lowered[len(prefix):].strip()
                return lowered
        return lowered

    def _has_task_prefix(self, user_msg: str) -> bool:
        lowered = user_msg.strip().lower()
        for prefix in self.conversation_cfg.get("task_prefixes", []):
            if lowered.startswith(prefix):
                return True
        return False

    def _remember_chat(self, role: str, content: str):
        if not content:
            return
        self.chat_history.append({"role": role, "content": content})
        max_history = int(self.conversation_cfg.get("max_history", 12))
        if max_history > 0 and len(self.chat_history) > max_history:
            self.chat_history = self.chat_history[-max_history:]

    def _is_confirmation(self, user_msg: str) -> bool:
        lowered = user_msg.strip().lower()
        for kw in self.conversation_cfg.get("confirm_keywords", []):
            if lowered == kw or kw in lowered:
                return True
        return False

    def _is_decline(self, user_msg: str) -> bool:
        lowered = user_msg.strip().lower()
        for kw in self.conversation_cfg.get("decline_keywords", []):
            if lowered == kw or kw in lowered:
                return True
        return False

    def _looks_like_action_request(self, user_msg: str) -> bool:
        lowered = user_msg.strip().lower()
        if lowered.startswith(("can you", "could you", "please", "i need", "i want")):
            return True
        for kw in self.conversation_cfg.get("action_keywords", []):
            if kw in lowered:
                return True
        if self.conversation_cfg.get("action_on_questions", False) and self._should_act_on_question(lowered):
            return True
        return False

    def _action_needs_confirmation(self, user_msg: str) -> bool:
        """Confirmation is disabled globally."""
        return False

    def _is_question(self, user_msg: str) -> bool:
        if not user_msg:
            return False
        text = user_msg.strip().lower()
        if text.endswith("?"):
            return True
        question_starts = (
            "what", "what's", "whats", "why", "how", "when", "where", "who",
            "which", "should", "can", "could", "is", "are", "do", "does", "did"
        )
        return text.startswith(question_starts)

    def _starts_with_any(self, text: str, prefixes: list) -> bool:
        if not text or not prefixes:
            return False
        for prefix in prefixes:
            if text.startswith(prefix):
                return True
        return False

    def _is_opinion_question(self, user_msg: str) -> bool:
        text = user_msg.strip().lower()
        for kw in self.conversation_cfg.get("opinion_keywords", []):
            if kw in text:
                return True
        return False

    def _is_action_question(self, user_msg: str) -> bool:
        text = user_msg.strip().lower()
        action_terms = (
            "time", "date", "day", "today", "now",
            "weather", "forecast", "temperature", "humidity", "rain", "snow",
            "latest", "current", "news", "price", "stock", "market", "score", "schedule",
            "how to", "best way", "recipe", "steps", "instructions"
        )
        return any(term in text for term in action_terms)

    def _should_act_on_question(self, user_msg: str) -> bool:
        text = user_msg.strip().lower()
        if not self._is_question(text):
            return False
        if self._starts_with_any(text, self.conversation_cfg.get("question_force_chat_prefixes", [])):
            return False
        if self._starts_with_any(text, self.conversation_cfg.get("question_force_action_prefixes", [])):
            return True

        mode = (self.conversation_cfg.get("question_action_mode") or "smart").lower()
        if mode == "off":
            return False
        if mode == "always":
            return True
        if self._is_opinion_question(text):
            return False
        return self._is_action_question(text)

    def _route_question_skill(self, user_msg: str) -> str | None:
        text = user_msg.strip().lower()
        registry = self.librarian.skill_registry or {}

        def has(*terms: str) -> bool:
            return any(term in text for term in terms)

        if has("cwd", "current working directory", "working directory", "pwd"):
            return "powershell" if "powershell" in registry else None
        if has("time", "date", "clock", "timestamp"):
            return "powershell" if "powershell" in registry else None
        if has("weather", "forecast", "temperature", "rain", "snow", "humidity"):
            return "research_ops" if "research_ops" in registry else None
        if has("best", "how to", "recipe", "cook", "steak", "recommend", "should i", "tips", "guide"):
            return "research_ops" if "research_ops" in registry else None
        if has("latest", "current", "news", "price", "stock", "market"):
            return "research_ops" if "research_ops" in registry else None

        default_skill = self.conversation_cfg.get("question_default_skill") or ""
        if default_skill and default_skill in registry:
            return default_skill
        return None

    def _build_question_plan(self, user_msg: str) -> dict:
        skill_id = self._route_question_skill(user_msg)
        if skill_id:
            task = f"use {skill_id} to answer: {user_msg}"
        else:
            task = f"answer question: {user_msg}"
        return {
            "goal": f"Answer question: {user_msg}",
            "steps": [{"id": 1, "task": task, "status": "pending"}],
        }

    def _is_busy(self) -> bool:
        intent = self.state.get("intent")
        plan = self.state.get("plan")
        if bool(plan) or (intent not in (None, "", "None")):
            return True
        return any(task and not task.done() for task in self.user_tasks)

    def _track_user_task(self, task: asyncio.Task) -> None:
        self.user_tasks.add(task)
        def _cleanup(t: asyncio.Task) -> None:
            self.user_tasks.discard(t)
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                return
            if exc:
                print(f"[Cognitive Fault] User task failed: {exc}", flush=True)
        task.add_done_callback(_cleanup)

    def _next_message_id(self) -> int:
        self._message_seq += 1
        return self._message_seq

    def _maybe_ack_user(self, msg_id: int | None = None) -> None:
        if not self.conversation_cfg.get("immediate_ack", True):
            return
        suffix = f" #{msg_id}" if msg_id is not None else ""
        self._print_internal("System", f"(received{suffix})")

    def _print_user(self, message: str) -> None:
        if message is None:
            return
        print(f"[Dexter][Jeffrey] {message}", flush=True)

    def _print_internal(self, channel: str, message: str) -> None:
        if message is None:
            return
        channel = channel.strip() if channel else "System"
        print(f"[Dexter][{channel}] {message}", flush=True)

    def _summarize_artifacts(self, artifacts: List[Dict[str, Any]], limit: int = 3) -> List[str]:
        lines: List[str] = []
        for art in artifacts:
            payload = art.get("payload")
            if isinstance(payload, dict) and "insights" in payload:
                entries = payload.get("insights") or []
            elif isinstance(payload, list):
                entries = payload
            else:
                entries = [payload]
            for entry in entries:
                if len(lines) >= limit:
                    break
                if isinstance(entry, dict):
                    content = entry.get("content") or entry.get("summary") or str(entry)
                    typ = entry.get("type") or "note"
                    conf = entry.get("confidence")
                    conf_str = f"{float(conf):.2f}" if conf is not None else "?"
                    lines.append(f"{typ} (c={conf_str}): {str(content).strip()}")
                else:
                    lines.append(str(entry).strip())
            if len(lines) >= limit:
                break
        return lines

    def _format_internal_panels(
        self,
        internal: Dict[str, Any],
        injection_payload: Optional[Dict[str, Any]],
    ) -> str:
        sources = (injection_payload or {}).get("sources") or {}
        think_items = list((sources.get("think_tank") or {}).get("artifacts") or [])
        reasoning_items = list((sources.get("reasoning_trm") or {}).get("artifacts") or [])

        think_lines = self._summarize_artifacts(think_items, limit=4)
        reasoning_lines = self._summarize_artifacts(reasoning_items, limit=3)

        tasks = internal.get("tasks") or []
        forge_lines = []
        if tasks:
            forge_lines = [str(t).strip() for t in tasks if str(t).strip()]

        parts: List[str] = []
        if think_lines or reasoning_lines:
            parts.append("[Dexter][ThinkTank+Reasoning]")
            if think_lines:
                parts.append("ThinkTank:")
                parts.extend([f"- {line}" for line in think_lines])
            if reasoning_lines:
                parts.append("Reasoning TRM:")
                parts.extend([f"- {line}" for line in reasoning_lines])

        parts.append("[Dexter][ForgeIntent]")
        if forge_lines:
            parts.extend([f"- {line}" for line in forge_lines])
        else:
            parts.append("- none")

        return "\n".join(parts)

    def _ensure_brain_db(self) -> None:
        """Ensure the brain database exists and schema is installed."""
        try:
            db_path = memory_db_path()
            if not os.path.isabs(db_path):
                db_path = str(self.repo_root / db_path)
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            ensure_brain_schema(db_path)
        except Exception as exc:
            print(f"[Brain] Database init failed: {exc}", flush=True)

    async def _answer_quick_question(self, user_msg: str) -> str | None:
        return None

    def _speak_system(self, text: str):
        if self.conversation_cfg.get("speak_system_messages", False):
            speak_out_loud(text)

    def _auto_tune_trm_thresholds(self, accuracy: float) -> dict | None:
        cfg = self.trm_tune_cfg
        if not cfg.get("enabled"):
            return None
        if not self.trm_accuracy.ready():
            return None
        self.trm_tune_counter += 1
        if cfg["every_n"] > 1 and (self.trm_tune_counter % cfg["every_n"] != 0):
            return None

        exec_th = float(self.trm_tool_policy.execute_threshold)
        shadow_th = float(self.trm_tool_policy.shadow_threshold)
        target = cfg["target"]
        hysteresis = cfg["hysteresis"]
        step = cfg["step"]

        tuned = False
        if accuracy >= (target + hysteresis):
            exec_th = max(cfg["min_execute"], exec_th - step)
            shadow_th = max(cfg["min_shadow"], shadow_th - (step / 2))
            tuned = True
        elif accuracy <= (target - hysteresis):
            exec_th = min(cfg["max_execute"], exec_th + step)
            shadow_th = min(cfg["max_shadow"], shadow_th + (step / 2))
            tuned = True

        # Ensure ordering
        if shadow_th >= exec_th:
            shadow_th = max(cfg["min_shadow"], exec_th - 0.05)

        if tuned:
            self.trm_tool_policy.execute_threshold = exec_th
            self.trm_tool_policy.shadow_threshold = shadow_th

        return {
            "tuned": tuned,
            "execute_threshold": exec_th,
            "shadow_threshold": shadow_th,
        } if tuned else None

    def _tool_failed(self, result: dict) -> bool:
        if not isinstance(result, dict):
            return True
        if result.get("ok") is False or result.get("success") is False:
            return True
        payload = result.get("result")
        if payload is False:
            return True
        if isinstance(payload, dict) and payload.get("success") is False:
            return True
        return False

    def _tool_exists(self, tool_name: str) -> bool:
        if not tool_name:
            return False
        try:
            registry = tool_executor._build_registry()
            if tool_name in registry:
                return True
            if "." not in tool_name:
                return False
            module_name, func_name = tool_name.rsplit(".", 1)
            tools_dir = getattr(tool_executor, "TOOLS_DIR", str(self.repo_root / "skills"))
            file_path = os.path.join(tools_dir, f"{module_name}.py")
            if not os.path.exists(file_path):
                return False
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                return False
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, func_name, None) is not None
        except Exception:
            return False

    def _missing_required_args(self, tool_name: str, arguments: Dict[str, Any]) -> List[str]:
        if not tool_name:
            return []
        try:
            registry = tool_executor._build_registry()
            entry = registry.get(tool_name)
            if not entry:
                return []
            func = entry[2]
            template = build_tool_template(tool_name, func, provided=arguments or {})
            missing = []
            for key, value in (template.get("arguments") or {}).items():
                if isinstance(value, str) and value == f"<{key}>":
                    missing.append(key)
            return missing
        except Exception:
            return []

    def _extract_tool_error(self, result: dict) -> tuple[str, str]:
        if not isinstance(result, dict):
            return ("Tool failed", "")
        error_msg = result.get("error") or ""
        tb = result.get("traceback") or ""
        payload = result.get("result")
        if isinstance(payload, dict):
            if not error_msg:
                error_msg = payload.get("error") or ""
        return (error_msg or "Tool failed", tb or "")

    async def _build_context_payload(self, task: str) -> dict:
        snapshot = await self.context_bundler.build_snapshot_async(
            intent=self.state.get("intent") or "",
            task=task or "",
            chat_history=self.chat_history,
            plan=self.state.get("plan"),
        )
        recent = self.context_bundles[-self.context_bundler.max_bundles:] if self.context_bundles else []
        
        # Query trained Memory TRM for relevant context
        trained_memories = []
        if self.use_trained_memory_trm and self.trained_memory_trm and self.trained_memory_trm.is_ready():
            try:
                mem_result = self.trained_memory_trm.query(task, memory_type="all")
                if mem_result.get("confidence", 0) >= self.trained_trm_confidence_threshold:
                    trained_memories = mem_result.get("memories", [])
                    if trained_memories:
                        print(f"[Trained Memory TRM] Retrieved {len(trained_memories)} memories", flush=True)
            except Exception as e:
                pass  # Silently fail, don't block on memory
        
        return {
            "snapshot": snapshot,
            "recent_bundles": recent,
            "trained_memories": trained_memories,
        }

    def _parse_orchestrator_response(self, raw: str) -> tuple[str, Dict[str, Any]]:
        if not raw:
            return ("", {})
        parsed = extract_json(raw)
        if isinstance(parsed, dict) and "user_response" in parsed:
            user_resp = str(parsed.get("user_response") or "").strip()
            internal = parsed.get("internal") or {}
            return (user_resp, internal if isinstance(internal, dict) else {})
        return (raw.strip(), {})

    async def _dispatch_internal_payload(self, internal: Dict[str, Any], user_msg: str):
        if not internal:
            return
        try:
            await self.response_tank.publish(
                source="orchestrator_internal",
                content={"type": "internal_payload", "payload": internal, "timestamp": time.time()},
                priority="medium",
            )
        except Exception:
            pass

        tasks = internal.get("tasks") or []
        questions = internal.get("questions_for_think_tank") or []
        mem_updates = internal.get("memory_updates") or []
        reasoning_requests = internal.get("reasoning_requests") or []

        if tasks:
            try:
                await self.context_curator.stage_structured_artifact(
                    source="orchestrator",
                    artifact_type="tasks",
                    payload=tasks,
                    confidence=0.8,
                    priority=6,
                    metadata={"origin": "orchestrator"},
                )
            except Exception:
                pass
            try:
                self.memory_ingestor.enqueue("orchestrator_tasks", json.dumps(tasks, ensure_ascii=False, default=str))
            except Exception:
                pass

        if mem_updates:
            try:
                self.memory_ingestor.enqueue("orchestrator_memory", json.dumps(mem_updates, ensure_ascii=False, default=str))
            except Exception:
                pass

        if questions:
            asyncio.create_task(self._schedule_think_tank(user_msg, internal_questions=questions))

        if reasoning_requests:
            for req in reasoning_requests:
                asyncio.create_task(self._schedule_reasoning_trm(str(req), context={"origin": "orchestrator"}))

    async def _trigger_orchestrator_injection(self, trigger_type: str, trigger_content: str) -> str:
        injection = await self.staged_context.trigger_and_inject(
            trigger_type=trigger_type,
            trigger_content=trigger_content,
        )
        self._last_injection_payload = injection
        if injection.get("sources"):
            await self.orchestrator_bundle.merge_injection(injection)
            self._print_internal("System", f"Staged context triggered: {len(injection['sources'])} sources")
            try:
                self._print_internal("System", "Injection bundle:\n" + json.dumps(injection, ensure_ascii=False, default=str))
            except Exception:
                pass
        return self.orchestrator_bundle.format_for_llm()

    async def _schedule_retrieval(self, query: str, trigger: str):
        if not query:
            return
        try:
            results = await asyncio.to_thread(retrieve_context, query, self.config, 12)
        except Exception:
            return
        if not results:
            return
        try:
            await self.context_curator.stage_structured_artifact(
                source="memory_retrieval",
                artifact_type="retrieval",
                payload=results,
                confidence=0.9,
                priority=7,
                metadata={"trigger": trigger, "query": query},
            )
        except Exception:
            pass

    async def _schedule_think_tank(self, user_msg: str, internal_questions: Optional[List[str]] = None):
        if not self.llm_think_tank.is_available():
            return
        context = {
            "user_input": user_msg,
            "bundle_context": self.think_tank_bundle.format_for_llm(),
        }
        try:
            insights = await self.llm_think_tank.broadcast(
                context=context,
                specific_questions=internal_questions,
                timeout=8.0,
            )
        except Exception:
            return
        # Add insights back into think-tank persistent bundle
        for insight in insights:
            try:
                await self.think_tank_bundle.add_artifact(
                    source=f"llm_{insight.source}",
                    artifact_type="llm_insight",
                    payload=insight.insights,
                    confidence=float(insight.confidence or 0.6),
                    priority=6,
                    metadata={"questions": insight.questions_for_dexter},
                )
            except Exception:
                pass
            try:
                self.memory_ingestor.enqueue(
                    f"llm_{insight.source}",
                    json.dumps(
                        {
                            "insights": insight.insights,
                            "questions": insight.questions_for_dexter,
                            "memory_needs": insight.memory_needs,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )
            except Exception:
                pass

    async def _schedule_reasoning_trm(self, prompt: str, context: Optional[Dict[str, Any]] = None):
        if not self.trained_reasoning_trm or not self.trained_reasoning_trm.is_ready():
            return
        try:
            result = self.trained_reasoning_trm.reason(prompt, context=context or {})
        except Exception:
            return
        try:
            await self.context_curator.stage_structured_artifact(
                source="reasoning_trm",
                artifact_type="scenario_thoughts",
                payload=result,
                confidence=float(result.get("confidence") or 0.6),
                priority=6,
                metadata={"trigger": "user_input"},
            )
        except Exception:
            pass
        try:
            self.memory_ingestor.enqueue("reasoning_trm", json.dumps(result, ensure_ascii=False, default=str))
        except Exception:
            pass

    async def _handle_tool_result(self, meta: dict, result: dict) -> None:
        self.state["last_action_result"] = result
        self.state["last_tool_call"] = {
            "name": meta.get("tool_name"),
            "arguments": meta.get("arguments") or {},
            "skill_id": meta.get("skill_id"),
            "task": meta.get("task"),
        }
        if self._tool_failed(result):
            error_msg, _ = self._extract_tool_error(result)
            self.state["last_tool_error"] = error_msg
        else:
            self.state["last_tool_error"] = None
        bundle = await self.context_bundler.build_bundle_async(
            intent=self.state.get("intent") or "",
            task=meta.get("task") or "",
            chat_history=self.chat_history,
            tool_result=result,
            tool_meta=meta or {},
            plan=self.state.get("plan"),
        )
        self.context_bundles.append(bundle)
        if self.context_bundler.max_bundles > 0 and len(self.context_bundles) > self.context_bundler.max_bundles:
            self.context_bundles = self.context_bundles[-self.context_bundler.max_bundles:]
        self.reasoning.ingest_context_bundle(bundle)
        self.evolution.ingest_context_bundle(bundle)

    async def _on_tool_result(self, upstream_bundle: dict) -> None:
        """
        Callback from Tool TRM Gateway - receives rich execution results immediately.
        
        This is where Dexter perceives tool results with full metadata:
        - success/failure
        - retries, arg changes
        - skill forged
        - timing breakdown
        - selection method (TRM/LLM/heuristic)
        """
        # Broadcast to API/dashboard immediately
        broadcast_thought("tool_execution", upstream_bundle)
        
        # Update state with rich result
        self.state["last_gateway_result"] = upstream_bundle
        
        # Extract key info
        success = upstream_bundle.get("success", False)
        tool = upstream_bundle.get("tool", "unknown")
        intent = upstream_bundle.get("original_intent", "")
        exec_summary = upstream_bundle.get("execution_summary", {})
        
        # Log for memory formation
        self._print_internal(
            "System",
            f"Tool Result: {tool} -> {'OK' if success else 'FAIL'} "
            f"(method={exec_summary.get('method', '?')}, "
            f"forged={exec_summary.get('skill_forged', False)})",
        )

        # Full, untruncated payload for the Activity Stream terminal
        try:
            self._print_internal("System", "Full tool result bundle:\n" + json.dumps(upstream_bundle, ensure_ascii=False, default=str))
        except Exception:
            pass
        
        # If skill was forged, announce it
        if exec_summary.get("skill_forged"):
            skill_name = exec_summary.get("skill_forged_name", "new skill")
            self._print_internal("System", f"✨ New skill created: {skill_name}")
            broadcast_thought("skill_created", {"name": skill_name, "intent": intent})
        
        # Stage a compact structured artifact for downstream injection (full payload is still logged above)
        try:
            result_payload = upstream_bundle.get("result")
            preview = result_payload
            if isinstance(preview, (dict, list)):
                preview = json.dumps(preview, ensure_ascii=False, default=str)
            preview = str(preview)
            if len(preview) > 800:
                preview = preview[:800] + "…"

            await self.staged_context.stage_artifact(
                source="tool_results",
                artifact_type="tool_result",
                payload={
                    "tool": tool,
                    "success": success,
                    "error": upstream_bundle.get("error"),
                    "method": exec_summary.get("method"),
                    "retries": exec_summary.get("retries"),
                    "result_preview": preview,
                },
                confidence=1.0 if success else 0.6,
                priority=9 if not success else 7,
                metadata={"trigger": "tool_result"},
            )
        except Exception:
            pass

        # Compact summary used for retrieval/aux pipelines.
        result_summary = f"{tool} -> {'SUCCESS' if success else 'FAIL'}"

        # Enqueue for orchestrator reaction. The reactor will trigger injection + LLM call.
        try:
            self._tool_result_queue.put_nowait(upstream_bundle)
        except Exception:
            pass

        # Background memory + retrieval
        try:
            self.memory_ingestor.enqueue("tool_result", json.dumps(upstream_bundle, ensure_ascii=False, default=str))
        except Exception:
            pass
        try:
            self.bucket_manager.enqueue(
                bucket_name="tool_results",
                event_type="tool_result",
                payload=upstream_bundle,
                source="tool_results",
                metadata={"via": "tool_gateway"},
            )
        except Exception:
            pass
        asyncio.create_task(self._schedule_retrieval(result_summary, "tool_result"))
        
        # Feed to memory system for episodic storage
        try:
            memory_record = {
                "type": "tool_execution",
                "tool": tool,
                "intent": intent,
                "success": success,
                "summary": exec_summary,
                "timestamp": time.time(),
            }
            # Queue for memory consolidation
            if hasattr(self, 'rolling_context'):
                await self.rolling_context.add_event("tool_result", memory_record)
        except Exception:
            pass
    
    async def _on_context_channel_trigger(self, bundle: list):
        """
        Called when messages arrive on Dexter's context channel.
        
        Forge handles execution:
        - results: Tool execution completed
        - clarification: Needs more info from Dexter
        - status: Progress updates
        - error: Something went wrong
        
        These messages get injected into Dexter's LLM context.
        """
        for msg in bundle:
            sender = msg["sender"]
            content = msg["message"]
            msg_type = msg["type"]
            metadata = msg.get("metadata", {})
            
            # Always log the raw inbound message to the Activity Stream.
            self._print_internal("System", f"Received from {sender} ({msg_type}): {content}")

            # Append-only capture for training + audit (does not block runtime).
            try:
                self.bucket_manager.enqueue(
                    bucket_name="channel_messages",
                    event_type="channel_message",
                    payload={
                        "from": sender,
                        "type": msg_type,
                        "content": content,
                        "metadata": metadata,
                        "timestamp": time.time(),
                    },
                    source="channel_messages",
                    metadata={"via": "context_channel"},
                )
            except Exception:
                pass

            # Surface Forge messages to the conversation terminal (as Dexter).
            if sender == "forge":
                if msg_type in ("clarification", "error", "result") or (msg_type == "response" and "?" in str(content)):
                    self._print_user(str(content))
            
            # Broadcast for API/dashboard
            broadcast_thought("channel_message", {
                "from": sender,
                "type": msg_type,
                "content": content,
            })
            
            # Store in state for LLM context injection
            if "channel_messages" not in self.state:
                self.state["channel_messages"] = []
            self.state["channel_messages"].append({
                "from": sender,
                "type": msg_type,
                "content": content,
                "timestamp": time.time(),
            })
            # Keep only last 10 messages
            self.state["channel_messages"] = self.state["channel_messages"][-10:]
            
            # If it's a result, also update via the normal result handling
            if msg_type == "result" and "result" in metadata:
                await self._on_tool_result(metadata["result"])
    
    async def _artifact_worker(self):
        """Convert ResponseTank messages into structured staged artifacts (non-blocking)."""
        try:
            subscriber = await self.response_tank.subscribe(
                subscriber_id="staged_bundle_bridge",
                source_patterns=["*"],
            )
        except Exception as exc:
            print(f"[StagedBundle] Subscribe failed: {exc}", flush=True)
            return

        async with subscriber as sub:
            while self.running:
                msg = await sub.receive(timeout=0.5)
                if not msg:
                    continue

                source = getattr(msg, "source", "") or ""
                content = getattr(msg, "content", None) or {}
                ctype = content.get("type") if isinstance(content, dict) else None

                try:
                    # Append-only capture of ALL artifacts (user/tool/think-tank/reasoning/etc).
                    # This should never block the runtime; slow IO is handled by the bucket flush worker.
                    try:
                        self.bucket_manager.enqueue(
                            bucket_name=source or "unknown",
                            event_type=str(ctype or "artifact"),
                            payload=content,
                            source=source or "unknown",
                            metadata={"via": "response_tank"},
                        )
                    except Exception:
                        pass

                    if source == "memory_trm" and ctype == "deep_context":
                        ctx = content.get("context") or {}
                        payload = {
                            "facts": (ctx.get("facts") or [])[:6],
                            "triples": (ctx.get("triples") or [])[:6],
                            "patterns": (ctx.get("patterns") or [])[:3],
                            "current_task": ctx.get("current_task"),
                            "current_project": ctx.get("current_project"),
                        }
                        await self.context_curator.stage_structured_artifact(
                            source="memory_trm",
                            artifact_type="deep_context",
                            payload=payload,
                            confidence=0.85,
                            priority=6,
                            timestamp=ctx.get("timestamp"),
                            metadata={"from": "response_tank"},
                        )
                        try:
                            self.memory_ingestor.enqueue("memory_trm", json.dumps(ctx, ensure_ascii=False, default=str))
                        except Exception:
                            pass
                    elif source.startswith("llm_") and ctype == "llm_insight":
                        insight = content.get("insight") or {}
                        payload = {
                            "source": insight.get("source"),
                            "confidence": insight.get("confidence"),
                            "insights": (insight.get("insights") or [])[:6],
                            "questions_for_dexter": (insight.get("questions_for_dexter") or [])[:3],
                            "memory_needs": (insight.get("memory_needs") or [])[:3],
                        }
                        await self.context_curator.stage_structured_artifact(
                            source="think_tank",
                            artifact_type="llm_insight",
                            payload=payload,
                            confidence=float(insight.get("confidence") or 0.6),
                            priority=5,
                            timestamp=insight.get("timestamp"),
                            metadata={"from": source},
                        )
                        try:
                            self.memory_ingestor.enqueue(source, json.dumps(payload, ensure_ascii=False, default=str))
                        except Exception:
                            pass
                except Exception:
                    pass

    async def send_to_forge(self, message: str, msg_type: str = "request"):
        """
        Send a message to Forge via context channel.

        Forge is Dexter's execution brain: Dexter sends intent, Forge decides tool+args.
        """
        async def _fire() -> None:
            try:
                await self.tool_gateway.inbound_channel.send(
                    sender="dexter",
                    message=message,
                    msg_type=msg_type,
                )
                # Trigger Forge processing without blocking caller.
                await self.tool_gateway.inbound_channel.trigger_injection()
            except Exception as exc:
                print(f"[Forge] Send failed: {exc}", flush=True)

        asyncio.create_task(_fire())
        # Pump Dexter-side channel handling without blocking caller.
        asyncio.create_task(self._pump_context_channel())

    async def _pump_context_channel(self, timeout: float = 0.05) -> None:
        """Best-effort, non-blocking pump of Forge replies into Dexter's context channel."""
        try:
            await self.context_channel.wait_for_message(timeout=timeout)
            await self.context_channel.trigger_injection()
        except Exception:
            return

    async def execute_via_gateway(self, intent: str, tool: str = None, args: dict = None) -> dict:
        """
        Direct gateway access for LLM or conversation layer.
        Accepts natural language OR structured tool calls.
        
        Args:
            intent: Natural language description of what to do (e.g., "open notepad")
            tool: Optional specific tool name if known
            args: Optional arguments if tool is specified
            
        Returns:
            Execution result dict
        """
        request = ToolRequest(
            intent=intent,
            tool_name=tool,
            arguments=args or {},
            context={"source": "direct_llm"},
        )
        result = await self.tool_gateway.execute(request)
        print(f"[Gateway] {result.tool_name or 'forge'} -> {'OK' if result.result.get('ok') else 'FAILED'}", flush=True)
        return result.result

    async def _action_worker(self):
        """Processes the current Plan via Tool TRM Gateway without blocking on execution."""
        while self.running:
            plan = self.state.get("plan") or {}
            steps = plan.get("steps", [])
            active_idx = int(self.state.get("active_step", 0) or 0)

            if not steps:
                await asyncio.sleep(1)
                continue

            if active_idx >= len(steps):
                self._print_internal("System", f"Objective '{self.state.get('intent')}' accomplished.")
                self._speak_system(f"Jeffrey, I have successfully accomplished the task: {self.state.get('intent')}.")
                async with self.state_lock:
                    self.state["intent"] = "None"
                    self.state["plan"] = {}
                try:
                    await self.rolling_context.set_current_task("")
                except Exception:
                    pass
                await asyncio.sleep(1)
                continue

            if self._plan_task and not self._plan_task.done():
                await asyncio.sleep(0.5)
                continue

            current_task = steps[active_idx]["task"]
            self._print_internal("System", f"Working: {current_task}")

            request = ToolRequest(
                intent=current_task,
                context={
                    "state_intent": self.state.get("intent", ""),
                    "step_index": active_idx,
                    "total_steps": len(steps),
                },
                last_error=self.state.get("last_tool_error"),
            )

            self._plan_task = asyncio.create_task(
                self._execute_plan_step(current_task, request, active_idx, len(steps))
            )
            await asyncio.sleep(0.5)

    async def _execute_plan_step(
        self,
        current_task: str,
        request: ToolRequest,
        active_idx: int,
        total_steps: int,
    ) -> None:
        try:
            gateway_result = await self.tool_gateway.execute(request)
        except Exception as exc:
            gateway_result = ToolResult(
                ok=False,
                result=None,
                error=str(exc),
                tool_used=None,
                arguments_used=request.arguments or {},
                metadata=ExecutionMetadata(),
            )
        finally:
            self._plan_task = None

        await self._process_plan_result(gateway_result, current_task, active_idx, total_steps)

    async def _process_plan_result(
        self,
        gateway_result: ToolResult,
        current_task: str,
        active_idx: int,
        total_steps: int,
    ) -> None:
        skill_id = gateway_result.metadata.trm_prediction or gateway_result.tool_used or "unknown"
        skill_conf = gateway_result.metadata.trm_confidence
        tool_name = gateway_result.tool_used or "unknown"
        arguments = gateway_result.arguments_used or {}
        result_data = {"ok": gateway_result.ok, "result": gateway_result.result, "error": gateway_result.error}
        call_source = gateway_result.metadata.source
        tool_conf = gateway_result.metadata.trm_confidence or 0.0
        call_info = {"name": tool_name, "arguments": arguments}

        self._print_internal("System", f"Selected Skill: {skill_id}")
        if tool_name:
            self._print_internal("System", f"Executed: {tool_name} -> {'OK' if result_data.get('ok') else 'FAILED'}")

        if self.training_log_cfg.get("enabled", True):
            extra = {
                "gateway_source": gateway_result.metadata.source,
                "gateway_forged": gateway_result.metadata.skill_forged,
            }
            enq_ok = enqueue_tool_call(
                path=self.training_log_path,
                intent=self.state.get("intent"),
                task=current_task,
                skill_id=skill_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result_data,
                call_source=call_source,
                skill_confidence=skill_conf,
                tool_confidence=tool_conf,
                extra=extra,
            )
            if not enq_ok:
                asyncio.create_task(
                    asyncio.to_thread(
                        log_tool_call,
                        path=self.training_log_path,
                        intent=self.state.get("intent"),
                        task=current_task,
                        skill_id=skill_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        result=result_data,
                        call_source=call_source,
                        skill_confidence=skill_conf,
                        tool_confidence=tool_conf,
                        extra=extra,
                    )
                )

        context_for_eval = await self._build_context_payload(current_task)
        decision = await self.reasoning.evaluate_step(
            active_idx,
            result_data,
            context_bundle=context_for_eval,
        )

        if self.experience_log_cfg.get("enabled", False):
            plan_snapshot = self.state.get("plan") if isinstance(self.state.get("plan"), dict) else {}
            extra_exp = {
                "call_source": call_source,
                "tool_confidence": tool_conf,
                "skill_confidence": skill_conf,
                "template_id": plan_snapshot.get("template_id"),
                "template_source": plan_snapshot.get("template_source"),
            }
            enq_ok = enqueue_experience(
                path=self.experience_log_path,
                intent=self.state.get("intent"),
                step_index=active_idx,
                task=current_task,
                skill_id=skill_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result_data,
                decision=decision,
                plan=plan_snapshot,
                extra=extra_exp,
            )
            if not enq_ok:
                asyncio.create_task(
                    asyncio.to_thread(
                        log_experience,
                        path=self.experience_log_path,
                        intent=self.state.get("intent"),
                        step_index=active_idx,
                        task=current_task,
                        skill_id=skill_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        result=result_data,
                        decision=decision,
                        plan=plan_snapshot,
                        extra=extra_exp,
                    )
                )

        try:
            self.bucket_manager.enqueue(
                bucket_name="history",
                event_type="history",
                payload={
                    "intent": self.state.get("intent"),
                    "step_index": active_idx,
                    "task": current_task,
                    "skill_id": skill_id,
                    "tool_call": call_info if skill_id != "unknown" else {},
                    "result": result_data,
                    "decision": decision,
                },
                source="history",
                metadata={"via": "action_worker"},
            )
        except Exception:
            pass

        async with self.state_lock:
            if decision == "CONTINUE":
                self.evolution.log_success(current_task, skill_id)
                self.state["active_step"] = active_idx + 1
                self._print_internal("System", f"Step {active_idx + 1} complete.")
            elif decision == "RE-PLAN":
                self._print_internal("System", "Encountered wall. Re-thinking...")
                new_plan = await self.reasoning.re_plan(result_data.get("error"), self.state)
                self.state["plan"] = new_plan
                self.state["active_step"] = 0
                try:
                    await self.context_curator.stage_structured_artifact(
                        source="reasoning_trm",
                        artifact_type="plan",
                        payload={
                            "goal": new_plan.get("goal"),
                            "steps": [s.get("task") for s in (new_plan.get("steps") or [])][:12],
                        },
                        confidence=0.7,
                        priority=6,
                        metadata={"event": "replan"},
                    )
                except Exception:
                    pass
            elif decision == "FINISH":
                self.evolution.log_success(current_task, skill_id)
                self._print_internal("System", "Goal achieved early.")
                self.state["plan"] = {}
                self.state["intent"] = "None"
                try:
                    await self.rolling_context.set_current_task("")
                except Exception:
                    pass

    async def _autonomy_worker(self):
        """
        Self-directed objective loop (scheduled, non-blocking).

        Runs only when:
        - enabled in config
        - user is idle
        - no plan is currently executing
        - no pending goal confirmation is waiting (even though we can disable confirmations)
        """
        proactive_cfg = self.config.get("self_evolution", {}) or {}
        cfg = self.autonomy_cfg or {}

        # Backwards compatible: allow the legacy flag to enable the new loop.
        enabled = bool(cfg.get("enabled", False)) or bool(proactive_cfg.get("autonomy_enabled", False))
        if self._autonomy_runtime_enabled is not None:
            enabled = bool(self._autonomy_runtime_enabled)
        if not enabled:
            return

        tick_seconds = float(cfg.get("tick_seconds", 15))
        idle_seconds = float(cfg.get("idle_seconds", 20))
        max_backlog = int(cfg.get("max_backlog", 8))
        generate_when_empty = bool(cfg.get("generate_when_empty", True))
        gen_cfg = cfg.get("generation", {}) or {}
        gen_slot = str(gen_cfg.get("slot", "orchestrator"))
        gen_max = int(gen_cfg.get("max_objectives", 5))
        proactive_interval = float(proactive_cfg.get("proactive_interval", 300))

        while self.running:
            await asyncio.sleep(tick_seconds)
            if not self.running:
                break

            # Respect user focus: never interrupt when the user is actively interacting.
            # Also: do not start autonomy until we've seen at least one user input since startup.
            if not self.user_last_input:
                continue
            if self.user_last_input and (time.time() - self.user_last_input) < self.user_focus_window:
                continue
            if self.user_last_input and (time.time() - self.user_last_input) < idle_seconds:
                continue

            # Update enabled dynamically (can be toggled at runtime via commands).
            enabled_now = bool(self.autonomy_cfg.get("enabled", False)) or bool(proactive_cfg.get("autonomy_enabled", False))
            if self._autonomy_runtime_enabled is not None:
                enabled_now = bool(self._autonomy_runtime_enabled)
            if not enabled_now:
                continue

            # Only run when Dexter is truly idle.
            async with self.state_lock:
                plan = self.state.get("plan") or {}
                if plan:
                    continue
                if self.pending_goal:
                    continue
                if self.state.get("intent") not in (None, "", "None"):
                    # Intent may linger; consider idle if no plan and no intent.
                    pass

            # Ensure backlog is populated (LLM-based generation).
            if not self._autonomy_backlog and generate_when_empty:
                if (time.time() - self._autonomy_last_generation) >= proactive_interval:
                    self._autonomy_last_generation = time.time()
                    try:
                        print("[Autonomy] Backlog empty; generating new objectives...", flush=True)
                        ideas = await self.reasoning.generate_proactive_backlog(
                            system_context=self.state,
                            slot=gen_slot,
                            max_objectives=gen_max,
                        )
                        if ideas:
                            for idea in ideas:
                                if len(self._autonomy_backlog) >= max_backlog:
                                    break
                                s = str(idea).strip()
                                if s:
                                    self._autonomy_backlog.append(s)
                            self._save_autonomy_backlog()
                            broadcast_thought("autonomy_backlog", {"count": len(self._autonomy_backlog), "ideas": self._autonomy_backlog[:5]})
                    except Exception as exc:
                        print(f"[Autonomy] Objective generation failed: {exc}", flush=True)

            if not self._autonomy_backlog:
                continue

            # Pop the next objective and execute it as a full plan.
            objective = self._autonomy_backlog.pop(0)
            self._save_autonomy_backlog()

            async with self.state_lock:
                if self.state.get("plan"):
                    continue
                print(f"[Autonomy] Objective: {objective}", flush=True)
                self._speak_system(f"Jeffrey, I am autonomously executing: {objective}.")
                broadcast_thought("autonomy_objective", objective)
                self.state["intent"] = objective
                self.state["plan"] = await self.reasoning.create_plan(objective, self.state)
                self.state["active_step"] = 0
                try:
                    await self.staged_context.stage_artifact(
                        source="reasoning_trm",
                        artifact_type="plan",
                        payload={
                            "goal": self.state["plan"].get("goal"),
                            "steps": [s.get("task") for s in (self.state["plan"].get("steps") or [])][:12],
                            "template_id": self.state["plan"].get("template_id"),
                        },
                        confidence=float(self.state["plan"].get("template_confidence") or 0.75),
                        priority=6,
                        metadata={"event": "plan_created", "origin": "autonomy"},
                    )
                except Exception:
                    pass
                try:
                    await self.rolling_context.set_current_task(objective)
                except Exception:
                    pass


    async def _big_brain_worker(self):
        """Periodically syncs low-importance memory to the Big Brain."""
        cfg = self.config.get("big_brain", {}) or {}
        if not cfg.get("enabled", False):
            return
        sync_on_startup = bool(cfg.get("sync_on_startup", False))
        interval = int(cfg.get("sync_interval_sec", 0))

        async def run_sync():
            from core.big_brain_sync import sync_all
            try:
                result = await asyncio.to_thread(sync_all, self.config)
                broadcast_thought("big_brain_sync", result)
            except Exception as exc:
                print(f"[BigBrain] Sync failed: {exc}", flush=True)

        if sync_on_startup:
            await run_sync()
        if interval <= 0:
            return

        while self.running:
            await asyncio.sleep(interval)
            if not self.running:
                break
            await run_sync()


    async def _process_user_message(self, user_msg: str) -> None:
        if not user_msg:
            return
        self.user_last_input = time.time()
        msg_id = self._next_message_id()
        self._maybe_ack_user(msg_id)

        if user_msg.lower() in ["exit", "quit", "shutdown"]:
            self._print_user("Powering down. Operational data secured.")
            self.running = False
            return

        # Autonomy control (no confirmation; instant).
        cmd = user_msg.strip().lower()
        if cmd in ("/autonomy", "autonomy", "autonomy status"):
            self._print_user(f"Autonomy status: {json.dumps(self._autonomy_status(), ensure_ascii=False)}")
            return
        if cmd in ("autonomy on", "/autonomy on"):
            self._autonomy_runtime_enabled = True
            self._print_user("Autonomy enabled=true")
            return
        if cmd in ("autonomy off", "/autonomy off"):
            self._autonomy_runtime_enabled = False
            self._print_user("Autonomy enabled=false")
            return
        if cmd in ("autonomy next", "/autonomy next"):
            # Force immediate generation on next tick.
            self._autonomy_last_generation = 0.0
            self._print_user("Autonomy next generation forced")
            return
        if cmd in ("autonomy clear", "/autonomy clear"):
            self._autonomy_backlog = []
            self._save_autonomy_backlog()
            self._print_user("Autonomy backlog cleared")
            return

        # Config inspection + updates (fast-path, no LLM)
        cmd_lower = cmd.strip().lower()
        if cmd_lower in ("config show", "/config show", "config status", "/config status"):
            summary = self._format_config_summary()
            self._print_user(f"Config Summary: {summary}")
            return
        if cmd_lower in ("config reload", "/config reload"):
            new_cfg = self._load_json(self._config_path)
            if isinstance(self.config, dict):
                self.config.clear()
                self.config.update(new_cfg)
            else:
                self.config = new_cfg
            self._refresh_runtime_config()
            self._print_user("Config reloaded. Some changes require restart.")
            return
        if cmd_lower.startswith("config get") or cmd_lower.startswith("/config get"):
            raw = cmd.split(None, 2)
            path = raw[2].strip() if len(raw) >= 3 else ""
            value = self._get_config_value(path)
            self._print_user(f"Config {path}: {json.dumps(value, ensure_ascii=False, default=str)}")
            return
        if cmd_lower.startswith("config set") or cmd_lower.startswith("/config set"):
            raw = cmd.split(None, 2)
            remainder = raw[2].strip() if len(raw) >= 3 else ""
            if "=" in remainder:
                path, value_raw = remainder.split("=", 1)
            else:
                parts = remainder.split(None, 1)
                path = parts[0] if parts else ""
                value_raw = parts[1] if len(parts) > 1 else ""
            value = self._parse_config_value(value_raw)
            if not path:
                self._print_user("Config set failed: missing path.")
                return
            if self._set_config_value(path.strip(), value):
                self._refresh_runtime_config()
                saved = self._persist_config()
                suffix = "saved" if saved else "save_failed"
                self._print_user(f"Config updated: {path.strip()} = {json.dumps(value, ensure_ascii=False, default=str)} ({suffix}).")
            else:
                self._print_user(f"Config set failed: {path.strip()}")
            return

        # No routing heuristics: do not rewrite or classify user messages.
        force_chat = False

        # Always broadcast raw user input immediately (no parser/gating).
        try:
            asyncio.create_task(
                self.response_tank.publish(
                    source="user",
                    content={"type": "user_input", "user_input": user_msg, "timestamp": time.time(), "force_chat": force_chat},
                    priority="high",
                )
            )
            asyncio.create_task(self.rolling_context.add_turn("user", user_msg))
        except Exception:
            pass

    async def _tool_result_reactor(self) -> None:
        """
        Background worker that turns tool results into orchestrator reactions.

        Requirements:
        - Tool results MUST trigger an orchestrator LLM call, otherwise Dexter won't react.
        - Injection must include the tool result + the reason it was called (original intent).

        Strategy:
        - Debounce bursts of tool results and react once per batch.
        - Trigger staged-context injection so results become available to the orchestrator.
        - Call orchestrator with an internal event message that instructs it to respond
          to the user and/or schedule next tasks.
        """
        debounce_s = float(os.getenv("DEXTER_TOOL_RESULT_DEBOUNCE_SEC", "0.05") or "0.05")
        max_batch = int(os.getenv("DEXTER_TOOL_RESULT_BATCH", "20") or "20")

        while self.running:
            try:
                first = await self._tool_result_queue.get()
            except Exception:
                await asyncio.sleep(0.05)
                continue

            batch: List[dict] = [first]
            try:
                # Small debounce window to coalesce multi-step tool bursts.
                await asyncio.sleep(max(0.0, debounce_s))
                while len(batch) < max_batch:
                    try:
                        batch.append(self._tool_result_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                    except Exception:
                        break
            except Exception:
                pass

            # Build a compact summary for the injection trigger and the internal event.
            summaries: List[str] = []
            for b in batch:
                try:
                    t = b.get("tool") or "unknown"
                    ok = bool(b.get("success", False))
                    intent = str(b.get("original_intent") or "").strip()
                    if intent:
                        intent = _compact_text(intent, max_words=30, max_chars=160)
                        summaries.append(f"{t}={'OK' if ok else 'FAIL'} intent={intent!r}")
                    else:
                        summaries.append(f"{t}={'OK' if ok else 'FAIL'}")
                except Exception:
                    continue
            trigger_content = " | ".join(summaries[:5]) if summaries else "tool_result"

            injection_text = ""
            injection_payload = None
            try:
                injection_text = await self._trigger_orchestrator_injection("tool_result", trigger_content)
                injection_payload = self._last_injection_payload
            except Exception:
                injection_text = ""
                injection_payload = None

            # Snapshot chat history for context, without adding a synthetic "user" turn.
            try:
                async with self.user_msg_lock:
                    chat_snapshot = list(self.chat_history)
            except Exception:
                chat_snapshot = list(self.chat_history)

            # The orchestrator must react: explain what happened and decide what to do next.
            internal_event_msg = (
                "INTERNAL EVENT: One or more tool calls just completed.\n"
                "Use the injected context (tool results + metadata) to:\n"
                "1) Explain outcome to Jeffrey briefly.\n"
                "2) If follow-up actions are needed, add them to internal.tasks.\n"
                "3) If the tool failed, propose a recovery and schedule it.\n"
                f"Tool batch: {trigger_content}"
            )

            task = asyncio.create_task(
                self._run_orchestrator_chat(
                    user_msg=internal_event_msg,
                    chat_history=chat_snapshot,
                    context_injection=injection_text,
                    msg_id=None,
                    fast=True,
                    injection_payload=injection_payload,
                )
            )
            self._track_user_task(task)

        # Stage user input into orchestrator + think tank bundles (fire-and-forget).
        injection_task: Optional[asyncio.Task] = None
        try:
            asyncio.create_task(
                self.staged_context.stage_artifact(
                    source="user",
                    artifact_type="user_input",
                    payload=user_msg,
                    confidence=1.0,
                    priority=10,
                    metadata={"event": "user_input", "force_chat": force_chat},
                )
            )
            asyncio.create_task(
                self.think_tank_bundle.add_artifact(
                    source="user",
                    artifact_type="user_input",
                    payload=user_msg,
                    confidence=1.0,
                    priority=9,
                    metadata={"event": "user_input"},
                )
            )
            # Stage last carry state and current runtime status for orchestrator visibility.
            last_carry = self.state.get("carry_state")
            if last_carry:
                asyncio.create_task(
                    self.staged_context.stage_artifact(
                        source="orchestrator",
                        artifact_type="carry_state",
                        payload=last_carry,
                        confidence=0.9,
                        priority=9,
                        metadata={"event": "carry_state"},
                    )
                )
            asyncio.create_task(
                self.staged_context.stage_artifact(
                    source="system",
                    artifact_type="runtime_status",
                    payload=self._build_runtime_status(),
                    confidence=0.9,
                    priority=8,
                    metadata={"event": "runtime_status"},
                )
            )
            injection_task = asyncio.create_task(self._trigger_orchestrator_injection("user_input", user_msg))
        except Exception:
            pass

        # Background pipelines (non-blocking)
        try:
            self.memory_ingestor.enqueue("user", user_msg, {"type": "user_input"})
        except Exception:
            pass
        asyncio.create_task(self._schedule_retrieval(user_msg, "user_input"))
        asyncio.create_task(self._schedule_think_tank(user_msg))
        asyncio.create_task(self._schedule_reasoning_trm(user_msg, context={"trigger": "user_input"}))

        # NOTE: We intentionally do not "fast-path" user input directly to Forge.
        # The orchestrator must always see and respond to the user first; tool execution can
        # be requested via the orchestrator's internal payload and run asynchronously.





        try:
            # Snapshot chat history with the user message appended so parallel requests
            # still see the most recent input even if responses return out of order.
            async with self.user_msg_lock:
                self._remember_chat("user", user_msg)
                chat_snapshot = list(self.chat_history)
        except Exception:
            chat_snapshot = list(self.chat_history)

        injection_payload = self._last_injection_payload
        injection_text = ""
        if injection_task is not None:
            try:
                await asyncio.sleep(0)  # yield once; do not block
                if injection_task.done():
                    injection_text = injection_task.result()
            except Exception:
                pass

        task = asyncio.create_task(
            self._run_orchestrator_chat(
                user_msg=user_msg,
                chat_history=chat_snapshot,
                context_injection=injection_text,
                msg_id=msg_id,
                fast=self._is_busy(),
                injection_payload=injection_payload,
            )
        )
        self._track_user_task(task)
        return

    async def _run_orchestrator_chat(
        self,
        user_msg: str,
        chat_history: List[Dict[str, str]],
        context_injection: str,
        msg_id: int | None = None,
        fast: bool = False,
        injection_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Run the orchestrator LLM call without blocking other user inputs."""
        start = time.time()
        raw_response = await self.reasoning.chat_response(
            user_msg,
            chat_history=chat_history,
            slot=self.conversation_cfg.get("chat_slot", "orchestrator"),
            pending_goal=None,
            fast=fast,
            context_injection=context_injection,
        )
        duration_ms = (time.time() - start) * 1000.0
        user_response, internal = self._parse_orchestrator_response(raw_response)
        if not user_response:
            user_response = raw_response

        try:
            async with self.user_msg_lock:
                self._remember_chat("assistant", user_response)
        except Exception:
            pass

        try:
            await self.rolling_context.add_turn("assistant", user_response)
        except Exception:
            pass

        tag = f"(#{msg_id}) " if msg_id is not None else ""
        self._print_user(f"{tag}{user_response}")
        self._print_internal("System", f"{tag}LLM response_time_ms={duration_ms:.0f}")
        internal_panel = self._format_internal_panels(internal, injection_payload)
        if internal_panel:
            self._print_internal("System", internal_panel)
        broadcast_thought(
            "chat_response",
            {
                "chat": user_response,
                "think_tank_reasoning": internal_panel,
                "forge_intent": internal.get("tasks") or [],
                "message_id": msg_id,
            },
        )
        asyncio.create_task(self._dispatch_internal_payload(internal, user_msg))
        # Fire-and-forget: send only orchestrator-declared tasks to Forge.
        try:
            intents = internal.get("tasks") or []
            if intents:
                for intent in intents:
                    asyncio.create_task(self.send_to_forge(str(intent)))
        except Exception:
            pass
        if self.conversation_cfg.get("speak_chat", False):
            speak_out_loud(user_response)
        # Persist per-turn carry state to working memory (non-blocking thread).
        carry_state = self._build_carry_state(user_msg, user_response, internal)
        self.state["carry_state"] = carry_state
        await self._append_working_memory(carry_state)


    async def _conversation_worker(self):
        """Handles user interaction and high-level goal setting via terminal."""
        while self.running:
            try:
                # Run blocking input in a thread so the event loop stays responsive.
                user_msg = await asyncio.to_thread(input, "[Jeffrey] > ")
                if not user_msg:
                    continue

                cmd = user_msg.strip()
                if cmd.lower() == "/m":
                    # Interactive picker runs in the input thread; never block the event loop.
                    await asyncio.to_thread(self._ui_pick_orchestrator_model)
                    continue
                if cmd.lower() == "/mtank":
                    picked = await asyncio.to_thread(self._ui_pick_think_tank_model)
                    if picked:
                        slot_name, model = picked
                        if self._apply_think_tank_model(slot_name, model):
                            # Restart to apply new model/provider config into active advisors.
                            if self._think_tank_reload_task is None or self._think_tank_reload_task.done():
                                self._think_tank_reload_task = asyncio.create_task(self._restart_think_tank())
                    continue
                if cmd.lower() == "/rreset":
                    try:
                        if self.trained_reasoning_trm and self.trained_reasoning_trm.is_ready():
                            self.trained_reasoning_trm.reset_carry()
                            self._print_internal("System", "[ReasoningTRM] Carry reset")
                        else:
                            self._print_internal("System", "[ReasoningTRM] Not available")
                    except Exception:
                        pass
                    continue
                if cmd.lower().startswith("/r "):
                    prompt = cmd[3:].strip()
                    if not prompt:
                        continue
                    if not self.trained_reasoning_trm or not self.trained_reasoning_trm.is_ready():
                        self._print_internal("System", "[ReasoningTRM] Not available")
                        continue
                    try:
                        res = await asyncio.to_thread(self.trained_reasoning_trm.reason, prompt, {"origin": "direct_chat"})
                    except Exception as exc:
                        self._print_internal("System", f"[ReasoningTRM] Error: {exc}")
                        continue
                    steps = res.get("reasoning_steps") or []
                    concl = res.get("conclusion")
                    conf = res.get("confidence")
                    if steps:
                        body = "\n".join([f"- {s}" for s in steps[:12]])
                        extra = "\n... (truncated)" if len(steps) > 12 else ""
                        self._print_user(f"[ReasoningTRM]\n{body}{extra}")
                    elif concl:
                        self._print_user(f"[ReasoningTRM] {concl}")
                    else:
                        self._print_user("[ReasoningTRM] (no output)")
                    try:
                        self._print_internal("System", f"[ReasoningTRM] confidence={float(conf or 0.0):.2f} carry_steps={res.get('carry_steps')}")
                    except Exception:
                        pass
                    continue
                task = asyncio.create_task(self._process_user_message(user_msg))
                self._track_user_task(task)

            except EOFError:
                # Allows non-interactive runs (piped stdin) to exit cleanly.
                self.running = False
                return
            except Exception as e:
                print(f"[Cognitive Fault] {e}")
            
            await asyncio.sleep(0.1)

    async def startup(self, initial_intent: str = None):
        """Boot Dexter and optionally set an initial goal."""
        self._run_bootstrap()
        # Must be called from within a running event loop.
        maybe_enable_asyncio_debug(self.config)
        self._loop_lag_monitor.start()
        if self._tool_result_reactor_task is None or self._tool_result_reactor_task.done():
            self._tool_result_reactor_task = asyncio.create_task(self._tool_result_reactor())
        if initial_intent:
            self._print_internal("System", f"Preset Intent: {initial_intent}")
            self.state["intent"] = initial_intent
            self.state["plan"] = await self.reasoning.create_plan(initial_intent, self.state)
            try:
                await self.rolling_context.set_current_task(initial_intent)
            except Exception:
                pass

        self._print_internal("System", "Starting Cerebral API...")
        threading.Thread(target=start_api_server, daemon=True).start()

        await self._start_trm_stack()
        await self.async_executor.start()

        # Start Tool TRM Gateway
        await self.tool_gateway.start()

        await self._stage_startup_status()
        
        # Register to receive execution results immediately
        self.tool_gateway.register_result_callback(self._on_tool_result)
        
        # Start Online Trainers (dual-weight real-time learning)
        self.online_trainers = init_online_trainers(self.config)
        for trm_type, trainer in (self.online_trainers or {}).items():
            trainer.start()
            self._print_internal("System", f"Online TRM Trainer started ({trm_type})")
        
        self._print_internal("System", "Waking up parallel workers...")
        try:
            await asyncio.gather(
                self._artifact_worker(),
                self._action_worker(),
                self._conversation_worker(),
                self._autonomy_worker(),
                self._big_brain_worker(),
            )
        finally:
            try:
                await self._loop_lag_monitor.stop()
            except Exception:
                pass
            # Best-effort shutdown of background pipelines.
            try:
                await self.memory_ingestor.stop()
            except Exception:
                pass
            try:
                await self.memory_trm.stop()
            except Exception:
                pass
            try:
                await self.tool_gateway.stop()
            except Exception:
                pass
            try:
                await self.async_executor.stop()
            except Exception:
                pass
            try:
                await self.memory_db_writer.stop()
            except Exception:
                pass
            try:
                await self.bucket_manager.stop()
            except Exception:
                pass


def main():
    """Main entry point with dual terminal support."""
    parser = argparse.ArgumentParser(description='Dexter AI Assistant')
    parser.add_argument('--stream-mode', action='store_true', 
                        help='Run as stream terminal (receives all output)')
    parser.add_argument('--no-spawn', action='store_true',
                        help='Do not spawn stream terminal automatically')
    parser.add_argument('--single', action='store_true',
                        help='Run in single terminal mode (legacy)')
    args = parser.parse_args()
    
    if args.stream_mode:
        # This is the stream terminal - just display incoming data
        _run_stream_terminal()
        return
    
    if args.single:
        # Legacy single terminal mode
        dexter = Dexter()
        asyncio.run(dexter.startup())
        return
    
    # Start stream server in background
    threading.Thread(target=_stream_server, daemon=True).start()
    time.sleep(0.1)  # Let server start
    
    # Spawn stream terminal unless disabled
    if not args.no_spawn:
        _spawn_stream_terminal()
        time.sleep(0.5)  # Let stream terminal connect
    
    # Redirect stdout to stream writer (conversation mode - filtered output)
    sys.stdout = StreamWriter(sys.__stdout__, is_conversation_terminal=True)
    
    print("═" * 70)
    print("  DEXTER - Conversation Terminal")
    print("═" * 70)
    print()
    print("  This is your clean conversation space with Dexter.")
    print("  The Activity Stream terminal shows all system logs.")
    print()
    print("  Type your message and press Enter to chat.")
    print()
    print("═" * 70)
    print()
    
    dexter = Dexter()
    asyncio.run(dexter.startup())


if __name__ == "__main__":
    main()
