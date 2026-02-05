#!/usr/bin/env python3
"""
tool_trm_gateway.py - The Tool TRM Gateway

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                         DEXTER (Main LLM)                               │
│                              │                                          │
│                    [sends raw request]                                  │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    TOOL TRM GATEWAY                              │   │
│  │  ┌──────────────────┐    ┌──────────────────┐                    │   │
│  │  │  Forge    │◄───│   TOOL TRM       │                    │   │
│  │  │  (LLM - DECIDES) │    │  (OBSERVER ONLY) │                    │   │
│  │  │                  │    │                  │                    │   │
│  │  │  • Understands   │────▶ • Watches        │                    │   │
│  │  │  • Selects tools │    │ • Learns         │                    │   │
│  │  │  • Structures    │    │ • Builds weights │                    │   │
│  │  │    call args     │    │ • Never decides  │                    │   │
│  │  └────────┬─────────┘    └──────────────────┘                    │   │
│  │           │                                                      │   │
│  │           ▼                                                      │   │
│  │     [structured call]                                            │   │
│  │           │                                                      │   │
│  │           ▼                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐ │   │
│  │  │                    EXECUTOR                                 │ │   │
│  │  │  • Parses Teacher output                                    │ │   │
│  │  │  • Runs tools                                               │ │   │
│  │  │  • Returns results                                          │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Key Design Principle:
  - Tool TRM is an INFANT - it ONLY OBSERVES, never decides
  - Forge (LLM) makes ALL decisions
  - TRM watches Teacher's decisions and learns from them
  - Over time, TRM builds knowledge and can eventually graduate
"""

import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import core.tool_executor as tool_executor
from core.async_executor import AsyncToolExecutor
from core.trained_trm_loader import get_tool_trm, ToolTRM
from core.persistent_bundle import PersistentArtifactBundle


BASE_DIR = Path(__file__).resolve().parent.parent
SKILLS_DIR = BASE_DIR / "skills"

# Online trainer - imported lazily to avoid circular deps
_online_trainer = None

def _get_online_trainer():
    """Lazy load online trainer."""
    global _online_trainer
    if _online_trainer is None:
        try:
            from core.trm_online_trainer import get_online_trainer
            _online_trainer = get_online_trainer()
        except ImportError:
            pass
    return _online_trainer


@dataclass
class ToolRequest:
    """A request to the Tool TRM Gateway."""
    # Can be structured or natural language
    intent: str  # Natural language: "open notepad"
    tool: Optional[str] = None  # Structured: "shell.run_command"
    args: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # "high", "normal", "low"
    source: str = "dexter"  # Who sent this request
    conversation_id: Optional[str] = None  # Track conversation context
    last_error: Optional[str] = None  # Previous error for retry context


@dataclass 
class ExecutionMetadata:
    """Rich metadata about how execution proceeded."""
    # Selection path
    selection_method: str = "unknown"  # "trm", "llm_teacher", "heuristic", "forge"
    trm_confidence: float = 0.0
    trm_prediction: Optional[str] = None
    llm_teacher_used: bool = False
    
    # Source tracking
    source: str = "dexter"  # Who initiated: "dexter", "jeffrey", "external", "api"
    
    # Execution details
    retries: int = 0
    args_modified: bool = False
    original_args: Dict[str, Any] = field(default_factory=dict)
    final_args: Dict[str, Any] = field(default_factory=dict)
    
    # Skill creation
    skill_forged: bool = False
    skill_forged_name: Optional[str] = None
    skill_forge_duration_ms: int = 0
    
    # Dependencies
    deps_installed: List[str] = field(default_factory=list)
    deps_install_duration_ms: int = 0
    
    # Timing
    total_duration_ms: int = 0
    selection_duration_ms: int = 0
    execution_duration_ms: int = 0


@dataclass 
class ToolResult:
    """
    Rich result from Tool TRM execution.
    Contains everything Dexter needs to learn and remember.
    """
    ok: bool
    result: Any = None
    error: Optional[str] = None
    tool_used: Optional[str] = None
    arguments_used: Dict[str, Any] = field(default_factory=dict)
    
    # Rich metadata for learning
    metadata: ExecutionMetadata = field(default_factory=ExecutionMetadata)
    
    # Legacy compat
    skill_created: Optional[str] = None
    dependencies_installed: List[str] = field(default_factory=list)
    duration_ms: int = 0
    execution_path: str = "unknown"
    
    def to_upstream_bundle(self) -> Dict[str, Any]:
        """
        Package result for upstream consumption by Dexter.
        This is what gets injected into Dexter's context.
        """
        return {
            "success": self.ok,
            "tool": self.tool_used,
            "arguments": self.arguments_used,
            "result": self.result,
            "error": self.error,
            "execution_summary": {
                "method": self.metadata.selection_method,
                "trm_confidence": self.metadata.trm_confidence,
                "llm_teacher_used": self.metadata.llm_teacher_used,
                "retries": self.metadata.retries,
                "args_modified": self.metadata.args_modified,
                "skill_forged": self.metadata.skill_forged,
                "skill_forged_name": self.metadata.skill_forged_name,
                "deps_installed": self.metadata.deps_installed,
            },
            "timing": {
                "total_ms": self.metadata.total_duration_ms,
                "selection_ms": self.metadata.selection_duration_ms,
                "execution_ms": self.metadata.execution_duration_ms,
            },
        }


@dataclass
class TeacherContext:
    """
    Context bundle for the LLM Teacher.
    Gives the teacher everything it needs to make good decisions.
    """
    # Available capabilities
    available_skills: List[str] = field(default_factory=list)
    skill_descriptions: Dict[str, str] = field(default_factory=dict)
    
    # Conversation history (last N turns)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    max_history: int = 10
    
    # Current state
    current_intent: str = ""
    last_tool_result: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    
    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def to_prompt_context(self) -> str:
        """Format context for LLM prompt."""
        parts = []
        
        # Skills
        if self.available_skills:
            parts.append(f"Available tools ({len(self.available_skills)}):")
            for skill in self.available_skills[:30]:  # Top 30
                desc = self.skill_descriptions.get(skill, "")
                if desc:
                    parts.append(f"  - {skill}: {desc[:80]}")
                else:
                    parts.append(f"  - {skill}")
        
        # Recent conversation
        if self.conversation_history:
            parts.append("\nRecent conversation:")
            for turn in self.conversation_history[-5:]:
                parts.append(f"  [{turn['role']}]: {turn['content']}")
        
        # Last result
        if self.last_tool_result:
            ok = self.last_tool_result.get("ok", False)
            tool = self.last_tool_result.get("tool", "unknown")
            parts.append(f"\nLast execution: {tool} -> {'SUCCESS' if ok else 'FAILED'}")
        
        if self.last_error:
            parts.append(f"Last error: {self.last_error}")
        
        return "\n".join(parts)


class ContextChannel:
    """
    A communication channel between two LLM brains.
    
    Messages sent to a channel accumulate in its bundle.
    When triggered, the bundle is injected into the owning LLM's context.
    
    TRMs (Memory, Reasoning, Tool) can SUBSCRIBE to channels to learn from
    all communication flowing through the system.
    
    This enables async, decoupled LLM-to-LLM communication:
    - Dexter writes to Forge's channel → triggers Tool LLM
    - Forge writes to Dexter's channel → triggers Dexter's LLM
    - All TRMs subscribed to channels learn from every message
    """
    
    def __init__(self, owner: str, on_trigger: callable = None):
        self.owner = owner  # Who owns this channel (receives messages)
        self.bundle: List[Dict[str, Any]] = []  # Accumulated messages
        self.on_trigger = on_trigger  # Callback when bundle is injected
        self._lock = asyncio.Lock()
        self._pending = asyncio.Event()
        
        # Subscribers - TRMs and other learners that want to see all messages
        self._subscribers: List[callable] = []
    
    def subscribe(self, callback: callable):
        """
        Subscribe to all messages on this channel.
        
        TRMs use this to learn from all communication:
        - Memory TRM: stores conversations, builds episodic memory
        - Reasoning TRM: learns from request→result patterns
        - Tool TRM: learns from tool selection decisions
        
        Callback receives: (sender, message, msg_type, metadata)
        """
        self._subscribers.append(callback)
        print(f"[ContextChannel:{self.owner}] New subscriber registered", flush=True)
    
    def unsubscribe(self, callback: callable):
        """Remove a subscriber."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    async def _notify_subscribers(self, msg: Dict[str, Any]):
        """Notify all subscribers of a new message."""
        for subscriber in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(
                        msg["sender"],
                        msg["message"],
                        msg["type"],
                        msg.get("metadata", {}),
                    )
                else:
                    subscriber(
                        msg["sender"],
                        msg["message"],
                        msg["type"],
                        msg.get("metadata", {}),
                    )
            except Exception as e:
                print(f"[ContextChannel:{self.owner}] Subscriber error: {e}", flush=True)
    
    async def send(self, sender: str, message: str, msg_type: str = "request", metadata: Dict = None):
        """
        Send a message to this channel.
        
        Args:
            sender: Who sent this (e.g., "dexter", "forge")
            message: The content
            msg_type: "request", "response", "clarification", "result", "error"
            metadata: Optional extra data
        """
        msg = {
            "sender": sender,
            "message": message,
            "type": msg_type,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        
        async with self._lock:
            self.bundle.append(msg)
            self._pending.set()
        
        # Notify subscribers immediately (they learn in real-time)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._notify_subscribers(msg))
        except RuntimeError:
            await self._notify_subscribers(msg)
    
    async def trigger_injection(self) -> List[Dict[str, Any]]:
        """
        Trigger injection - returns bundle and clears it.
        Called when the owning LLM is ready to receive.
        """
        async with self._lock:
            bundle = self.bundle.copy()
            self.bundle.clear()
            self._pending.clear()
            on_trigger = self.on_trigger

        if on_trigger and bundle:
            try:
                if asyncio.iscoroutinefunction(on_trigger):
                    await on_trigger(bundle)
                else:
                    on_trigger(bundle)
            except Exception as e:
                print(f"[ContextChannel:{self.owner}] Trigger error: {e}", flush=True)

        return bundle
    
    def has_pending(self) -> bool:
        """Check if there are pending messages."""
        return len(self.bundle) > 0
    
    async def wait_for_message(self, timeout: float = None) -> bool:
        """Wait for a message to arrive."""
        try:
            await asyncio.wait_for(self._pending.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def format_for_injection(self) -> str:
        """Format pending messages for LLM context injection."""
        if not self.bundle:
            return ""
        
        lines = [f"[Messages from {self.owner} channel]"]
        for msg in self.bundle:
            sender = msg["sender"]
            content = msg["message"]
            msg_type = msg["type"]
            lines.append(f"[{sender}|{msg_type}]: {content}")
        
        return "\n".join(lines)


class ToolTRMGateway:
    """
    The Tool TRM Gateway - Central execution intelligence.
    
    LLM-to-LLM Communication Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     DEXTER (Reasoning)                       │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │             Dexter's Context Bundle                  │    │
    │  │  (receives: results, clarifications, status)         │    │
    │  └──────────────────────▲──────────────────────────────┘    │
    │                         │ send_to_dexter()                   │
    └─────────────────────────┼───────────────────────────────────┘
                              │
    ┌─────────────────────────┼───────────────────────────────────┐
    │                 Forge (Execution)                     │
    │                         │                                    │
    │  ┌──────────────────────▼──────────────────────────────┐    │
    │  │            Forge's Context Bundle             │    │
    │  │  (receives: requests from Dexter)                    │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                         │                                    │
    │                         ▼                                    │
    │              Parse → Execute → Return                        │
    └─────────────────────────────────────────────────────────────┘
    
    - Dexter writes to inbound_channel → triggers Forge
    - Forge writes to outbound_channel → triggers Dexter
    - Both brains communicate through language, decoupled by channels
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gateway_cfg = config.get("tool_trm_gateway", {}) or {}
        self.tool_access_cfg = config.get("tool_access", {}) or {}
        
        # Executor pool
        exec_cfg = config.get("executor", {}) or {}
        self._executor = AsyncToolExecutor(
            max_workers=int(exec_cfg.get("max_workers", 20)),
            result_handler=self._on_execution_complete,
        )
        
        # Trained Tool TRM for intelligent selection
        self._trm: Optional[ToolTRM] = None
        self._trm_enabled = bool(self.gateway_cfg.get("trm_enabled", True))
        self._trm_confidence_threshold = float(self.gateway_cfg.get("confidence_threshold", 0.3))
        
        # LLM Teacher - helps while TRM learns
        self._llm_teacher_enabled = bool(self.gateway_cfg.get("llm_teacher_enabled", True))
        self._llm_teacher_slot = self.gateway_cfg.get("llm_teacher_slot", "forge")
        self._llm_takeover_threshold = float(self.gateway_cfg.get("trm_takeover_threshold", 0.7))
        self._tool_agent = None  # Lazy loaded
        
        # Teacher context - maintains conversation state for LLM
        self._teacher_context = TeacherContext()
        
        # Forge's context bundle - simpler than Dexter's
        # Contains: current mission, available skills, recent results
        self._teacher_bundle = {
            "current_mission": None,
            "current_objective": None,
            "available_skills": [],
            "recent_results": [],  # Last 5 tool results
            "pending_request": None,
        }
        self._forge_bundle = PersistentArtifactBundle("forge")
        
        # Skill registry cache
        self._skill_registry: Dict[str, Any] = {}
        self._skill_docs: Dict[str, str] = {}  # Skill docstrings
        self._last_registry_refresh = 0
        self._registry_ttl = 30  # Refresh every 30s

        # Tool access control (hide/disallow tools from Forge).
        # This is the lever that lets Dexter keep memory/knowledge graph writes internal-only.
        self._deny_tools = set(self.tool_access_cfg.get("deny_tools") or [])
        self._allow_tools = set(self.tool_access_cfg.get("allow_tools") or [])
        self._deny_prefixes = tuple(self.tool_access_cfg.get("deny_prefixes") or [])
        
        # Execution history for learning
        self._execution_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        
        # Skill forge integration
        self._forge_enabled = bool(self.gateway_cfg.get("auto_forge", True))
        self._forge_on_miss = bool(self.gateway_cfg.get("forge_on_miss", True))
        self._auto_retry_enabled = bool(self.gateway_cfg.get("auto_retry_on_fail", True))
        self._max_retries = int(self.gateway_cfg.get("max_retries", 1))
        self._retry_backoff_sec = float(self.gateway_cfg.get("retry_backoff_sec", 0.2))
        
        # Dependency auto-install
        self._auto_install = bool(config.get("dependency_auto_install", {}).get("enabled", True))
        
        # Result callbacks - for upstream notification
        self._result_callbacks: List[callable] = []
        
        # Stats
        self._stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "skills_forged": 0,
            "deps_installed": 0,
            "trm_selections": 0,
            "llm_teacher_selections": 0,
            "heuristic_selections": 0,
        }
        
        self._running = False
        
        # Context Channels for LLM-to-LLM communication
        # Dexter writes to this channel to request tool actions
        self.inbound_channel = ContextChannel(
            owner="forge",
            on_trigger=self._on_inbound_trigger,
        )
        # Forge writes to this channel to respond to Dexter
        self.outbound_channel: Optional[ContextChannel] = None  # Set by Dexter

    def _tool_allowed(self, tool_name: Optional[str]) -> bool:
        if not tool_name:
            return False
        if self._allow_tools and tool_name in self._allow_tools:
            return True
        if tool_name in self._deny_tools:
            return False
        if self._deny_prefixes and tool_name.startswith(self._deny_prefixes):
            return False
        return True

    def _filter_registry(self, registry: Dict[str, Any]) -> Dict[str, Any]:
        if not registry:
            return {}
        return {k: v for k, v in registry.items() if self._tool_allowed(k)}
    
    def connect_to_dexter(self, dexter_channel: ContextChannel):
        """
        Connect to Dexter's context channel for bidirectional communication.
        
        The Forge will write responses/clarifications/results to this channel,
        which triggers injection into Dexter's LLM context.
        """
        self.outbound_channel = dexter_channel
        print("[Tool TRM Gateway] Connected to Dexter's context channel", flush=True)
    
    def set_mission(self, mission: str, objective: str = None):
        """
        Set the current mission/objective for Forge's context.
        Called by Dexter to keep Forge aligned with goals.
        """
        self._teacher_bundle["current_mission"] = mission
        if objective:
            self._teacher_bundle["current_objective"] = objective
    
    def _build_teacher_context_bundle(self, request: str) -> str:
        """
        Build the Forge's context bundle for LLM injection.
        
        Contains:
        - Current mission/objective
        - Available tools (with full function names like shell.run)
        - Recent tool results
        - Current request from Dexter
        """
        parts = []

        # Persistent forge memory (skills inventory, execution patterns, etc.)
        try:
            persistent = self._forge_bundle.format_for_llm()
            if persistent:
                parts.append(persistent)
        except Exception:
            pass
        
        # Mission/objective
        if self._teacher_bundle["current_mission"]:
            parts.append(f"Mission: {self._teacher_bundle['current_mission']}")
        if self._teacher_bundle["current_objective"]:
            parts.append(f"Current Objective: {self._teacher_bundle['current_objective']}")
        
        # Available tools - show ACTUAL function names (tool_prefix.function)
        # NOTE: _skill_registry is a mapping of fully-qualified tool names -> callables.
        if self._skill_registry:
            parts.append(f"\nAvailable Tools ({len(self._skill_registry)} total):")

            # Prefer showing the most relevant tools first (improves Teacher choices)
            preferred_prefixes = (
                "shell.",
                "powershell.",
                "file_ops.",
                "file_system.",
            )
            preferred = [t for t in self._skill_registry.keys() if t.startswith(preferred_prefixes)]
            others = [t for t in self._skill_registry.keys() if t not in preferred]

            tool_count = 0
            for tool_name in (preferred + others):
                doc = self._skill_docs.get(tool_name, "")
                doc_first = (doc.split("\n", 1)[0] if doc else "")
                if doc_first:
                    parts.append(f"  • {tool_name}: {doc_first}")
                else:
                    parts.append(f"  • {tool_name}")
                tool_count += 1
                if tool_count >= 60:  # cap to avoid token explosion
                    parts.append("  ... (more tools available)")
                    break
        
        # Recent results
        if self._teacher_bundle["recent_results"]:
            parts.append("\nRecent Tool Results:")
            for r in self._teacher_bundle["recent_results"][-3:]:
                status = "✓" if r.get("ok") else "✗"
                parts.append(f"  {status} {r.get('tool', '?')}: {r.get('summary', '')[:80]}")
        
        # Current request
        parts.append(f"\n[DEXTER REQUESTS]: {request}")
        
        return "\n".join(parts)

    def _queue_forge_artifact(
        self,
        source: str,
        artifact_type: str,
        payload: Any,
        confidence: float = 0.7,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Non-blocking persist into Forge's bundle."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            loop.create_task(
                self._forge_bundle.add_artifact(
                    source=source,
                    artifact_type=artifact_type,
                    payload=payload,
                    confidence=confidence,
                    priority=priority,
                    metadata=metadata,
                )
            )
        except Exception:
            pass
    
    async def _on_inbound_trigger(self, bundle: List[Dict[str, Any]]):
        """
        Called when messages from Dexter are ready to be processed.
        
        This is the TRIGGER for Forge's LLM injection:
        - Build context bundle with mission, skills, recent results
        - Inject into Tool LLM
        - Process request
        - Send result back to Dexter
        """
        # Process each message from Dexter
        for msg in bundle:
            sender = msg["sender"]
            content = msg["message"]
            msg_type = msg["type"]
            
            print(f"[Forge] Received from {sender}: {content}", flush=True)

            # Persist the inbound request into Forge's bundle
            self._queue_forge_artifact(
                source="dexter",
                artifact_type="request",
                payload=content,
                confidence=0.9,
                priority=8,
                metadata={"msg_type": msg_type},
            )
            
            # Build context bundle for Forge LLM injection
            context_bundle = self._build_teacher_context_bundle(content)
            self._teacher_bundle["pending_request"] = content
            
            print(f"[Forge] Context bundle injected ({len(context_bundle)} chars)", flush=True)
            
            # Process the request
            if msg_type in ("request", "execute"):
                result = await self.execute(content)
                
                # Record result in bundle for future context
                self._teacher_bundle["recent_results"].append({
                    "ok": result.ok,
                    "tool": result.tool_used,
                    # Keep full payload for stream visibility / future teacher context.
                    "summary": json.dumps(result.to_upstream_bundle(), ensure_ascii=False, default=str),
                    "timestamp": time.time(),
                })
                # Keep only last 5 results
                self._teacher_bundle["recent_results"] = self._teacher_bundle["recent_results"][-5:]
                
                # Send result back to Dexter's channel
                if self.outbound_channel:
                    await self.outbound_channel.send(
                        sender="forge",
                        message=self._format_result_for_dexter(result),
                        msg_type="result",
                        metadata={"result": result.to_upstream_bundle()},
                    )

                # Persist tool result into Forge bundle (non-blocking)
                self._queue_forge_artifact(
                    source="forge",
                    artifact_type="tool_result",
                    payload={
                        "tool": result.tool_used,
                        "ok": result.ok,
                        "error": result.error,
                        "summary": result.to_upstream_bundle(),
                    },
                    confidence=1.0 if result.ok else 0.6,
                    priority=7,
                )
    
    def _format_result_for_dexter(self, result: ToolResult) -> str:
        """Format a tool result for Dexter's conversation channel (keep it short)."""
        tool = result.tool_used or "unknown"
        if result.ok:
            return f"Done. ({tool})"
        err = (result.error or "unknown error").strip()
        return f"Tool failed ({tool}): {err}"
    
    async def send_to_dexter(self, message: str, msg_type: str = "response", metadata: Dict = None):
        """
        Send a message to Dexter's context channel.
        
        Used for:
        - Clarification questions
        - Status updates
        - Execution results
        - Errors
        """
        if self.outbound_channel:
            await self.outbound_channel.send(
                sender="forge",
                message=message,
                msg_type=msg_type,
                metadata=metadata,
            )
        else:
            print(f"[Forge] No outbound channel to Dexter: {message}", flush=True)
    
    def register_result_callback(self, callback: callable):
        """Register a callback to receive execution results."""
        self._result_callbacks.append(callback)
    
    async def _notify_result(self, request: ToolRequest, result: ToolResult):
        """Notify all registered callbacks of execution result."""
        upstream_bundle = result.to_upstream_bundle()
        upstream_bundle["original_intent"] = request.intent
        upstream_bundle["request_source"] = request.source

        # Persist request + result for Forge memory (covers direct gateway usage too)
        try:
            self._queue_forge_artifact(
                source=request.source or "dexter",
                artifact_type="request",
                payload=request.intent,
                confidence=0.85,
                priority=7,
                metadata={"origin": "notify_result"},
            )
            self._queue_forge_artifact(
                source="forge",
                artifact_type="tool_result",
                payload={
                    "tool": result.tool_used,
                    "ok": result.ok,
                    "error": result.error,
                    "summary": upstream_bundle,
                },
                confidence=1.0 if result.ok else 0.6,
                priority=6,
                metadata={"origin": "notify_result"},
            )
        except Exception:
            pass
        
        # Fire-and-forget callbacks: result processing must not block tool execution.
        # (Dexter should remain responsive even if staging/injection is slow.)
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except Exception:
            loop = None

        def _log_task_result(t: asyncio.Task) -> None:
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                return
            except Exception:
                return
            if exc:
                print(f"[Gateway] Result callback task error: {exc}", flush=True)

        for callback in self._result_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    if loop:
                        task = loop.create_task(callback(upstream_bundle))
                        task.add_done_callback(_log_task_result)
                    else:
                        # No loop: best-effort sync run (should be rare).
                        await callback(upstream_bundle)
                else:
                    if loop:
                        loop.run_in_executor(None, callback, upstream_bundle)
                    else:
                        callback(upstream_bundle)
            except Exception as e:
                print(f"[Gateway] Result callback error: {e}", flush=True)
    
    async def direct_request(self, raw_input: str, source: str = "external") -> ToolResult:
        """
        Direct communication with the Tool Agent.
        
        For external callers (Jeffrey over the network, other systems) to talk
        directly to the Forge without going through Dexter's LLM.
        
        The caller speaks naturally to the Forge, who understands and executes.
        
        Args:
            raw_input: Natural language request or any format the caller wants
            source: Who is calling (for logging/tracking)
            
        Returns:
            ToolResult with full execution metadata
        """
        # Mark as external source for tracking
        result = await self.execute(raw_input)
        # Add source tracking
        if result.metadata:
            result.metadata.source = source
        return result
    
    async def start(self):
        """Start the Tool TRM Gateway."""
        if self._running:
            return
        
        self._running = True
        await self._executor.start()
        
        # Load Tool TRM
        if self._trm_enabled:
            try:
                self._trm = get_tool_trm()
                if self._trm.is_ready():
                    print("[Tool TRM Gateway] Neural Tool TRM online", flush=True)
                else:
                    print("[Tool TRM Gateway] Tool TRM not ready, using heuristics", flush=True)
            except Exception as e:
                print(f"[Tool TRM Gateway] TRM load failed: {e}", flush=True)
        
        # Initial registry load
        self._refresh_registry()
        
        print(f"[Tool TRM Gateway] Online - {len(self._skill_registry)} skills available", flush=True)
    
    async def stop(self):
        """Stop the gateway."""
        self._running = False
        await self._executor.shutdown()
        print(f"[Tool TRM Gateway] Offline - Stats: {self._stats}", flush=True)
    
    def _extract_raw_input(self, request: Union[ToolRequest, Dict, str]) -> str:
        """
        Extract raw input from Dexter to pass to the Forge.
        
        The Teacher is an LLM - it will understand whatever Dexter sends.
        We just need to convert it to a string for the Teacher to read.
        """
        if isinstance(request, str):
            return request
        
        if isinstance(request, ToolRequest):
            # Reconstruct what Dexter was trying to say
            if request.intent:
                return request.intent
            elif request.tool:
                return f"Call {request.tool} with args: {json.dumps(request.args)}"
            else:
                return str(request)
        
        if isinstance(request, dict):
            # Try to get the human-readable intent
            if "intent" in request:
                return request["intent"]
            if "message" in request:
                return request["message"]
            if "query" in request:
                return request["query"]
            # Otherwise serialize it - Teacher will understand JSON
            return json.dumps(request)
        
        return str(request)
    
    async def _resolve_via_teacher(self, raw_input: str) -> Dict[str, Any]:
        """
        Send raw input to the Forge (LLM) and get back a structured call.
        
        The Teacher understands Dexter's intent naturally (LLM-to-LLM).
        
        ARCHITECTURE NOTE:
        - The Tool TRM is an OBSERVER ONLY - it never makes decisions
        - The Forge (LLM) makes ALL decisions
        - The TRM watches, listens, and learns from the Teacher's decisions
        - Over time, as the TRM learns, it can eventually graduate to making decisions
          (but that's a future milestone - for now it's purely observational)
        
        Returns: {"tool": "name", "args": {...}, "method": "...", ...}
        """
        trm_confidence = 0.0
        trm_prediction = None
        
        # 1. Let TRM OBSERVE (but NOT decide) - record what it would have predicted
        #    This is purely for learning - we want to compare TRM's guess vs Teacher's decision
        if self._trm and self._trm.is_ready():
            try:
                result = self._trm.select_tool(raw_input)
                trm_confidence = result.get("confidence", 0)
                trm_prediction = result.get("tool")
                
                # Log what TRM would have done (for training comparison)
                print(f"[Tool TRM] Observing - would predict: {trm_prediction} (conf: {trm_confidence:.2f})", flush=True)
                
                # TRM is an INFANT - it does NOT get to make decisions yet
                # It only watches the Teacher and learns
                
            except Exception as e:
                print(f"[Tool TRM] Observation error: {e}", flush=True)
        
        # 2. Ask LLM Teacher - THE TEACHER MAKES ALL DECISIONS
        if self._llm_teacher_enabled:
            teacher_result = await self._ask_llm_teacher(raw_input)
            
            if teacher_result and teacher_result.get("tool"):
                teacher_tool = teacher_result["tool"]
                
                # Parse the Teacher's output (not Dexter's!)
                parsed = self._parse_any_format(teacher_result)
                
                # Record for TRM training - TRM LEARNS from this!
                # This is how the infant TRM builds up its knowledge
                self._record_teacher_decision(
                    intent=raw_input,
                    trm_prediction=trm_prediction,
                    trm_confidence=trm_confidence,
                    llm_decision=teacher_tool,
                )
                
                # Log what TRM learned (comparison)
                if trm_prediction:
                    if trm_prediction == teacher_tool:
                        print(f"[Tool TRM] ✓ Correct prediction! Learning reinforced.", flush=True)
                    else:
                        print(f"[Tool TRM] ✗ Teacher chose {teacher_tool}, TRM guessed {trm_prediction}. Learning...", flush=True)
                else:
                    print(f"[Tool TRM] Learning new pattern: '{raw_input[:50]}...' -> {teacher_tool}", flush=True)
                
                # Log conversation
                self._teacher_context.add_turn("tool_agent", f"Selected: {teacher_tool}")
                
                return {
                    "tool": parsed.tool,
                    "args": parsed.args,
                    "method": "llm_teacher",
                    "trm_confidence": trm_confidence,
                    "trm_prediction": trm_prediction,
                    "llm_used": True,
                }
        
        # 3. Heuristic fallback (if Teacher is disabled/failed)
        tool_name, args = self._heuristic_match(raw_input, {})
        if tool_name:
            # Still record for TRM learning
            self._record_teacher_decision(
                intent=raw_input,
                trm_prediction=trm_prediction,
                trm_confidence=trm_confidence,
                llm_decision=tool_name,
            )
            return {
                "tool": tool_name,
                "args": args,
                "method": "heuristic",
                "trm_confidence": trm_confidence,
                "trm_prediction": trm_prediction,
                "llm_used": False,
            }
        
        # 4. No match - will trigger skill forge downstream
        return {
            "tool": None,
            "args": {},
            "method": "no_match",
            "trm_confidence": trm_confidence,
            "trm_prediction": trm_prediction,
            "llm_used": False,
        }
    
    def _parse_any_format(self, request: Union[ToolRequest, Dict, str]) -> ToolRequest:
        """
        Parse the Forge's (or TRM's) structured output into a ToolRequest.
        
        IMPORTANT: This parses the TEACHER's output, NOT Dexter's raw input.
        Dexter talks to the Teacher in natural language.
        The Teacher outputs structured calls that this parser understands.
        
        Handles:
        - ToolRequest object (pass through)
        - Dict with various key names (tool, function, action, etc.)
        - String with JSON embedded
        - OpenAI function_call format
        - Anthropic tool_use format
        """
        import re
        
        # Already a ToolRequest
        if isinstance(request, ToolRequest):
            return request
        
        # Dict - normalize various key names
        if isinstance(request, dict):
            # Extract intent from various possible keys
            intent = (
                request.get("intent") or 
                request.get("query") or 
                request.get("message") or 
                request.get("input") or 
                request.get("text") or 
                request.get("prompt") or
                ""
            )
            
            # Extract tool name from various possible keys
            tool = (
                request.get("tool") or
                request.get("function") or
                request.get("name") or
                request.get("tool_name") or
                request.get("function_name") or
                request.get("action") or
                None
            )
            
            # Extract args from various possible keys  
            args = {}
            for key in ["args", "arguments", "parameters", "params"]:
                if key in request and isinstance(request[key], dict):
                    args = request[key]
                    break
                elif key in request and isinstance(request[key], str):
                    try:
                        args = json.loads(request[key])
                        break
                    except:
                        pass
            if "input" in request and isinstance(request["input"], dict):
                args = request["input"]
            
            # Handle OpenAI function_call format
            if "function_call" in request:
                fc = request["function_call"]
                if isinstance(fc, dict):
                    tool = fc.get("name", tool)
                    try:
                        args = json.loads(fc.get("arguments", "{}"))
                    except:
                        args = {}
            
            # Handle Anthropic tool_use format
            if "tool_use" in request:
                tu = request["tool_use"]
                if isinstance(tu, dict):
                    tool = tu.get("name", tool)
                    args = tu.get("input", args)
            
            return ToolRequest(
                intent=intent,
                tool=tool,
                args=args,
                context=request.get("context", {}),
                priority=request.get("priority", "normal"),
                source=request.get("source", "dexter"),
            )
        
        # String - parse various formats
        if isinstance(request, str):
            text = request.strip()
            intent = text
            tool = None
            args = {}
            
            # Try to parse entire string as JSON first
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return self._parse_any_format(parsed)
                except:
                    pass
            
            # Try to extract JSON from the string
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # Markdown JSON block
                r'```\s*(\{.*?\})\s*```',  # Generic code block with JSON
                r'(\{[^{}]*"(?:function|tool|name|action)"[^{}]*\})',  # Embedded JSON with tool key
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    try:
                        json_str = match.group(1) if match.lastindex else match.group(0)
                        parsed = json.loads(json_str)
                        # Recursively parse the extracted dict
                        return self._parse_any_format(parsed)
                    except:
                        pass
            
            # Try XML/tag format: <tool name="...">args</tool>
            xml_match = re.search(
                r'<(?:tool|function|call)\s+(?:name|function)=["\']([^"\']+)["\']>([^<]*)</(?:tool|function|call)>',
                text, re.IGNORECASE
            )
            if xml_match:
                tool = xml_match.group(1)
                arg_text = xml_match.group(2).strip()
                # Try to parse args as JSON
                try:
                    args = json.loads(arg_text)
                except:
                    if arg_text:
                        args = {"input": arg_text}
                intent = f"call {tool}"
            
            # Try markdown tool block: ```tool\nshell.run notepad\n```
            md_match = re.search(r'```(?:tool|function)\s*\n([^\n]+)(?:\n([^`]*))?```', text, re.IGNORECASE)
            if md_match:
                tool_line = md_match.group(1).strip()
                arg_text = (md_match.group(2) or "").strip()
                
                # Parse tool line: "tool_name arg1 arg2" or "tool_name"
                parts = tool_line.split(None, 1)
                tool = parts[0] if parts else None
                if len(parts) > 1:
                    try:
                        args = json.loads(parts[1])
                    except:
                        args = {"input": parts[1]}
                intent = f"call {tool}"
            
            # Try function_call: prefix
            fc_match = re.search(r'function_call:\s*(\{.*\})', text, re.DOTALL | re.IGNORECASE)
            if fc_match:
                try:
                    parsed = json.loads(fc_match.group(1))
                    return self._parse_any_format({"function_call": parsed})
                except:
                    pass
            
            # If no structured format found, treat as natural language intent
            return ToolRequest(intent=intent, tool=tool, args=args)
        
        # Fallback
        return ToolRequest(intent=str(request))
    
    def _refresh_registry(self):
        """Refresh the skill registry and update teacher context."""
        now = time.time()
        if now - self._last_registry_refresh < self._registry_ttl:
            return
        
        self._skill_registry = self._filter_registry(tool_executor._build_registry())
        self._last_registry_refresh = now
        
        # Update teacher context with available skills
        self._teacher_context.available_skills = list(self._skill_registry.keys())
        
        # Load skill docstrings for better teacher context
        self._load_skill_docs()
        self._teacher_context.skill_descriptions = self._skill_docs

        # Persist tool inventory for Forge's long-lived context
        try:
            inventory = sorted(self._skill_registry.keys())
            self._queue_forge_artifact(
                source="system",
                artifact_type="tool_inventory",
                payload={"count": len(inventory), "tools": inventory},
                confidence=0.8,
                priority=6,
                metadata={"event": "registry_refresh"},
            )
        except Exception:
            pass
    
    def _load_skill_docs(self):
        """Load docstrings for all skills."""
        for tool_name in self._skill_registry:
            if tool_name in self._skill_docs:
                continue
            try:
                func = self._skill_registry[tool_name]
                doc = func.__doc__ or ""
                self._skill_docs[tool_name] = doc.split("\n")[0] if doc else ""
            except Exception:
                self._skill_docs[tool_name] = ""
    
    async def execute(self, request: Union[ToolRequest, Dict, str]) -> ToolResult:
        """
        Execute a tool request with full metadata tracking.
        
        Flow:
        1. Dexter sends raw input (any format - natural language, JSON, XML, whatever)
        2. The Forge LLM understands Dexter's intent (LLM-to-LLM, no parsing)
        3. Forge outputs a structured call
        4. Parser parses the Teacher's output
        5. Executor runs the parsed call
        
        The Main LLM never talks to the executor directly - it talks to the Teacher.
        
        Returns ToolResult with rich metadata for upstream learning.
        """
        self._stats["total_requests"] += 1
        started = time.monotonic()

        # Capture raw input from Dexter (DO NOT parse it - Teacher will understand it)
        base_raw_input = self._extract_raw_input(request)

        # Refresh registry if stale
        self._refresh_registry()

        max_retries = self._max_retries if self._auto_retry_enabled else 0
        last_error: Optional[str] = None

        for attempt in range(max_retries + 1):
            selection_started = time.monotonic()
            meta = ExecutionMetadata()
            meta.retries = attempt

            attempt_input = base_raw_input
            if last_error:
                attempt_input = (
                    f"{base_raw_input}\n"
                    f"Previous error: {last_error}\n"
                    f"Retry attempt {attempt + 1}: fix the tool choice or arguments."
                )

            # Add request to teacher conversation history
            self._teacher_context.add_turn("dexter", attempt_input)

            # Ask the Forge (or TRM) to understand and structure the call
            resolve_result = await self._resolve_via_teacher(attempt_input)

            # NOW parse the Teacher's structured output
            tool_name = resolve_result.get("tool")
            args = resolve_result.get("args", {})
            skill_created = None
            forge_duration = 0

            # Access control: never allow Forge to call disallowed tools even if it guesses one.
            if tool_name and not self._tool_allowed(tool_name):
                minimal_request = ToolRequest(intent=base_raw_input)
                meta.selection_method = resolve_result.get("method", "teacher")
                meta.llm_teacher_used = resolve_result.get("llm_used", False)
                meta.total_duration_ms = int((time.monotonic() - started) * 1000)
                result = ToolResult(
                    ok=False,
                    error=f"Tool '{tool_name}' is blocked by tool_access policy",
                    metadata=meta,
                    duration_ms=meta.total_duration_ms,
                    execution_path="blocked",
                )
                await self._notify_result(minimal_request, result)
                return result

            meta.selection_method = resolve_result.get("method", "teacher")
            meta.trm_confidence = resolve_result.get("trm_confidence", 0)
            meta.trm_prediction = resolve_result.get("trm_prediction")
            meta.llm_teacher_used = resolve_result.get("llm_used", False)

            # Update stats - note: TRM is observer-only, so "trm_selected" won't happen
            if meta.selection_method == "llm_teacher":
                self._stats["llm_teacher_selections"] += 1
            elif meta.selection_method == "heuristic":
                self._stats["heuristic_selections"] += 1

            # Track TRM observation accuracy (did it predict correctly?)
            if meta.trm_prediction:
                self._stats.setdefault("trm_observations", 0)
                self._stats["trm_observations"] += 1
                if meta.trm_prediction == resolve_result.get("tool"):
                    self._stats.setdefault("trm_correct_predictions", 0)
                    self._stats["trm_correct_predictions"] += 1

            try:
                # Path: No skill found - forge one
                if not tool_name and self._forge_on_miss:
                    forge_start = time.monotonic()
                    tool_name, skill_created = await self._forge_skill(attempt_input)
                    forge_duration = int((time.monotonic() - forge_start) * 1000)

                    if tool_name:
                        meta.selection_method = "forge"
                        meta.skill_forged = True
                        meta.skill_forged_name = skill_created
                        meta.skill_forge_duration_ms = forge_duration
                        self._stats["skills_forged"] += 1

                        # Log skill creation for TRM training
                        self._log_skill_creation(attempt_input, tool_name, skill_created)

                meta.selection_duration_ms = int((time.monotonic() - selection_started) * 1000)

                if not tool_name:
                    if attempt < max_retries:
                        last_error = f"No skill found for: {base_raw_input}"
                        if self._retry_backoff_sec > 0:
                            await asyncio.sleep(self._retry_backoff_sec)
                        continue
                    minimal_request = ToolRequest(intent=base_raw_input)
                    result = ToolResult(
                        ok=False,
                        error=f"No skill found for: {base_raw_input}",
                        metadata=meta,
                        duration_ms=int((time.monotonic() - started) * 1000),
                        execution_path="no_skill",
                    )
                    await self._notify_result(minimal_request, result)
                    return result

                # Build request for callbacks
                final_request = ToolRequest(intent=base_raw_input, tool=tool_name, args=args)
                meta.final_args = args

                # Execute the tool
                exec_started = time.monotonic()
                print(f"[Tool TRM Gateway] Executing: {tool_name}({args})", flush=True)
                result = await self._executor.execute(tool_name, args)
                meta.execution_duration_ms = int((time.monotonic() - exec_started) * 1000)

                # Check for missing dependencies
                if not result.get("ok") and "ModuleNotFoundError" in str(result.get("error", "")):
                    dep_start = time.monotonic()
                    dep_result = await self._handle_missing_dependency(result, tool_name, args)
                    if dep_result:
                        meta.deps_installed = dep_result.get("installed", [])
                        meta.deps_install_duration_ms = int((time.monotonic() - dep_start) * 1000)
                        result = dep_result.get("result", result)
                        self._stats["deps_installed"] += len(meta.deps_installed)

                # Record execution for training
                self._record_execution(final_request, tool_name, args, result, meta.selection_method)

                # Update teacher context with result
                self._teacher_context.last_tool_result = {"ok": result.get("ok"), "tool": tool_name}
                if not result.get("ok"):
                    self._teacher_context.last_error = result.get("error", "")
                self._teacher_context.add_turn("tool_agent", f"{tool_name} -> {'OK' if result.get('ok') else 'FAILED'}")

                # Ingest result into TRM carry state immediately
                if self._trm and self._trm.is_ready():
                    try:
                        ingest_metadata = {
                            "skill_forged": meta.skill_forged,
                            "skill_forged_name": meta.skill_forged_name,
                            "retries": meta.retries,
                        }
                        self._trm.ingest_result(
                            task=base_raw_input,
                            tool_used=tool_name,
                            success=result.get("ok", False),
                            error=result.get("error"),
                            metadata=ingest_metadata,
                        )
                    except Exception as e:
                        print(f"[Gateway] TRM ingest error: {e}", flush=True)

                meta.total_duration_ms = int((time.monotonic() - started) * 1000)

                if not result.get("ok") and attempt < max_retries:
                    last_error = str(result.get("error") or "unknown error")
                    if self._retry_backoff_sec > 0:
                        await asyncio.sleep(self._retry_backoff_sec)
                    continue

                # Build final result
                if result.get("ok"):
                    self._stats["successful"] += 1
                    final_result = ToolResult(
                        ok=True,
                        result=result.get("result"),
                        tool_used=tool_name,
                        arguments_used=args,
                        metadata=meta,
                        skill_created=skill_created,
                        dependencies_installed=meta.deps_installed,
                        duration_ms=meta.total_duration_ms,
                        execution_path=meta.selection_method,
                    )
                else:
                    self._stats["failed"] += 1
                    final_result = ToolResult(
                        ok=False,
                        error=result.get("error"),
                        tool_used=tool_name,
                        arguments_used=args,
                        metadata=meta,
                        skill_created=skill_created,
                        dependencies_installed=meta.deps_installed,
                        duration_ms=meta.total_duration_ms,
                        execution_path=meta.selection_method,
                    )

                # Notify upstream immediately
                await self._notify_result(final_request, final_result)
                return final_result

            except Exception as e:
                if attempt < max_retries:
                    last_error = str(e)
                    if self._retry_backoff_sec > 0:
                        await asyncio.sleep(self._retry_backoff_sec)
                    continue
                self._stats["failed"] += 1
                meta.total_duration_ms = int((time.monotonic() - started) * 1000)
                minimal_request = ToolRequest(intent=base_raw_input)
                result = ToolResult(
                    ok=False,
                    error=str(e),
                    metadata=meta,
                    duration_ms=meta.total_duration_ms,
                    execution_path="exception",
                )
                await self._notify_result(minimal_request, result)
                return result

        # Should never reach here
        minimal_request = ToolRequest(intent=base_raw_input)
        fallback = ToolResult(
            ok=False,
            error="Tool gateway retry loop exhausted without a result",
            metadata=ExecutionMetadata(),
            duration_ms=int((time.monotonic() - started) * 1000),
            execution_path="retry_exhausted",
        )
        await self._notify_result(minimal_request, fallback)
        return fallback
    
    def _log_skill_creation(self, intent: str, tool_name: str, skill_id: Optional[str]):
        """Log skill creation for TRM training on skill forging."""
        try:
            log_path = BASE_DIR / "artifacts" / "trm_training" / "skill_creations.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                record = {
                    "timestamp": time.time(),
                    "intent": intent,
                    "tool_created": tool_name,
                    "skill_id": skill_id,
                }
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
    
    async def _ask_llm_teacher(
        self, 
        raw_input: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Send Dexter's raw input to the Forge LLM.
        
        The Teacher is an LLM - it understands Dexter's intent naturally.
        No parsing of Dexter's input needed. The Teacher outputs structured calls.
        
        Context bundle is injected with: mission, skills, recent results.
        """
        # Lazy load tool agent
        if self._tool_agent is None:
            try:
                from core.tool_agent import ToolAgent
                self._tool_agent = ToolAgent(self.config)
            except ImportError:
                return None
        
        # Build Forge's context bundle (mission, skills, results)
        teacher_bundle = self._build_teacher_context_bundle(raw_input)
        
        # Build prompt with injected context bundle
        prompt = f"""You are Forge, the execution brain for Dexter AI. You convert intent into LOCAL TOOL calls.

You do not chat with the user. You ONLY output structured tool selection JSON.

[CONTEXT BUNDLE - INJECTED]
{teacher_bundle}
[END CONTEXT BUNDLE]

Rules:
- You MUST choose a tool name EXACTLY as shown in "Available Tools".
- Do NOT invent tool names.
- Prefer safe, local tools (shell/powershell/file ops) over network/browser/agent tools unless explicitly requested.
- If the user asks to run a terminal command (e.g. dir, ls, cd, pip, python), use shell.run with args: {{"command": "..."}}.
- Output MUST be ONLY valid JSON, no prose, no markdown.

Return JSON:
{{"tool": "tool_name", "args": {{ ... }} }}
Or:
{{"tool": null, "reason": "..."}}"""

        try:
            response = await self._tool_agent.quick_query(
                prompt,
                slot=self._llm_teacher_slot,
                max_tokens=200,
            )
            
            # Parse JSON from response (robust)
            from core.utils import extract_json
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            print(f"[Gateway] LLM Teacher query error: {e}", flush=True)
        
        return None
    
    def _record_teacher_decision(
        self,
        intent: str,
        trm_prediction: Optional[str],
        trm_confidence: float,
        llm_decision: str,
    ):
        """
        Record teacher's decision - THIS IS HOW THE TRM LEARNS.
        
        Every time the Teacher makes a decision, we:
        1. Log it to disk for batch training later
        2. Feed it to the online trainer for immediate weight updates
        3. Let the TRM ingest it into its carry state
        
        The TRM is an infant observer - this is how it builds knowledge.
        """
        record = {
            "timestamp": time.time(),
            "intent": intent,
            "trm_prediction": trm_prediction,
            "trm_confidence": trm_confidence,
            "teacher_decision": llm_decision,
            "trm_was_correct": trm_prediction == llm_decision,
        }
        
        # 1. Log to training file (for batch training)
        try:
            log_path = BASE_DIR / "artifacts" / "trm_training" / "teacher_decisions.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
        
        # 2. Feed to online trainer for immediate learning
        trainer = _get_online_trainer()
        if trainer:
            try:
                # Format: intent -> tool (this is what TRM should learn)
                trainer.observe_teacher_decision(intent, llm_decision, record)
            except Exception as e:
                print(f"[Tool TRM] Online training error: {e}", flush=True)
        
        # 3. Let TRM ingest (if it has carry state)
        if self._trm and self._trm.is_ready():
            try:
                # TRM ingests the Teacher's decision as a learning example
                self._trm.ingest(
                    f"<LEARN> Intent: {intent} -> Tool: {llm_decision}",
                    metadata={
                        "source": "teacher",
                        "my_prediction": trm_prediction,
                        "was_correct": trm_prediction == llm_decision,
                    }
                )
            except Exception:
                pass  # TRM may not have ingest method yet
    
    def _heuristic_match(
        self, 
        intent: str, 
        provided_args: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Simple heuristic matching for common intents."""
        intent_lower = intent.lower()
        args = dict(provided_args)
        
        # Shell/command execution
        if any(kw in intent_lower for kw in ["dir", "ls", "list files", "list directory"]):
            if "shell.run" in self._skill_registry:
                args["command"] = "dir"
                return "shell.run", args

        if any(kw in intent_lower for kw in ["run", "execute", "open", "launch", "start"]):
            # Extract the command
            for prefix in ["run ", "execute ", "open ", "launch ", "start "]:
                if intent_lower.startswith(prefix):
                    cmd = intent[len(prefix):].strip()
                    if cmd:
                        args["command"] = cmd
                        # Prefer shell.run (this repo's shell tool)
                        if "shell.run" in self._skill_registry:
                            return "shell.run", args
                        if "powershell.run" in self._skill_registry:
                            return "powershell.run", args
                        if "shell.run_command" in self._skill_registry:
                            return "shell.run_command", args
        
        # File operations
        if any(kw in intent_lower for kw in ["read file", "write file", "create file", "delete file"]):
            if "file_system.read_file" in self._skill_registry and "read" in intent_lower:
                return "file_system.read_file", args
            if "file_system.write_file" in self._skill_registry and "write" in intent_lower:
                return "file_system.write_file", args
        
        # System info
        if any(kw in intent_lower for kw in ["cpu", "memory", "disk", "system"]):
            if "system_ops.get_system_info" in self._skill_registry:
                return "system_ops.get_system_info", args
        
        # Memory operations
        if any(kw in intent_lower for kw in ["remember", "recall", "memory", "store"]):
            if "memory_ops.store_memory" in self._skill_registry and any(kw in intent_lower for kw in ["remember", "store"]):
                return "memory_ops.store_memory", args
            if "memory_ops.recall_memory" in self._skill_registry and "recall" in intent_lower:
                return "memory_ops.recall_memory", args
        
        # Keyword search in tool names
        for tool_name in self._skill_registry:
            tool_lower = tool_name.lower().replace("_", " ").replace(".", " ")
            # Check if any significant word from intent matches tool
            intent_words = set(intent_lower.split())
            tool_words = set(tool_lower.split())
            if len(intent_words & tool_words) >= 1:
                return tool_name, args
        
        return None, args
    
    async def _forge_skill(self, intent: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Forge a new skill to handle an intent.
        Returns: (tool_name, skill_file_created)
        """
        if not self._forge_enabled:
            return None, None
        
        print(f"[Tool TRM Gateway] Forging new skill for: {intent}", flush=True)
        
        try:
            from core.skill_forge import SkillForge
            forge = SkillForge(self.config)
            
            # Generate skill name from intent
            words = intent.lower().split()[:3]
            skill_name = "_".join(w for w in words if w.isalnum())[:20] + "_ops"
            if len(skill_name) < 5:
                skill_name = f"skill_{int(time.time())}_ops"
            
            result = await forge.forge_tool(intent, skill_name)
            
            if result.get("ok"):
                skill_id = result.get("skill_id")
                # Refresh registry to include new skill
                self._skill_registry = tool_executor._build_registry()
                self._last_registry_refresh = time.time()
                
                # Find a function in the new skill
                for tool_name in self._skill_registry:
                    if tool_name.startswith(skill_name):
                        print(f"[Tool TRM Gateway] Forged: {tool_name}", flush=True)
                        return tool_name, skill_id
                
                return None, skill_id
            else:
                print(f"[Tool TRM Gateway] Forge failed: {result.get('error')}", flush=True)
                return None, None
                
        except Exception as e:
            print(f"[Tool TRM Gateway] Forge error: {e}", flush=True)
            return None, None
    
    async def _handle_missing_dependency(
        self,
        result: Dict[str, Any],
        tool_name: str,
        args: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Handle missing module errors by installing dependencies."""
        if not self._auto_install:
            return None
        
        error = str(result.get("error", ""))
        
        # Extract module name from error
        import re
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error)
        if not match:
            return None
        
        module_name = match.group(1).split(".")[0]
        print(f"[Tool TRM Gateway] Auto-installing: {module_name}", flush=True)
        
        try:
            from core.dependency_installer import install_for_missing_modules
            install_result = await asyncio.to_thread(install_for_missing_modules, [module_name], set(), set())
            
            if install_result.get("success"):
                # Retry execution
                retry_result = await self._executor.execute(tool_name, args)
                return {
                    "installed": [module_name],
                    "result": retry_result,
                }
        except Exception as e:
            print(f"[Tool TRM Gateway] Install failed: {e}", flush=True)
        
        return None
    
    def _record_execution(
        self,
        request: ToolRequest,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
        execution_path: str,
    ):
        """Record execution for learning and feed to online trainer."""
        record = {
            "timestamp": time.time(),
            "intent": request.intent,
            "tool": tool_name,
            "args": args,
            "success": result.get("ok", False),
            "error": result.get("error"),
            "path": execution_path,
            "duration_ms": result.get("duration_ms", 0),
        }
        
        self._execution_history.append(record)
        if len(self._execution_history) > self._max_history:
            self._execution_history = self._execution_history[-self._max_history:]
        
        # Feed to online trainer for real-time learning
        trainer = _get_online_trainer()
        if trainer:
            trainer.record_execution(
                intent=request.intent,
                tool_name=tool_name,
                arguments=args,
                success=result.get("ok", False),
                context=request.context,
            )
    
    async def _on_execution_complete(self, meta: Dict[str, Any], result: Dict[str, Any]):
        """Callback when execution completes."""
        # Can be used for additional monitoring/learning
        pass
    
    # ----- Convenience Methods -----
    
    async def run(self, intent: str, **kwargs) -> ToolResult:
        """Shorthand: execute with natural language intent."""
        return await self.execute(ToolRequest(intent=intent, args=kwargs))
    
    async def call(self, tool: str, **kwargs) -> ToolResult:
        """Shorthand: execute a specific tool."""
        return await self.execute(ToolRequest(intent="", tool=tool, args=kwargs))
    
    def get_available_skills(self) -> List[str]:
        """Get list of available skills."""
        self._refresh_registry()
        return list(self._skill_registry.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return dict(self._stats)


# Global instance
_gateway: Optional[ToolTRMGateway] = None


def get_gateway(config: Optional[Dict[str, Any]] = None) -> ToolTRMGateway:
    """Get or create the global Tool TRM Gateway."""
    global _gateway
    if _gateway is None:
        if config is None:
            config = {}
        _gateway = ToolTRMGateway(config)
    return _gateway


async def execute(request: Union[ToolRequest, Dict, str], config: Optional[Dict] = None) -> ToolResult:
    """Execute a tool request via the gateway."""
    gateway = get_gateway(config)
    if not gateway._running:
        await gateway.start()
    return await gateway.execute(request)
