import asyncio
import time
import platform
import json
import hashlib
import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

BUNDLE_PERSIST_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "context_bundles"
)


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }


class RollingContextBundle:
    def __init__(
        self,
        max_turns: int = 10,
        max_decisions: int = 20,
        user_name: str = "Jeffrey Gliksman"
    ):
        self.max_turns = max_turns
        self.max_decisions = max_decisions
        self.user_name = user_name
        self._lock = asyncio.Lock()
        self._bundle = self._initialize_bundle()
        self._persist_dir = BUNDLE_PERSIST_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._bundle_path = self._persist_dir / "rolling_bundle.json"
        self._load_bundle()
    
    def _initialize_bundle(self) -> Dict[str, Any]:
        return {
            "recent_conversation": [],
            "current_directory": str(Path.cwd()),
            "operating_system": platform.system(),
            "current_project": None,
            "current_task": None,
            "working_directory": str(Path.cwd()),
            "available_tools": [],
            "available_skills": [],
            "enabled_llms": [],
            "user_name": self.user_name,
            "user_preferences": {
                "response_style": "direct",
                "verbosity": "concise"
            },
            "user_mode": "casual",
            "pending_tool_calls": {},
            "recent_decisions": [],
            "confidence_trend": [],
            "last_action": None,
            "last_action_time": 0.0,
            "mission": "Care for Jeffrey Gliksman and ensure his health, safety, happiness, and prosperity.",
            "core_capabilities": [
                "Execute tools asynchronously",
                "Query knowledge graph",
                "Plan multi-step tasks",
                "Learn from experience",
                "Manage multiple concurrent operations"
            ],
            "safety_constraints": [
                "Don't execute harmful commands",
                "Ask before making changes to user files",
                "Protect user privacy and data",
                "Verify before destructive actions"
            ],
            "session_start": time.time(),
            "turn_count": 0,
            "errors_encountered": 0
        }
    
    async def get(self) -> Dict[str, Any]:
        async with self._lock:
            return self._bundle.copy()
    
    async def get_fast_subset(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "user_input": self._get_last_user_input(),
                "current_directory": self._bundle["current_directory"],
                "current_task": self._bundle["current_task"],
                "user_mode": self._bundle["user_mode"],
                "available_tools": self._bundle["available_tools"],
                "last_action": self._bundle["last_action"]
            }
    
    def _get_last_user_input(self) -> str:
        for turn in reversed(self._bundle["recent_conversation"]):
            if turn["role"] == "user":
                return turn["content"]
        return ""
    
    async def add_turn(self, role: str, content: str):
        async with self._lock:
            turn = ConversationTurn(
                role=role,
                content=content,
                timestamp=time.time()
            )
            
            self._bundle["recent_conversation"].append(turn.to_dict())
            
            if len(self._bundle["recent_conversation"]) > self.max_turns:
                self._bundle["recent_conversation"] = \
                    self._bundle["recent_conversation"][-self.max_turns:]
            
            if role == "user":
                self._bundle["turn_count"] += 1
            self._schedule_save(copy.deepcopy(self._bundle))
    
    async def update_state(self, updates: Dict[str, Any]):
        async with self._lock:
            for key, value in updates.items():
                if key in self._bundle:
                    self._bundle[key] = value
            self._schedule_save(copy.deepcopy(self._bundle))
    
    async def set_current_task(self, task: str):
        async with self._lock:
            self._bundle["current_task"] = task
            self._schedule_save(copy.deepcopy(self._bundle))
    
    async def set_current_project(self, project: str):
        async with self._lock:
            self._bundle["current_project"] = project
            self._schedule_save(copy.deepcopy(self._bundle))
    
    async def set_working_directory(self, path: str):
        async with self._lock:
            self._bundle["working_directory"] = path
            self._bundle["current_directory"] = path
            self._schedule_save(copy.deepcopy(self._bundle))
    
    async def add_capability(self, capability: str, capability_type: str):
        async with self._lock:
            if capability_type == "tool":
                if capability not in self._bundle["available_tools"]:
                    self._bundle["available_tools"].append(capability)
            elif capability_type == "skill":
                if capability not in self._bundle["available_skills"]:
                    self._bundle["available_skills"].append(capability)
            self._schedule_save(copy.deepcopy(self._bundle))
    
    async def set_enabled_llms(self, llms: List[str]):
        async with self._lock:
            self._bundle["enabled_llms"] = llms
            self._schedule_save(copy.deepcopy(self._bundle))
    
    async def add_pending_tool_call(self, call_id: str, tool_name: str, arguments: Dict[str, Any]):
        async with self._lock:
            self._bundle["pending_tool_calls"][call_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "status": "pending",
                "timestamp": time.time()
            }
    
    async def update_tool_call_status(
        self,
        call_id: str,
        status: str,
        result: Optional[Any] = None
    ):
        pending_decision: Optional[Dict[str, Any]] = None
        async with self._lock:
            if call_id in self._bundle["pending_tool_calls"]:
                self._bundle["pending_tool_calls"][call_id]["status"] = status
                self._bundle["pending_tool_calls"][call_id]["result"] = result
                if status in ("completed", "failed"):
                    tool_call = self._bundle["pending_tool_calls"].pop(call_id)
                    pending_decision = {
                        "type": "tool_call",
                        "tool_name": tool_call["tool_name"],
                        "status": status,
                        "timestamp": time.time()
                    }
            snapshot = copy.deepcopy(self._bundle)
        if pending_decision:
            await self.add_recent_decision(pending_decision)
        else:
            self._schedule_save(snapshot)
    
    async def set_last_action(self, action: str):
        async with self._lock:
            self._bundle["last_action"] = action
            self._bundle["last_action_time"] = time.time()
    
    async def add_recent_decision(self, decision: Dict[str, Any]):
        async with self._lock:
            self._bundle["recent_decisions"].append(decision)
            
            if len(self._bundle["recent_decisions"]) > self.max_decisions:
                self._bundle["recent_decisions"] = \
                    self._bundle["recent_decisions"][-self.max_decisions:]
            self._schedule_save(copy.deepcopy(self._bundle))
    
    async def record_confidence(self, confidence: float):
        async with self._lock:
            self._bundle["confidence_trend"].append({
                "confidence": confidence,
                "timestamp": time.time()
            })
            
            if len(self._bundle["confidence_trend"]) > 1000:
                self._bundle["confidence_trend"] = \
                    self._bundle["confidence_trend"][-1000:]
            self._schedule_save(copy.deepcopy(self._bundle))
    
    async def update_from_deep_context(self, deep_context: Dict[str, Any]):
        updates = {}
        
        if "critical_facts" in deep_context:
            updates["recent_facts"] = deep_context["critical_facts"][:5]
        
        if "enabled_llms" in deep_context:
            updates["enabled_llms"] = deep_context["enabled_llms"]
        
        if "new_tools" in deep_context:
            new_tools = deep_context["new_tools"]
            current_tools = set(self._bundle["available_tools"])
            updates["available_tools"] = list(current_tools | set(new_tools))
        
        if "inferred_user_mode" in deep_context:
            updates["user_mode"] = deep_context["inferred_user_mode"]
        
        if updates:
            await self.update_state(updates)
    
    async def get_context_summary(self) -> str:
        async with self._lock:
            parts = []
            
            if self._bundle["current_task"]:
                parts.append(f"Task: {self._bundle['current_task']}")
            if self._bundle["current_project"]:
                parts.append(f"Project: {self._bundle['current_project']}")
            parts.append(f"Directory: {self._bundle['current_directory']}")
            parts.append(f"Turns: {self._bundle['turn_count']}")
            parts.append(f"Mode: {self._bundle['user_mode']}")
            parts.append(f"Tools: {len(self._bundle['available_tools'])} available")
            parts.append(f"LLMs: {len(self._bundle['enabled_llms'])} enabled")
            
            pending = len(self._bundle["pending_tool_calls"])
            if pending > 0:
                parts.append(f"Pending: {pending} tool calls")
            
            return " | ".join(parts)
    
    async def reset(self):
        async with self._lock:
            self._bundle = self._initialize_bundle()
            self._save_bundle()

    def _load_bundle(self):
        if not self._bundle_path.exists():
            return
        try:
            with self._bundle_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._bundle.update(data)
        except Exception as exc:
            print(f"[RollingBundle] Could not load persisted bundle: {exc}", flush=True)

    def _save_bundle(self):
        try:
            with self._bundle_path.open("w", encoding="utf-8") as fh:
                json.dump(self._bundle, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[RollingBundle] Could not persist bundle: {exc}", flush=True)

    async def _save_bundle_async(self, snapshot: Dict[str, Any]):
        await asyncio.to_thread(self._write_bundle_snapshot, snapshot)

    def _write_bundle_snapshot(self, snapshot: Dict[str, Any]):
        try:
            with self._bundle_path.open("w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[RollingBundle] Could not persist bundle: {exc}", flush=True)

    def _schedule_save(self, snapshot: Dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._write_bundle_snapshot(snapshot)
            return
        loop.create_task(self._save_bundle_async(snapshot))


class StagedContextBundle:
    """
    A staging area for background context updates.

    Background modules should publish structured artifacts (facts/triples/plans/risks)
    with timestamps + confidence. This bundle stores them, dedupes them, and applies a
    deterministic prune policy. Staged content accumulates until a TRIGGER occurs:
    - User input arrives
    - Tool result returns

    On trigger, staged content is packaged into an injection bundle for the next LLM call.
    """

    def __init__(self, main_bundle: RollingContextBundle):
        self.main_bundle = main_bundle
        self._lock = asyncio.Lock()
        self._persist_dir = (
            Path(__file__).resolve().parents[1] / "artifacts" / "context_bundles"
        )
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._bundle_path = self._persist_dir / "staged_bundle.json"

        # Staged artifacts by source
        self._staged: Dict[str, List[Dict[str, Any]]] = {
            "memory_trm": [],
            "reasoning_trm": [],
            "subconscious": [],
            "tool_results": [],
            "channel_messages": [],
        }

        # Token budgets per source (prevents domination). Tokens are approximated.
        self._token_budgets = {
            "memory_trm": 500,
            "reasoning_trm": 400,
            "subconscious": 600,
            "tool_results": 800,
            "channel_messages": 400,
        }

        self._load_staged()
        self._on_inject_callbacks: List[callable] = []
        self._total_staged = 0
        self._total_injected = 0
        self._save_staged()

    def _artifact_id(self, source: str, artifact_type: str, payload: Any) -> str:
        base = {"source": source, "type": artifact_type, "payload": payload}
        raw = json.dumps(base, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _estimate_tokens(self, artifact: Dict[str, Any]) -> int:
        try:
            raw = json.dumps(artifact, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            raw = str(artifact)
        return max(1, len(raw) // 4)

    def _sort_key(self, a: Dict[str, Any]):
        return (
            -int(a.get("priority", 5)),
            -float(a.get("confidence", 0.0)),
            -float(a.get("timestamp", 0.0)),
            str(a.get("id", "")),
        )

    async def stage(
        self,
        source: str,
        content: Any,
        priority: int = 5,
        metadata: Dict = None,
        artifact_type: Optional[str] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
    ):
        """Stage either raw text or a structured artifact for later injection."""
        if artifact_type is None and isinstance(content, dict):
            # Allow callers to pass a fully-formed artifact-like dict.
            raw_artifact = content
            artifact_type = raw_artifact.get("type") or raw_artifact.get("artifact_type")
            if confidence is None:
                confidence = raw_artifact.get("confidence")
            if timestamp is None:
                timestamp = raw_artifact.get("timestamp")
            if "payload" in raw_artifact:
                content = raw_artifact.get("payload")

        if artifact_type is None:
            artifact_type = "note"
        if confidence is None:
            confidence = 0.7

        await self.stage_artifact(
            source=source,
            artifact_type=str(artifact_type),
            payload=content,
            confidence=float(confidence),
            priority=int(priority),
            timestamp=timestamp,
            metadata=metadata,
        )

    async def stage_artifact(
        self,
        source: str,
        artifact_type: str,
        payload: Any,
        confidence: float = 0.7,
        priority: int = 5,
        timestamp: Optional[float] = None,
        metadata: Dict = None,
    ):
        # Precompute an approximate token estimate outside the lock.
        # (Keeps trimming fast and avoids heavy JSON dumps on the event loop.)
        try:
            tok_est = max(1, len(str(payload)) // 4)
        except Exception:
            tok_est = 1

        async with self._lock:
            if source not in self._staged:
                self._staged[source] = []

            ts = float(timestamp if timestamp is not None else time.time())
            conf = float(confidence if confidence is not None else 0.7)
            art_id = self._artifact_id(source, artifact_type, payload)

            artifact = {
                "id": art_id,
                "type": str(artifact_type),
                "payload": payload,
                "confidence": conf,
                "priority": int(priority),
                "timestamp": ts,
                "metadata": metadata or {},
                "_tokens": tok_est,
            }

            # Deterministic merge: replace only if strictly better/newer.
            items = self._staged[source]
            replaced = False
            for idx, existing in enumerate(items):
                if existing.get("id") == art_id:
                    if self._sort_key(existing) <= self._sort_key(artifact):
                        # Existing is better (or identical); keep it.
                        replaced = True
                        break
                    items[idx] = artifact
                    replaced = True
                    break
            if not replaced:
                items.append(artifact)

            self._total_staged += 1
            self._trim_to_budget(source)
            # Shallow snapshot of lists for persistence; avoids deepcopy under lock.
            snapshot = {k: list(v) for k, v in self._staged.items()}
        self._schedule_save_staged(snapshot)

    def _trim_to_budget(self, source: str):
        """Deterministic prune: sort by priority/confidence/recency/id, then keep within budget."""
        items = self._staged.get(source, [])
        budget = int(self._token_budgets.get(source, 500))

        items.sort(key=self._sort_key)

        total_tokens = 0
        kept: List[Dict[str, Any]] = []
        for item in items:
            item_tokens = int(item.get("_tokens") or self._estimate_tokens(item))
            if total_tokens + item_tokens <= budget:
                kept.append(item)
                total_tokens += item_tokens

        self._staged[source] = kept

    async def trigger_and_inject(self, trigger_type: str, trigger_content: str = "") -> Dict[str, Any]:
        async with self._lock:
            injection = {
                "trigger": trigger_type,
                "trigger_content": str(trigger_content)[:500],
                "timestamp": time.time(),
                "sources": {},
                "artifacts": [],
            }

            for source, items in self._staged.items():
                if not items:
                    continue
                items.sort(key=self._sort_key)
                injection["sources"][source] = {
                    "count": len(items),
                    "artifacts": list(items),
                }
                injection["artifacts"].extend(items)

            total_injected = sum(len(items) for items in self._staged.values())
            self._total_injected += total_injected
            for source in self._staged:
                self._staged[source] = []

            callbacks = list(self._on_inject_callbacks)

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(injection)
                else:
                    await asyncio.to_thread(callback, injection)
            except Exception as e:
                print(f"[StagedBundle] Inject callback error: {e}", flush=True)

        # Save a shallow snapshot; avoid deepcopy on the event loop.
        try:
            snapshot = {k: list(v) for k, v in self._staged.items()}
        except Exception:
            snapshot = {}
        self._schedule_save_staged(snapshot)
        return injection

    def _save_staged(self):
        try:
            raw = {
                "staged": self._staged,
                "token_budgets": self._token_budgets,
            }
            with self._bundle_path.open("w", encoding="utf-8") as fh:
                json.dump(raw, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[StagedBundle] Persist failed: {exc}", flush=True)

    async def _save_staged_async(self, snapshot: Dict[str, Any]):
        await asyncio.to_thread(self._write_staged_snapshot, snapshot)

    def _write_staged_snapshot(self, snapshot: Dict[str, Any]):
        try:
            raw = {
                "staged": snapshot,
                "token_budgets": self._token_budgets,
            }
            with self._bundle_path.open("w", encoding="utf-8") as fh:
                json.dump(raw, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[StagedBundle] Persist failed: {exc}", flush=True)

    def _schedule_save_staged(self, snapshot: Dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._write_staged_snapshot(snapshot)
            return
        loop.create_task(self._save_staged_async(snapshot))

    def _load_staged(self):
        if not self._bundle_path.exists():
            return
        try:
            with self._bundle_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            staged = data.get("staged")
            if isinstance(staged, dict):
                for key, items in staged.items():
                    if isinstance(items, list):
                        self._staged[key] = items[:]
            budgets = data.get("token_budgets")
            if isinstance(budgets, dict):
                for key, val in budgets.items():
                    try:
                        self._token_budgets[key] = int(val)
                    except Exception:
                        continue
        except Exception as exc:
            print(f"[StagedBundle] Could not load persisted data: {exc}", flush=True)

    def on_inject(self, callback: callable):
        self._on_inject_callbacks.append(callback)

    async def get_staged_summary(self) -> str:
        async with self._lock:
            parts = []
            for source, items in self._staged.items():
                if items:
                    parts.append(f"{source}: {len(items)}")
            return " | ".join(parts) if parts else "empty"

    def format_for_llm(self, injection: Dict[str, Any]) -> str:
        lines = ["[Background Context Update]", f"Trigger: {injection.get('trigger', '')}"]

        sources = injection.get("sources", {}) or {}
        for source in sorted(sources.keys()):
            data = sources[source] or {}
            items = list(data.get("artifacts") or [])
            if not items:
                continue

            lines.append(f"\n[{source.replace('_', ' ').title()}] ({len(items)} artifacts):")

            # Group by type for readability (deterministic)
            by_type: Dict[str, List[Dict[str, Any]]] = {}
            for a in items:
                by_type.setdefault(str(a.get("type") or "note"), []).append(a)
            for t in sorted(by_type.keys()):
                lines.append(f"- {t}:")
                by_type[t].sort(key=self._sort_key)
                for a in by_type[t][:10]:
                    payload = a.get("payload")
                    if isinstance(payload, (dict, list)):
                        payload_str = json.dumps(payload, ensure_ascii=False, default=str)
                    else:
                        payload_str = str(payload)
                    payload_str = payload_str.strip().replace("\n", " ")
                    lines.append(
                        f"  • (c={a.get('confidence', 0):.2f}) {payload_str[:500]}"
                    )

        return "\n".join(lines)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_staged": self._total_staged,
            "total_injected": self._total_injected,
            "currently_staged": sum(len(items) for items in self._staged.values()),
        }


_global_bundle: Optional[RollingContextBundle] = None
_global_staged: Optional[StagedContextBundle] = None


def get_global_bundle() -> RollingContextBundle:
    global _global_bundle
    if _global_bundle is None:
        _global_bundle = RollingContextBundle()
    return _global_bundle


def get_staged_bundle() -> StagedContextBundle:
    """Get the global staged context bundle."""
    global _global_staged
    if _global_staged is None:
        _global_staged = StagedContextBundle(get_global_bundle())
    return _global_staged
