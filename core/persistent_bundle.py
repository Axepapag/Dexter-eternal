import asyncio
import hashlib
import json
import time
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional


class PersistentArtifactBundle:
    """
    Persistent artifact bundle for a specific LLM (orchestrator / think_tank / forge).
    Stores structured artifacts and retains them across injections.
    """

    def __init__(
        self,
        name: str,
        token_budgets: Optional[Dict[str, int]] = None,
        persist_dir: Optional[Path] = None,
    ):
        self.name = name
        self._lock = asyncio.Lock()
        self._persist_dir = (
            persist_dir
            or Path(__file__).resolve().parents[1] / "artifacts" / "context_bundles"
        )
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._bundle_path = self._persist_dir / f"{name}_persistent.json"
        self._items: Dict[str, List[Dict[str, Any]]] = {}
        self._token_budgets = token_budgets or {}
        self._total_added = 0
        self._load()

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

    async def add_artifact(
        self,
        source: str,
        artifact_type: str,
        payload: Any,
        confidence: float = 0.7,
        priority: int = 5,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        async with self._lock:
            if source not in self._items:
                self._items[source] = []

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
            }

            items = self._items[source]
            replaced = False
            for idx, existing in enumerate(items):
                if existing.get("id") == art_id:
                    if self._sort_key(existing) <= self._sort_key(artifact):
                        replaced = True
                        break
                    items[idx] = artifact
                    replaced = True
                    break
            if not replaced:
                items.append(artifact)

            self._total_added += 1
            self._trim_to_budget(source)
            snapshot = copy.deepcopy(self._items)
        self._schedule_save(snapshot)

    async def merge_injection(self, injection: Dict[str, Any]):
        if not injection:
            return
        sources = injection.get("sources") or {}
        for source, data in sources.items():
            for artifact in data.get("artifacts") or []:
                await self.add_artifact(
                    source=source,
                    artifact_type=artifact.get("type", "note"),
                    payload=artifact.get("payload"),
                    confidence=float(artifact.get("confidence", 0.7)),
                    priority=int(artifact.get("priority", 5)),
                    timestamp=float(artifact.get("timestamp", time.time())),
                    metadata=artifact.get("metadata") or {},
                )

    def _trim_to_budget(self, source: str):
        items = self._items.get(source, [])
        budget = int(self._token_budgets.get(source, 1500))
        items.sort(key=self._sort_key)
        total_tokens = 0
        kept: List[Dict[str, Any]] = []
        for item in items:
            item_tokens = self._estimate_tokens(item)
            if total_tokens + item_tokens <= budget:
                kept.append(item)
                total_tokens += item_tokens
        self._items[source] = kept

    def format_for_llm(self) -> str:
        lines = [f"[Persistent Context: {self.name}]"]
        sources = self._items or {}
        for source in sorted(sources.keys()):
            items = list(sources[source])
            if not items:
                continue
            lines.append(f"\n[{source.replace('_', ' ').title()}] ({len(items)} artifacts):")
            by_type: Dict[str, List[Dict[str, Any]]] = {}
            for a in items:
                by_type.setdefault(str(a.get("type") or "note"), []).append(a)
            for t in sorted(by_type.keys()):
                lines.append(f"- {t}:")
                by_type[t].sort(key=self._sort_key)
                for a in by_type[t][:15]:
                    payload = a.get("payload")
                    if isinstance(payload, (dict, list)):
                        payload_str = json.dumps(payload, ensure_ascii=False, default=str)
                    else:
                        payload_str = str(payload)
                    payload_str = payload_str.strip().replace("\n", " ")
                    lines.append(
                        f"  • (c={a.get('confidence', 0):.2f}) {payload_str[:800]}"
                    )
        return "\n".join(lines)

    def _save(self):
        try:
            raw = {
                "items": self._items,
                "token_budgets": self._token_budgets,
                "total_added": self._total_added,
            }
            with self._bundle_path.open("w", encoding="utf-8") as fh:
                json.dump(raw, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[PersistentBundle:{self.name}] Persist failed: {exc}", flush=True)

    async def _save_async(self, snapshot: Dict[str, Any], total_added: int):
        await asyncio.to_thread(self._write_snapshot, snapshot, total_added)

    def _write_snapshot(self, snapshot: Dict[str, Any], total_added: int):
        try:
            raw = {
                "items": snapshot,
                "token_budgets": self._token_budgets,
                "total_added": total_added,
            }
            with self._bundle_path.open("w", encoding="utf-8") as fh:
                json.dump(raw, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[PersistentBundle:{self.name}] Persist failed: {exc}", flush=True)

    def _schedule_save(self, snapshot: Dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._write_snapshot(snapshot, self._total_added)
            return
        loop.create_task(self._save_async(snapshot, self._total_added))

    def _load(self):
        if not self._bundle_path.exists():
            return
        try:
            with self._bundle_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            items = data.get("items")
            if isinstance(items, dict):
                self._items = {k: v[:] for k, v in items.items()}
            budgets = data.get("token_budgets")
            if isinstance(budgets, dict):
                self._token_budgets.update({k: int(v) for k, v in budgets.items()})
            self._total_added = int(data.get("total_added", 0))
        except Exception as exc:
            print(f"[PersistentBundle:{self.name}] Load failed: {exc}", flush=True)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_added": self._total_added,
            "currently_stored": sum(len(items) for items in self._items.values()),
        }
