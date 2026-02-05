#!/usr/bin/env python3
"""
EvolutionEngine - The Self-Improvement Loop for Dexter
Handles retraining of TRMs based on successful interactions.
"""

import json
import os
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.train_librarian import train_librarian_model
from core.generate_librarian_dataset import DatasetGenerator
from core.fragment_store import store_fragment
from core.brain_schema import ensure_brain_schema

class EvolutionEngine:
    def __init__(self, repo_root: Path, config: Dict[str, Any]):
        self.repo_root = repo_root
        self.config = config
        self.training_data_path = repo_root / "memory" / "librarian_training_data.jsonl"
        self.evolution_cfg = config.get("self_evolution", {})

        self.new_data_count = 0
        self.retrain_threshold = self.evolution_cfg.get("retrain_threshold", 5)
        self.big_brain_cfg = config.get("big_brain", {}) or {}

    def refresh_from_brain_db(self):
        """
        Pulls new data from the Big Brain export channel and refreshes locals.
        """
        if not self.big_brain_cfg.get("enabled", False):
            print("[Evolution] Big Brain sync disabled. Local growth only.")
            return

        from core.big_brain_sync import import_updates

        print("[Evolution] Pulling updates from Big Brain...", flush=True)
        try:
            result = import_updates(self.config)
        except Exception as exc:
            print(f"[Evolution] Big Brain import failed: {exc}", flush=True)
            return
        imported = result.get("imported", 0) if isinstance(result, dict) else 0
        print(f"[Evolution] Big Brain import complete. Imported: {imported}.", flush=True)

    def log_success(self, goal: str, skill_id: str):
        """Logs a successful interaction to the training dataset."""
        if not goal or not skill_id or skill_id == "unknown":
            return

        print(f"[Evolution] Logging successful interaction: {skill_id} -> {goal}", flush=True)
        
        entry = {"goal": goal, "skill_id": skill_id}
        with open(self.training_data_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            
        self.new_data_count += 1
        
        if self.new_data_count >= self.retrain_threshold:
            self.trigger_retraining()

    async def upgrade_instincts(self):
        """Regenerates the entire instinct dataset and retrains the TRM."""
        print("[Evolution] Starting Full Neural Upgrade...", flush=True)
        
        # 1. Discover new skills
        from core.skill_librarian import SkillLibrarian
        lib = SkillLibrarian(self.repo_root)
        lib.discover_skills()
        
        # 2. Run Generator
        print("[Evolution] Brainstorming new phrasings via Cloud Brain...", flush=True)
        gen = DatasetGenerator(
            config_path=self.repo_root / "configs" / "tool_agent_config.json",
            index_path=self.repo_root / "memory" / "skill_index.json"
        )
        await gen.run(self.training_data_path)
        
        # 3. Retrain
        return self.trigger_retraining()

    async def maintenance(self):
        """Standard health check: Sync to Cloud -> Prune Local."""
        from core.big_brain_sync import sync_all
        from skills.memory_pruner import prune_local_memory

        print("[Evolution] Starting Maintenance: Syncing Big Brain...", flush=True)
        await asyncio.to_thread(sync_all, self.config)

        print("[Evolution] Starting Maintenance: Pruning Local Working Memory...", flush=True)
        await asyncio.to_thread(prune_local_memory, self.config)

    def trigger_retraining(self):
        """Initiates the retraining of sub-cortical models."""
        # Optional: Sync with cloud before training
        if self.big_brain_cfg.get("enabled", False):
            self.refresh_from_brain_db()

        print("[Evolution] Upgrading offline intelligence from local successes...", flush=True)
        success = train_librarian_model(self.repo_root, epochs=30)
        
        if success:
            print("[Evolution] Retraining complete. Models updated.", flush=True)
            self.new_data_count = 0
            
            # Start Background Maintenance (Async)
            asyncio.create_task(self.maintenance())
            return True
        return False

    def ingest_context_bundle(self, bundle: Dict[str, Any]) -> None:
        """Persist raw tool results + context for future learning."""
        if not bundle:
            return
        path = self.repo_root / "memory" / "context_bundles.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(bundle, ensure_ascii=False) + "\n")
        except Exception:
            pass
        try:
            from skills.memory_ops import _db_path
            db_path = _db_path()
            ensure_brain_schema(db_path)
            import sqlite3

            conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
            text = json.dumps(bundle, ensure_ascii=False, default=str)
            store_fragment(
                conn=conn,
                text=text,
                parent_type="bundle",
                parent_id=str(bundle.get("ts", "")),
                source="context_bundle",
                context={"intent": bundle.get("intent"), "task": bundle.get("task")},
                config=self.config,
            )
            conn.close()
        except Exception:
            pass

    async def forge_and_integrate(self, gap_task: str) -> Optional[str]:
        """Uses the SkillForge to create a new tool for an unhandled task."""
        from core.skill_forge import SkillForge
        forge = SkillForge(self.config)
        
        # Determine a name for the new skill
        suggested_name = gap_task.lower().replace(" ", "_").split("_")[0] + "_ops"
        if len(suggested_name) < 5: suggested_name = "new_skill_ops"

        max_retries = int(self.evolution_cfg.get("forge_max_retries", 2))
        last_error = None
        for attempt in range(max_retries + 1):
            result = await forge.forge_tool(gap_task, suggested_name)
            if result.get("ok"):
                skill_id = result.get("skill_id")
                # Log this success so it's prioritized next time
                self.log_success(gap_task, skill_id)
                return skill_id
            last_error = result.get("error")
            # fallback name on retry
            suggested_name = f"new_skill_{int(time.time())}_ops"

        print(f"[Evolution] Skill forge failed after retries: {last_error}")
        return None

    async def repair_skill(
        self,
        task: str,
        skill_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        error: str,
        traceback_text: str,
        attempt: int = 1,
    ) -> Dict[str, Any]:
        """Attempts to repair a failing skill file via SkillForge."""
        if not self.evolution_cfg.get("auto_repair_enabled", False):
            return {"ok": False, "error": "auto_repair_disabled"}
        if not skill_id or skill_id == "unknown":
            return {"ok": False, "error": "invalid_skill"}

        from core.skill_forge import SkillForge
        forge = SkillForge(self.config)
        return await forge.repair_skill(
            skill_id=skill_id,
            task=task,
            tool_name=tool_name,
            arguments=arguments or {},
            error=error or "Unknown error",
            traceback_text=traceback_text or "",
            attempt=attempt,
        )

    def audit_capabilities(self, intent: str, predicted_skill: str, actual_result: Dict[str, Any]):
        """
        Future: Use the Cloud brain to analyze if a prediction was 'lucky' 
        or if we need to 'Forge' a new skill for this specific intent.
        """
        pass
