#!/usr/bin/env python3
"""
SkillLibrarian - The Knowledge Indexer for Dexter
Translates high-level goals into specific skill/tool IDs.
"""

import os
import sys
import json
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import ast
import core.tool_agent_provider as agent_provider

# Fix module pathing
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    import torch
except ImportError:
    torch = None

class SkillLibrarian:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.skills_path = self.repo_root / "skills"
        self.index_path = self.repo_root / "memory" / "skill_index.json"
        
        # Load Config
        with open(self.repo_root / "configs" / "core_config.json", "r") as f:
            self.config = json.load(f)

        trm_path = self.config.get("trm_config", {}).get("librarian_checkpoint")
        self.model_path = self.repo_root / trm_path if trm_path else None
        self.skill_registry = {}
        self.model_loaded = False
        
        # Load Index
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.skill_registry = json.load(f)

        # Auto-discover skills if enabled or index missing/stale
        cfg = self.config.get("skill_librarian", {}) or {}
        self.auto_discover = cfg.get("auto_discover", True)
        self.refresh_on_startup = cfg.get("refresh_on_startup", True)
        if self.auto_discover and (not self.index_path.exists() or (self.refresh_on_startup and self._index_needs_refresh())):
            self.discover_skills()
        
        # Load Librarian TRM
        if self.model_path and self.model_path.exists():
            try:
                import torch
                from core.trm_base import BaseTRM, TRMConfig
                
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.vocab = checkpoint["vocab"]
                self.skill_map = checkpoint["skill_map"]
                self.id_map = {v: k for k, v in self.skill_map.items()}
                
                # Reconstruct Model
                config = checkpoint.get("config", TRMConfig(num_classes=len(self.skill_map)))
                self.model = BaseTRM(config)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                
                self.model_loaded = True
                print("[Librarian] Neural Index Online.")
            except Exception as e:
                print(f"[Librarian] Failed to load Neural Index: {e}")

    def _relative_path_str(self, path: Path) -> str:
        rel_path = os.path.relpath(path, self.repo_root)
        return Path(rel_path).as_posix()

    def discover_skills(self):
        """Scans the skills folder and updates the skill index with detailed function maps."""
        print(f"[Librarian] Performing deep scan of {self.skills_path} for capabilities...")
        new_registry = {}
        for file in self.skills_path.glob("*.py"):
            if file.name.startswith("_"): continue
            skill_id = file.stem
            path_str = self._relative_path_str(file)
            
            try:
                import importlib.util
                import inspect
                
                # Dynamic load to read metadata via inspection
                spec = importlib.util.spec_from_file_location(skill_id, str(file))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                skill_desc = inspect.getdoc(module) or f"Handles {skill_id} operations."
                function_map = []
                
                for name, func in inspect.getmembers(module, inspect.isfunction):
                    if name.startswith("_"): continue
                    doc = inspect.getdoc(func) or "No description."
                    function_map.append({
                        "name": name,
                        "description": doc.split("\n")[0]
                    })
                
                new_registry[skill_id] = {
                    "path": path_str,
                    "description": skill_desc,
                    "functions": function_map
                }
            except Exception as e:
                print(f"[Librarian] Error indexing {skill_id} (import). Falling back to AST scan: {e}")
                try:
                    with open(file, "r", encoding="utf-8") as fh:
                        tree = ast.parse(fh.read(), filename=str(file))
                    skill_desc = ast.get_docstring(tree) or f"Handles {skill_id} operations."
                    function_map = []
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                            doc = ast.get_docstring(node) or "No description."
                            function_map.append({
                                "name": node.name,
                                "description": doc.split("\n")[0]
                            })
                    new_registry[skill_id] = {
                        "path": path_str,
                        "description": skill_desc,
                        "functions": function_map
                    }
                except Exception as parse_err:
                    print(f"[Librarian] AST scan failed for {skill_id}: {parse_err}")
                    new_registry[skill_id] = {"path": path_str, "description": "Error loading.", "functions": []}
            
        self.skill_registry = new_registry
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.skill_registry, f, indent=2)
        print(f"[Librarian] Deep index refreshed. {len(self.skill_registry)} skills mapped.")

    def predict_skill(self, goal: str) -> str:
        """Use the TRM to find the best skill ID, with a Commander Override and Heuristics for robust steering."""
        goal_lower = goal.lower()

        # --- COMMANDER OVERRIDE (Expert Reflex) ---
        # If the goal explicitly mentions a skill name, prioritize it.
        for skill_id in self.skill_registry.keys():
            if f"use {skill_id}" in goal_lower or f"run {skill_id}" in goal_lower or f"with {skill_id}" in goal_lower:
                print(f"[Librarian] Commander Override detected: {skill_id}")
                return skill_id
        
        # --- INTELLIGENCE HEURISTIC ---
        # Penalize memory_ops if the goal implies real-time or external data
        research_keywords = ["search", "find", "latest", "market", "trends", "current", "sectors", "research"]
        if any(kw in goal_lower for kw in research_keywords) and "memory" not in goal_lower:
            # Only prioritize if research_ops actually exists in registry
            if "research_ops" in self.skill_registry:
                print(f"[Librarian] Heuristic: Goal implies external research. Prioritizing research_ops.")
                return "research_ops"

        if not self.model_loaded or torch is None:
            matches = self.find_relevant_skill(goal)
            return matches[0] if matches else "unknown"
            
        # Quick Tokenize
        words = goal.lower().split()
        ids = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in words]
        if len(ids) < 32: ids += [0] * (32 - len(ids))
        else: ids = ids[:32]
        
        # Neural Inference
        with torch.no_grad():
            input_tensor = torch.tensor([ids], dtype=torch.long)
            logits = self.model(input_tensor)
            skill_idx = torch.argmax(logits, dim=1).item()
        
        return self.id_map.get(skill_idx, "unknown")

    def predict_skill_with_confidence(self, goal: str) -> tuple[str, float]:
        """
        Returns (skill_id, confidence). Confidence is max softmax prob when TRM is loaded.
        """
        skill = self.predict_skill(goal)
        if not self.model_loaded or torch is None or skill == "unknown":
            return skill, 0.0

        # Compute confidence
        words = goal.lower().split()
        ids = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in words]
        if len(ids) < 32:
            ids += [0] * (32 - len(ids))
        else:
            ids = ids[:32]

        with torch.no_grad():
            input_tensor = torch.tensor([ids], dtype=torch.long)
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf = float(probs.max().item())
        return skill, conf

    async def choose_skill_llm(
        self,
        goal: str,
        slot: str = "dexter",
        context_bundle: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Use an LLM to select the most appropriate skill if TRM confidence is low.
        """
        from core.llm_slots import resolve_llm_slot
        p_name, resolved_cfg, p_model = resolve_llm_slot(self.config, slot)
        provider = agent_provider.AsyncAIProvider(p_name, resolved_cfg)

        # Build a compact skill list
        skills = []
        for skill_id, info in self.skill_registry.items():
            desc = info.get("description", "")
            skills.append({"id": skill_id, "description": desc})

        bundle_text = ""
        if context_bundle:
            bundle_text = f"\nContext Bundle (full, untruncated JSON):\n{json.dumps(context_bundle, ensure_ascii=False, default=str)}\n"

        prompt = f"""
Goal: {goal}
{bundle_text}

Available skills (id + description):
{json.dumps(skills, ensure_ascii=False)}

Choose the single best skill id. Return ONLY the id string.
"""
        messages = [
            agent_provider.ChatMessage(role="system", content="You are a skill router. Return ONLY the skill id."),
            agent_provider.ChatMessage(role="user", content=prompt),
        ]
        response = await provider.chat(messages, p_model)
        if response.success:
            choice = response.content.strip().split()[0].strip()
            if choice in self.skill_registry:
                return choice
        return "unknown"

    def find_relevant_skill(self, goal: str) -> List[str]:
        """
        Baseline: Keyword matching.
        Evolution: Train a TRM to do this based on the docstrings.
        """
        # (This is where the Librarian TRM will eventually sit)
        matches = []
        for skill_id, info in self.skill_registry.items():
            if any(word in goal.lower() for word in skill_id.split("_")):
                matches.append(skill_id)
        return matches

    def _index_needs_refresh(self) -> bool:
        """Check if any skill file is newer than the index."""
        try:
            if not self.index_path.exists():
                return True
            index_mtime = self.index_path.stat().st_mtime
            for file in self.skills_path.glob("*.py"):
                if file.name.startswith("_"):
                    continue
                if file.stat().st_mtime > index_mtime:
                    return True
        except Exception:
            return True
        return False

    def health_report(self) -> Dict[str, Any]:
        """
        Returns a report of skill import health.
        {
          "total": int,
          "ok": int,
          "failed": int,
          "details": {skill_id: {"status": "ok"|"failed", "error": "..."}}
        }
        """
        report: Dict[str, Any] = {"total": 0, "ok": 0, "failed": 0, "details": {}}
        for file in self.skills_path.glob("*.py"):
            if file.name.startswith("_"):
                continue
            skill_id = file.stem
            report["total"] += 1
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(skill_id, str(file))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                report["ok"] += 1
                report["details"][skill_id] = {"status": "ok"}
            except Exception as exc:
                from core.dependency_installer import extract_missing_module
                missing = extract_missing_module(exc)
                report["failed"] += 1
                report["details"][skill_id] = {"status": "failed", "error": str(exc), "missing_module": missing}
        return report

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    lib = SkillLibrarian(repo_root)
    lib.discover_skills()
    test_goal = "Record a new fact about my favorite color"
    predicted = lib.predict_skill(test_goal)
    print(f"Goal: {test_goal}")
    print(f"Predicted Skill: {predicted}")
