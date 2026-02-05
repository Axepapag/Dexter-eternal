from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PlanTemplateLibrary:
    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("plan_templates", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.min_score = int(cfg.get("min_trigger_score", 1))
        self.repo_root = Path(__file__).resolve().parent.parent
        path = cfg.get("path", "memory/plan_templates.json")
        self.path = self._resolve_path(path)
        self.templates: List[Dict[str, Any]] = []
        self.template_index: Dict[str, Dict[str, Any]] = {}
        self.trm_index_map: Dict[int, str] = {}
        self._load()

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.repo_root / candidate

    def _load(self) -> None:
        if not self.enabled or not self.path.exists():
            self.templates = []
            self.template_index = {}
            self.trm_index_map = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            self.templates = []
            self.template_index = {}
            self.trm_index_map = {}
            return

        raw_templates = payload.get("templates", []) if isinstance(payload, dict) else []
        templates: List[Dict[str, Any]] = []
        for idx, raw in enumerate(raw_templates):
            if not isinstance(raw, dict):
                continue
            template_id = str(raw.get("id") or f"template_{idx}")
            trm_index = raw.get("trm_index")
            if trm_index is None:
                trm_index = idx
            template = {
                "id": template_id,
                "name": str(raw.get("name") or template_id),
                "description": str(raw.get("description") or ""),
                "triggers": list(raw.get("triggers") or []),
                "steps": list(raw.get("steps") or []),
                "trm_index": int(trm_index),
            }
            templates.append(template)

        self.templates = templates
        self.template_index = {t["id"]: t for t in templates}
        self.trm_index_map = {t["trm_index"]: t["id"] for t in templates}

    def get_by_id(self, template_id: str) -> Optional[Dict[str, Any]]:
        return self.template_index.get(template_id)

    def match_template(self, goal: str) -> Tuple[Optional[Dict[str, Any]], int]:
        if not self.templates or not goal:
            return None, 0
        text = goal.lower()
        best = None
        best_score = 0
        for template in self.templates:
            score = 0
            template_id = template.get("id", "").lower()
            template_name = template.get("name", "").lower()
            if template_id and template_id in text:
                score += 2
            if template_name and template_name in text:
                score += 1
            for trigger in template.get("triggers", []):
                trigger_text = str(trigger).lower()
                if trigger_text and trigger_text in text:
                    score += 1
            if score > best_score:
                best_score = score
                best = template
        if best_score < self.min_score:
            return None, best_score
        return best, best_score

    def build_plan(self, goal: str, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        steps: List[Dict[str, Any]] = []
        format_ctx = {
            "goal": goal,
            "context": context or {},
        }
        for idx, step in enumerate(template.get("steps", []), start=1):
            task = self._format_step(step, format_ctx)
            steps.append({"id": idx, "task": task, "status": "pending"})
        return {
            "goal": goal,
            "template_id": template.get("id"),
            "template_name": template.get("name"),
            "steps": steps,
        }

    def _format_step(self, step: Any, context: Dict[str, Any]) -> str:
        text = str(step)
        try:
            return text.format(**context)
        except KeyError:
            return text.replace("{goal}", context.get("goal", ""))
        except Exception:
            return text
