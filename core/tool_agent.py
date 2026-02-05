#!/usr/bin/env python3
"""
ToolAgent - Mid-level Reasoning
Translates high-level tasks into specific tool calls with parameters.
"""

import os
import json
import inspect
import importlib.util
from typing import Dict, Any, Optional, List
from pathlib import Path
import core.tool_agent_provider as agent_provider

class ToolAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        repo_root = Path(__file__).resolve().parent.parent
        skills_dir = config.get("skills_dir")
        if not skills_dir:
            skills_dir = str(repo_root / "skills")
        if not os.path.isabs(skills_dir):
            skills_dir = str(repo_root / skills_dir)
        self.skills_dir = Path(skills_dir)
        self.active_slot = config.get("mode", "slot")

    async def _get_brain(self, slot: str = "forge") -> tuple:
        from core.llm_slots import resolve_llm_slot
        p_name, resolved_cfg, p_model = resolve_llm_slot(self.config, slot)
        return agent_provider.AsyncAIProvider(p_name, resolved_cfg), p_model

    def _get_tool_metadata(self, skill_id: str) -> str:
        """Reads the docstrings of all functions in a skill file."""
        file_path = self.skills_dir / f"{skill_id}.py"
        if not file_path.exists():
            return f"Skill {skill_id} not found."
            
        try:
            spec = importlib.util.spec_from_file_location(skill_id, str(file_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            methods = []
            for name, func in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("_"): continue
                doc = inspect.getdoc(func) or "No description."
                sig = str(inspect.signature(func))
                methods.append(f"- {name}{sig}: {doc}")
            
            return "\n".join(methods)
        except Exception as e:
            return f"Error reading skill {skill_id}: {e}"

    async def parameterize(
        self,
        task: str,
        skill_id: str,
        context: Dict[str, Any],
        context_bundle: Optional[Dict[str, Any]] = None,
        preferred_tool_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Uses an LLM to decide which function and which arguments to use for a skill.
        """
        print(f"[ToolAgent] Parameterizing {skill_id} for task: {task}...", flush=True)
        provider, model = await self._get_brain("forge")
        try:
            metadata = self._get_tool_metadata(skill_id)

            bundle_text = ""
            if context_bundle:
                bundle_text = f"\nContext Bundle (full, untruncated JSON):\n{json.dumps(context_bundle, ensure_ascii=False, default=str)}\n"

            preferred_block = ""
            if preferred_tool_name:
                preferred_block = (
                    f"Preferred Tool (must use): {preferred_tool_name}\n"
                    "Use this exact tool name and only fill arguments for it.\n"
                )

            prompt = f"""
Task: {task}
Skill: {skill_id}
{preferred_block}Available Functions in this Skill:
{metadata}

Current Context: {context}
{bundle_text}

Select the best function and provide the arguments.
Return ONLY a valid JSON object:
{{
  "name": "{skill_id}.function_name",
  "arguments": {{
    "arg1": "value1",
    ...
  }}
}}
"""
            messages = [
                agent_provider.ChatMessage(role="system", content="You are Dexter's Tool Parameterizer. You output ONLY JSON."),
                agent_provider.ChatMessage(role="user", content=prompt)
            ]

            from core.api import broadcast_thought
            broadcast_thought("tool_call_selection", {"skill_id": skill_id, "task": task})

            response = await provider.chat(messages, model)

            if response.success:
                from core.utils import extract_json
                parsed = extract_json(response.content)
                if parsed:
                    broadcast_thought("tool_parameterized", parsed)
                    return parsed
                else:
                    print(f"[ToolAgent] Raw Output (Failed Parse): {response.content}")
                    broadcast_thought("error", f"Tool parameterization failed to parse JSON for {skill_id}")

            return {"name": f"{skill_id}.process_list" if skill_id == "system_ops" else f"{skill_id}.run_test", "arguments": {}}
        finally:
            try:
                await provider.close()
            except Exception:
                pass

    async def quick_query(
        self,
        prompt: str,
        slot: str = "forge",
        max_tokens: int = 150,
    ) -> str:
        """
        Quick LLM query for simple tasks like tool selection.
        """
        provider, model = await self._get_brain(slot)
        try:
            messages = [
                agent_provider.ChatMessage(role="user", content=prompt)
            ]

            response = await provider.chat(messages, model, max_tokens=max_tokens)

            if response.success:
                return response.content
            return ""
        finally:
            try:
                await provider.close()
            except Exception:
                pass
