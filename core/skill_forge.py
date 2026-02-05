#!/usr/bin/env python3
"""
Dexter Skill Forge: Autonomous Tool Creation
Identifies capability gaps and recursively writes new tools to solve them.
"""

import os
import sys
import json
import asyncio
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from core import tool_agent_provider
from core import forge_manager

class SkillForge:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        default_root = Path(__file__).resolve().parent.parent
        repo_root = config.get("repo_root", str(default_root))
        self.repo_root = Path(repo_root).resolve()
        self.skills_dir = self.repo_root / "skills"
        
        from core.llm_slots import resolve_llm_slot
        p_name, resolved_cfg, p_model = resolve_llm_slot(config, "forge")
        self.provider = tool_agent_provider.AsyncAIProvider(p_name, resolved_cfg)
        self.model = p_model

    def _relative_path_str(self, path: Path) -> str:
        rel_path = os.path.relpath(path, self.repo_root)
        return Path(rel_path).as_posix()

    async def forge_tool(self, gap_description: str, suggested_name: str) -> Dict[str, Any]:
        """
        Generate a new tool based on a gap description.
        returns: {"ok": bool, "code": str, "path": str, "error": str}
        """
        from core.api import broadcast_thought
        broadcast_thought("thinking", f"Forging new skill: {suggested_name} to bridge gap: {gap_description}")
        print(f"[Forge] Attempting to forge skill: {suggested_name}")
        
        system_prompt = f"""
You are the Dexter Skill Forge. Your mission is to write high-quality, standalone Python tools for the Dexter agent ecosystem.

Dexter Identity: Dexter is a high-level AGI coordinator focusing on Jeffrey Gliksman's prosperity.

Rules:
1. Every tool must be a single Python file in the `skills` directory.
2. Every tool file SHOULD have a `__tool_prefix__` variable (e.g., `__tool_prefix__ = "media_ops"`) to group its functions.
3. Functions should be well-documented with docstrings.
4. Return results as a dictionary: {{"success": True, "data": ...}} or {{"success": False, "error": "..."}}.
5. Avoid user interaction (no input(), no GUIs).
6. Use standard libraries or common ones like requests, psutil, beautifulsoup4, etc.

Example:
__tool_prefix__ = "system"

def get_uptime() -> Dict[str, Any]:
    '''Returns the system uptime in seconds.'''
    import time
    try:
        import psutil
        uptime = time.time() - psutil.boot_time()
        return {{"success": True, "uptime": uptime}}
    except Exception as e:
        return {{"success": False, "error": str(e)}}
"""

        user_prompt = f"Gap Description: {gap_description}\nSuggested Skill Name: {suggested_name}\n\nPlease write the full Python code for this skill. Wrap the code in ```python blocks."
        
        messages = [
            tool_agent_provider.ChatMessage(role="system", content=system_prompt),
            tool_agent_provider.ChatMessage(role="user", content=user_prompt)
        ]
        
        response = await self.provider.chat(messages, self.model)
        
        if not response.success:
            broadcast_thought("error", f"Skill Forge failed: {response.content}")
            return {"ok": False, "error": f"LLM Error: {response.content}"}
            
        code = self._extract_code(response.content)
            
        # Determine path
        module_name = suggested_name.split(".")[0] if "." in suggested_name else suggested_name
        file_path = self.skills_dir / f"{module_name}.py"
        rel_file_path = self._relative_path_str(file_path)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Update Skill Index
            broadcast_thought("thinking", f"New skill {suggested_name} forged at {rel_file_path}. Integrating...")
            
            # We don't need forge_manager here if we use the Librarian's discovery
            from core.skill_librarian import SkillLibrarian
            lib = SkillLibrarian(self.repo_root)
            lib.discover_skills()
            if module_name not in lib.skill_registry:
                return {"ok": False, "error": "Skill written but not registered. Check import errors."}
            
            broadcast_thought("skill_created", {"skill_id": module_name, "path": rel_file_path})
            
            return {
                "ok": True,
                "code": code,
                "path": rel_file_path,
                "skill_id": module_name
            }
        except Exception as e:
            broadcast_thought("error", f"Skill Forge write error: {e}")
            return {"ok": False, "error": str(e)}

    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    async def repair_skill(
        self,
        skill_id: str,
        task: str,
        tool_name: str,
        arguments: Dict[str, Any],
        error: str,
        traceback_text: str,
        attempt: int = 1,
    ) -> Dict[str, Any]:
        from core.api import broadcast_thought
        file_path = self.skills_dir / f"{skill_id}.py"
        rel_file_path = self._relative_path_str(file_path)
        if not file_path.exists():
            return {"ok": False, "error": f"Skill file not found: {rel_file_path}"}

        original = file_path.read_text(encoding="utf-8")
        backup_dir = self.skills_dir / "_backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / f"{skill_id}_{int(time.time())}.py"
        backup_path.write_text(original, encoding="utf-8")
        rel_backup_path = self._relative_path_str(backup_path)

        system_prompt = """
You are the Dexter Skill Repair agent.
You will fix the existing Python skill file based on the error and context.
Rules:
1. Return the full corrected Python code for the same file.
2. Preserve __tool_prefix__ and all public function names unless they are broken.
3. Avoid interactive input or GUI usage.
4. Keep dependencies minimal and standard when possible.
5. Ensure decorated functions keep correct metadata (use functools.wraps when wrapping).
"""

        user_prompt = f"""
Task: {task}
Skill: {skill_id}
Tool: {tool_name}
Arguments: {json.dumps(arguments or {}, ensure_ascii=False, indent=2)}
Error: {error}
Traceback: {traceback_text}
Attempt: {attempt}

Current skill code:
```python
{original}
```

Return ONLY the corrected Python code.
"""

        broadcast_thought("thinking", f"Repairing skill {skill_id} after error: {error}")

        messages = [
            tool_agent_provider.ChatMessage(role="system", content=system_prompt.strip()),
            tool_agent_provider.ChatMessage(role="user", content=user_prompt.strip())
        ]

        response = await self.provider.chat(messages, self.model)
        if not response.success:
            broadcast_thought("error", f"Skill Repair failed: {response.content}")
            return {"ok": False, "error": response.content, "backup": rel_backup_path}

        code = self._extract_code(response.content)
        if not code:
            return {"ok": False, "error": "Repair returned empty code", "backup": rel_backup_path}

        try:
            file_path.write_text(code, encoding="utf-8")
            from core.skill_librarian import SkillLibrarian
            lib = SkillLibrarian(self.repo_root)
            lib.discover_skills()
            if skill_id not in lib.skill_registry:
                return {"ok": False, "error": "Repaired skill did not register", "backup": rel_backup_path}
            broadcast_thought("skill_repaired", {"skill_id": skill_id, "path": rel_file_path})
            return {"ok": True, "path": rel_file_path, "backup": rel_backup_path, "skill_id": skill_id}
        except Exception as e:
            return {"ok": False, "error": str(e), "backup": rel_backup_path}
