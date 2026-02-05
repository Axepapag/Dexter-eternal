#!/usr/bin/env python3
"""
generate_librarian_dataset.py - Brainstorms training data for the Librarian TRM.
Uses the Cloud Brain to imagine how users would ask for specific skills.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add core to path
sys.path.append(str(Path(__file__).parent))
import tool_agent_provider

class DatasetGenerator:
    def __init__(self, config_path: Path, index_path: Path):
        # Load the full tool agent config for API keys
        self.config = tool_agent_provider.load_tool_agent_config(str(config_path))
        self.index_path = index_path
        with open(index_path, "r", encoding="utf-8") as f:
            self.skills = json.load(f)
        
        # Use Cloud Brain for high-quality phrasings
        _, provider_cfg = tool_agent_provider.resolve_provider_config({
            **self.config, 
            "provider": {"name": "ollama_cloud", "model": "deepseek-v3.1:671b-cloud"}
        })
        self.provider = tool_agent_provider.AsyncAIProvider("ollama_cloud", provider_cfg)
        self.model = "deepseek-v3.1:671b-cloud"

    async def brainstorm_skill(self, skill_id: str, description: str) -> List[str]:
        """Ask the LLM for 20 diverse ways to request a skill."""
        print(f"[Generator] Brainstorming phrasings for: {skill_id}...")
        
        prompt = f"""
Skill Name: {skill_id}
Description: {description}

Please generate 20 diverse, natural language phrases a user might say to trigger this skill. 
Return ONLY a JSON list of strings. No conversational filler.
"""
        messages = [
            tool_agent_provider.ChatMessage(role="system", content="You are the Dexter Brainstorming Engine. You generate high-quality synthetic training data."),
            tool_agent_provider.ChatMessage(role="user", content=prompt)
        ]
        
        response = await self.provider.chat(messages, self.model)
        if response.success:
            try:
                # Cleanup common AI markdown
                clean_content = response.content.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_content)
            except Exception as e:
                print(f"Error parsing JSON for {skill_id}: {e}")
                return []
        return []

    async def run(self, output_path: Path):
        print(f"[Generator] Starting generation at {output_path}...")
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for skill_id, info in self.skills.items():
                phrasings = await self.brainstorm_skill(skill_id, info["description"])
                if not phrasings:
                    print(f"[Generator] WARNING: No phrasings for {skill_id}")
                    continue
                
                for phrase in phrasings:
                    f.write(json.dumps({"goal": phrase, "skill_id": skill_id}) + "\n")
                    count += 1
                f.flush() # Ensure it's on disk
                print(f"[Generator] Added {len(phrasings)} phrasings for {skill_id}. Total: {count}")
        
        print(f"[Generator] Finished. Total dataset size: {count}")

async def main():
    repo_root = Path(__file__).resolve().parent.parent
    gen = DatasetGenerator(
        config_path=repo_root / "configs" / "tool_agent_config.json",
        index_path=repo_root / "memory" / "skill_index.json"
    )
    await gen.run(repo_root / "memory" / "librarian_training_data.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
