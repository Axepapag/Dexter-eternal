#!/usr/bin/env python3
"""
Manual Evolve - Forces an immediate synchronization and retraining of Dexter's local models.
"""

import sys
import json
from pathlib import Path

# Add core to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from core.evolution_engine import EvolutionEngine

async def main():
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "configs" / "core_config.json"
    
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    print("--- DEXTER NEURAL UPGRADE STARTING ---")
    engine = EvolutionEngine(repo_root, config)
    
    # Run the full upgrade
    await engine.upgrade_instincts()
    
    print("--- DEXTER NEURAL UPGRADE COMPLETE ---")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
