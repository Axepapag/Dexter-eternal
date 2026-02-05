#!/usr/bin/env python3
"""
Knowledge Discovery Skill
Dexter mines his own memories to build a structured Knowledge Graph.
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Dict, Any

def mine_memories(limit: int = 10) -> Dict[str, Any]:
    """Dexter analyzes recent facts to extract structured triples."""
    # Find repo root
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = Path(os.path.dirname(tools_dir))
    
    # Load config
    config_path = repo_root / "configs" / "core_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    from core.discovery_engine import DiscoveryEngine
    engine = DiscoveryEngine(config)
    
    return asyncio.run(engine.discover_relationships(limit=limit))

def run_test():
    return mine_memories(limit=10)
