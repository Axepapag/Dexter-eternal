#!/usr/bin/env python3
"""
Graph Query Skill
Allows Dexter to perform multi-hop reasoning over his structured knowledge graph.
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Dict, Any

async def async_ask_graph(question: str) -> Dict[str, Any]:
    """Internal async version for API use."""
    # Find repo root
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = Path(os.path.dirname(tools_dir))
    
    # Load config
    config_path = repo_root / "configs" / "core_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    from core.graph_reasoner import GraphReasoner
    reasoner = GraphReasoner(config)
    
    return await reasoner.answer_question(question)

def ask_graph(question: str) -> Dict[str, Any]:
    """Queries Dexter's Knowledge Graph to answer complex, interconnected questions."""
    return asyncio.run(async_ask_graph(question))

def run_test():
    return ask_graph("What do you know about Gliksbot and how he relates to Dexter?")
