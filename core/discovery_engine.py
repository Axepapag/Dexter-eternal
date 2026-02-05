#!/usr/bin/env python3
"""
DiscoveryEngine - The Knowledge Miner for Dexter
Mines raw facts and transforms them into structured knowledge (triples).
"""

import os
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List
import core.tool_agent_provider as agent_provider
from core.brain_schema import ensure_brain_schema
from core.importance import upsert_importance
from core.vector_store import store_embedding
from skills.memory_ops import search as search_facts
from skills.kg_ops import add_triple

class DiscoveryEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def _get_brain(self, slot: str = "orchestrator") -> tuple:
        from core.llm_slots import resolve_llm_slot
        p_name, resolved_cfg, p_model = resolve_llm_slot(self.config, slot)
        return agent_provider.AsyncAIProvider(p_name, resolved_cfg), p_model

    async def discover_relationships(self, limit: int = 10) -> Dict[str, Any]:
        """
        Fetches raw facts and uses the LLM to discover triples.
        """
        print(f"[Discovery] Mining up to {limit} recent facts for relationships...", flush=True)
        fact_result = search_facts(query="%", limit=limit)
        if not fact_result.get("success") or not fact_result.get("results"):
            return {"success": False, "error": "No facts found."}
        
        facts_text = "\n".join([f"- {f['text']}" for f in fact_result["results"]])
        provider, model = await self._get_brain("orchestrator")
        
        prompt = f"Analyze these facts and extract (Subject, Predicate, Object) triples. Return JSON list only.\nFacts:\n{facts_text}"
        messages = [
            agent_provider.ChatMessage(role="system", content="You are Dexter's Knowledge Miner. Output JSON list of triples."),
            agent_provider.ChatMessage(role="user", content=prompt)
        ]
        
        try:
            response = await provider.chat(messages, model)
        finally:
            try:
                await provider.close()
            except Exception:
                pass

        if not response.success:
            return {"success": False, "error": response.content or "LLM call failed"}
            
        from core.utils import extract_json
        raw_triples = extract_json(response.content) or []

        normalized: List[Dict[str, Any]] = []
        if isinstance(raw_triples, list):
            for t in raw_triples:
                if isinstance(t, dict):
                    sub = (t.get("subject") or t.get("sub") or t.get("s") or "").strip()
                    pred = (t.get("predicate") or t.get("pred") or t.get("p") or "").strip()
                    obj = (t.get("object") or t.get("obj") or t.get("o") or "").strip()
                    conf = float(t.get("confidence", 0.7) or 0.7)
                elif isinstance(t, list) and len(t) >= 3:
                    sub, pred, obj = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
                    conf = 0.7
                else:
                    continue
                if sub and pred and obj:
                    normalized.append({"subject": sub, "predicate": pred, "object": obj, "confidence": conf})

        added = 0
        for t in normalized:
            if add_triple(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                source="DiscoveryEngine",
            )["success"]:
                added += 1

        return {
            "success": True,
            "mined_facts": len(fact_result.get("results") or []),
            "discovered_triples": len(normalized),
            "added": added,
            "triples": normalized,
        }

    async def discover_entities(self, limit: int = 50) -> Dict[str, Any]:
        """
        Scans triples to build unique, canonical entity profiles.
        """
        print("[Discovery] Resolving unique entities from knowledge graph...", flush=True)
        import sqlite3
        from skills.memory_ops import _db_path
        
        db_path = _db_path()
        ensure_brain_schema(db_path)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT DISTINCT subject FROM triples UNION SELECT DISTINCT object FROM triples LIMIT ?", (limit,))
        raw_entities = [row[0] for row in c.fetchall()]
        conn.close()

        if not raw_entities: return {"success": False, "error": "No graph data yet."}

        provider, model = await self._get_brain("cloud")
        prompt = f"Categorize these entities and provide brief descriptions. Return JSON object: {{'name': {{'type': '...', 'description': '...'}} }}.\nEntities: {raw_entities}"
        messages = [
            agent_provider.ChatMessage(role="system", content="You are Dexter's Entity Resolver. Output JSON only."),
            agent_provider.ChatMessage(role="user", content=prompt)
        ]

        response = await provider.chat(messages, model)
        from core.utils import extract_json
        profiles = extract_json(response.content) or {}

        # Save to entities table
        conn = sqlite3.connect(_db_path())
        c = conn.cursor()
        import time
        added = 0
        for name, info in profiles.items():
            try:
                ts = time.time()
                c.execute(
                    "INSERT OR REPLACE INTO entities (name, type, description, ts, source) VALUES (?, ?, ?, ?, ?)",
                    (name, info.get("type"), info.get("description"), ts, "DiscoveryEngine"),
                )
                upsert_importance(
                    conn=conn,
                    table="entities",
                    key_col="name",
                    key_val=name,
                    created_ts=ts,
                    source="DiscoveryEngine",
                    pinned=False,
                    config=self.config,
                )
                store_embedding(
                    conn=conn,
                    item_type="entity",
                    item_id=name,
                    text=f"{name} {info.get('type', '')} {info.get('description', '')}",
                    config=self.config,
                    source="DiscoveryEngine",
                )
                added += 1
            except: pass
        conn.commit()
        conn.close()
        return {"success": True, "resolved": added}

    async def discover_patterns(self, limit: int = 100) -> Dict[str, Any]:
        """
        Analyzes historical successful execution sequences to find repeating patterns.
        """
        print("[Discovery] Mining execution history for patterns...", flush=True)
        import sqlite3
        from skills.memory_ops import _db_path
        
        conn = sqlite3.connect(_db_path())
        c = conn.cursor()
        c.execute("SELECT intent, task, skill_id, tool_call FROM history WHERE decision IN ('CONTINUE', 'FINISH') ORDER BY ts DESC LIMIT ?", (limit,))
        history_rows = c.fetchall()
        conn.close()

        if not history_rows: return {"success": False, "error": "History is empty."}

        # Structure history by intent
        intent_map = {}
        for intent, task, skill, call in history_rows:
            if intent not in intent_map: intent_map[intent] = []
            intent_map[intent].append({"task": task, "skill": skill, "call": call})

        provider, model = await self._get_brain("cloud")
        prompt = f"Analyze these execution sequences and extract high-value 'Patterns'. A pattern is a mapping from a specific type of user intent to a sequence of tool-call signatures. Output JSON list of: {{'trigger_intent': '...', 'steps_json': '...'}}.\nHistory:\n{json.dumps(intent_map)}"
        messages = [
            agent_provider.ChatMessage(role="system", content="You are Dexter's Pattern Miner. Output JSON only."),
            agent_provider.ChatMessage(role="user", content=prompt)
        ]

        response = await provider.chat(messages, model)
        from core.utils import extract_json
        patterns = extract_json(response.content) or []

        db_path = _db_path()
        ensure_brain_schema(db_path)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        added = 0
        for p in patterns:
            try:
                ts = time.time()
                steps_json = json.dumps(p.get("steps_json"))
                c.execute(
                    "INSERT INTO patterns (trigger_intent, steps_json, ts, source) VALUES (?, ?, ?, ?)",
                    (p.get("trigger_intent"), steps_json, ts, "DiscoveryEngine"),
                )
                pattern_id = c.lastrowid
                upsert_importance(
                    conn=conn,
                    table="patterns",
                    key_col="id",
                    key_val=pattern_id,
                    created_ts=ts,
                    source="DiscoveryEngine",
                    pinned=False,
                    config=self.config,
                )
                store_embedding(
                    conn=conn,
                    item_type="pattern",
                    item_id=pattern_id,
                    text=f"{p.get('trigger_intent', '')} {steps_json}",
                    config=self.config,
                    source="DiscoveryEngine",
                )
                added += 1
            except: pass
        conn.commit()
        conn.close()
        return {"success": True, "added": added}

    async def consolidate_memory(self):
        """Runs the full pipeline: Facts -> Triples -> Entities -> Patterns."""
        print("--- STARTING MEMORY CONSOLIDATION ---", flush=True)
        # Unit tests should not require external LLMs or API keys.
        if os.getenv("PYTEST_CURRENT_TEST") or bool(self.config.get("offline_mode", False)):
            return {
                "triples_added": 0,
                "entities_resolved": 0,
                "patterns_discovered": 0,
                "offline": True,
            }
        rel_res = await self.discover_relationships(limit=20)
        ent_res = await self.discover_entities()
        pat_res = await self.discover_patterns()
        return {
            "triples_added": rel_res.get("added", 0),
            "entities_resolved": ent_res.get("resolved", 0),
            "patterns_discovered": pat_res.get("added", 0)
        }
