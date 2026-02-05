#!/usr/bin/env python3
"""
GraphReasoner - The Semantic Query Engine for Dexter
Uses the Knowledge Graph (triples/entities) to answer complex questions.
"""

import os
import json
import asyncio
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
import core.tool_agent_provider as agent_provider
from core.brain_schema import ensure_brain_schema
from core.importance import record_access
from skills.memory_ops import _db_path

class GraphReasoner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def _get_brain(self, slot: str = "orchestrator") -> tuple:
        from core.llm_slots import resolve_llm_slot
        p_name, resolved_cfg, p_model = resolve_llm_slot(self.config, slot)
        return agent_provider.AsyncAIProvider(p_name, resolved_cfg), p_model

    def _get_context(self, entity_name: str, depth: int = 1) -> List[Dict[str, Any]]:
        """Fetches neighboring triples for an entity."""
        db_path = _db_path()
        ensure_brain_schema(db_path)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Simple depth-1 context
        c.execute("SELECT id, subject, predicate, object FROM triples WHERE subject LIKE ? OR object LIKE ? LIMIT 50",
                  (f"%{entity_name}%", f"%{entity_name}%"))
        rows = c.fetchall()
        results = []
        for row in rows:
            record_access(conn, "triples", "id", row[0], config=self.config)
            results.append({"subject": row[1], "predicate": row[2], "object": row[3]})
        conn.close()

        return results

    def _get_priority_context(self) -> List[Dict[str, Any]]:
        """Fetches mission-critical identity and mission triples."""
        db_path = _db_path()
        ensure_brain_schema(db_path)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT id, subject, predicate, object FROM triples WHERE source IN ('Identity_Anchor', 'Core_Mission')")
        rows = c.fetchall()
        results = []
        for row in rows:
            record_access(conn, "triples", "id", row[0], config=self.config)
            results.append({"subject": row[1], "predicate": row[2], "object": row[3]})
        conn.close()
        return results

    async def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Uses the graph to find facts and an LLM to reason over them.
        """
        print(f"[Reasoner] Analyzing question: {question}...", flush=True)
        
        # 1. Identify key entities in the question via LLM
        provider, model = await self._get_brain("orchestrator")
        try:
            extract_prompt = f"Identify the primary entities (nouns/concepts) in this question: '{question}'. Return ONLY a JSON list of strings."
            messages = [
                agent_provider.ChatMessage(role="system", content="You are Dexter's Entity Extractor. Output JSON only."),
                agent_provider.ChatMessage(role="user", content=extract_prompt)
            ]

            response = await provider.chat(messages, model)
            from core.utils import extract_json
            entities = extract_json(response.content) or []
            print(f"[Reasoner] Extracted entities: {entities}")

            # 2. Gather Graph Context
            all_triples = self._get_priority_context() # Always start with core mission/identity
            for ent in entities:
                all_triples.extend(self._get_context(ent))

            if not all_triples:
                return {"success": False, "error": "No related facts found in knowledge graph."}

            # 3. Final Reasoning
            context_text = "\n".join([f"- ({t['subject']}, {t['predicate']}, {t['object']})" for t in all_triples])

            reason_prompt = f"""
Question: {question}

I found the following relationships in my Knowledge Graph:
{context_text}

Using ONLY the graph data provided, answer the question accurately. 
If the graph doesn't contain the answer, say so.
"""
            messages = [
                agent_provider.ChatMessage(role="system", content="You are Dexter's Graph Reasoner. You answer based on structured knowledge."),
                agent_provider.ChatMessage(role="user", content=reason_prompt)
            ]

            final_resp = await provider.chat(messages, model)
            return {
                "success": True,
                "answer": final_resp.content,
                "entities_found": entities,
                "triples_consulted": len(all_triples)
            }
        finally:
            try:
                await provider.close()
            except Exception:
                pass
