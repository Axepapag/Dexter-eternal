import asyncio
import json
import time
import sqlite3
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from core.response_tank import ResponseTank, get_global_tank
from core.rolling_context_bundle import RollingContextBundle
from core.trm_base import BaseTRM, TRMConfig
from core import embedding_client


class MemoryTRM(nn.Module):
    def __init__(
        self,
        config: TRMConfig,
        response_tank: ResponseTank,
        rolling_context: RollingContextBundle,
        dexter_config: Dict[str, Any]
    ):
        super().__init__()
        self.config = config
        self._tank = response_tank
        self._rolling_context = rolling_context
        self._dexter_config = dexter_config
        
        # Database
        self._db_path = dexter_config.get("database_path", "brain.db")
        self._conn: Optional[sqlite3.Connection] = None
        self._db_ready = False
        
        # Embedding configuration
        self._embedding_cfg = dexter_config.get("embeddings", {})
        self._embedding_enabled = self._embedding_cfg.get("enabled", True)
        
        # Cache for queries
        self._query_cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # State
        self._conversation_state: Dict[str, Any] = {}
        self._subscriber = None
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        self._message_queue = asyncio.Queue()
        
        # Statistics
        self._query_count = 0
        self._cache_hits = 0
    
    async def start(self):
        """Start the Memory TRM with database connection."""
        print("[Memory TRM] Connecting to database...")
        
        # Connect to database
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            print(f"[Memory TRM] Connected to {self._db_path}")
            
            # Check tables
            cursor = self._conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"[Memory TRM] Available tables: {', '.join(tables)}")
            self._db_ready = True
            
        except Exception as e:
            print(f"[Memory TRM] Failed to connect to database: {e}")
            self._conn = None
            self._db_ready = False
        finally:
            try:
                if self._conn:
                    self._conn.close()
            except Exception:
                pass
            self._conn = None
        
        # Check embedding service
        if self._embedding_enabled:
            try:
                # Test embedding
                test_embed = await embedding_client.embed_text_async("test", self._dexter_config)
                if test_embed:
                    print("[Memory TRM] Embedding service working")
                else:
                    print("[Memory TRM] Embedding service returned None, disabling")
                    self._embedding_enabled = False
            except Exception as e:
                print(f"[Memory TRM] Embedding service failed: {e}")
                self._embedding_enabled = False
        else:
            print("[Memory TRM] Embeddings disabled")
        
        # Start listening to broadcasts
        print("[Memory TRM] Online and listening to all broadcasts")
        
        from core.response_tank import Subscriber
        self._subscriber = await self._tank.subscribe(
            subscriber_id="memory_trm",
            source_patterns=["*"]
        )
        
        self._running = True
        self._processing_task = asyncio.create_task(self._read_from_subscriber())
        asyncio.create_task(self._process_broadcasts())
    
    async def stop(self):
        """Stop the Memory TRM."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        if self._subscriber:
            await self._tank.unsubscribe("memory_trm")
        if self._conn:
            self._conn.close()
        
        print(f"[Memory TRM] Offline - Processed {self._query_count} queries, {self._cache_hits} cache hits")
    
    async def _read_from_subscriber(self):
        """Read messages from subscriber and put into internal queue."""
        from core.response_tank import Subscriber
        
        try:
            async with self._subscriber as subscriber:
                while self._running:
                    msg = await subscriber.receive(timeout=0.5)
                    if msg:
                        await self._message_queue.put(msg)
        except Exception as e:
            if self._running:
                print(f"[Memory TRM] Error reading from subscriber: {e}")
    
    async def _process_broadcasts(self):
        """Process messages from the queue."""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=0.5)
                
                if msg:
                    await self._handle_message(msg)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[Memory TRM] Error processing broadcast: {e}")
    
    async def _handle_message(self, message):
        """Handle a single broadcast message."""
        source = message.source
        content = message.content
        
        if source.startswith("user"):
            await self._handle_user_message(content)
        elif source.startswith("llm_"):
            await self._handle_llm_insight(content)
        elif source.startswith("tool_policy_trm"):
            await self._handle_tool_action(content)
    
    async def _handle_user_message(self, content: Dict[str, Any]):
        """Handle user message - trigger deep context building."""
        user_input = content.get("user_input", "")
        if not user_input:
            return
        
        print(f"[Memory TRM] User input: {user_input}")
        
        # Query databases
        memories = await self._query_all_databases(user_input)
        
        # Build deep context
        deep_context = await self._build_deep_context(user_input, memories)
        
        # Update Rolling Context
        await self._rolling_context.update_from_deep_context(deep_context)
        
        # Publish deep context for LLMs
        await self._publish_deep_context(deep_context)
    
    async def _handle_llm_insight(self, content: Dict[str, Any]):
        """Handle LLM insight - add to knowledge graph."""
        insight = content.get("insight", {})
        if not insight:
            return
        
        new_facts = []
        if "content" in insight:
            if insight.get("type") == "fact":
                new_facts.append(insight["content"])
        
        if new_facts:
            await self._rolling_context.update_state({
                "recent_facts": new_facts[:5]
            })
            print(f"[Memory TRM] Added {len(new_facts)} facts from LLM insight")
    
    async def _handle_tool_action(self, content: Dict[str, Any]):
        """Handle tool action - track and learn."""
        pass
    
    async def _query_all_databases(
        self,
        query: str,
        max_results: int = 5
    ) -> Dict[str, List]:
        """Query all databases in parallel."""
        if not self._db_ready:
            return {"episodes": [], "facts": [], "triples": [], "patterns": []}
        
        self._query_count += 1
        
        # Check cache
        cache_key = f"{query}_{max_results}"
        if cache_key in self._query_cache:
            cached_time, cached_result = self._query_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                self._cache_hits += 1
                print(f"[Memory TRM] Cache hit for query")
                return cached_result
        
        # Query each database
        results = {}
        episodes, facts, triples, patterns = await asyncio.gather(
            self._query_episodes(query, max_results),
            self._query_facts(query, max_results),
            self._query_triples(query, max_results),
            self._query_patterns(query, max_results),
        )
        results["episodes"] = episodes
        results["facts"] = facts
        results["triples"] = triples
        results["patterns"] = patterns
        
        # Cache results
        self._query_cache[cache_key] = (time.time(), results)
        
        return results
    
    async def _query_episodes(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Query episodes from history table."""
        if not self._db_ready:
            return []

        try:
            query_embedding = None
            if self._embedding_enabled:
                query_embedding = await embedding_client.embed_text_async(query, self._dexter_config)

            results = await asyncio.to_thread(
                self._query_episodes_sync,
                query,
                max_results,
                query_embedding,
            )

            print(f"[Memory TRM] Found {len(results)} episodes")
            return results

        except Exception as e:
            print(f"[Memory TRM] Error querying episodes: {e}")
            return []
    
    async def _query_facts(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Query facts from knowledge graph."""
        if not self._db_ready:
            return []

        try:
            query_embedding = None

            if self._embedding_enabled:
                query_embedding = await embedding_client.embed_text_async(query, self._dexter_config)
            results = await asyncio.to_thread(
                self._query_facts_sync,
                query,
                max_results,
                query_embedding,
            )

            print(f"[Memory TRM] Found {len(results)} facts")
            return results

        except Exception as e:
            print(f"[Memory TRM] Error querying facts: {e}")
            return []
    
    async def _query_triples(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Query triples from knowledge graph."""
        try:
            results = await asyncio.to_thread(
                self._query_triples_sync,
                query,
                max_results,
            )
            
            print(f"[Memory TRM] Found {len(results)} triples")
            return results
            
        except Exception as e:
            print(f"[Memory TRM] Error querying triples: {e}")
            return []
    
    async def _query_patterns(
        self,
        query: str,
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Query patterns from memory."""
        try:
            results = await asyncio.to_thread(
                self._query_patterns_sync,
                query,
                max_results,
            )
            
            print(f"[Memory TRM] Found {len(results)} patterns")
            return results
            
        except Exception as e:
            print(f"[Memory TRM] Error querying patterns: {e}")
            return []
    
    def _open_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _find_similar_vectors_sync(self, conn: sqlite3.Connection, query_embedding: List[float], max_results: int = 5) -> List[int]:
        """Find similar vectors using cosine similarity (sync)."""
        if not query_embedding:
            return []

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT item_id, embedding FROM vectors")

            import numpy as np

            query_vec = np.array(query_embedding, dtype=np.float32)
            similarities = []

            for row in cursor.fetchall():
                item_id = row["item_id"]
                raw_embedding = row["embedding"]
                if raw_embedding is None:
                    continue
                if isinstance(raw_embedding, (bytes, bytearray, memoryview)):
                    stored_embedding = np.frombuffer(raw_embedding, dtype=np.float32)
                elif isinstance(raw_embedding, str):
                    try:
                        stored_embedding = np.array(json.loads(raw_embedding), dtype=np.float32)
                    except Exception:
                        continue
                elif isinstance(raw_embedding, list):
                    stored_embedding = np.array(raw_embedding, dtype=np.float32)
                else:
                    continue

                if len(stored_embedding) == len(query_vec):
                    similarity = np.dot(query_vec, stored_embedding) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_embedding) + 1e-8)
                    similarities.append((similarity, item_id))

            similarities.sort(reverse=True)
            return [item_id for _, item_id in similarities[:max_results]]
        except Exception as e:
            print(f"[Memory TRM] Error finding similar vectors: {e}")
            return []

    def _query_episodes_sync(self, query: str, max_results: int, query_embedding: Optional[List[float]]) -> List[Dict[str, Any]]:
        conn = self._open_conn()
        try:
            cursor = conn.cursor()
            if query_embedding:
                similar_item_ids = self._find_similar_vectors_sync(conn, query_embedding, max_results)
                if similar_item_ids:
                    placeholders = ", ".join("?" for _ in similar_item_ids)
                    cursor.execute(f"""
                        SELECT id, intent, task, decision, ts
                        FROM history
                        WHERE id IN ({placeholders})
                    """, similar_item_ids)
                    rows = cursor.fetchall()
                else:
                    cursor.execute("""
                        SELECT id, intent, task, decision, ts
                        FROM history
                        WHERE task LIKE ? OR intent LIKE ?
                        ORDER BY ts DESC
                        LIMIT ?
                    """, (f"%{query}%", f"%{query}%", max_results))
                    rows = cursor.fetchall()
            else:
                cursor.execute("""
                    SELECT id, intent, task, decision, ts
                    FROM history
                    WHERE task LIKE ? OR intent LIKE ?
                    ORDER BY ts DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", max_results))
                rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "type": "episode",
                    "content": f"{row['intent']}: {row['task']} -> {row['decision']}",
                    "intent": row["intent"],
                    "task": row["task"],
                    "decision": row["decision"],
                    "timestamp": row["ts"],
                    "similarity": 0.8,
                })
            return results
        finally:
            conn.close()

    def _query_facts_sync(self, query: str, max_results: int, query_embedding: Optional[List[float]]) -> List[Dict[str, Any]]:
        conn = self._open_conn()
        try:
            cursor = conn.cursor()
            similar_item_ids: List[int] = []
            if query_embedding:
                similar_item_ids = self._find_similar_vectors_sync(conn, query_embedding, max_results)

            if similar_item_ids:
                placeholders = ", ".join("?" for _ in similar_item_ids)
                cursor.execute(f"""
                    SELECT id, text, importance
                    FROM facts
                    WHERE id IN ({placeholders})
                    ORDER BY importance DESC
                    LIMIT ?
                """, (*similar_item_ids, max_results))
            else:
                cursor.execute("""
                    SELECT id, text, importance
                    FROM facts
                    WHERE text LIKE ?
                    ORDER BY importance DESC
                    LIMIT ?
                """, (f"%{query}%", max_results))

            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "type": "fact",
                    "text": row["text"],
                    "importance": row["importance"],
                    "relevance": row["importance"],
                })
            return results
        finally:
            conn.close()

    def _query_triples_sync(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        conn = self._open_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, subject, predicate, object, confidence
                FROM triples
                WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ?
                ORDER BY confidence DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", max_results))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "type": "triple",
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "confidence": row["confidence"],
                    "relevance": row["confidence"],
                })
            return results
        finally:
            conn.close()

    def _query_patterns_sync(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        conn = self._open_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, trigger_intent, steps_json, usage_count
                FROM patterns
                WHERE trigger_intent LIKE ?
                ORDER BY usage_count DESC
                LIMIT ?
            """, (f"%{query}%", max_results))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "type": "pattern",
                    "trigger": row["trigger_intent"],
                    "steps": row["steps_json"],
                    "confidence": min(1.0, row["usage_count"] / 10.0),
                    "relevance": min(1.0, row["usage_count"] / 10.0),
                })
            return results
        finally:
            conn.close()
    
    async def _build_deep_context(
        self,
        user_input: str,
        memories: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Build rich deep context from retrieved memories."""
        rolling = await self._rolling_context.get()
        
        deep_context = {
            "user_input": user_input,
            "timestamp": time.time(),
            "conversation_turn": rolling["turn_count"],
            "recent_conversation": rolling["recent_conversation"],
            "semantic_episodes": memories.get("episodes", []),
            "facts": memories.get("facts", []),
            "triples": memories.get("triples", []),
            "patterns": memories.get("patterns", []),
            "current_directory": rolling["current_directory"],
            "current_task": rolling["current_task"],
            "current_project": rolling["current_project"],
            "user_mode": rolling["user_mode"],
            "available_tools": rolling["available_tools"],
            "available_skills": rolling["available_skills"],
            "enabled_llms": rolling["enabled_llms"],
            "environment": {
                "past": rolling["recent_conversation"][-3:] if len(rolling["recent_conversation"]) > 3 else rolling["recent_conversation"],
                "present": rolling["last_action"],
                "future": rolling["current_task"]
            }
        }
        
        return deep_context
    
    async def _publish_deep_context(self, deep_context: Dict[str, Any]):
        """Publish deep context to Response Tank for LLMs."""
        await self._tank.publish(
            source="memory_trm",
            content={
                "type": "deep_context",
                "context": deep_context
            },
            priority="high"
        )


def create_memory_trm(
    dexter_config: Dict[str, Any],
    response_tank: ResponseTank,
    rolling_context: RollingContextBundle
) -> MemoryTRM:
    """Create and initialize the Memory TRM."""
    trm_config = TRMConfig(
        vocab_size=5000,
        seq_len=128,
        hidden_size=256,
        num_heads=4,
        num_layers=2,
        H_cycles=2,
        L_cycles=2
    )
    
    trm = MemoryTRM(
        config=trm_config,
        response_tank=response_tank,
        rolling_context=rolling_context,
        dexter_config=dexter_config
    )
    
    return trm
