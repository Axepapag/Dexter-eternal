#!/usr/bin/env python3
"""
context_bundle.py - Build rich, untruncated context bundles for Dexter.
"""

import json
import os
import re
import sqlite3
import time
import asyncio
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.importance import record_access
from core.vector_store import search_similar

from skills.memory_ops import _db_path as _memory_db_path


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "so", "to",
    "of", "in", "on", "at", "by", "for", "with", "from", "about", "as",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "you", "your", "yours", "me", "my", "mine", "we", "our",
    "i", "he", "she", "they", "them", "his", "her", "their", "what", "why",
    "how", "when", "where", "who", "which", "do", "does", "did", "can", "could",
    "should", "would", "will", "shall", "may", "might", "not", "no", "yes",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class ContextBundler:
    def __init__(self, repo_root: Path, config: Dict[str, Any]):
        self.config = config
        cfg = config.get("context_bundle", {}) or {}
        self.max_chat_turns = int(cfg.get("max_chat_turns", 12))
        self.max_facts = int(cfg.get("max_facts", 20))
        self.max_triples = int(cfg.get("max_triples", 30))
        self.max_keywords = int(cfg.get("max_keywords", 12))
        self.max_bundles = int(cfg.get("max_bundles", 5))
        self.session = cfg.get("session") or os.getenv("BRAIN_SESSION", "core")
        self.repo_root = repo_root

    def build_bundle(
        self,
        intent: str,
        task: str,
        chat_history: List[Dict[str, str]],
        tool_result: Optional[Dict[str, Any]] = None,
        tool_meta: Optional[Dict[str, Any]] = None,
        plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chat_tail = chat_history[-self.max_chat_turns:] if self.max_chat_turns > 0 else []
        texts = [intent or "", task or ""]
        texts.extend([m.get("content", "") for m in chat_tail if m.get("content")])
        keywords = self._extract_keywords(texts)
        facts = self._fetch_facts(keywords)
        triples = self._fetch_triples(keywords)
        vectors = self._fetch_vectors(" ".join(texts))
        bundle = {
            "ts": time.time(),
            "intent": intent,
            "task": task,
            "keywords": keywords,
            "conversation_tail": chat_tail,
            "facts": facts,
            "triples": triples,
            "vectors": vectors,
            "tool_meta": tool_meta or {},
            "tool_result": tool_result,
            "plan": plan or {},
        }
        return bundle

    async def build_bundle_async(
        self,
        intent: str,
        task: str,
        chat_history: List[Dict[str, str]],
        tool_result: Optional[Dict[str, Any]] = None,
        tool_meta: Optional[Dict[str, Any]] = None,
        plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.build_bundle,
            intent,
            task,
            chat_history,
            tool_result,
            tool_meta,
            plan,
        )

    def build_snapshot(
        self,
        intent: str,
        task: str,
        chat_history: List[Dict[str, str]],
        plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.build_bundle(
            intent=intent,
            task=task,
            chat_history=chat_history,
            tool_result=None,
            tool_meta={"phase": "pre_tool"},
            plan=plan,
        )

    async def build_snapshot_async(
        self,
        intent: str,
        task: str,
        chat_history: List[Dict[str, str]],
        plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.build_snapshot,
            intent,
            task,
            chat_history,
            plan,
        )

    def _extract_keywords(self, texts: Iterable[str]) -> List[str]:
        counts: Counter = Counter()
        for text in texts:
            for token in _tokenize(text or ""):
                counts[token] += 1
        if not counts:
            return []
        return [w for w, _ in counts.most_common(self.max_keywords)]

    def _fetch_facts(self, keywords: List[str]) -> List[Dict[str, Any]]:
        if not keywords or self.max_facts <= 0:
            return []
        per_term = max(1, self.max_facts // max(len(keywords), 1))
        results: List[Dict[str, Any]] = []
        seen = set()
        conn = sqlite3.connect(_memory_db_path(), timeout=10, check_same_thread=False)
        c = conn.cursor()
        try:
            for term in keywords:
                c.execute(
                    "SELECT id, text, session, ts, importance, access_count, last_accessed, source "
                    "FROM facts WHERE text LIKE ? ORDER BY id DESC LIMIT ?",
                    (f"%{term}%", per_term),
                )
                for row in c.fetchall():
                    if row[0] in seen:
                        continue
                    seen.add(row[0])
                    record_access(conn, "facts", "id", row[0], config=self.config)
                    results.append({
                        "id": row[0],
                        "text": row[1],
                        "session": row[2],
                        "ts": row[3],
                        "importance": row[4],
                        "access_count": row[5],
                        "last_accessed": row[6],
                        "source": row[7],
                    })
                    if len(results) >= self.max_facts:
                        break
                if len(results) >= self.max_facts:
                    break
        finally:
            conn.close()
        return results

    def _fetch_triples(self, keywords: List[str]) -> List[Dict[str, Any]]:
        if not keywords or self.max_triples <= 0:
            return []
        per_term = max(1, self.max_triples // max(len(keywords), 1))
        results: List[Dict[str, Any]] = []
        seen = set()
        conn = sqlite3.connect(_memory_db_path(), timeout=10, check_same_thread=False)
        c = conn.cursor()
        try:
            for term in keywords:
                c.execute(
                    "SELECT id, subject, predicate, object, ts, confidence, source, context, session, importance, access_count, last_accessed "
                    "FROM triples WHERE subject LIKE ? OR object LIKE ? ORDER BY id DESC LIMIT ?",
                    (f"%{term}%", f"%{term}%", per_term),
                )
                for row in c.fetchall():
                    key = (row[1], row[2], row[3], row[4])
                    if key in seen:
                        continue
                    seen.add(key)
                    record_access(conn, "triples", "id", row[0], config=self.config)
                    try:
                        ctx = json.loads(row[7]) if row[7] else {}
                    except json.JSONDecodeError:
                        ctx = {}
                    results.append({
                        "subject": row[1],
                        "predicate": row[2],
                        "object": row[3],
                        "ts": row[4],
                        "confidence": row[5],
                        "source": row[6],
                        "context": ctx,
                        "session": row[8],
                        "importance": row[9],
                        "access_count": row[10],
                        "last_accessed": row[11],
                    })
                    if len(results) >= self.max_triples:
                        break
                if len(results) >= self.max_triples:
                    break
        finally:
            conn.close()
        return results

    def _fetch_vectors(self, query_text: str) -> List[Dict[str, Any]]:
        cfg = self.config.get("embeddings", {}) or {}
        if not cfg.get("enabled", False):
            return []
        bundle_cfg = self.config.get("context_bundle", {}) or {}
        limit = int(bundle_cfg.get("max_vectors", 0))
        if limit <= 0:
            return []
        scan_limit = int(bundle_cfg.get("vector_scan_limit", 500))
        conn = sqlite3.connect(_memory_db_path(), timeout=10, check_same_thread=False)
        try:
            return search_similar(
                conn=conn,
                query_text=query_text,
                config=self.config,
                limit=limit,
                scan_limit=scan_limit,
                item_types=bundle_cfg.get("vector_item_types"),
            )
        finally:
            conn.close()
