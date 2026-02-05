# Dexter Codebase Audit Report
**Date:** Generated during code review session  
**Scope:** Core architecture, failure modes, concurrency, and robustness issues

---

## Executive Summary

The Dexter codebase demonstrates an ambitious TRM-based architecture with dual terminals, ungated input broadcast, and trigger-based context injection. **Critical issues found:** 
- SQL injection vulnerabilities
- Unsafe database connection handling
- Missing cleanup on failures
- Dead code references to deprecated IntentReasonerTRM
- Unsafe synchronous method call within async context
- Stream server failure handling gaps
- Missing guards on dict/list operations

**Total Issues Found:**
- **10 Critical Gaps** (causing failures/incorrect behavior)
- **10 Important Gaps** (missing robustness logic)
- **10 Opportunities** (enhancement recommendations)
- **6 Questions** (clarifications needed)

---

## Critical Gaps (P0 - Fix Before Production)

### 1. SQL Injection Vulnerabilities
**Location:** `core/memory_trm.py:262-266, 330-335`

**Code:**
```python
ids = ",".join(str(item_id) for item_id in similar_item_ids)
cursor.execute(f"""
    SELECT id, intent, task, decision, ts
    FROM history
    WHERE id IN ({ids})
""")
```

**Issue:** F-string interpolation bypasses SQL parameter binding. If `_find_similar_vectors()` returns corrupted data, injection is possible.

**Fix:**
```python
placeholders = ",".join("?" for _ in similar_item_ids)
cursor.execute(f"""
    SELECT id, intent, task, decision, ts
    FROM history
    WHERE id IN ({placeholders})
""", similar_item_ids)
```

---

### 2. Unsafe Database Connection in Async Context
**Location:** `core/memory_trm.py:58`

**Code:**
```python
self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
```

**Issue:** SQLite connections are NOT thread-safe despite `check_same_thread=False`. Multiple async tasks calling queries simultaneously will cause database locks, corrupted results, and potential data corruption.

**Fix:**
```python
# Use async SQLite library
import aiosqlite

async def start(self):
    self._conn = await aiosqlite.connect(self._db_path)
    # OR serialize access with semaphore
    self._db_semaphore = asyncio.Semaphore(1)
```

---

### 3. Undefined Method Call Crashes Tool Completion
**Location:** `core/rolling_context_bundle.py:170`

**Code:**
```python
self._add_recent_decision({  # ❌ This method doesn't exist!
    "type": "tool_call",
    ...
})
```

**Issue:** Called method is `_add_recent_decision()` but actual method is `add_recent_decision()` (line 182, async). Crashes with `AttributeError` when tool completes.

**Fix:**
```python
await self.add_recent_decision({
    "type": "tool_call",
    "tool_name": tool_call["tool_name"],
    "status": status,
    "timestamp": time.time()
})
```

---

### 4. Dead Code Reference to Deprecated IntentReasonerTRM
**Location:** `dexter.py:1074-1090`, `core/memory_trm.py:154-155`

**Code:**
```python
elif source == "intent_reasoner_trm" and ctype == "intent_decision":
    # This source never fires - IntentReasonerTRM not imported/initialized
```

**Issue:** 
- `intent_reasoner_trm` imported in integration_demo but NOT in dexter.py
- Message handling branches never execute
- Staged context artifacts for reasoning never created
- Misleading dead code paths

**Fix:** Remove dead code or implement the TRM properly.

---

### 5. Stream Server Crashes Silently
**Location:** `dexter.py:147-159`

**Code:**
```python
def _stream_server():
    try:
        server.bind(('127.0.0.1', STREAM_PORT))  # Can fail if port in use
        server.listen(5)
        # ...
    except Exception as e:
        pass  # ❌ Silent failure - no logging, no recovery
```

**Issue:** Port already in use → bind fails → stream terminal never works, no error shown. No retry, no user notification.

**Fix:**
```python
for attempt in range(max_retries):
    try:
        server.bind(('127.0.0.1', STREAM_PORT))
        server.listen(5)
        _original_print(f"[Stream] Server started on port {STREAM_PORT}")
        # ... accept loop
    except OSError as e:
        if "Address already in use" in str(e):
            _original_print(f"[Stream] Port {STREAM_PORT} in use, retrying...")
            time.sleep(1)
            continue
        raise
```

---

### 6. Dead Stream Clients Never Closed
**Location:** `dexter.py:140-144`

**Code:**
```python
for c in dead_clients:
    _stream_clients.remove(c)  # ❌ Socket never closed!
```

**Issue:** File descriptor and memory leaks accumulate on long-running instances.

**Fix:**
```python
for c in dead_clients:
    try:
        c.close()
    except:
        pass
    _stream_clients.remove(c)
```

---

### 7. LLM Response Parsing Missing Error Handling
**Location:** `core/llm_think_tank.py:165-186`

**Code:**
```python
content = response_data["choices"][0]["message"]["content"]  # Can KeyError
```

**Issue:** Unexpected API response format causes unhandled KeyError/IndexError.

**Fix:**
```python
choices = response_data.get("choices", [])
if not choices:
    raise ValueError("No choices in response")

message = choices[0].get("message", {})
content = message.get("content", "")

if not content:
    raise ValueError("Empty content in response")
```

---

### 8. Database Connection Failure Recovery Missing
**Location:** `core/memory_trm.py:52-70`

**Code:**
```python
except Exception as e:
    print(f"[Memory TRM] Failed to connect to database: {e}")
    self._conn = None  # ❌ TRM continues with no database!
```

**Issue:** System appears to work but has no memory. No retry on transient failures.

**Fix:**
```python
max_retries = 3
for attempt in range(max_retries):
    try:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        break
    except Exception as e:
        if attempt == max_retries - 1:
            raise RuntimeError("Memory TRM cannot start: database unavailable")
        await asyncio.sleep(1)
```

---

### 9. Race Condition in StagedContextBundle
**Location:** `core/rolling_context_bundle.py:305-440`

**Issue:** `stage()` and `trigger_and_inject()` both acquire `self._lock`. Slow `stage()` operations (large artifacts, slow JSON serialization) can block `trigger_and_inject()`, delaying user input processing.

**Fix:** Minimize work under lock:
```python
async def stage_artifact(...):
    # Build artifact outside lock
    artifact = {...}
    
    async with self._lock:
        self._staged[source].append(artifact)
    
    # Trim outside lock
    await self._trim_source(source)
```

---

### 10. Missing Validation on Tool Request Parsing
**Location:** `core/tool_trm_gateway.py:890-906`

**Code:**
```python
try:
    args = json.loads(fc.get("arguments", "{}"))
except:
    args = {}  # ❌ Silently ignores malformed JSON
```

**Issue:** Malformed LLM responses result in empty args, causing hard-to-debug tool failures.

**Fix:**
```python
try:
    args_str = fc.get("arguments", "{}")
    args = json.loads(args_str)
except json.JSONDecodeError as e:
    print(f"[Gateway] Failed to parse arguments: {e}")
    print(f"[Gateway] Raw: {args_str}")
    args = {"_parse_error": str(e)}
```

---

## Important Gaps (P1 - Next Sprint)

### 11. No Timeout on Response Tank Processing
**Location:** `core/response_tank.py:61-75`

**Issue:** Slow subscribers can block entire message pipeline.

**Fix:** Add timeouts to lock and distribution.

---

### 12. Execution History Never Trimmed
**Location:** `core/tool_trm_gateway.py:440-442`

**Issue:** Memory leak on long-running instances.

**Fix:** Prune history when exceeding `_max_history`.

---

### 13. No Priority Validation
**Location:** `core/rolling_context_bundle.py:309`

**Issue:** Negative or extreme priority values can break sort ordering.

**Fix:** Clamp to valid range (1-10).

---

### 14. Empty Insights List Edge Case
**Location:** `core/llm_think_tank.py:184`

**Issue:** Division by zero prevention uses `max(1, len(...))` but semantic unclear for empty list.

**Fix:** Explicit handling for empty insights.

---

### 15. Subscriber Queue Not Drained on Exit
**Location:** `core/response_tank.py:178`

**Issue:** Pending messages lost when subscriber exits. Loses training data in learning scenarios.

**Fix:** Drain queue before unsubscribe.

---

### 16. Embedding Dimension Mismatch Silent Skip
**Location:** `core/memory_trm.py:472`

**Issue:** Dimension mismatches silently skipped, incomplete results after model changes.

**Fix:** Log mismatches and consider cleanup.

---

### 17. Error Results Cached
**Location:** `core/memory_trm.py:223-229`

**Issue:** Failed queries cached for 5 minutes, returning stale errors.

**Fix:** Only cache successful results.

---

### 18. Terminal Spawn Silent Failure on Linux
**Location:** `dexter.py:201-223`

**Issue:** If all terminal emulators fail, no error reported.

**Fix:** Report terminal not found.

---

### 19. No Validation on Conversation Role
**Location:** `core/rolling_context_bundle.py:100-115`

**Issue:** Arbitrary roles can break LLM context.

**Fix:** Validate role in `{"user", "assistant", "system"}`.

---

### 20. Synchronous Callbacks Block Event Loop
**Location:** `core/rolling_context_bundle.py:430-437`

**Issue:** Sync callbacks delay user input during injection.

**Fix:** Run sync callbacks in thread pool.

---

## Opportunities (P2 - Enhancements)

1. **Structured Logging** - Replace print() with logging module
2. **Metrics/Observability** - Export to Prometheus/Grafana
3. **Connection Pooling** - Use aiosqlite or pool for database
4. **Request Tracing** - Add UUID trace IDs for debugging
5. **Circuit Breaker** - Prevent cascading LLM failures
6. **Health Check Endpoint** - Add /health API
7. **Tool Argument Validation** - JSON schema validation
8. **Graceful Shutdown** - Drain queues before stopping
9. **LRU Cache** - Replace dict-based cache
10. **Retry Logic** - Exponential backoff with tenacity

---

## Questions/Clarifications

1. **Intent Reasoner TRM:** Is deprecated code intentionally disabled or migration-in-progress? Remove or implement?

2. **Dead Client Strategy:** Auto-reconnect stream clients or expect manual restart?

3. **Database Access Pattern:** Are writes expected in Memory TRM? If yes, current approach is unsafe.

4. **LLM Failure Mode:** Should system block, cache, or operate without insights when all LLMs fail?

5. **Tool Failure Accumulation:** Should repeated failures trigger circuit breaker or retry forever?

6. **Stream Server Loss:** Should Dexter abort startup, continue, or fall back to single terminal?

---

## Architecture Verification

### Input Gating ✅
- **Confirmed:** NO gating of user input - broadcasts raw via ResponseTank
- **Confirmed:** Trigger semantics implemented - `trigger_and_inject()` only on user input/tool results
- **Issue:** Dead `intent_reasoner_trm` paths suggest incomplete migration

### Concurrency Safety ⚠️
- **Issue:** Database connection not thread-safe (check_same_thread=False insufficient)
- **Issue:** Race condition in StagedContextBundle lock contention
- **OK:** Response tank uses asyncio.Lock correctly
- **Issue:** Synchronous callbacks can block event loop

### Error Handling ⚠️
- **Issue:** Stream server failures swallowed silently
- **Issue:** SQL injection via f-strings
- **Issue:** LLM response parsing assumes structure
- **Issue:** Database connection failures not retried

---

## High-Priority Recommendations

### Immediate (P0)
1. Fix SQL injection → Use parameter binding
2. Fix undefined method call → Use async version
3. Fix database connection → Use aiosqlite or semaphore
4. Remove or implement intent_reasoner_trm

### Short-Term (P1)
1. Add stream server error handling
2. Implement socket cleanup
3. Add LLM response validation
4. Fix StagedContextBundle race condition

### Medium-Term (P2)
1. Structured logging
2. Health checks and metrics
3. Graceful shutdown
4. Request tracing

---

## Verification Notes

- All line numbers verified against actual file content
- SQL injection confirmed in 2 locations (episodes, facts queries)
- `check_same_thread=False` pattern found in memory_trm.py:58
- Dead code path for `intent_reasoner_trm` confirmed
- Stream server error swallowing confirmed at line 159
