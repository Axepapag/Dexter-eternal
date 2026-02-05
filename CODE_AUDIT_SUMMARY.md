# Dexter Code Audit - Summary Report

**Audit Date:** 2025  
**Scope:** Dual terminals, ungated input broadcast, trigger injection, Forge routing, Tool TRM gateway, Memory/ThinkTank artifacts, logging

---

## Summary

The codebase successfully implements ungated input broadcast and trigger-based injection semantics as intended. However, **10 critical gaps** threaten production stability: SQL injection vulnerabilities, unsafe async database access, undefined method crashes, dead code paths, stream server silent failures, resource leaks, and missing error guards. The architecture's core intent is sound, but robustness issues around failure modes, concurrency, and cleanup require immediate attention.

---

## Critical Gaps

### 1. **SQL Injection Vulnerabilities** (P0 - Security)
**Location:** `core/memory_trm.py:262-266, 330-335`

F-string interpolation of `similar_item_ids` bypasses SQL parameter binding:
```python
ids = ",".join(str(item_id) for item_id in similar_item_ids)
cursor.execute(f"""SELECT id, intent, task FROM history WHERE id IN ({ids})""")
```

**Why it matters:** Corrupted vector search results or malicious data can inject SQL.

**Fix:**
```python
placeholders = ",".join("?" for _ in similar_item_ids)
cursor.execute(f"SELECT id, intent FROM history WHERE id IN ({placeholders})", similar_item_ids)
```

---

### 2. **Unsafe Database Access in Async Context** (P0 - Data Corruption)
**Location:** `core/memory_trm.py:58`

```python
self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
```

**Why it matters:** SQLite is NOT thread-safe despite `check_same_thread=False`. Multiple async tasks calling `_query_episodes()`, `_query_facts()`, etc. simultaneously will cause database locks, corrupted results, or data corruption.

**Fix:**
```python
import aiosqlite
async def start(self):
    self._conn = await aiosqlite.connect(self._db_path)
# OR use semaphore:
self._db_semaphore = asyncio.Semaphore(1)
```

---

### 3. **Undefined Method Crashes Tool Completion** (P0 - Runtime Crash)
**Location:** `core/rolling_context_bundle.py:170`

```python
self._add_recent_decision({  # ❌ No such method
    "type": "tool_call",
    ...
})
```

**Why it matters:** `AttributeError` crashes tool completion flow. Actual method is `add_recent_decision()` (async, line 182).

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

### 4. **Dead Code Paths - Deprecated IntentReasonerTRM** (P0 - Misleading Logic)
**Location:** `dexter.py:248-250`, `core/memory_trm.py:154-155`

```python
# dexter.py:248
# IntentReasonerTRM is deprecated: raw user input must broadcast directly to all modules.
self.intent_trm_enabled = False

# BUT memory_trm.py:154 still listens:
elif source.startswith("intent_reasoner_trm"):
    await self._handle_intent_decision(content)  # Never fires!
```

**Why it matters:** Dead branches never execute, wasting CPU and misleading debugging.

**Fix:** Remove all `intent_reasoner_trm` references or complete migration.

---

### 5. **Stream Server Silent Failures** (P0 - User Experience)
**Location:** `dexter.py:147-159`

```python
def _stream_server():
    try:
        server.bind(('127.0.0.1', STREAM_PORT))  # Can fail
        server.listen(5)
        # ... loop
    except Exception as e:
        pass  # ❌ Silent failure
```

**Why it matters:** Port conflict → bind fails → stream terminal never works, no error logged.

**Fix:**
```python
for attempt in range(3):
    try:
        server.bind(('127.0.0.1', STREAM_PORT))
        _original_print(f"[Stream] Server started on port {STREAM_PORT}")
        break
    except OSError as e:
        if "Address already in use" in str(e):
            _original_print(f"[Stream] Port {STREAM_PORT} in use, retrying...")
            time.sleep(1)
        else:
            raise
```

---

### 6. **Resource Leak - Dead Stream Clients Never Closed** (P0 - Memory Leak)
**Location:** `dexter.py:140-144`

```python
for c in dead_clients:
    _stream_clients.remove(c)  # ❌ Socket never closed!
```

**Why it matters:** File descriptors accumulate over time.

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

### 7. **LLM Response Parsing Missing Guards** (P0 - Crash on Unexpected Response)
**Location:** `core/llm_think_tank.py:167`

```python
content = response_data["choices"][0]["message"]["content"]  # KeyError/IndexError
```

**Why it matters:** Unexpected API format (rate limits, errors) crashes LLM advisor.

**Fix:**
```python
choices = response_data.get("choices", [])
if not choices:
    raise ValueError("No choices in response")
message = choices[0].get("message", {})
content = message.get("content", "")
if not content:
    raise ValueError("Empty content")
```

---

### 8. **No Database Connection Retry** (P0 - Permanent Memory Loss)
**Location:** `core/memory_trm.py:68-70`

```python
except Exception as e:
    print(f"[Memory TRM] Failed to connect to database: {e}")
    self._conn = None  # ❌ System continues without memory
```

**Why it matters:** Transient failure (file locked, disk full) → permanent loss of memory system.

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

### 9. **Race Condition in StagedContextBundle** (P0 - Input Lag)
**Location:** `core/rolling_context_bundle.py:305-440`

Lock held during expensive operations (JSON serialization, artifact deduplication):
```python
async def stage_artifact(...):
    async with self._lock:  # ⏳ Blocks trigger_and_inject
        # ... build artifact
        # ... dedupe
        # ... trim to budget
```

**Why it matters:** Slow `stage()` from background module delays `trigger_and_inject()` for user input.

**Fix:** Minimize work under lock:
```python
# Build artifact outside lock
artifact = {...}
async with self._lock:
    self._staged[source].append(artifact)
# Trim outside lock
await self._trim_source(source)
```

---

### 10. **Malformed JSON Silent Ignore** (P0 - Tool Failures)
**Location:** `core/tool_trm_gateway.py:896-898`

```python
try:
    args = json.loads(fc.get("arguments", "{}"))
except:
    args = {}  # ❌ Silently ignores malformed JSON
```

**Why it matters:** LLM generates invalid JSON → args empty → tool fails mysteriously.

**Fix:**
```python
try:
    args_str = fc.get("arguments", "{}")
    args = json.loads(args_str)
except json.JSONDecodeError as e:
    print(f"[Gateway] Failed to parse arguments: {e}")
    print(f"[Gateway] Raw: {args_str}")
    args = {"_parse_error": str(e), "_raw": args_str}
```

---

## Important Gaps

### 11. No Timeout on Response Tank Processing
**Location:** Response tank message distribution  
**Impact:** Slow subscribers block pipeline  
**Fix:** Add timeout to lock acquisition

### 12. Execution History Never Trimmed
**Location:** `core/tool_trm_gateway.py:440-442`  
**Impact:** Memory leak on long-running instances  
**Fix:** Prune when exceeding `_max_history`

### 13. No Priority Value Validation
**Location:** `core/rolling_context_bundle.py:309`  
**Impact:** Negative/extreme priority breaks sort  
**Fix:** Clamp to 1-10 range

### 14. Empty Insights List Edge Case
**Location:** `core/llm_think_tank.py:184`  
**Impact:** Division by `max(1, len([]))` semantically unclear  
**Fix:** Explicit handling for 0 insights

### 15. Subscriber Queue Not Drained on Exit
**Location:** Response tank unsubscribe  
**Impact:** Loses pending training data  
**Fix:** Drain queue before unsubscribe

### 16. Embedding Dimension Mismatch Silent Skip
**Location:** `core/memory_trm.py:472`  
**Impact:** Incomplete results after model change  
**Fix:** Log mismatches, cleanup old embeddings

### 17. Error Results Cached
**Location:** `core/memory_trm.py:224-229`  
**Impact:** Failed queries cached 5 minutes  
**Fix:** Only cache successful results

### 18. Terminal Spawn Silent Failure (Linux)
**Location:** `dexter.py:214-223`  
**Impact:** No error if all emulators fail  
**Fix:** Report terminal not found

### 19. No Conversation Role Validation
**Location:** `core/rolling_context_bundle.py:100`  
**Impact:** Arbitrary roles break LLM context  
**Fix:** Validate `role in {"user", "assistant", "system"}`

### 20. Synchronous Callbacks Block Event Loop
**Location:** `core/rolling_context_bundle.py:432-436`  
**Impact:** Delays user input during injection  
**Fix:** Run sync callbacks in thread pool

---

## Opportunities

1. **Structured Logging:** Replace `print()` with logging module (levels, rotation, filters)
2. **Metrics/Observability:** Export to Prometheus/Grafana for production monitoring
3. **Connection Pooling:** Use aiosqlite or connection pool for database
4. **Request Tracing:** Add UUID trace IDs across components for debugging
5. **Circuit Breaker:** Prevent cascading failures when LLM APIs down
6. **Health Check Endpoint:** Add `/health` API for monitoring
7. **Tool Argument Validation:** JSON schema validation before execution
8. **Graceful Shutdown:** Drain queues and close connections on SIGTERM
9. **LRU Cache:** Replace dict-based cache with `functools.lru_cache`
10. **Retry Logic:** Add exponential backoff with `tenacity` library

---

## Questions

1. **Intent Reasoner TRM:** Is deprecated code intentionally disabled or migration-in-progress? Should it be removed or implemented?

2. **Dead Client Strategy:** Should stream clients auto-reconnect, or is manual restart expected?

3. **Database Write Pattern:** Are writes expected in Memory TRM? If yes, current approach is unsafe for async writes.

4. **LLM Failure Mode:** When all LLMs fail, should system block, use cached insights, or continue without?

5. **Tool Failure Accumulation:** Should repeated tool failures trigger circuit breaker or retry forever?

6. **Stream Server Loss:** Should Dexter abort startup, continue without stream, or fall back to single terminal?

---

## Architecture Verification

### Input Gating ✅
- **Confirmed:** NO gating of user input - raw broadcast via ResponseTank  
- **Confirmed:** Trigger-only injection - `trigger_and_inject()` only on user input/tool results  
- **Issue:** Dead `intent_reasoner_trm` paths suggest incomplete migration

### Concurrency Safety ⚠️
- **Issue:** Database connection not thread-safe (`check_same_thread=False` insufficient)  
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

### Immediate (This Week)
1. ✅ Fix SQL injection → Use parameter binding
2. ✅ Fix undefined method call → Use async version
3. ✅ Fix database connection → Use aiosqlite or semaphore
4. ✅ Remove or implement intent_reasoner_trm

### Short-Term (Next Sprint)
1. Add stream server error handling and retry
2. Implement socket cleanup for dead clients
3. Add LLM response validation guards
4. Fix StagedContextBundle race condition

### Medium-Term (Next Month)
1. Structured logging with log levels
2. Health checks and metrics endpoints
3. Graceful shutdown handlers
4. Request tracing with correlation IDs

---

**Full detailed report:** See `CODEBASE_AUDIT_REPORT.md`
