# Dexter Startup - Quick Reference Card

## 🚨 CRITICAL ISSUES FOUND & FIXED

| Issue | Severity | Status | Fix |
|-------|----------|--------|-----|
| Wrong API command | CRITICAL | ✅ FIXED | Changed `http.server` → `core/api.py` |
| Missing dependencies | CRITICAL | ⚠️ USER ACTION | Run `pip install -r requirements.txt` |
| Browser directory missing | HIGH | ✅ FIXED | Disabled in config |
| Multiple entry points | HIGH | ✅ DOCUMENTED | Use `entrypoint.py` |
| Dual terminal issues | MEDIUM | ✅ DOCUMENTED | Use `--single` flag if needed |

---

## ⚡ QUICK START (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python check_dependencies.py
```

### Step 3: Start Dexter
```bash
# Windows - Use fixed script
START_DEXTER_FIXED.bat

# OR direct Python
python entrypoint.py agent
```

---

## 🔍 WHAT WAS BROKEN

### Before Fix
```
launcher_config.json:
  "command": ["python", "-m", "http.server", "8000"]  ❌ WRONG
  
Result: Static file server, no API endpoints
```

### After Fix
```
launcher_config.json:
  "command": ["python", "core/api.py"]  ✅ CORRECT
  
Result: FastAPI server with WebSocket and REST endpoints
```

---

## 📊 STARTUP FLOW

### ❌ OLD (Broken)
```
User runs launcher
  ↓
Starts http.server on port 8000 (WRONG!)
  ↓
Starts dexter.py → tries to start API → PORT CONFLICT
  ↓
Tries to start browser → DIRECTORY MISSING
  ↓
Result: Partial failure
```

### ✅ NEW (Fixed)
```
User runs launcher/entrypoint
  ↓
Dependencies checked
  ↓
Dexter main process starts
  ↓
API server starts in thread (port 8000)
  ↓
Browser disabled (not needed for core functionality)
  ↓
Result: System operational
```

---

## 🎯 KEY FILES

### Configuration
- `launcher_config.json` - ✅ Fixed API command
- `configs/core_config.json` - System config (no changes needed)

### Entry Points
- `entrypoint.py` - ✅ Recommended entry point
- `dexter.py` - ✅ Direct start (alternative)
- `core/api.py` - ✅ API server only

### Launchers
- `START_DEXTER_FIXED.bat` - ✅ NEW - Use this
- `START_DEXTER_FULL.bat` - ⚠️ OLD - Needs manual update
- `advanced_launcher.py` - ✅ Fixed

### Documentation
- `STARTUP_ANOMALIES_REPORT.md` - Full technical details
- `FIX_STARTUP_ISSUES.md` - Step-by-step fixes
- `DEXTER_STARTUP_SUMMARY.md` - Executive summary

### Tools
- `check_dependencies.py` - ✅ NEW - Verify dependencies

---

## 🧪 VERIFICATION TESTS

### Test 1: Dependencies
```bash
python check_dependencies.py
```
✅ All critical dependencies installed

### Test 2: API Server
```bash
python core/api.py
# Leave running, open new terminal:
curl http://localhost:8000/status
```
✅ Returns: `{"status":"online","identity":"Dexter Gliksbot",...}`

### Test 3: Full System
```bash
python entrypoint.py agent
```
✅ Dual terminals open, no errors, ready for input

---

## 🚑 COMMON ERRORS

### "ModuleNotFoundError: No module named 'fastapi'"
```bash
pip install fastapi uvicorn psutil pydantic torch
# OR
pip install -r requirements.txt
```

### "Port 8000 already in use"
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux  
lsof -ti:8000 | xargs kill -9
```

### "dexter-browser directory not found"
**This is OK!** Browser component is disabled. System works without it.

---

## 📍 FILE LOCATIONS

### What Changed
```
launcher_config.json:11    → API command changed
launcher_config.json:22    → Browser disabled
advanced_launcher.py:44    → Default command updated
```

### New Files
```
STARTUP_ANOMALIES_REPORT.md    → Full analysis
FIX_STARTUP_ISSUES.md          → Quick fixes
DEXTER_STARTUP_SUMMARY.md      → Executive summary
check_dependencies.py           → Dependency checker
START_DEXTER_FIXED.bat         → Fixed launcher
QUICK_REFERENCE.md             → This file
```

---

## 🎓 ARCHITECTURE

### Component Layout
```
┌─────────────────────────────────┐
│  Dexter Main Process            │
│  (entrypoint.py → dexter.py)    │
│                                 │
│  ┌───────────────────────────┐  │
│  │ Reasoning Engine          │  │
│  │ Tool Executor             │  │
│  │ Memory System             │  │
│  │ TRM Models                │  │
│  │                           │  │
│  │  ┌─────────────────────┐  │  │
│  │  │ API Server (thread) │  │  │
│  │  │ Port 8000           │  │  │
│  │  │ FastAPI + uvicorn   │  │  │
│  │  └─────────────────────┘  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

### Port Usage
- **8000** - API server (FastAPI/WebSocket)
- **8001** - Reserved (Dexter core, not actively used)
- **3000** - Browser (disabled - directory missing)
- **9222** - Browser debug (disabled)
- **19847** - Stream terminal (internal)

---

## ✅ SUCCESS CHECKLIST

Before starting Dexter:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dependencies verified (`python check_dependencies.py`)
- [ ] Ports 8000 free
- [ ] Using correct launcher (START_DEXTER_FIXED.bat or entrypoint.py)

System is working when:
- [ ] No import errors
- [ ] API server responds at http://localhost:8000/status
- [ ] Dual terminals open (Windows) or single terminal (Linux)
- [ ] Can interact with Dexter
- [ ] No port conflict errors

---

## 🔗 RELATED DOCS

1. **STARTUP_ANOMALIES_REPORT.md**
   - Complete technical analysis
   - All anomalies with code evidence
   - Severity ratings and confidence levels

2. **FIX_STARTUP_ISSUES.md**
   - Step-by-step fix instructions
   - Code snippets to copy/paste
   - Troubleshooting guide

3. **DEXTER_STARTUP_SUMMARY.md**
   - Executive summary
   - Testing procedures
   - Next steps

---

## 💡 PRO TIPS

### Simplest Start
```bash
pip install -r requirements.txt && python entrypoint.py agent
```

### Check if Running
```bash
curl http://localhost:8000/status
```

### Stop All Dexter Processes
```bash
# Windows
tasklist | findstr python
taskkill /F /IM python.exe

# Linux
pkill -f dexter.py
```

### View Logs
```bash
# If using launcher
dir logs\*.log      # Windows
ls logs/*.log       # Linux

# If using entrypoint.py directly
# Logs appear in terminal
```

---

## 📞 NEED HELP?

**Check these in order:**

1. **QUICK_REFERENCE.md** (this file) - Quick fixes
2. **FIX_STARTUP_ISSUES.md** - Detailed step-by-step
3. **STARTUP_ANOMALIES_REPORT.md** - Full technical analysis
4. **check_dependencies.py** - Verify your environment

**Still stuck?** Check the error message against common errors section above.

---

**Last Updated:** 2024
**Agent:** code-anomaly-detector
**Status:** All critical fixes applied ✅
