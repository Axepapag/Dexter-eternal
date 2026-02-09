# Quick Fix Guide for Dexter Startup Issues

## CRITICAL FIXES - Apply Immediately

### Fix 1: Correct API Server Command in launcher_config.json

**Current (BROKEN):**
```json
"api_server": {
  "enabled": true,
  "command": ["python", "-m", "http.server", "8000"],
  "working_dir": ".",
  "wait_time": 3
}
```

**Fixed:**
```json
"api_server": {
  "enabled": true,
  "command": ["python", "core/api.py"],
  "working_dir": ".",
  "wait_time": 3
}
```

**Location:** `/launcher_config.json` lines 9-14

---

### Fix 2: Install Missing Python Dependencies

**Run this command:**
```bash
pip install -r requirements.txt
```

**Required packages:**
- fastapi
- uvicorn[standard]
- psutil
- torch
- pydantic
- aiohttp
- and others from requirements.txt

---

### Fix 3: Disable Missing Browser Component

**Edit launcher_config.json:**
```json
"browser": {
  "enabled": false,
  "command": ["npm", "start"],
  "working_dir": "dexter-browser",
  "wait_time": 10
}
```

**Location:** `/launcher_config.json` lines 22-27

---

### Fix 4: Update START_DEXTER_FULL.bat

**Find line 76:**
```batch
start "Dexter API Server" cmd /k "cd /d %~dp0 && python -m http.server 8000 > logs\api_server.log 2>&1"
```

**Replace with:**
```batch
start "Dexter API Server" cmd /k "cd /d %~dp0 && python core\api.py > logs\api_server.log 2>&1"
```

---

## RECOMMENDED: Simplify Architecture

### Option A: Let Dexter Core Start API (Recommended)

The `dexter.py` already starts the API server internally at line 3097:
```python
threading.Thread(target=start_api_server, daemon=True).start()
```

**So you can:**

1. **Remove the api_server component entirely from launcher_config.json**

2. **Only start dexter_core:**
```json
"components": {
  "dexter_core": {
    "enabled": true,
    "command": ["python", "entrypoint.py", "agent"],
    "working_dir": ".",
    "wait_time": 5
  },
  "browser": {
    "enabled": false
  }
}
```

3. **Simplified START_DEXTER_FULL.bat:**
```batch
@echo off
title Dexter Launcher
echo Installing/checking dependencies...
python -m pip install -r requirements.txt --quiet

echo Starting Dexter (includes API server)...
python entrypoint.py agent
```

---

## Verification Steps

### 1. Test API Server Standalone
```bash
python core/api.py
```
Expected output:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Then visit: http://localhost:8000/status
Should return: `{"status": "online", "identity": "Dexter Gliksbot", "user": "Jeffrey Gliksman"}`

### 2. Test Full System
```bash
python entrypoint.py agent
```

Expected:
- Two terminal windows open (if on Windows)
- API server starts on port 8000
- No port conflicts
- System enters conversation mode

### 3. Test Advanced Launcher
```bash
python advanced_launcher.py --start
```

Expected:
- Components start without errors
- Logs appear in `logs/` directory
- Status shows components running

---

## Common Errors and Solutions

### Error: ModuleNotFoundError: No module named 'fastapi'
**Solution:** Run `pip install -r requirements.txt`

### Error: Port 8000 already in use
**Solution:** 
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux
lsof -ti:8000 | xargs kill -9
```

### Error: dexter-browser directory not found
**Solution:** Disable browser component in launcher_config.json (set enabled: false)

### Error: Two API servers starting
**Solution:** Remove api_server from launcher_config.json, only run dexter_core

---

## File Modifications Summary

### Files to Edit:
1. **launcher_config.json** - Lines 10-13, 23
2. **START_DEXTER_FULL.bat** - Line 76
3. **advanced_launcher.py** - Line 44 (if using advanced launcher)

### Files to Create (Optional):
1. **check_deps.py** - Dependency verification script
2. **simple_start.bat** - Minimal startup script

---

## Minimal Working Configuration

**launcher_config.json:**
```json
{
  "ports": {
    "api_server": 8000,
    "dexter_core": 8001
  },
  "components": {
    "dexter_core": {
      "enabled": true,
      "command": ["python", "entrypoint.py", "agent"],
      "working_dir": ".",
      "wait_time": 5
    }
  },
  "max_retries": 3,
  "startup_timeout": 30,
  "log_level": "INFO"
}
```

**simple_start.bat:**
```batch
@echo off
pip install -r requirements.txt --quiet
python entrypoint.py agent
pause
```

---

## Next Steps

1. ✅ Apply Fix 1: Update launcher_config.json API command
2. ✅ Apply Fix 2: Install dependencies
3. ✅ Apply Fix 3: Disable browser component
4. ✅ Test: Run `python core/api.py` to verify API works
5. ✅ Test: Run `python entrypoint.py agent` to verify full system
6. ⚠️ Consider: Creating/restoring the dexter-browser frontend

---

## If You Want the GUI

The system expects a Next.js dashboard in `dexter-browser/` that:
- Connects to WebSocket at `ws://localhost:8000/ws`
- Displays real-time thoughts/events
- Queries REST endpoints

**To create a minimal dashboard:**
```bash
npx create-next-app dexter-browser
cd dexter-browser
npm install socket.io-client
# Create components that connect to ws://localhost:8000/ws
npm start
```

See core/api.py for available endpoints:
- GET /status
- GET /triples
- POST /ask
- WebSocket /ws

---

## Support

For detailed analysis, see: `STARTUP_ANOMALIES_REPORT.md`
