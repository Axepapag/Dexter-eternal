@echo off
REM Fixed Dexter Startup Script
REM This script addresses the startup issues identified in STARTUP_ANOMALIES_REPORT.md

title Dexter - Fixed Startup
color 0B

echo ================================================
echo     DEXTER - FIXED STARTUP SCRIPT
echo ================================================
echo.

REM Step 1: Check Python availability
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)
python --version
echo.

REM Step 2: Check dependencies
echo [2/5] Checking Python dependencies...
python check_dependencies.py
if errorlevel 1 (
    echo.
    echo ERROR: Critical dependencies missing!
    echo.
    set /p install="Install dependencies now? (y/n): "
    if /i "%install%"=="y" (
        echo Installing dependencies from requirements.txt...
        python -m pip install -r requirements.txt
        if errorlevel 1 (
            echo ERROR: Failed to install dependencies
            pause
            exit /b 1
        )
        echo Dependencies installed successfully!
    ) else (
        echo Cannot start without dependencies. Exiting.
        pause
        exit /b 1
    )
)
echo.

REM Step 3: Create logs directory
echo [3/5] Setting up environment...
if not exist "logs" (
    mkdir logs
    echo Created logs directory
)
echo.

REM Step 4: Clear ports
echo [4/5] Checking and clearing ports...

REM Function to kill process on port
setlocal EnableDelayedExpansion
for %%p in (8000 8001) do (
    echo Checking port %%p...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p " 2^>nul') do (
        set PID=%%a
        if defined PID (
            echo   Port %%p in use by PID !PID! - killing...
            taskkill /f /pid !PID! >nul 2>&1
            timeout /t 1 >nul
        )
    )
)
endlocal
echo Ports cleared.
echo.

REM Step 5: Start Dexter
echo [5/5] Starting Dexter system...
echo.
echo ================================================
echo     STARTING DEXTER
echo ================================================
echo.
echo Note: Two terminal windows will open:
echo   1. Activity Stream - Shows all system logs
echo   2. Conversation - Clean chat interface
echo.
echo The API server will start on port 8000
echo Visit http://localhost:8000/status to verify
echo.
echo Starting in 3 seconds...
timeout /t 3 >nul

REM Start Dexter using entrypoint.py (recommended method)
python entrypoint.py agent

REM If the above fails, user will see the error
if errorlevel 1 (
    echo.
    echo ================================================
    echo     STARTUP FAILED
    echo ================================================
    echo.
    echo Dexter failed to start. Check the error above.
    echo.
    echo Common issues:
    echo   - Missing dependencies: Run check_dependencies.py
    echo   - Port conflicts: Check if port 8000 is in use
    echo   - Configuration errors: Check configs/core_config.json
    echo.
    echo For detailed troubleshooting, see:
    echo   STARTUP_ANOMALIES_REPORT.md
    echo   FIX_STARTUP_ISSUES.md
    echo.
    pause
    exit /b 1
)

pause
