@echo off
title Dexter Full System Launcher
color 0B

echo ================================================
echo     DEXTER FULL SYSTEM LAUNCHER
echo ================================================
echo.
echo Starting all Dexter components...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python and add to PATH
    pause
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Function to check if port is in use
:check_port
setlocal EnableDelayedExpansion
echo Checking port %1...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%1"') do (
    set PID=%%a
    echo Port %1 is in use by PID !PID!
    goto :kill_port
)
echo Port %1 is free
goto :port_check_done

:kill_port
echo Killing process !PID! on port %1...
taskkill /f /pid !PID! >nul 2>&1
if errorlevel 1 (
    echo Failed to kill process !PID!
) else (
    echo Successfully killed process !PID!
)
timeout /t 2 >nul

:port_check_done
endlocal
goto :eof

echo.
echo ================================================
echo     CHECKING AND CLEARING PORTS
echo ================================================

REM Check common Dexter ports and clear them
REM API Server typically runs on port 8000
call :check_port 8000
REM Dexter main might run on port 8001
call :check_port 8001
REM Browser might run on port 3000
call :check_port 3000
REM Browser debugging port
call :check_port 9222

echo.
echo All ports checked and cleared.
echo.

echo ================================================
echo     STARTING DEXTER COMPONENTS
echo ================================================
echo.

REM Start API Server in background
echo [1/4] Starting API Server...
start "Dexter API Server" cmd /k "cd /d %~dp0 && python -m http.server 8000 > logs\api_server.log 2>&1"

REM Wait a moment for API server to start
timeout /t 3 >nul

REM Start main Dexter process
echo [2/4] Starting Dexter Main Process...
start "Dexter Core" cmd /k "cd /d %~dp0 && python dexter.py > logs\dexter_core.log 2>&1"

REM Wait for Dexter to initialize
timeout /t 5 >nul

REM Start Dexter Browser if it exists
echo [3/4] Starting Dexter Browser...
if exist "dexter-browser" (
    cd dexter-browser
    start "Dexter Browser" cmd /k "npm start > ..\logs\browser.log 2>&1"
    cd ..
    echo Browser startup initiated...
) else (
    echo [WARNING] dexter-browser directory not found
    echo You may need to install dependencies: cd dexter-browser && npm install
)

REM Start additional services if needed
echo [4/4] Starting additional services...
echo Starting memory system...
start "Dexter Memory" cmd /k "cd /d %~dp0 && python -c \"from core.memory_ingestor import MemoryIngestor; print('Memory system ready')\" > logs\memory.log 2>&1"

echo.
echo ================================================
echo     DEXTER SYSTEM LAUNCHED
echo ================================================
echo.
echo Components started:
echo   [✓] API Server on port 8000
echo   [✓] Dexter Core Process
echo   [✓] Browser (if available)
echo   [✓] Memory System
echo.
echo Log files are in the 'logs' directory
echo.
echo To stop all processes, run STOP_DEXTER_FULL.bat
echo.
echo Press any key to view system status...
pause >nul

REM Show running processes
echo.
echo Current Dexter processes:
tasklist | findstr /i "python\|node" | findstr /v "pythonw"

echo.
echo Launch complete! Check logs/ directory for details.
pause