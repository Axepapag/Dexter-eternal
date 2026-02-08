@echo off
title Dexter System Status Checker
color 0A

echo ================================================
echo     DEXTER SYSTEM STATUS CHECKER
echo ================================================
echo.

REM Function to check if a port is in use
:check_port_status
setlocal EnableDelayedExpansion
echo Checking port %1...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%1"') do (
    echo   [IN USE] Port %1 - PID: %%a
    set PORT_USED=1
    goto :port_check_done_%1
)
echo   [FREE] Port %1
set PORT_USED=0

:port_check_done_%1
endlocal
goto :eof

REM Function to check if a process is running
:check_process
echo Checking for %1 processes...
tasklist /fi "imagename eq %1" 2>nul | findstr /i "%1" >nul
if not errorlevel 1 (
    echo   [RUNNING] %1 processes found
    for /f "tokens=2" %%a in ('tasklist /fi "imagename eq %1" ^| findstr /i "%1"') do (
        echo      PID: %%a
    )
) else (
    echo   [STOPPED] No %1 processes found
)
goto :eof

echo.
echo ================================================
echo     PORT STATUS
echo ================================================

REM Check common ports
call :check_port_status 8000
call :check_port_status 8001
call :check_port_status 3000
call :check_port_status 9222

echo.
echo ================================================
echo     PROCESS STATUS
echo ================================================

REM Check for Python processes (Dexter Core, API Server)
call :check_process python.exe
call :check_process pythonw.exe

REM Check for Node processes (Dexter Browser)
call :check_process node.exe

REM Check for http-server processes
call :check_process http-server.exe

echo.
echo ================================================
echo     SYSTEM RESOURCES
echo ================================================

REM Check memory usage
echo Memory Usage:
for /f "tokens=2 delims=:" %%a in ('wmic OS get TotalVisibleMemorySize /value') do set TOTAL_MEM=%%a
for /f "tokens=2 delims=:" %%a in ('wmic OS get FreePhysicalMemory /value') do set FREE_MEM=%%a
set /a USED_MEM=%TOTAL_MEM%-%FREE_MEM%
set /a USED_PERCENT=(%USED_MEM%*100)/%TOTAL_MEM%
echo   Total Memory: %TOTAL_MEM% KB
echo   Free Memory:  %FREE_MEM% KB
echo   Used Memory:  %USED_MEM% KB (%USED_PERCENT%%%)

echo.
REM Check disk space
echo Disk Space:
for /f "tokens=3" %%a in ('dir /-c ^| find "bytes free"') do set FREE_DISK=%%a
echo   Free Space: %FREE_DISK% bytes

echo.
echo ================================================
echo     LOG FILE STATUS
echo ================================================

REM Check if log files exist and show their sizes
echo Checking log files:
if exist "logs\api_server.log" (
    for %%A in ("logs\api_server.log") do echo   [API] api_server.log - %%~zA bytes
) else (
    echo   [API] api_server.log - Not found
)

if exist "logs\dexter_core.log" (
    for %%A in ("logs\dexter_core.log") do echo   [CORE] dexter_core.log - %%~zA bytes
) else (
    echo   [CORE] dexter_core.log - Not found
)

if exist "logs\browser.log" (
    for %%A in ("logs\browser.log") do echo   [BROWSER] browser.log - %%~zA bytes
) else (
    echo   [BROWSER] browser.log - Not found
)

echo.
echo ================================================
echo     SUMMARY
echo ================================================

echo.
echo System Health Assessment:
echo.

REM Overall status
set SYSTEM_OK=1

REM Check if core processes are running
tasklist /fi "imagename eq python.exe" 2>nul | findstr /i "python" >nul
if errorlevel 1 (
    echo [WARNING] No Python processes running - Dexter core may not be started
    set SYSTEM_OK=0
) else (
    echo [OK] Python processes running
)

REM Check if ports are accessible
netstat -ano | findstr ":8000" >nul
if errorlevel 1 (
    echo [WARNING] Port 8000 not in use - API server may not be running
) else (
    echo [OK] Port 8000 in use
)

if %SYSTEM_OK%==1 (
    echo.
    echo [SUCCESS] Dexter system appears to be running normally
) else (
    echo.
    echo [WARNING] Some Dexter components may not be running properly
)

echo.
echo ================================================
echo.
echo To start the full system, run: START_DEXTER_FULL.bat
echo To stop all processes, run: STOP_DEXTER_FULL.bat
echo.
pause