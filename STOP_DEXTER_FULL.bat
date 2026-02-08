@echo off
title Dexter Full System Stopper
color 0C

echo ================================================
echo     DEXTER FULL SYSTEM STOPPER
echo ================================================
echo.
echo Stopping all Dexter components...
echo.

REM Function to kill process by name
:kill_process
echo Killing %1 processes...
taskkill /f /im "%1" >nul 2>&1
if errorlevel 1 (
    echo No %1 processes found running
) else (
    echo Successfully killed %1 processes
)
goto :eof

echo.
echo ================================================
echo     STOPPING DEXTER COMPONENTS
echo ================================================
echo.

REM Kill Python processes (Dexter Core, API Server, etc.)
call :kill_process python.exe
call :kill_process pythonw.exe

REM Kill Node processes (Dexter Browser)
call :kill_process node.exe

REM Kill any other common processes
call :kill_process http-server.exe

echo.
echo ================================================
echo     CHECKING PORTS
echo ================================================

REM Check if ports are still in use
echo Checking if ports are now free...
netstat -ano | findstr ":8000 " >nul
if not errorlevel 1 (
    echo WARNING: Port 8000 is still in use
)

netstat -ano | findstr ":3000 " >nul
if not errorlevel 1 (
    echo WARNING: Port 3000 is still in use
)

netstat -ano | findstr ":9222 " >nul
if not errorlevel 1 (
    echo WARNING: Port 9222 is still in use
)

echo.
echo ================================================
echo     CLEANUP COMPLETE
echo ================================================
echo.
echo All Dexter processes have been stopped.
echo.
echo If ports are still in use, you may need to:
echo 1. Restart your computer
echo 2. Manually kill remaining processes
echo 3. Check for other applications using these ports
echo.
pause