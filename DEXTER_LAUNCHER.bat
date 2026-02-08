@echo off
title Dexter Quick Launcher
color 0B

echo ================================================
echo     DEXTER QUICK LAUNCHER
echo ================================================
echo.
echo Choose an option:
echo   1. Start Full System
echo   2. Stop All Processes
echo   3. Check System Status
echo   4. Open Advanced Launcher
echo   5. Exit
echo.

:menu
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto start_system
if "%choice%"=="2" goto stop_system
if "%choice%"=="3" goto check_status
if "%choice%"=="4" goto advanced_launcher
if "%choice%"=="5" goto exit
echo Invalid choice. Please try again.
goto menu

:start_system
echo.
echo Starting Dexter Full System...
echo.
call START_DEXTER_FULL.bat
pause
goto menu

:stop_system
echo.
echo Stopping All Dexter Processes...
echo.
call STOP_DEXTER_FULL.bat
pause
goto menu

:check_status
echo.
echo Checking Dexter System Status...
echo.
call CHECK_DEXTER_STATUS.bat
pause
goto menu

:advanced_launcher
echo.
echo Opening Advanced Launcher...
echo.
echo Available commands:
echo   python advanced_launcher.py --start    (Start system)
echo   python advanced_launcher.py --stop     (Stop system)
echo   python advanced_launcher.py --status   (Show status)
echo.
python advanced_launcher.py --status
pause
goto menu

:exit
echo.
echo Goodbye!
exit /b 0