@echo off
REM PyGent Factory Status Console Launcher
REM This script starts the status console after ensuring the backend is running

echo.
echo ================================================================================
echo                        PYGENT FACTORY STATUS CONSOLE
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if aiohttp is installed
python -c "import aiohttp" >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install aiohttp
)

REM Start the status console
echo Starting status console...
echo Backend URL: http://localhost:8000
echo Press Ctrl+C to exit
echo.

python status_console.py --backend-url http://localhost:8000 --refresh-interval 3

pause
