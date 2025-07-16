@echo off
echo 🚀 PYGENT FACTORY SYSTEM STARTUP
echo ================================
echo.

cd /d "d:\mcp\pygent-factory"

echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

echo 🏥 Running system startup checklist...
python start_system.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ SYSTEM READY!
    echo 🐉 You can now run: python dragon_task_execution.py
    echo.
    pause
) else (
    echo.
    echo ❌ SYSTEM STARTUP FAILED
    echo 🔧 Check the output above for issues
    echo.
    pause
)
