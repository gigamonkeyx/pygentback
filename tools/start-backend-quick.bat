@echo off
echo Starting PyGent Factory Backend...
cd /d "d:\mcp\pygent-factory"

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting backend server on port 8000...
python main.py --mode server --host 0.0.0.0 --port 8000

pause
