@echo off
REM Install missing dependencies for PyGent Factory

echo Installing missing dependencies...

REM Change to project directory
cd /d D:\mcp\pygent-factory

REM Activate virtual environment
call .\venv\Scripts\activate.bat

REM Install missing dependencies
echo Installing watchdog...
pip install watchdog

echo Installing sentence-transformers...
pip install sentence-transformers

echo Installing jsonschema...
pip install jsonschema

echo Installing scikit-learn...
pip install scikit-learn

echo Installing optuna...
pip install optuna

echo Installing rich...
pip install rich

echo Installing aiofiles...
pip install aiofiles

echo Installing httpx...
pip install httpx

echo Installing requests...
pip install requests

echo Installing numpy...
pip install numpy

echo Installing pandas...
pip install pandas

echo Installing matplotlib...
pip install matplotlib

echo Installing plotly...
pip install plotly

echo Installing dash...
pip install dash

echo Installing streamlit...
pip install streamlit

echo Installing pytest...
pip install pytest

echo Installing pytest-asyncio...
pip install pytest-asyncio

echo Installing coverage...
pip install coverage

echo Installing black...
pip install black

echo Installing flake8...
pip install flake8

echo Installing mypy...
pip install mypy

echo All dependencies installed successfully!
pause
