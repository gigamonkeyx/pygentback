@echo off
REM Windows Launch Script for PyGent Factory
REM Prevents common hanging issues

REM Set environment variables
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONPATH=%~dp0src

REM Change to script directory
cd /d "%~dp0"

REM Launch with proper Windows settings
python %*
