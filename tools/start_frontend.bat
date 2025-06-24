@echo off
echo 🚀 Starting PyGent Factory Frontend...
cd ui
echo Current directory: %CD%
echo.
echo Installing dependencies...
call npm install
echo.
echo Starting development server...
call npm run dev
pause
