@echo off
echo 🚀 PyGent Factory GPU Build
echo.

REM Check Docker
echo Checking Docker...
D:\Docker\resources\bin\docker.exe version
if %ERRORLEVEL% neq 0 (
    echo ❌ Docker not available
    exit /b 1
)

REM Check GPU
echo Checking GPU...
nvidia-smi --query-gpu=name --format=csv,noheader
if %ERRORLEVEL% neq 0 (
    echo ⚠️ GPU not detected
)

REM Create directories
echo Creating directories...
if not exist "D:\docker-data\models" mkdir "D:\docker-data\models"
if not exist "D:\docker-data\cache" mkdir "D:\docker-data\cache"

REM Build image
echo Building Docker image...
D:\Docker\resources\bin\docker.exe build ^
    --file Dockerfile.test ^
    --tag pygent-test:gpu ^
    .

if %ERRORLEVEL% eq 0 (
    echo ✅ Build completed successfully!
    echo Testing GPU access...
    D:\Docker\resources\bin\docker.exe run --rm --gpus all pygent-test:gpu
) else (
    echo ❌ Build failed
    exit /b 1
)

echo 🎉 Build process completed!
