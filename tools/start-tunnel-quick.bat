@echo off
echo Starting Cloudflare Tunnel...
cd /d "d:\mcp\pygent-factory"

echo Checking if backend is running...
netstat -an | findstr :8000
if %errorlevel% neq 0 (
    echo WARNING: Backend not detected on port 8000
    echo Make sure to start the backend first!
    timeout /t 5
)

echo Starting tunnel with configuration...
.\cloudflared-new.exe tunnel --config cloudflared-config.yml run

pause
