@echo off
echo Starting Cloudflare Tunnel...
cd /d "d:\mcp\pygent-factory"
.\cloudflared-new.exe tunnel --config cloudflared-config.yml run
pause
