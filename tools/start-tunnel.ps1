# PyGent Factory Tunnel Startup Script
# This script starts the Cloudflared tunnel for production deployment

Write-Host "🚇 Starting PyGent Factory Cloudflared Tunnel..." -ForegroundColor Green

# Check if backend is running
Write-Host "🔍 Checking if PyGent Factory backend is running..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Backend is running and healthy" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Backend responded with status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Backend is not running or not responding" -ForegroundColor Red
    Write-Host "Please start the backend first with: python main.py server" -ForegroundColor Yellow
    exit 1
}

# Check if Ollama is running
Write-Host "🔍 Checking if Ollama is running..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Ollama is running and accessible" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Ollama responded with status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Ollama is not running or not responding" -ForegroundColor Red
    Write-Host "Please start Ollama first with: ollama serve" -ForegroundColor Yellow
    Write-Host "Continuing anyway - tunnel will still work for non-AI endpoints..." -ForegroundColor Yellow
}

# Start the tunnel
Write-Host "🚀 Starting Cloudflared tunnel..." -ForegroundColor Green
Write-Host "📋 Tunnel will be available at:" -ForegroundColor Cyan
Write-Host "   - API: https://api.timpayne.net" -ForegroundColor White
Write-Host "   - WebSocket: wss://ws.timpayne.net" -ForegroundColor White
Write-Host "   - Health: https://health.timpayne.net" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the tunnel" -ForegroundColor Yellow
Write-Host ""

try {
    # Start tunnel with configuration file
    cloudflared tunnel --config cloudflared-config.yml run
} catch {
    Write-Host "❌ Failed to start tunnel: $_" -ForegroundColor Red
    Write-Host "Please check your configuration and try again" -ForegroundColor Yellow
    exit 1
}
