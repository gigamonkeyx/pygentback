# PyGent Factory Cloudflare Tunnel Startup Script

Write-Host "🚀 Starting PyGent Factory Cloudflare Tunnel..." -ForegroundColor Green

# Tunnel token for pygent-factory tunnel
$tunnelToken = "eyJhIjoiNTc1YjU5MzY5OTEyNDE2NGExZjVlMDJmMDVkMDZiODkiLCJzIjoiRDFuT0Q1ZzExbXl6RHNORXdBaWhTZ1MxemoxTmdrY0lHWTAreGNrR2oyRT0iLCJ0IjoiNDY4NmJlZDQtNDYwOC00ZGUwLTljODMtNGEwNjFkNmEwZGFlIn0="

Write-Host "📋 Tunnel Information:" -ForegroundColor Yellow
Write-Host "  - Tunnel ID: 4686bed4-4608-4de0-9c83-4a061d6a0dae" -ForegroundColor Cyan
Write-Host "  - Tunnel Name: pygent-factory" -ForegroundColor Cyan
Write-Host "  - API Endpoint: api.timpayne.net -> localhost:8000" -ForegroundColor Cyan
Write-Host "  - WebSocket Endpoint: ws.timpayne.net -> localhost:8000" -ForegroundColor Cyan

Write-Host "⚠️  Make sure PyGent Factory backend is running on localhost:8000" -ForegroundColor Yellow

Write-Host "🔗 Starting tunnel connection..." -ForegroundColor Green

# Start the tunnel using the token
cloudflared tunnel --token $tunnelToken

Write-Host "🛑 Tunnel stopped." -ForegroundColor Red