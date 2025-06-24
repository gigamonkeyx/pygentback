# Simple PyGent Factory Deployment Test

Write-Host "🏭 PyGent Factory Deployment Test" -ForegroundColor Green

# Test backend
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Backend is healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Backend is not responding" -ForegroundColor Red
}

# Test Ollama
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Ollama is healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Ollama is not responding" -ForegroundColor Red
}

# Test research workflow
try {
    $payload = '{"query": "test", "max_papers": 1}'
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/workflows/research-analysis" -Method POST -Headers @{'Content-Type'='application/json'} -Body $payload -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Research workflow is working" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Research workflow failed" -ForegroundColor Red
}

Write-Host "🎉 Test complete!" -ForegroundColor Green
