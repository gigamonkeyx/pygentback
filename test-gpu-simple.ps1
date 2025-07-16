# Simple GPU test script
function docker { & "D:\Docker\resources\bin\docker.exe" @args }

Write-Host "🧪 Testing GPU setup..." -ForegroundColor Cyan

# Test 1: Basic Docker GPU access
Write-Host "Test 1: Basic GPU access" -ForegroundColor Yellow
try {
    $result = docker run --rm --gpus all nvidia/cuda:12.9-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    Write-Host "✅ GPU detected: $result" -ForegroundColor Green
} catch {
    Write-Error "❌ GPU test failed: $_"
    exit 1
}

# Test 2: PyTorch GPU test
Write-Host "Test 2: PyTorch GPU test" -ForegroundColor Yellow
try {
    $pythonCmd = 'import torch; print("CUDA available:", torch.cuda.is_available()); print("GPU count:", torch.cuda.device_count()); print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")'
    $result = docker run --rm --gpus all pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime python -c $pythonCmd
    Write-Host "✅ PyTorch GPU test result:" -ForegroundColor Green
    Write-Host $result -ForegroundColor White
} catch {
    Write-Warning "⚠️ PyTorch GPU test failed: $_"
}

Write-Host "🎉 GPU tests completed!" -ForegroundColor Green
