# Simple GPU Build Script for PyGent Factory
param(
    [string]$Target = "production",
    [switch]$Verbose
)

# Set up Docker function
function docker { & "D:\Docker\resources\bin\docker.exe" @args }

Write-Host "🚀 PyGent Factory Simple GPU Build" -ForegroundColor Cyan
Write-Host "Target: $Target" -ForegroundColor Green

# Check Docker
try {
    $dockerVersion = docker version --format "{{.Client.Version}}"
    Write-Host "✅ Docker version: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Error "❌ Docker not found"
    exit 1
}

# Check GPU
try {
    nvidia-smi --query-gpu=name --format=csv,noheader | ForEach-Object {
        Write-Host "✅ GPU: $_" -ForegroundColor Green
    }
} catch {
    Write-Warning "⚠️ GPU not detected"
}

# Create directories
$dirs = @("D:/docker-data/models", "D:/docker-data/cache")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created: $dir" -ForegroundColor Green
    }
}

# Build command
Write-Host "🔨 Building Docker image..." -ForegroundColor Yellow

$buildArgs = @(
    "buildx", "build"
    "--file", "Dockerfile.gpu"
    "--target", $Target
    "--tag", "pygent-factory:gpu-latest"
    "--build-arg", "CUDA_VERSION=12.9"
    "--build-arg", "PYTHON_VERSION=3.11"
    "--build-arg", "TORCH_CUDA_ARCH_LIST=8.6"
    "--build-arg", "BUILD_TYPE=$Target"
    "."
)

if ($Verbose) {
    Write-Host "Command: docker $($buildArgs -join ' ')" -ForegroundColor Magenta
}

# Execute build
try {
    docker @buildArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Build completed successfully!" -ForegroundColor Green
        
        # Show image info
        docker images pygent-factory:gpu-latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
        
    } else {
        Write-Error "❌ Build failed with exit code: $LASTEXITCODE"
        exit $LASTEXITCODE
    }
} catch {
    Write-Error "❌ Build failed: $_"
    exit 1
}

Write-Host "🎉 Build completed!" -ForegroundColor Green
$runCommand = "docker run --rm --gpus all -p 8000:8000 pygent-factory:gpu-latest"
Write-Host "To run: $runCommand" -ForegroundColor Cyan
