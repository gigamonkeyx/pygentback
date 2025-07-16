# PyGent Factory GPU Runtime Script
# Comprehensive Docker Compose orchestration with GPU support

param(
    [string]$Mode = "dev",  # dev, prod, test, benchmark
    [switch]$Build,
    [switch]$Pull,
    [switch]$Down,
    [switch]$Logs,
    [switch]$Monitor,
    [switch]$Clean,
    [string]$Service = "",
    [switch]$Verbose
)

# Set up Docker function
function docker { & "D:\Docker\resources\bin\docker.exe" @args }

Write-Host "🚀 PyGent Factory GPU Runtime" -ForegroundColor Cyan
Write-Host "Mode: $Mode" -ForegroundColor Green

# Configuration based on mode
$composeFiles = @("-f", "docker-compose.gpu.yml")
switch ($Mode) {
    "dev" {
        $composeFiles += @("-f", "docker-compose.override.yml")
        Write-Host "🔧 Development mode with live reloading" -ForegroundColor Yellow
    }
    "prod" {
        Write-Host "🏭 Production mode" -ForegroundColor Yellow
    }
    "test" {
        $composeFiles += @("-f", "docker-compose.test.yml")
        Write-Host "🧪 Testing mode" -ForegroundColor Yellow
    }
    "benchmark" {
        $composeFiles += @("-f", "docker-compose.benchmark.yml")
        Write-Host "📊 Benchmark mode" -ForegroundColor Yellow
    }
}

# Clean up if requested
if ($Clean) {
    Write-Host "🧹 Cleaning up..." -ForegroundColor Yellow
    docker compose @composeFiles down --volumes --remove-orphans
    docker system prune -f
    docker volume prune -f
    Write-Host "✅ Cleanup completed" -ForegroundColor Green
    return
}

# Stop services if requested
if ($Down) {
    Write-Host "🛑 Stopping services..." -ForegroundColor Yellow
    docker compose @composeFiles down --remove-orphans
    Write-Host "✅ Services stopped" -ForegroundColor Green
    return
}

# Show logs if requested
if ($Logs) {
    Write-Host "📋 Showing logs..." -ForegroundColor Yellow
    if ($Service) {
        docker compose @composeFiles logs -f $Service
    } else {
        docker compose @composeFiles logs -f
    }
    return
}

# Monitor services
if ($Monitor) {
    Write-Host "📊 Opening monitoring dashboard..." -ForegroundColor Yellow
    Start-Process "http://localhost:9090"  # Prometheus
    Start-Process "http://localhost:8000"  # Main app
    if ($Mode -eq "dev") {
        Start-Process "http://localhost:8888"  # Jupyter
    }
    return
}

# Pre-flight checks
Write-Host "🔍 Running pre-flight checks..." -ForegroundColor Yellow

# Check Docker
try {
    $dockerVersion = docker version --format "{{.Client.Version}}"
    Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Error "❌ Docker not available"
    exit 1
}

# Check GPU
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    Write-Host "✅ GPU: $gpuInfo" -ForegroundColor Green
} catch {
    Write-Warning "⚠️ GPU not detected"
}

# Check available disk space
$freeSpace = (Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='D:'").FreeSpace / 1GB
if ($freeSpace -lt 10) {
    Write-Warning "⚠️ Low disk space on D: drive: $([math]::Round($freeSpace, 2)) GB"
} else {
    Write-Host "✅ Disk space: $([math]::Round($freeSpace, 2)) GB available" -ForegroundColor Green
}

# Create necessary directories
$dirs = @("D:/docker-data/models", "D:/docker-data/cache", "./notebooks", "./sql/dev")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created: $dir" -ForegroundColor Green
    }
}

# Pull images if requested
if ($Pull) {
    Write-Host "📥 Pulling images..." -ForegroundColor Yellow
    docker compose @composeFiles pull
}

# Build if requested or if images don't exist
if ($Build -or $Pull) {
    Write-Host "🔨 Building images..." -ForegroundColor Yellow
    $buildArgs = @("build")
    if ($Verbose) { $buildArgs += "--verbose" }
    if ($Service) { $buildArgs += $Service }
    
    docker compose @composeFiles @buildArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Build failed"
        exit $LASTEXITCODE
    }
}

# Start services
Write-Host "🚀 Starting services..." -ForegroundColor Yellow

$upArgs = @("up", "-d")
if ($Service) {
    $upArgs += $Service
} else {
    # Remove orphans and recreate containers
    $upArgs += @("--remove-orphans", "--force-recreate")
}

docker compose @composeFiles @upArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Failed to start services"
    exit $LASTEXITCODE
}

# Wait for services to be healthy
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service status
Write-Host "📊 Service Status:" -ForegroundColor Yellow
docker compose @composeFiles ps

# Test GPU access
Write-Host "🧪 Testing GPU access..." -ForegroundColor Yellow
try {
    $gpuTest = docker compose @composeFiles exec -T pygent-factory python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
    Write-Host "✅ GPU Test Result:" -ForegroundColor Green
    Write-Host $gpuTest -ForegroundColor White
} catch {
    Write-Warning "⚠️ GPU test failed: $_"
}

# Show useful URLs
Write-Host "🌐 Available Services:" -ForegroundColor Cyan
Write-Host "  Main Application: http://localhost:8000" -ForegroundColor White
Write-Host "  WebSocket API:    ws://localhost:8001" -ForegroundColor White
Write-Host "  Metrics:          http://localhost:8002/metrics" -ForegroundColor White
Write-Host "  Prometheus:       http://localhost:9090" -ForegroundColor White
Write-Host "  TensorBoard:      http://localhost:6006" -ForegroundColor White

if ($Mode -eq "dev") {
    Write-Host "  Jupyter Lab:      http://localhost:8888" -ForegroundColor White
    Write-Host "  Debug Port:       localhost:5678" -ForegroundColor White
}

Write-Host "🎉 PyGent Factory is running!" -ForegroundColor Green
Write-Host "Use 'docker compose $($composeFiles -join ' ') logs -f' to view logs" -ForegroundColor Cyan
