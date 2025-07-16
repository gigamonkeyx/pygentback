# PyGent Factory GPU Build Script
# Comprehensive Docker build leveraging all advanced features
# Optimized for NVIDIA RTX 3080 with CUDA 12.9

param(
    [string]$Target = "production",
    [string]$Platform = "linux/amd64",
    [switch]$Push,
    [switch]$Load,
    [switch]$NoCache,
    [switch]$Verbose,
    [string]$Registry = "localhost:5000",
    [string]$Tag = "latest"
)

# Set up Docker function
function docker { & "D:\Docker\resources\bin\docker.exe" @args }

Write-Host "🚀 PyGent Factory GPU Build Script" -ForegroundColor Cyan
Write-Host "Target: $Target | Platform: $Platform | Tag: $Tag" -ForegroundColor Green

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow

# Check Docker
try {
    $dockerVersion = docker version --format "{{.Client.Version}}"
    Write-Host "✅ Docker version: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Error "❌ Docker not found or not running"
    exit 1
}

# Check NVIDIA runtime
try {
    $runtimes = docker info --format "{{.Runtimes}}"
    if ($runtimes -match "nvidia") {
        Write-Host "✅ NVIDIA runtime available" -ForegroundColor Green
    } else {
        Write-Warning "⚠️ NVIDIA runtime not found"
    }
} catch {
    Write-Warning "⚠️ Could not check NVIDIA runtime"
}

# Check GPU
try {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | ForEach-Object {
        Write-Host "✅ GPU: $_" -ForegroundColor Green
    }
} catch {
    Write-Warning "⚠️ NVIDIA GPU not detected"
}

# Create necessary directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
$dirs = @("D:/docker-data/models", "D:/docker-data/cache", "./monitoring", "./sql/init")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created: $dir" -ForegroundColor Green
    }
}

# Check secrets
Write-Host "🔐 Checking secrets..." -ForegroundColor Yellow
$secrets = @("huggingface_token.txt", "openai_api_key.txt", "github_token.txt")
foreach ($secret in $secrets) {
    $secretPath = "./secrets/$secret"
    $examplePath = "./secrets/$secret.example"
    
    if (!(Test-Path $secretPath)) {
        if (Test-Path $examplePath) {
            Write-Warning "⚠️ $secret not found. Copy from $examplePath and add your token"
        } else {
            Write-Warning "⚠️ $secret not found"
        }
    } else {
        Write-Host "✅ Secret found: $secret" -ForegroundColor Green
    }
}

# Build arguments
$buildArgs = @(
    "--build-arg", "BUILDKIT_INLINE_CACHE=1"
    "--build-arg", "CUDA_VERSION=12.9"
    "--build-arg", "PYTHON_VERSION=3.11"
    "--build-arg", "TORCH_CUDA_ARCH_LIST=8.6"
    "--build-arg", "BUILD_TYPE=$Target"
    "--build-arg", "ENABLE_OPTIMIZATIONS=true"
)

# Cache configuration
$cacheArgs = @()
if (!$NoCache) {
    $cacheArgs += @(
        "--cache-from", "type=registry,ref=$Registry/pygent-factory:buildcache"
        "--cache-from", "type=local,src=D:/docker-data/cache"
        "--cache-to", "type=registry,ref=$Registry/pygent-factory:buildcache,mode=max"
        "--cache-to", "type=local,dest=D:/docker-data/cache,mode=max"
    )
}

# Secret mounts
$secretArgs = @()
$secrets | ForEach-Object {
    $secretPath = "./secrets/$_"
    if (Test-Path $secretPath) {
        $secretName = $_.Replace(".txt", "")
        $secretArgs += @("--secret", "id=$secretName,src=$secretPath")
    }
}

# Platform and output
$platformArgs = @("--platform", $Platform)
$outputArgs = @()
if ($Push) {
    $outputArgs += @("--push")
} elseif ($Load) {
    $outputArgs += @("--load")
} else {
    $outputArgs += @("--output", "type=docker")
}

# Tags
$tagArgs = @(
    "--tag", "pygent-factory:gpu-$Tag"
    "--tag", "pygent-factory:gpu-$Target-$Tag"
    "--tag", "$Registry/pygent-factory:gpu-$Tag"
)

# Build command
Write-Host "🔨 Building Docker image..." -ForegroundColor Yellow
Write-Host "Target: $Target" -ForegroundColor Cyan

$buildCmd = @(
    "buildx", "build"
    "--file", "Dockerfile.gpu"
    "--target", $Target
) + $buildArgs + $cacheArgs + $secretArgs + $platformArgs + $outputArgs + $tagArgs + @(".")

if ($Verbose) {
    Write-Host "Command: docker $($buildCmd -join ' ')" -ForegroundColor Magenta
}

# Execute build
try {
    docker @buildCmd
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Build completed successfully!" -ForegroundColor Green
        
        # Show image info
        Write-Host "📊 Image information:" -ForegroundColor Yellow
        docker images pygent-factory:gpu-$Tag --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        
        # GPU test
        Write-Host "🧪 Testing GPU access..." -ForegroundColor Yellow
        try {
            $pythonCmd = 'import torch; print("CUDA available:", torch.cuda.is_available()); print("GPU count:", torch.cuda.device_count()); print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")'
            $testResult = docker run --rm --gpus all pygent-factory:gpu-$Tag python -c $pythonCmd
            Write-Host "✅ GPU test result:" -ForegroundColor Green
            Write-Host $testResult -ForegroundColor White
        } catch {
            Write-Warning "⚠️ GPU test failed: $_"
        }
        
    } else {
        Write-Error "❌ Build failed with exit code: $LASTEXITCODE"
        exit $LASTEXITCODE
    }
} catch {
    Write-Error "❌ Build failed: $_"
    exit 1
}

Write-Host "🎉 Build process completed!" -ForegroundColor Green
Write-Host "To run the container with GPU support:" -ForegroundColor Cyan
Write-Host "docker run --rm --gpus all -p 8000:8000 pygent-factory:gpu-$Tag" -ForegroundColor White
