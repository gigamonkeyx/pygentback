# PyGent Factory GPU Setup

🚀 **Comprehensive GPU-accelerated Docker deployment for PyGent Factory**

Optimized for **NVIDIA RTX 3080** with **CUDA 12.9** support, leveraging the latest Docker BuildKit features and advanced optimization techniques.

## 🎯 Features

### 🔥 GPU Optimizations
- **NVIDIA RTX 3080** specific optimizations (Ampere architecture)
- **CUDA 12.9** with TensorRT acceleration
- **Mixed precision training** (FP16/BF16)
- **Flash Attention** and **xFormers** for transformer efficiency
- **GPU memory optimization** with smart caching

### 🏗️ Advanced Docker Features
- **Multi-stage builds** with BuildKit 1.10 syntax
- **Cache mounts** for dependencies (pip, apt, CUDA)
- **Secret mounts** for secure API keys
- **Bind mounts** for efficient development
- **Multi-platform builds** (AMD64/ARM64)
- **Registry caching** for CI/CD optimization

### 📊 Monitoring & Observability
- **Prometheus** metrics collection
- **GPU monitoring** with DCGM exporter
- **TensorBoard** integration
- **Health checks** and auto-recovery
- **Performance profiling** tools

## 🚀 Quick Start

### 1. Prerequisites

```powershell
# Check GPU
nvidia-smi

# Check Docker with GPU support
docker run --rm --gpus all nvidia/cuda:12.9-base-ubuntu22.04 nvidia-smi
```

### 2. Setup Secrets

```powershell
# Copy example files and add your tokens
cp secrets/huggingface_token.txt.example secrets/huggingface_token.txt
cp secrets/openai_api_key.txt.example secrets/openai_api_key.txt
cp secrets/github_token.txt.example secrets/github_token.txt

# Edit the files with your actual tokens
```

### 3. Build and Run

```powershell
# Development mode (with Jupyter, debugging, live reload)
.\run-gpu.ps1 -Mode dev -Build

# Production mode
.\run-gpu.ps1 -Mode prod -Build

# Benchmark mode
.\run-gpu.ps1 -Mode benchmark -Build
```

## 🛠️ Advanced Usage

### Build Options

```powershell
# Build specific target
.\build-gpu.ps1 -Target production -Tag v1.0.0

# Multi-platform build
.\build-gpu.ps1 -Platform "linux/amd64,linux/arm64" -Push

# No cache build
.\build-gpu.ps1 -NoCache -Verbose

# Push to registry
.\build-gpu.ps1 -Push -Registry "your-registry.com"
```

### Runtime Options

```powershell
# Start specific service
.\run-gpu.ps1 -Service pygent-factory

# View logs
.\run-gpu.ps1 -Logs -Service pygent-factory

# Monitor services
.\run-gpu.ps1 -Monitor

# Clean up everything
.\run-gpu.ps1 -Clean
```

### Manual Docker Commands

```powershell
# Set up Docker function
function docker { & "D:\Docker\resources\bin\docker.exe" @args }

# Build with all optimizations
docker buildx build \
  --file Dockerfile.gpu \
  --target production \
  --platform linux/amd64 \
  --cache-from type=registry,ref=localhost:5000/pygent-factory:buildcache \
  --cache-to type=registry,ref=localhost:5000/pygent-factory:buildcache,mode=max \
  --secret id=huggingface_token,src=./secrets/huggingface_token.txt \
  --tag pygent-factory:gpu-latest \
  .

# Run with GPU support
docker run --rm --gpus all \
  -p 8000:8000 \
  -v ${PWD}:/app \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  pygent-factory:gpu-latest
```

## 📊 Monitoring

### Access Points
- **Main App**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **TensorBoard**: http://localhost:6006
- **Jupyter Lab**: http://localhost:8888 (dev mode)
- **GPU Metrics**: http://localhost:9400/metrics

### Key Metrics
- GPU utilization and memory usage
- CUDA kernel execution times
- Model inference latency
- Memory allocation patterns
- Cache hit rates

## 🔧 Configuration

### Environment Variables

```bash
# GPU Configuration
NVIDIA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=8.6

# Performance Tuning
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
OMP_NUM_THREADS=8

# Application Settings
PYGENT_GPU_ENABLED=true
PYGENT_BATCH_SIZE=32
PYGENT_PRECISION=fp16
```

### Volume Mounts

```yaml
volumes:
  # High-performance model storage
  - D:/docker-data/models:/app/models
  
  # GPU cache persistence
  - cuda-cache:/app/.cuda-cache
  - torch-cache:/root/.cache/torch
  
  # Development bind mounts
  - ./src:/app/src:cached
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```powershell
   # Check NVIDIA runtime
   docker info | Select-String nvidia
   
   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:12.9-base nvidia-smi
   ```

2. **Out of Memory**
   ```bash
   # Reduce batch size
   export PYGENT_BATCH_SIZE=16
   
   # Enable memory optimization
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
   ```

3. **Build Cache Issues**
   ```powershell
   # Clear build cache
   docker builder prune -f
   
   # Rebuild without cache
   .\build-gpu.ps1 -NoCache
   ```

4. **Permission Issues**
   ```powershell
   # Fix volume permissions
   docker compose exec pygent-factory chown -R pygent:pygent /app
   ```

## 📈 Performance Tips

### GPU Optimization
- Use **mixed precision** (FP16) for 2x speedup
- Enable **TensorRT** for inference optimization
- Use **Flash Attention** for transformer models
- Optimize **batch sizes** for your GPU memory

### Docker Optimization
- Use **cache mounts** for faster builds
- Leverage **multi-stage builds** for smaller images
- Use **bind mounts** for development
- Enable **BuildKit** for advanced features

### System Optimization
- Ensure adequate **cooling** for sustained performance
- Monitor **power limits** and **thermal throttling**
- Use **NVMe SSD** for model storage
- Allocate sufficient **system RAM** (32GB+ recommended)

## 🔗 Related Files

- `docker-compose.gpu.yml` - Main GPU-enabled compose file
- `Dockerfile.gpu` - Multi-stage GPU-optimized Dockerfile
- `docker-compose.override.yml` - Development overrides
- `build-gpu.ps1` - Advanced build script
- `run-gpu.ps1` - Runtime orchestration script
- `requirements-gpu.txt` - GPU-specific Python dependencies

## 📚 Documentation

- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
- [BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [PyTorch GPU Optimization](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
