# 🎨 ThreeStudio 3D Generation Docker Container
# Eliminates all compilation issues and provides clean environment

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Set environment variables to reduce warnings
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Suppress Python warnings
ENV PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY threestudio/requirements.txt .

# Install Python packages with fixed versions to avoid conflicts
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install ThreeStudio requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy threestudio source code
COPY threestudio/ /workspace/threestudio/

# Install threestudio
WORKDIR /workspace/threestudio
RUN pip install -e .

# Create cache directories
RUN mkdir -p /workspace/cache/huggingface && \
    mkdir -p /workspace/cache/torch && \
    mkdir -p /workspace/outputs

# Set environment variables for cache
ENV HF_HOME=/workspace/cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/cache/huggingface
ENV HF_DATASETS_CACHE=/workspace/cache/huggingface
ENV TORCH_HOME=/workspace/cache/torch

# Create startup script to suppress warnings
RUN echo '#!/bin/bash\n\
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning"\n\
export CUDA_LAUNCH_BLOCKING=0\n\
export TORCH_USE_CUDA_DSA=1\n\
cd /workspace/threestudio\n\
python "$@"' > /workspace/run_clean.sh && \
    chmod +x /workspace/run_clean.sh

# Set working directory back to threestudio
WORKDIR /workspace/threestudio

# Expose port for Gradio/web interfaces
EXPOSE 7860

# Default command
CMD ["/workspace/run_clean.sh", "launch.py", "--help"]
