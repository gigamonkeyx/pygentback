"""
GPU Configuration and Management System
Provides unified GPU detection, initialization, and optimization for PyGent Factory.
"""
import logging
import platform
import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

@dataclass
class GPUConfig:
    """GPU configuration and capabilities."""
    device: str = "cpu"
    device_name: str = "CPU"
    memory_total: int = 0  # MB
    memory_available: int = 0  # MB
    compute_capability: Optional[tuple] = None
    supports_mixed_precision: bool = False
    cuda_version: Optional[str] = None

class GPUManager:
    """Manages GPU resources and provides unified access."""
    
    def __init__(self):
        self.config = None
        self.device = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize GPU manager and detect available hardware."""
        
        try:
            # Detect GPU capabilities
            self.config = self._detect_gpu_capabilities()
            self.device = torch.device(self.config.device)
            
            # Configure PyTorch optimizations
            self._configure_pytorch_optimizations()
            
            self._initialized = True
            
            if self.config.device != "cpu":
                logger.info(f"GPU Manager initialized: {self.config.device_name} ({self.config.device})")
                logger.info(f"GPU Memory: {self.config.memory_available}MB available / {self.config.memory_total}MB total")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU manager: {e}")
            # Fallback to CPU
            self.config = GPUConfig(device="cpu", device_name="CPU")
            self.device = torch.device("cpu")
            self._initialized = True
            logger.info("Falling back to CPU computation")
            return False
    
    def _detect_gpu_capabilities(self) -> GPUConfig:
        """Detect available GPU capabilities."""
        
        # Check for CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # Use first GPU by default
                device_idx = 0
                device_name = torch.cuda.get_device_name(device_idx)
                
                # Get memory info
                memory_total = torch.cuda.get_device_properties(device_idx).total_memory // (1024**2)  # MB
                torch.cuda.empty_cache()
                memory_available = (torch.cuda.get_device_properties(device_idx).total_memory - 
                                  torch.cuda.memory_allocated(device_idx)) // (1024**2)
                
                # Get compute capability
                props = torch.cuda.get_device_properties(device_idx)
                compute_capability = (props.major, props.minor)
                
                # Check mixed precision support (requires compute capability >= 7.0)
                supports_mixed_precision = compute_capability >= (7, 0)
                
                # Get CUDA version
                cuda_version = torch.version.cuda
                
                return GPUConfig(
                    device=f"cuda:{device_idx}",
                    device_name=device_name,
                    memory_total=memory_total,
                    memory_available=memory_available,
                    compute_capability=compute_capability,
                    supports_mixed_precision=supports_mixed_precision,
                    cuda_version=cuda_version
                )
        
        # Check for ROCm (AMD)
        if hasattr(torch, 'hip') and torch.hip.is_available():
            device_count = torch.hip.device_count()
            if device_count > 0:
                device_idx = 0
                device_name = torch.hip.get_device_name(device_idx)
                
                return GPUConfig(
                    device=f"hip:{device_idx}",
                    device_name=device_name,
                    supports_mixed_precision=False  # Conservative default for ROCm
                )
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return GPUConfig(
                device="mps",
                device_name="Apple Silicon GPU",
                supports_mixed_precision=False  # MPS doesn't support all mixed precision ops
            )
        
        # Fallback to CPU
        cpu_count = psutil.cpu_count() if psutil else 1
        cpu_name = platform.processor() or f"{cpu_count}-core CPU"
        
        return GPUConfig(
            device="cpu",
            device_name=cpu_name
        )
    
    def _configure_pytorch_optimizations(self):
        """Configure PyTorch optimizations based on detected hardware."""
        
        if self.config.device.startswith("cuda"):
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            # Enable mixed precision if supported
            if self.config.supports_mixed_precision:
                logger.info("Mixed precision training enabled")
        
        elif self.config.device == "mps":
            # MPS-specific optimizations
            logger.info("Apple Silicon MPS optimizations enabled")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory usage information."""
        
        if not self._initialized:
            return {"error": "GPU manager not initialized"}
        
        if self.config.device.startswith("cuda"):
            try:
                allocated = torch.cuda.memory_allocated() // (1024**2)  # MB
                cached = torch.cuda.memory_reserved() // (1024**2)  # MB
                free = self.config.memory_total - allocated
                
                return {
                    "device": self.config.device,
                    "total_mb": self.config.memory_total,
                    "allocated_mb": allocated,
                    "cached_mb": cached,
                    "free_mb": free,
                    "utilization": allocated / self.config.memory_total
                }
            except Exception as e:
                return {"error": f"Failed to get CUDA memory info: {e}"}
        
        return {
            "device": self.config.device,
            "message": "Memory info not available for non-CUDA devices"
        }
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        
        if self.config.device.startswith("cuda"):
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared")
        
    def get_device(self) -> torch.device:
        """Get the configured PyTorch device."""
        
        if not self._initialized:
            return torch.device("cpu")
        
        return self.device
    
    def get_config(self) -> GPUConfig:
        """Get GPU configuration."""
        
        if not self._initialized:
            return GPUConfig()
        
        return self.config
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        
        return self._initialized and self.config.device != "cpu"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        
        info = {
            "gpu_initialized": self._initialized,
            "device": self.config.device if self.config else "unknown",
            "device_name": self.config.device_name if self.config else "unknown",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count()
            })
        
        return info

# Global GPU manager instance
gpu_manager = GPUManager()
