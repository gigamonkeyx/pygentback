#!/usr/bin/env python3
"""
GPU-Optimized Ollama Integration

Enhanced Ollama integration with RTX 3080 optimization, model management,
and performance monitoring for PyGent Factory.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass

import aiohttp
import psutil

from .gpu_optimization import gpu_optimizer
from .gpu_config import gpu_manager

logger = logging.getLogger(__name__)


@dataclass
class OllamaModelConfig:
    """Ollama model configuration with GPU optimization"""
    name: str
    size_gb: float
    context_length: int
    gpu_layers: int = -1  # -1 = all layers on GPU
    gpu_memory_required_gb: float = 0.0
    recommended_batch_size: int = 1
    supports_streaming: bool = True
    optimization_level: str = "balanced"  # conservative, balanced, aggressive


@dataclass
class OllamaPerformanceMetrics:
    """Ollama performance metrics"""
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_eval_time_ms: float
    eval_time_ms: float
    total_time_ms: float
    tokens_per_second: float
    gpu_utilization: float
    memory_usage_mb: float
    timestamp: datetime


class OllamaGPUManager:
    """GPU-optimized Ollama manager"""
    
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session = None
        self.is_initialized = False
        
        # Model management
        self.available_models: Dict[str, OllamaModelConfig] = {}
        self.loaded_models: List[str] = []
        self.current_model: Optional[str] = None
        
        # Performance tracking
        self.performance_history: List[OllamaPerformanceMetrics] = []
        self.model_performance: Dict[str, List[float]] = {}  # model -> [tokens/sec]
        
        # GPU optimization
        self.gpu_optimized = False
        self.max_gpu_memory_gb = 0.0
        
        # RTX 3080 optimized models
        self.rtx_3080_models = {
            "qwen3:8b": OllamaModelConfig(
                name="qwen3:8b",
                size_gb=5.2,
                context_length=32768,
                gpu_layers=-1,
                gpu_memory_required_gb=6.0,
                recommended_batch_size=4,
                optimization_level="aggressive"
            ),
            "deepseek-r1:8b": OllamaModelConfig(
                name="deepseek-r1:8b", 
                size_gb=5.2,
                context_length=16384,
                gpu_layers=-1,
                gpu_memory_required_gb=6.0,
                recommended_batch_size=2,
                optimization_level="balanced"
            ),
            "llama3.1:8b": OllamaModelConfig(
                name="llama3.1:8b",
                size_gb=4.7,
                context_length=8192,
                gpu_layers=-1,
                gpu_memory_required_gb=5.5,
                recommended_batch_size=4,
                optimization_level="aggressive"
            ),
            "codellama:7b": OllamaModelConfig(
                name="codellama:7b",
                size_gb=3.8,
                context_length=4096,
                gpu_layers=-1,
                gpu_memory_required_gb=4.5,
                recommended_batch_size=6,
                optimization_level="aggressive"
            )
        }
    
    async def initialize(self) -> bool:
        """Initialize GPU-optimized Ollama manager"""
        try:
            logger.info("Initializing GPU-optimized Ollama manager...")
            
            # Initialize GPU optimization
            if not gpu_optimizer.is_initialized:
                await gpu_optimizer.initialize()
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for model loading
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Check Ollama service
            if not await self._check_ollama_service():
                logger.error("Ollama service not available")
                return False
            
            # Detect available GPU memory
            self._detect_gpu_capabilities()
            
            # Load available models
            await self._load_available_models()
            
            # Optimize models for GPU
            await self._optimize_models_for_gpu()
            
            # Start performance monitoring
            asyncio.create_task(self._performance_monitor())
            
            self.is_initialized = True
            logger.info("GPU-optimized Ollama manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama GPU manager: {e}")
            return False
    
    async def _check_ollama_service(self) -> bool:
        """Check if Ollama service is running"""
        try:
            async with self.session.get(f"{self.base_url}/api/version") as response:
                if response.status == 200:
                    version_info = await response.json()
                    logger.info(f"Ollama service detected: {version_info.get('version', 'unknown')}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Ollama service check failed: {e}")
            return False
    
    def _detect_gpu_capabilities(self):
        """Detect GPU capabilities for Ollama optimization"""
        if gpu_manager.is_available():
            # Calculate available GPU memory for models
            total_memory_gb = gpu_manager.config.memory_total / 1024  # Convert MB to GB
            
            # Reserve memory for system and other processes
            available_memory_gb = total_memory_gb * 0.8  # Use 80% for models
            
            self.max_gpu_memory_gb = available_memory_gb
            self.gpu_optimized = True
            
            logger.info(f"GPU memory available for models: {available_memory_gb:.1f}GB")
        else:
            logger.warning("No GPU available for Ollama optimization")
    
    async def _load_available_models(self):
        """Load list of available models from Ollama"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    
                    for model in models:
                        model_name = model.get("name", "")
                        if model_name in self.rtx_3080_models:
                            self.available_models[model_name] = self.rtx_3080_models[model_name]
                        else:
                            # Create basic config for unknown models
                            size_bytes = model.get("size", 0)
                            size_gb = size_bytes / (1024**3) if size_bytes else 4.0
                            
                            self.available_models[model_name] = OllamaModelConfig(
                                name=model_name,
                                size_gb=size_gb,
                                context_length=4096,
                                gpu_memory_required_gb=size_gb * 1.2
                            )
                    
                    logger.info(f"Loaded {len(self.available_models)} available models")
                    
        except Exception as e:
            logger.error(f"Failed to load available models: {e}")
    
    async def _optimize_models_for_gpu(self):
        """Optimize models for GPU usage"""
        if not self.gpu_optimized:
            return
        
        for model_name, config in self.available_models.items():
            if config.gpu_memory_required_gb <= self.max_gpu_memory_gb:
                # Model fits in GPU memory
                config.gpu_layers = -1  # All layers on GPU
                logger.info(f"Model {model_name} optimized for full GPU usage")
            else:
                # Partial GPU usage
                memory_ratio = self.max_gpu_memory_gb / config.gpu_memory_required_gb
                estimated_layers = int(32 * memory_ratio)  # Estimate based on typical layer count
                config.gpu_layers = max(1, estimated_layers)
                logger.info(f"Model {model_name} optimized for {config.gpu_layers} GPU layers")
    
    async def generate(self, 
                      prompt: str, 
                      model: Optional[str] = None,
                      stream: bool = False,
                      **kwargs) -> Dict[str, Any]:
        """Generate response with GPU optimization"""
        if not self.is_initialized:
            raise RuntimeError("Ollama GPU manager not initialized")
        
        # Select optimal model
        target_model = model or self._select_optimal_model(prompt)
        if not target_model:
            raise RuntimeError("No suitable model available")
        
        # Ensure model is loaded with GPU optimization
        await self._ensure_model_loaded(target_model)
        
        # Prepare optimized request
        request_data = self._prepare_optimized_request(target_model, prompt, stream, **kwargs)
        
        # Generate with performance tracking
        start_time = time.time()
        
        try:
            with gpu_optimizer.optimized_inference(target_model):
                if stream:
                    return await self._generate_stream(request_data)
                else:
                    return await self._generate_single(request_data)
        finally:
            # Record performance metrics
            total_time = (time.time() - start_time) * 1000  # ms
            await self._record_performance(target_model, prompt, total_time)
    
    def _select_optimal_model(self, prompt: str) -> Optional[str]:
        """Select optimal model based on prompt and GPU capabilities"""
        # Simple heuristic: prefer smaller models for short prompts, larger for complex tasks
        prompt_length = len(prompt)
        
        if prompt_length < 500:
            # Short prompt - use fastest model
            candidates = ["codellama:7b", "llama3.1:8b", "qwen3:8b"]
        elif "reasoning" in prompt.lower() or "analysis" in prompt.lower():
            # Complex reasoning - use reasoning model
            candidates = ["deepseek-r1:8b", "qwen3:8b", "llama3.1:8b"]
        else:
            # General purpose
            candidates = ["qwen3:8b", "llama3.1:8b", "deepseek-r1:8b"]
        
        # Return first available candidate
        for candidate in candidates:
            if candidate in self.available_models:
                return candidate
        
        # Fallback to any available model
        return next(iter(self.available_models.keys())) if self.available_models else None
    
    async def _ensure_model_loaded(self, model_name: str):
        """Ensure model is loaded with GPU optimization"""
        if model_name not in self.loaded_models:
            logger.info(f"Loading model {model_name} with GPU optimization...")
            
            # Prepare model loading request with GPU settings
            config = self.available_models.get(model_name)
            if config and self.gpu_optimized:
                # Set GPU layers in environment or model config
                load_request = {
                    "name": model_name,
                    "options": {
                        "num_gpu": config.gpu_layers,
                        "num_ctx": config.context_length
                    }
                }
                
                try:
                    async with self.session.post(
                        f"{self.base_url}/api/pull",
                        json={"name": model_name}
                    ) as response:
                        if response.status == 200:
                            self.loaded_models.append(model_name)
                            self.current_model = model_name
                            logger.info(f"Model {model_name} loaded successfully")
                        
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
    
    def _prepare_optimized_request(self, model: str, prompt: str, stream: bool, **kwargs) -> Dict[str, Any]:
        """Prepare optimized request for GPU inference"""
        config = self.available_models.get(model)
        
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {}
        }
        
        if config:
            # Set optimized parameters
            request_data["options"].update({
                "num_ctx": config.context_length,
                "num_batch": config.recommended_batch_size,
                "num_gpu": config.gpu_layers
            })
            
            # RTX 3080 specific optimizations
            if gpu_optimizer.rtx_3080_detected and config.optimization_level == "aggressive":
                request_data["options"].update({
                    "num_thread": 8,  # Optimize for RTX 3080
                    "use_mlock": True,
                    "use_mmap": True
                })
        
        # Merge user-provided options
        request_data["options"].update(kwargs.get("options", {}))
        
        return request_data
    
    async def _generate_single(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate single response"""
        async with self.session.post(
            f"{self.base_url}/api/generate",
            json=request_data
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama API error: {response.status}")
            
            return await response.json()
    
    async def _generate_stream(self, request_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response"""
        async with self.session.post(
            f"{self.base_url}/api/generate",
            json=request_data
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama API error: {response.status}")
            
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        yield data
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    async def _record_performance(self, model: str, prompt: str, total_time_ms: float):
        """Record performance metrics"""
        try:
            # Estimate tokens (rough approximation)
            prompt_tokens = len(prompt.split())
            
            # Calculate tokens per second
            tokens_per_second = prompt_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
            
            # Get GPU metrics
            gpu_util = 0.0
            memory_usage = 0.0
            
            if gpu_manager.is_available():
                memory_info = gpu_manager.get_memory_info()
                memory_usage = memory_info.get("allocated_mb", 0)
            
            # Create performance record
            metrics = OllamaPerformanceMetrics(
                model_name=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,  # Would need response to calculate
                total_tokens=prompt_tokens,
                prompt_eval_time_ms=total_time_ms * 0.3,  # Estimate
                eval_time_ms=total_time_ms * 0.7,  # Estimate
                total_time_ms=total_time_ms,
                tokens_per_second=tokens_per_second,
                gpu_utilization=gpu_util,
                memory_usage_mb=memory_usage,
                timestamp=datetime.utcnow()
            )
            
            self.performance_history.append(metrics)
            
            # Update model performance tracking
            if model not in self.model_performance:
                self.model_performance[model] = []
            self.model_performance[model].append(tokens_per_second)
            
            # Keep only recent performance data
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            if len(self.model_performance[model]) > 100:
                self.model_performance[model] = self.model_performance[model][-100:]
                
        except Exception as e:
            logger.error(f"Failed to record performance metrics: {e}")
    
    async def _performance_monitor(self):
        """Background performance monitoring"""
        while self.is_initialized:
            try:
                # Monitor GPU utilization during inference
                if gpu_manager.is_available() and self.current_model:
                    memory_info = gpu_manager.get_memory_info()
                    utilization = memory_info.get("utilization", 0)
                    
                    if utilization > 0.95:
                        logger.warning(f"High GPU utilization during {self.current_model} inference: {utilization:.1%}")
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models"""
        summary = {
            "gpu_optimized": self.gpu_optimized,
            "max_gpu_memory_gb": self.max_gpu_memory_gb,
            "loaded_models": self.loaded_models,
            "current_model": self.current_model,
            "model_performance": {}
        }
        
        for model, performance_data in self.model_performance.items():
            if performance_data:
                summary["model_performance"][model] = {
                    "avg_tokens_per_second": sum(performance_data) / len(performance_data),
                    "max_tokens_per_second": max(performance_data),
                    "min_tokens_per_second": min(performance_data),
                    "sample_count": len(performance_data)
                }
        
        return summary
    
    async def cleanup(self):
        """Cleanup resources"""
        self.is_initialized = False
        if self.session:
            await self.session.close()


# Global GPU-optimized Ollama manager
ollama_gpu_manager = OllamaGPUManager()
