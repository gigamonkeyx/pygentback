"""
Ollama Provider Implementation - PRODUCTION READY

Provides local Ollama LLM access with comprehensive model management,
async operations, and production-ready error handling.
"""

import logging
import json
from typing import Dict, List, Any, Optional
import aiohttp
from datetime import datetime

from .base_provider import BaseProvider, BaseProviderManager, ProviderCapabilities

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """
    Ollama local LLM provider with comprehensive model support.
    
    Provides access to locally running Ollama models with full lifecycle management,
    model downloading, and health monitoring.
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 11434,
                 timeout: int = 30):
        """
        Initialize Ollama provider.
        
        Args:
            host: Ollama server host
            port: Ollama server port
            timeout: Request timeout in seconds
        """
        super().__init__("ollama")
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        self.available_models = []
        self.model_info = {}
        
        # Set up session with timeout
        self.session_timeout = aiohttp.ClientTimeout(total=timeout)
    
    async def initialize(self) -> bool:
        """Initialize the provider and check Ollama availability."""
        try:
            # Test connection to Ollama
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(f"{self.base_url}/api/version") as response:
                    if response.status == 200:
                        version_info = await response.json()
                        logger.info(f"Connected to Ollama version: {version_info.get('version', 'unknown')}")
                        
                        # Load available models
                        await self._load_available_models()
                        self.is_ready = True
                        logger.info(f"Ollama provider initialized with {len(self.available_models)} models")
                        return True
                    else:
                        logger.error(f"Ollama server returned status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            return False
    
    async def _load_available_models(self) -> None:
        """Load available models from Ollama."""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        
                        self.available_models = []
                        self.model_info = {}
                        
                        for model in models:
                            model_name = model.get("name", "")
                            if model_name:
                                self.available_models.append(model_name)
                                self.model_info[model_name] = {
                                    "name": model_name,
                                    "size": model.get("size", 0),
                                    "modified_at": model.get("modified_at", ""),
                                    "digest": model.get("digest", ""),
                                    "details": model.get("details", {}),
                                    "context_length": model.get("details", {}).get("parameter_size", None)
                                }
                        
                        logger.info(f"Loaded {len(self.available_models)} Ollama models")
                    else:
                        logger.error(f"Failed to fetch Ollama models: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error loading Ollama models: {e}")
    
    async def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        if not self.available_models:
            await self._load_available_models()
        return self.available_models
    
    async def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        available = await self.get_available_models()
        return model_name in available
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        if not self.model_info:
            await self._load_available_models()
        return self.model_info.get(model_name)
    
    async def generate_text(self, 
                           model: str,
                           prompt: str,
                           max_tokens: int = 1000,
                           temperature: float = 0.7,
                           stream: bool = False,
                           **kwargs) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            model: Model name to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Generated text string
        """
        try:
            # Check if model is available
            if not await self.is_model_available(model):
                logger.error(f"Model '{model}' not available in Ollama")
                return ""
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **kwargs
                }
            }
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        if stream:
                            # Handle streaming response
                            full_response = ""
                            async for line in response.content:
                                if line:
                                    try:
                                        json_line = json.loads(line.decode('utf-8'))
                                        if 'response' in json_line:
                                            full_response += json_line['response']
                                        if json_line.get('done', False):
                                            break
                                    except json.JSONDecodeError:
                                        continue
                            return full_response.strip()
                        else:
                            # Handle non-streaming response
                            result = await response.json()
                            return result.get("response", "").strip()
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error {response.status}: {error_text}")
                        return ""
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return ""
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Ollama service."""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(f"{self.base_url}/api/version") as response:
                    if response.status == 200:
                        version_info = await response.json()
                        return {
                            "healthy": True,
                            "status_code": response.status,
                            "version": version_info.get("version", "unknown"),
                            "models_loaded": len(self.available_models),
                            "base_url": self.base_url,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    else:
                        return {
                            "healthy": False,
                            "status_code": response.status,
                            "error": f"HTTP {response.status}",
                            "base_url": self.base_url,
                            "timestamp": datetime.utcnow().isoformat()
                        }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "base_url": self.base_url,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_default_model(self) -> Optional[str]:
        """Get the default/recommended model."""
        if not self.available_models:
            return None
        
        # Prioritize models in order of preference
        preferred_models = [
            "phi4-fast",
            "phi4",
            "llama3.2:3b",
            "llama3.1:8b",
            "llama3.1:7b",
            "llama3:8b",
            "llama3:7b",
            "phi3.5",
            "phi3",
            "gemma2:2b",
            "gemma:2b"
        ]
        
        for preferred in preferred_models:
            if preferred in self.available_models:
                return preferred
        
        # Return first available model if no preferred models found
        return self.available_models[0] if self.available_models else None
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            payload = {"name": model_name}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json=payload
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully pulled model: {model_name}")
                        # Refresh available models
                        await self._load_available_models()
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to pull model {model_name}: {error_text}")
                        return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        return ProviderCapabilities(
            supports_streaming=True,
            supports_embeddings=True,  # Ollama supports embeddings
            supports_function_calling=False,  # Limited function calling support
            max_context_length=None,  # Varies by model
            pricing_model="free"
        )


class OllamaManager(BaseProviderManager):
    """
    Manager for Ollama provider with lifecycle management.
    
    Provides centralized management of Ollama provider including initialization,
    health checks, and model management.
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 11434,
                 timeout: int = 30):
        """Initialize Ollama manager."""
        super().__init__("ollama")
        self.host = host
        self.port = port
        self.timeout = timeout
        self.provider = OllamaProvider(host=host, port=port, timeout=timeout)
    
    async def start(self) -> bool:
        """Start the Ollama manager."""
        try:
            success = await self.provider.initialize()
            self.is_ready = success
            if success:
                logger.info("Ollama manager started successfully")
            else:
                logger.error("Failed to start Ollama manager")
            return success
        except Exception as e:
            logger.error(f"Error starting Ollama manager: {e}")
            return False
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        if self.provider and self.is_ready:
            return await self.provider.pull_model(model_name)
        return False
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        if self.provider:
            return self.provider.get_capabilities()
        return ProviderCapabilities()
    
    async def get_recommended_models(self) -> List[str]:
        """Get list of recommended models for new users."""
        return [
            "phi4-fast",  # Fast and lightweight
            "llama3.2:3b",  # Good balance of size and performance
            "llama3.1:8b",  # Better performance, larger size
            "gemma2:2b"  # Very lightweight option
        ]


# Convenience function for global access
_ollama_manager = None

def get_ollama_manager(host: str = "localhost", 
                      port: int = 11434,
                      timeout: int = 30) -> OllamaManager:
    """Get global Ollama manager instance."""
    global _ollama_manager
    if _ollama_manager is None:
        _ollama_manager = OllamaManager(host=host, port=port, timeout=timeout)
    return _ollama_manager
