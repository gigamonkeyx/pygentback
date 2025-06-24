"""
OpenRouter Provider Implementation - PRODUCTION READY

Provides OpenRouter cloud LLM access with comprehensive model support,
async operations, and production-ready error handling.
"""

import logging
import os
from typing import Dict, List, Any, Optional
import aiohttp
from datetime import datetime

from .base_provider import BaseProvider, BaseProviderManager, ProviderCapabilities

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseProvider):
    """
    OpenRouter cloud LLM provider with comprehensive model support.
    
    Provides access to multiple cloud LLM providers through OpenRouter's unified API,
    including Claude, GPT-4, Gemini, and many others.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: str = "https://openrouter.ai/api/v1",
                 site_url: Optional[str] = None,
                 site_name: Optional[str] = None):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key (or from OPENROUTER_API_KEY env var)
            base_url: OpenRouter API base URL
            site_url: Your site URL for OpenRouter rankings (optional)
            site_name: Your site name for OpenRouter rankings (optional)
        """
        super().__init__("openrouter")
        # Use provided API key or the one you gave me as fallback
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or "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"
        self.base_url = base_url
        self.site_url = site_url or "https://pygent-factory.local"
        self.site_name = site_name or "PyGent Factory"
        self.available_models = []
        self.popular_models = {}
        
        # Set up default headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.site_url:
            self.headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            self.headers["X-Title"] = self.site_name
    
    async def initialize(self) -> bool:
        """Initialize the provider and fetch available models."""
        if not self.api_key:
            logger.error("OpenRouter API key not provided")
            return False
        
        try:
            # Test API connection and fetch models
            await self._load_available_models()
            self.is_ready = True
            logger.info(f"OpenRouter provider initialized with {len(self.available_models)} models")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter provider: {e}")
            return False
    
    async def _load_available_models(self) -> None:
        """Load available models from OpenRouter API, filtering for FREE models only."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Filter for FREE models only
                        free_models = []
                        self.popular_models = {}

                        for model in data.get("data", []):
                            model_id = model["id"]
                            pricing = model.get("pricing", {})

                            # Check if model is free (no cost for prompt or completion)
                            prompt_cost = float(pricing.get("prompt", "1"))
                            completion_cost = float(pricing.get("completion", "1"))

                            if prompt_cost == 0 and completion_cost == 0:
                                free_models.append(model_id)
                                self.popular_models[model_id] = {
                                    "name": model.get("name", model_id),
                                    "context_length": model.get("context_length", 4096),
                                    "pricing": "FREE",
                                    "owned_by": model.get("owned_by", "unknown")
                                }

                        self.available_models = free_models
                        logger.info(f"Loaded {len(self.available_models)} FREE OpenRouter models")
                    else:
                        logger.error(f"Failed to fetch OpenRouter models: HTTP {response.status}")
                        # Fallback to known free models
                        self._set_fallback_free_models()
        except Exception as e:
            logger.error(f"Error loading OpenRouter models: {e}")
            self._set_fallback_free_models()
    
    def _set_fallback_free_models(self) -> None:
        """Set fallback FREE models if API fetch fails."""
        self.popular_models = {
            "deepseek/deepseek-r1": {
                "name": "DeepSeek R1 (Free)",
                "context_length": 32768,
                "owned_by": "deepseek",
                "pricing": "FREE"
            },
            "google/gemma-2-9b-it:free": {
                "name": "Gemma 2 9B IT (Free)",
                "context_length": 8192,
                "owned_by": "google",
                "pricing": "FREE"
            },
            "meta-llama/llama-3.2-3b-instruct:free": {
                "name": "Llama 3.2 3B Instruct (Free)",
                "context_length": 131072,
                "owned_by": "meta",
                "pricing": "FREE"
            },
            "microsoft/phi-3-mini-128k-instruct:free": {
                "name": "Phi-3 Mini 128K Instruct (Free)",
                "context_length": 128000,
                "owned_by": "microsoft",
                "pricing": "FREE"
            },
            "qwen/qwen-2-7b-instruct:free": {
                "name": "Qwen 2 7B Instruct (Free)",
                "context_length": 32768,
                "owned_by": "alibaba",
                "pricing": "FREE"
            }
        }
        self.available_models = list(self.popular_models.keys())
    
    async def get_available_models(self) -> List[str]:
        """Get list of available model IDs."""
        if not self.available_models:
            await self._load_available_models()
        return self.available_models
    
    def get_popular_models(self) -> Dict[str, Dict[str, Any]]:
        """Get popular models with their metadata."""
        return self.popular_models
    
    async def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        available = await self.get_available_models()
        return model_name in available
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        if not self.popular_models:
            await self._load_available_models()
        return self.popular_models.get(model_name)
    
    async def generate_text(self, 
                           model: str,
                           prompt: str,
                           max_tokens: int = 1000,
                           temperature: float = 0.7,
                           stream: bool = False,
                           **kwargs) -> str:
        """
        Generate text using OpenRouter API.
        
        Args:
            model: Model ID to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Generated text string
        """
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
                **kwargs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenRouter API error {response.status}: {error_text}")
                        return ""
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            return ""
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on OpenRouter API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=self.headers
                ) as response:
                    return {
                        "healthy": response.status == 200,
                        "status_code": response.status,
                        "api_key_valid": bool(self.api_key),
                        "models_loaded": len(self.available_models),
                        "timestamp": datetime.utcnow().isoformat()
                    }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "api_key_valid": bool(self.api_key),
                "models_loaded": len(self.available_models),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_default_model(self) -> Optional[str]:
        """Get the default/recommended FREE model."""
        # Prioritize best free models
        preferred_free_models = [
            "deepseek/deepseek-r1",  # Primary recommendation - free and powerful
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemma-2-9b-it:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "qwen/qwen-2-7b-instruct:free"
        ]

        for preferred in preferred_free_models:
            if preferred in self.available_models:
                return preferred

        # Return first available free model if no preferred models found
        return self.available_models[0] if self.available_models else None
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        return ProviderCapabilities(
            supports_streaming=True,
            supports_embeddings=False,  # OpenRouter doesn't support embeddings directly
            supports_function_calling=True,  # Many OpenRouter models support function calling
            max_context_length=200000,  # Varies by model, Claude has very large context
            pricing_model="pay-per-use"
        )


class OpenRouterManager(BaseProviderManager):
    """
    Manager for OpenRouter provider with lifecycle management.
    
    Similar to OllamaManager, provides centralized management of OpenRouter
    provider including initialization, health checks, and model management.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenRouter manager."""
        super().__init__("openrouter")
        # Use provided API key or the one you gave me as fallback
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or "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"
        self.provider = OpenRouterProvider(api_key=self.api_key)
    
    async def start(self) -> bool:
        """Start the OpenRouter manager."""
        try:
            success = await self.provider.initialize()
            self.is_ready = success
            if success:
                logger.info("OpenRouter manager started successfully")
            else:
                logger.error("Failed to start OpenRouter manager")
            return success
        except Exception as e:
            logger.error(f"Error starting OpenRouter manager: {e}")
            return False
    
    def get_popular_models(self) -> Dict[str, Dict[str, Any]]:
        """Get popular models with metadata."""
        return self.provider.get_popular_models()
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        if self.provider:
            return self.provider.get_capabilities()
        return ProviderCapabilities()
    
    async def get_recommended_models(self) -> List[str]:
        """Get list of recommended FREE models for new users."""
        return [
            "deepseek/deepseek-r1",  # Primary recommendation - free and powerful
            "meta-llama/llama-3.2-3b-instruct:free",  # Good balance
            "google/gemma-2-9b-it:free",  # Google's free offering
            "microsoft/phi-3-mini-128k-instruct:free",  # Large context, free
            "qwen/qwen-2-7b-instruct:free"  # Alibaba's free model
        ]


# Convenience function for global access
_openrouter_manager = None

def get_openrouter_manager(api_key: Optional[str] = None) -> OpenRouterManager:
    """Get global OpenRouter manager instance."""
    global _openrouter_manager
    if _openrouter_manager is None:
        _openrouter_manager = OpenRouterManager(api_key=api_key)
    return _openrouter_manager
