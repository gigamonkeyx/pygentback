"""
Provider Registry - Central Provider Management (REFACTORED)

Manages all LLM providers (Ollama, OpenRouter, etc.) with unified interface,
health monitoring, and automatic failover capabilities.

SIMPLIFIED: Focused on provider management only, removed mock implementations.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProviderManager
from .ollama_provider import get_ollama_manager
from .openrouter_provider import get_openrouter_manager

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Central registry for managing all LLM providers.
    
    Provides unified access to multiple providers with health monitoring,
    automatic failover, and load balancing capabilities.
    """
    
    def __init__(self):
        """Initialize the provider registry."""
        self.providers: Dict[str, BaseProviderManager] = {}
        self.provider_status: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self, 
                        enable_ollama: bool = True,
                        enable_openrouter: bool = True,
                        ollama_config: Optional[Dict[str, Any]] = None,
                        openrouter_config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        Initialize all enabled providers.
        
        Args:
            enable_ollama: Whether to enable Ollama provider
            enable_openrouter: Whether to enable OpenRouter provider
            ollama_config: Optional configuration for Ollama
            openrouter_config: Optional configuration for OpenRouter
            
        Returns:
            Dict mapping provider names to initialization success status
        """
        async with self._lock:
            results = {}
            
            # Initialize Ollama if enabled
            if enable_ollama:
                try:
                    ollama_config = ollama_config or {}
                    ollama_manager = get_ollama_manager(
                        host=ollama_config.get("host", "localhost"),
                        port=ollama_config.get("port", 11434),
                        timeout=ollama_config.get("timeout", 30)
                    )
                    success = await ollama_manager.start()
                    self.providers["ollama"] = ollama_manager
                    results["ollama"] = success
                    logger.info(f"Ollama provider initialization: {'success' if success else 'failed'}")
                except Exception as e:
                    logger.error(f"Error initializing Ollama: {e}")
                    results["ollama"] = False
            
            # Initialize OpenRouter if enabled
            if enable_openrouter:
                try:
                    openrouter_config = openrouter_config or {}
                    openrouter_manager = get_openrouter_manager(
                        api_key=openrouter_config.get("api_key")
                    )
                    success = await openrouter_manager.start()
                    self.providers["openrouter"] = openrouter_manager
                    results["openrouter"] = success
                    logger.info(f"OpenRouter provider initialization: {'success' if success else 'failed'}")
                except Exception as e:
                    logger.error(f"Error initializing OpenRouter: {e}")
                    results["openrouter"] = False
            
            self.initialized = True
            await self._update_provider_status()
            
            return results
    
    async def _update_provider_status(self) -> None:
        """Update status information for all providers."""
        for name, provider in self.providers.items():
            try:
                health_info = await provider.health_check()
                models = await provider.get_available_models()
                
                self.provider_status[name] = {
                    "name": name,
                    "ready": provider.is_ready,
                    "health": health_info,
                    "model_count": len(models),
                    "models": models[:5],  # First 5 models for preview
                    "capabilities": provider.get_capabilities().__dict__ if hasattr(provider, 'get_capabilities') else {},
                    "last_updated": datetime.utcnow().isoformat()
                }
            except Exception as e:
                self.provider_status[name] = {
                    "name": name,
                    "ready": False,
                    "error": str(e),
                    "last_updated": datetime.utcnow().isoformat()
                }
    
    async def get_provider(self, provider_name: str) -> Optional[BaseProviderManager]:
        """Get a specific provider by name."""
        return self.providers.get(provider_name)
    
    async def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())
    
    async def get_ready_providers(self) -> List[str]:
        """Get list of ready provider names."""
        return [name for name, provider in self.providers.items() if provider.is_ready]
    
    async def get_all_models(self) -> Dict[str, List[str]]:
        """Get all available models from all providers."""
        all_models = {}
        for name, provider in self.providers.items():
            if provider.is_ready:
                try:
                    models = await provider.get_available_models()
                    all_models[name] = models
                except Exception as e:
                    logger.error(f"Error getting models from {name}: {e}")
                    all_models[name] = []
            else:
                all_models[name] = []
        return all_models
    
    async def is_model_available(self, model_name: str, provider_name: Optional[str] = None) -> Dict[str, bool]:
        """
        Check if a model is available across providers.
        
        Args:
            model_name: Name of the model to check
            provider_name: Optional specific provider to check
            
        Returns:
            Dict mapping provider names to availability status
        """
        availability = {}
        
        providers_to_check = [provider_name] if provider_name else self.providers.keys()
        
        for name in providers_to_check:
            provider = self.providers.get(name)
            if provider and provider.is_ready:
                try:
                    available = await provider.is_model_available(model_name)
                    availability[name] = available
                except Exception as e:
                    logger.error(f"Error checking model {model_name} on {name}: {e}")
                    availability[name] = False
            else:
                availability[name] = False
        
        return availability
    
    async def generate_text(self, 
                           provider_name: str,
                           model: str,
                           prompt: str,
                           **kwargs) -> str:
        """
        Generate text using a specific provider.
        
        Args:
            provider_name: Name of the provider to use
            model: Model name
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        provider = self.providers.get(provider_name)
        if not provider or not provider.is_ready:
            logger.error(f"Provider {provider_name} not available")
            return ""
        
        try:
            return await provider.generate_text(model, prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error generating text with {provider_name}/{model}: {e}")
            return ""
    
    async def generate_text_with_fallback(self, 
                                         model: str,
                                         prompt: str,
                                         preferred_providers: Optional[List[str]] = None,
                                         **kwargs) -> Dict[str, Any]:
        """
        Generate text with automatic fallback to other providers.
        
        Args:
            model: Model name to try
            prompt: Input prompt
            preferred_providers: Ordered list of preferred providers
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with result, provider used, and any errors
        """
        if not preferred_providers:
            preferred_providers = await self.get_ready_providers()
        
        errors = []
        
        for provider_name in preferred_providers:
            provider = self.providers.get(provider_name)
            if not provider or not provider.is_ready:
                errors.append(f"{provider_name}: not ready")
                continue
            
            # Check if model is available on this provider
            model_available = await provider.is_model_available(model)
            if not model_available:
                errors.append(f"{provider_name}: model '{model}' not available")
                continue
            
            # Try to generate
            try:
                result = await provider.generate_text(model, prompt, **kwargs)
                if result:  # Success
                    return {
                        "success": True,
                        "result": result,
                        "provider_used": provider_name,
                        "model_used": model,
                        "errors": errors
                    }
                else:
                    errors.append(f"{provider_name}: empty response")
            except Exception as e:
                errors.append(f"{provider_name}: {str(e)}")
        
        return {
            "success": False,
            "result": "",
            "provider_used": None,
            "model_used": model,
            "errors": errors
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for all providers."""
        await self._update_provider_status()
        
        ready_count = len(await self.get_ready_providers())
        total_count = len(self.providers)
        
        return {
            "initialized": self.initialized,
            "providers_ready": ready_count,
            "providers_total": total_count,
            "providers": self.provider_status,
            "system_healthy": ready_count > 0,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_model_recommendations(self, 
                                      agent_type: str = "general",
                                      include_free_only: bool = False) -> Dict[str, List[str]]:
        """
        Get model recommendations for different agent types.
        
        Args:
            agent_type: Type of agent (reasoning, coding, general, etc.)
            include_free_only: Whether to include only free models
            
        Returns:
            Dict mapping provider names to recommended model lists
        """
        recommendations = {}
        
        for name, provider in self.providers.items():
            if not provider.is_ready:
                continue
            
            try:
                if hasattr(provider, 'get_recommended_models'):
                    models = await provider.get_recommended_models()
                else:
                    # Fallback to available models
                    available = await provider.get_available_models()
                    models = available[:3] if available else []
                
                # Filter for free models if requested
                if include_free_only and name == "openrouter":
                    models = [m for m in models if ":free" in m or "free" in m.lower()]
                
                recommendations[name] = models
            except Exception as e:
                logger.error(f"Error getting recommendations from {name}: {e}")
                recommendations[name] = []
        
        return recommendations
    
    async def shutdown(self) -> None:
        """Shutdown all providers."""
        logger.info("Shutting down provider registry...")
        
        for name, provider in self.providers.items():
            try:
                await provider.stop()
                logger.info(f"Shutdown provider: {name}")
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
        
        self.providers.clear()
        self.provider_status.clear()
        self.initialized = False


# Global registry instance
_provider_registry = None

def get_provider_registry() -> ProviderRegistry:
    """Get global provider registry instance."""
    global _provider_registry
    if _provider_registry is None:
        _provider_registry = ProviderRegistry()
    return _provider_registry
