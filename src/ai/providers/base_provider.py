"""
Base Provider Interface

Defines the common interface for all LLM providers in the PyGent Factory system.
All providers (Ollama, OpenRouter, etc.) should implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, name: str):
        """Initialize the provider with a name."""
        self.name = name
        self.is_ready = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the provider.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """
        Get list of available model IDs.
        
        Returns:
            List[str]: Available model identifiers
        """
        pass
    
    @abstractmethod
    async def is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available.
        
        Args:
            model_name: Name/ID of the model to check
            
        Returns:
            bool: True if model is available, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_text(self, 
                           model: str,
                           prompt: str,
                           max_tokens: int = 1000,
                           temperature: float = 0.7,
                           stream: bool = False,
                           **kwargs) -> str:
        """
        Generate text using the specified model.
        
        Args:
            model: Model ID to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            str: Generated text
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the provider.
        
        Returns:
            Dict containing health status information
        """
        pass
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name/ID of the model
            
        Returns:
            Dict with model information or None if not available
        """
        # Default implementation - providers can override
        if await self.is_model_available(model_name):
            return {
                "name": model_name,
                "provider": self.name,
                "available": True,
                "context_length": None,
                "pricing": None
            }
        return None
    
    def get_default_model(self) -> Optional[str]:
        """
        Get the default/recommended model for this provider.
        
        Returns:
            str: Default model name or None if no models available
        """
        # Default implementation - providers should override
        return None


class BaseProviderManager(ABC):
    """Abstract base class for provider managers."""
    
    def __init__(self, provider_name: str):
        """Initialize the provider manager."""
        self.provider_name = provider_name
        self.is_ready = False
        self.provider: Optional[BaseProvider] = None
    
    @abstractmethod
    async def start(self) -> bool:
        """
        Start the provider manager.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        pass
    
    async def stop(self) -> None:
        """Stop the provider manager."""
        self.is_ready = False
        if self.provider:
            # Cleanup provider resources if needed
            pass
    
    async def get_available_models(self) -> List[str]:
        """Get available models from the provider."""
        if self.provider and self.is_ready:
            return await self.provider.get_available_models()
        return []
    
    async def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available."""
        if self.provider and self.is_ready:
            return await self.provider.is_model_available(model_name)
        return False
    
    async def generate_text(self, 
                           model: str,
                           prompt: str,
                           **kwargs) -> str:
        """Generate text using the provider."""
        if self.provider and self.is_ready:
            return await self.provider.generate_text(model, prompt, **kwargs)
        return ""
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        if self.provider:
            return await self.provider.health_check()
        return {
            "healthy": False,
            "error": "Provider not initialized",
            "provider": self.provider_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_default_model(self) -> Optional[str]:
        """Get default model."""
        if self.provider:
            return self.provider.get_default_model()
        return None


class ProviderCapabilities:
    """Represents the capabilities of a provider."""
    
    def __init__(self,
                 supports_streaming: bool = False,
                 supports_embeddings: bool = False,
                 supports_function_calling: bool = False,
                 max_context_length: Optional[int] = None,
                 pricing_model: Optional[str] = None):
        """Initialize provider capabilities."""
        self.supports_streaming = supports_streaming
        self.supports_embeddings = supports_embeddings
        self.supports_function_calling = supports_function_calling
        self.max_context_length = max_context_length
        self.pricing_model = pricing_model  # "free", "pay-per-use", "subscription", etc.
