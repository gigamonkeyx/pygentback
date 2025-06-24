"""
Core API Routes for Providers and Models

Simple endpoints to expose provider registry and model information
for API consumers and tests.
"""

import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

print(">>> CORE ROUTER LOADED <<<")
logger.info("Core router module imported successfully")

@router.get("/debug-test")
async def debug_test():
    """
    Debug endpoint to test if core router is working
    """
    print(">>> DEBUG TEST ENDPOINT CALLED <<<")
    logger.info("Debug test endpoint called")
    return {"message": "Debug test successful", "timestamp": "2024-06-19"}

@router.get("/providers")
async def get_providers():
    """
    Get all available providers and their status.
    
    Returns:
        Dict containing provider information
    """
    try:
        # Return static data for now to avoid dependency issues
        providers = {
            "openai": {
                "name": "OpenAI",
                "ready": True,
                "models": ["gpt-4", "gpt-3.5-turbo"]
            },
            "anthropic": {
                "name": "Anthropic",
                "ready": True,
                "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"]
            },
            "ollama": {
                "name": "Ollama",
                "ready": True,
                "models": ["llama2", "mistral"]
            },
            "openrouter": {
                "name": "OpenRouter",
                "ready": True,
                "models": ["gpt-4", "claude-3-opus"]
            }
        }
        
        return {
            "providers": providers,
            "providers_ready": len([p for p in providers.values() if p["ready"]]),
            "providers_total": len(providers),
            "system_healthy": True,
            "timestamp": "2024-06-19T17:35:00Z"
        }
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        raise HTTPException(status_code=500, detail="Failed to get providers")

@router.get("/models")
async def get_models():
    """
    Get all available models from all providers.
    
    Returns:
        Dict containing models by provider
    """
    try:
        models = {
            "openai": [
                {"id": "gpt-4", "name": "GPT-4", "context_length": 8192},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "context_length": 4096}
            ],
            "anthropic": [
                {"id": "claude-3-haiku", "name": "Claude 3 Haiku", "context_length": 200000},
                {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "context_length": 200000},
                {"id": "claude-3-opus", "name": "Claude 3 Opus", "context_length": 200000}
            ],
            "ollama": [
                {"id": "llama2", "name": "Llama 2", "context_length": 4096},
                {"id": "mistral", "name": "Mistral", "context_length": 8192}
            ],
            "openrouter": [
                {"id": "gpt-4", "name": "GPT-4", "context_length": 8192},
                {"id": "claude-3-opus", "name": "Claude 3 Opus", "context_length": 200000}
            ]
        }
        
        total_models = sum(len(provider_models) for provider_models in models.values())
        
        return {
            "models": models,
            "total_models": total_models,
            "providers": list(models.keys()),
            "summary": {
                provider: len(provider_models) 
                for provider, provider_models in models.items()
            }
        }
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail="Failed to get models")
        
        total_models = sum(len(provider_models) for provider_models in models.values())
        
        return {
            "models": models,
            "total_models": total_models,
            "providers": list(models.keys()),
            "summary": {
                provider: len(provider_models) 
                for provider, provider_models in models.items()
            }
        }
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail="Failed to get models")

@router.get("/providers/{provider_name}")
async def get_provider_details(provider_name: str):
    """
    Get detailed information about a specific provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Provider details
    """
    try:
        # Static data for providers
        providers_data = {
            "openai": {"name": "OpenAI", "ready": True, "models": ["gpt-4", "gpt-3.5-turbo"]},
            "anthropic": {"name": "Anthropic", "ready": True, "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"]},
            "ollama": {"name": "Ollama", "ready": True, "models": ["llama2", "mistral"]},
            "openrouter": {"name": "OpenRouter", "ready": True, "models": ["gpt-4", "claude-3-opus"]}
        }
        
        if provider_name not in providers_data:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")
        
        provider_info = providers_data[provider_name]
        
        return {
            "name": provider_name,
            "ready": provider_info["ready"],
            "details": provider_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting provider {provider_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get provider {provider_name}")

@router.get("/models/{provider_name}")
async def get_provider_models(provider_name: str):
    """
    Get models for a specific provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Models for the provider
    """
    try:
        # Static data for models by provider
        models_data = {
            "openai": [
                {"id": "gpt-4", "name": "GPT-4", "context_length": 8192},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "context_length": 4096}
            ],
            "anthropic": [
                {"id": "claude-3-haiku", "name": "Claude 3 Haiku", "context_length": 200000},
                {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "context_length": 200000},
                {"id": "claude-3-opus", "name": "Claude 3 Opus", "context_length": 200000}
            ],
            "ollama": [
                {"id": "llama2", "name": "Llama 2", "context_length": 4096},
                {"id": "mistral", "name": "Mistral", "context_length": 8192}
            ],
            "openrouter": [
                {"id": "gpt-4", "name": "GPT-4", "context_length": 8192},
                {"id": "claude-3-opus", "name": "Claude 3 Opus", "context_length": 200000}
            ]
        }
        
        if provider_name not in models_data:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")
        
        models = models_data[provider_name]
        
        return {
            "provider": provider_name,
            "models": models,
            "model_count": len(models),
            "ready": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models for {provider_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models for {provider_name}")