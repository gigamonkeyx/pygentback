"""
Core API routes for the PyGent Factory application.
"""
from fastapi import APIRouter
from typing import List, Dict, Any

router = APIRouter()

@router.get("/providers")
async def get_providers() -> List[Dict[str, Any]]:
    """Get available AI providers."""
    # Return a list of available providers
    providers = [
        {
            "id": "openai",
            "name": "OpenAI",
            "description": "OpenAI API provider",
            "models": ["gpt-4", "gpt-3.5-turbo"]
        },
        {
            "id": "anthropic",
            "name": "Anthropic",
            "description": "Anthropic Claude API provider",
            "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"]
        },
        {
            "id": "openrouter",
            "name": "OpenRouter",
            "description": "OpenRouter aggregated API provider",
            "models": ["gpt-4", "claude-3-opus", "llama-2-70b"]
        }
    ]
    return providers

@router.get("/models")
async def get_models() -> List[Dict[str, Any]]:
    """Get available AI models."""
    # Return a list of available models
    models = [
        {
            "id": "gpt-4",
            "name": "GPT-4",
            "provider": "openai",
            "description": "OpenAI's most capable model",
            "context_length": 8192
        },
        {
            "id": "gpt-3.5-turbo",
            "name": "GPT-3.5 Turbo",
            "provider": "openai",
            "description": "Fast and capable GPT-3.5 model",
            "context_length": 4096
        },
        {
            "id": "claude-3-haiku",
            "name": "Claude 3 Haiku",
            "provider": "anthropic",
            "description": "Fast and efficient Claude model",
            "context_length": 200000
        },
        {
            "id": "claude-3-sonnet",
            "name": "Claude 3 Sonnet",
            "provider": "anthropic",
            "description": "Balanced Claude model",
            "context_length": 200000
        },
        {
            "id": "claude-3-opus",
            "name": "Claude 3 Opus",
            "provider": "anthropic",
            "description": "Most capable Claude model",
            "context_length": 200000
        }
    ]
    return models
