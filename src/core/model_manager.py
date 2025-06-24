"""
Model Manager - Handles model selection and cost management

This module provides utilities for managing LLM models across providers,
including cost estimation, model recommendations, and user-friendly selection.
"""

from typing import Dict, List, Any, Optional


class ModelManager:
    """Manages model selection, recommendations, and cost estimation"""
    
    def __init__(self, ollama_manager=None, openrouter_manager=None):
        self.ollama_manager = ollama_manager
        self.openrouter_manager = openrouter_manager
    
    async def get_all_available_models(self, include_paid: bool = True) -> Dict[str, Any]:
        """Get all available models with metadata"""
        models = {
            "ollama": {"models": [], "status": "not_available"},
            "openrouter": {"models": [], "free_models": [], "paid_models": [], "status": "not_available"}
        }
        
        # Ollama models
        if self.ollama_manager and self.ollama_manager.is_ready:
            ollama_models = await self.ollama_manager.get_available_models()
            models["ollama"] = {
                "models": ollama_models,
                "status": "available",
                "cost": "free (local)",
                "note": "Requires local installation and download"
            }
        
        # OpenRouter models
        if self.openrouter_manager and self.openrouter_manager.is_ready:
            all_models = await self.openrouter_manager.get_available_models()
            free_models = [m for m in all_models if ":free" in m or "free" in m.lower()]
            paid_models = [m for m in all_models if m not in free_models] if include_paid else []
            
            models["openrouter"] = {
                "models": all_models,
                "free_models": free_models,
                "paid_models": paid_models,
                "status": "available",
                "default_free": "deepseek/deepseek-r1-0528-qwen3-8b:free"
            }
        
        return models
    
    def get_recommended_models_by_use_case(self) -> Dict[str, Dict[str, str]]:
        """Get model recommendations by use case"""
        return {
            "free_models": {
                "deepseek/deepseek-r1-0528-qwen3-8b:free": "General reasoning, coding - Free",
                "google/gemma-2-9b-it:free": "Balanced performance - Free", 
                "meta-llama/llama-3.1-8b-instruct:free": "Large context, coding - Free",
                "phi4-fast": "Local Ollama model - Free (if installed)"
            },
            "budget_models": {
                "anthropic/claude-3-haiku": "Fast, cheap, good quality - $0.25/$1.25 per 1M tokens",
                "openai/gpt-3.5-turbo": "Reliable, affordable - $0.50/$1.50 per 1M tokens"
            },
            "premium_models": {
                "anthropic/claude-3.5-sonnet": "Best reasoning/coding - $3.00/$15.00 per 1M tokens",
                "openai/gpt-4o": "Most reliable - $2.50/$10.00 per 1M tokens",
                "google/gemini-pro-1.5": "Huge context window - $1.25/$5.00 per 1M tokens"
            }
        }
    
    def get_default_config_for_agent(self, agent_type: str, budget: str = "free") -> Dict[str, Any]:
        """Get default configuration for an agent type and budget"""
        base_config = {
            "provider": "openrouter",
            "model_name": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Adjust for agent type
        type_adjustments = {
            "reasoning": {"temperature": 0.3, "max_tokens": 2000},
            "coding": {"temperature": 0.1, "max_tokens": 3000},
            "research": {"temperature": 0.2, "max_tokens": 4000},
            "analysis": {"temperature": 0.3, "max_tokens": 2000}
        }
        
        if agent_type in type_adjustments:
            base_config.update(type_adjustments[agent_type])
        
        # Adjust for budget
        if budget == "premium" and agent_type in ["reasoning", "coding", "analysis"]:
            base_config["model_name"] = "anthropic/claude-3.5-sonnet"
        elif budget == "budget":
            base_config["model_name"] = "anthropic/claude-3-haiku"
        # Keep free model for "free" budget
        
        return base_config
    
    def estimate_costs(self, model_name: str, tokens_per_day: int = 10000) -> Dict[str, Any]:
        """Estimate costs for a model"""
        if ":free" in model_name or "phi4" in model_name:
            return {
                "model": model_name,
                "is_free": True,
                "daily_cost": 0,
                "monthly_cost": 0,
                "note": "Free model"
            }
        
        # Simplified pricing map
        pricing = {
            "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
            "openai/gpt-4o": {"input": 2.5, "output": 10.0},
            "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
            "google/gemini-pro-1.5": {"input": 1.25, "output": 5.0}
        }
        
        if model_name not in pricing:
            return {"model": model_name, "error": "Pricing not available"}
        
        # Estimate costs (70% input, 30% output tokens)
        rates = pricing[model_name]
        input_tokens = int(tokens_per_day * 0.7)
        output_tokens = int(tokens_per_day * 0.3)
        
        daily_cost = (input_tokens / 1_000_000 * rates["input"]) + (output_tokens / 1_000_000 * rates["output"])
        
        return {
            "model": model_name,
            "is_free": False,
            "daily_cost": round(daily_cost, 4),
            "monthly_cost": round(daily_cost * 30, 2),
            "breakdown": f"{input_tokens:,} input + {output_tokens:,} output tokens"
        }
    
    async def validate_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a model configuration"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        provider = config.get("provider")
        model_name = config.get("model_name")
        
        if not provider:
            result["valid"] = False
            result["errors"].append("Provider is required")
            return result
        
        if not model_name:
            result["valid"] = False  
            result["errors"].append("Model name is required")
            return result
        
        if provider == "ollama":
            if not self.ollama_manager or not self.ollama_manager.is_ready:
                result["valid"] = False
                result["errors"].append("Ollama is not available")
            elif not await self.ollama_manager.is_model_available(model_name):
                result["valid"] = False
                available = await self.ollama_manager.get_available_models()
                result["errors"].append(f"Model '{model_name}' not available. Available: {', '.join(available[:3])}")
        
        elif provider == "openrouter":
            if not self.openrouter_manager or not self.openrouter_manager.is_ready:
                result["valid"] = False
                result["errors"].append("OpenRouter is not available")
            elif not await self.openrouter_manager.is_model_available(model_name):
                result["valid"] = False
                result["errors"].append(f"Model '{model_name}' not available on OpenRouter")
        
        else:
            result["valid"] = False
            result["errors"].append(f"Unknown provider '{provider}'. Use 'ollama' or 'openrouter'")
        
        return result
    
    def get_user_friendly_model_list(self, include_paid: bool = True) -> Dict[str, List[Dict[str, str]]]:
        """Get a user-friendly formatted model list"""
        recommendations = self.get_recommended_models_by_use_case()
        
        formatted = {
            "🆓 Free Models": [
                {"name": name, "description": desc} 
                for name, desc in recommendations["free_models"].items()
            ]
        }
        
        if include_paid:
            formatted["💰 Budget Models"] = [
                {"name": name, "description": desc}
                for name, desc in recommendations["budget_models"].items()
            ]
            formatted["⭐ Premium Models"] = [
                {"name": name, "description": desc}
                for name, desc in recommendations["premium_models"].items()
            ]
        
        return formatted
