#!/usr/bin/env python3
"""
Startup Model Manager
Integrates Hugging Face model discovery with local providers during system startup
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .model_discovery_service import (
    model_discovery_service, 
    ModelInfo, 
    ModelSnapshot,
    integrate_with_local_providers
)

logger = logging.getLogger(__name__)

class StartupModelManager:
    """Manages model discovery and configuration during system startup"""
    
    def __init__(self):
        self.model_snapshot: Optional[ModelSnapshot] = None
        self.integrated_models: Dict[str, ModelInfo] = {}
        self.provider_models: Dict[str, List[str]] = {}
        self.startup_complete = False

    async def initialize_models_on_startup(self, 
                                         provider_registry=None,
                                         force_refresh: bool = False) -> Dict[str, Any]:
        """Initialize model discovery during system startup"""
        
        logger.info("🔍 STARTING MODEL DISCOVERY ON STARTUP")
        logger.info("=" * 60)
        
        startup_results = {
            "success": False,
            "hf_models_discovered": 0,
            "local_models_found": 0,
            "integrated_models": 0,
            "best_models_by_capability": {},
            "errors": []
        }
        
        try:
            # Step 1: Discover models from Hugging Face
            logger.info("📋 Step 1: Discovering models from Hugging Face...")
            
            self.model_snapshot = await model_discovery_service.get_model_snapshot(
                force_refresh=force_refresh
            )
            
            startup_results["hf_models_discovered"] = len(self.model_snapshot.all_models)
            logger.info(f"   ✅ Discovered {startup_results['hf_models_discovered']} HF models")
            
            # Step 2: Get local provider models
            logger.info("📋 Step 2: Scanning local provider models...")
            
            if provider_registry:
                self.provider_models = await self._scan_local_providers(provider_registry)
            else:
                logger.warning("   ⚠️ No provider registry provided, using fallback scan")
                self.provider_models = await self._fallback_provider_scan()
            
            total_local = sum(len(models) for models in self.provider_models.values())
            startup_results["local_models_found"] = total_local
            logger.info(f"   ✅ Found {total_local} local models")
            
            # Step 3: Integrate HF knowledge with local models
            logger.info("📋 Step 3: Integrating HF knowledge with local models...")
            
            self.integrated_models = await integrate_with_local_providers(
                self.model_snapshot,
                self.provider_models.get("ollama", []),
                self.provider_models.get("openrouter", [])
            )
            
            startup_results["integrated_models"] = len(self.integrated_models)
            startup_results["best_models_by_capability"] = {
                capability: {
                    "name": model.name,
                    "provider": model.provider,
                    "performance_score": model.performance_score,
                    "is_free": model.is_free
                }
                for capability, model in self.integrated_models.items()
            }
            
            logger.info(f"   ✅ Integrated {startup_results['integrated_models']} capability mappings")
            
            # Step 4: Log the results
            await self._log_startup_results()
            
            self.startup_complete = True
            startup_results["success"] = True
            
            logger.info("🎉 MODEL DISCOVERY STARTUP COMPLETE!")
            
        except Exception as e:
            error_msg = f"Model discovery startup failed: {e}"
            logger.error(f"💥 {error_msg}")
            startup_results["errors"].append(error_msg)
        
        return startup_results

    async def _scan_local_providers(self, provider_registry) -> Dict[str, List[str]]:
        """Scan local providers for available models"""
        
        provider_models = {}
        
        try:
            # Get all models from provider registry
            all_models = await provider_registry.get_all_models()
            
            for provider, models in all_models.items():
                if models:
                    provider_models[provider] = models
                    logger.info(f"   📊 {provider}: {len(models)} models")
        
        except Exception as e:
            logger.error(f"   ❌ Error scanning providers: {e}")
        
        return provider_models

    async def _fallback_provider_scan(self) -> Dict[str, List[str]]:
        """Fallback provider scan when registry not available"""
        
        provider_models = {}
        
        # Try to scan Ollama directly
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        ollama_models = [m["name"] for m in data.get("models", [])]
                        provider_models["ollama"] = ollama_models
                        logger.info(f"   📊 ollama: {len(ollama_models)} models (direct scan)")
        except Exception as e:
            logger.warning(f"   ⚠️ Could not scan Ollama directly: {e}")
        
        # Try to get OpenRouter models (would need API key for full list)
        # For now, use known free models
        provider_models["openrouter"] = [
            "deepseek/deepseek-r1",
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemma-2-9b-it:free",
            "microsoft/phi-3-mini-128k-instruct:free"
        ]
        logger.info(f"   📊 openrouter: {len(provider_models['openrouter'])} models (known free)")
        
        return provider_models

    async def _log_startup_results(self):
        """Log detailed startup results"""
        
        logger.info("")
        logger.info("📊 MODEL DISCOVERY RESULTS:")
        logger.info("=" * 40)
        
        # Log best models by capability
        for capability, model in self.integrated_models.items():
            provider_icon = "🏠" if model.provider == "ollama" else "🌐" if model.provider == "openrouter" else "🤗"
            free_icon = "🆓" if model.is_free else "💰"
            
            logger.info(f"   🎯 {capability.upper()}: {model.name}")
            logger.info(f"      {provider_icon} Provider: {model.provider}")
            logger.info(f"      {free_icon} Cost: {'Free' if model.is_free else 'Paid'}")
            logger.info(f"      ⭐ Score: {model.performance_score:.2f}")
            logger.info("")
        
        # Log provider statistics
        logger.info("📈 PROVIDER STATISTICS:")
        for provider, models in self.provider_models.items():
            logger.info(f"   {provider}: {len(models)} models available")

    def get_best_model_for_capability(self, capability: str) -> Optional[ModelInfo]:
        """Get the best model for a capability (after startup)"""
        
        if not self.startup_complete:
            logger.warning("Model discovery not complete, returning None")
            return None
        
        return self.integrated_models.get(capability)

    def get_model_config_for_agent(self, agent_type: str, 
                                 preferred_provider: Optional[str] = None) -> Dict[str, Any]:
        """Get model configuration for an agent type"""
        
        # Map agent types to capabilities
        agent_capability_map = {
            "coding": "coding",
            "reasoning": "reasoning", 
            "research": "research",
            "general": "general",
            "evolution": "reasoning",
            "search": "research",
            "basic": "general",
            "nlp": "general"
        }
        
        capability = agent_capability_map.get(agent_type, "general")
        best_model = self.get_best_model_for_capability(capability)
        
        if not best_model:
            # Fallback configuration
            return {
                "provider": "openrouter",
                "model_name": "deepseek/deepseek-r1",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        
        # Filter by preferred provider if specified
        if preferred_provider and best_model.provider != preferred_provider:
            # Try to find alternative in preferred provider
            for cap_model in self.integrated_models.values():
                if (cap_model.provider == preferred_provider and 
                    capability in cap_model.capabilities):
                    best_model = cap_model
                    break
        
        # Create configuration
        config = {
            "provider": best_model.provider,
            "model_name": best_model.name,
            "temperature": self._get_temperature_for_capability(capability),
            "max_tokens": self._get_max_tokens_for_capability(capability)
        }
        
        return config

    def _get_temperature_for_capability(self, capability: str) -> float:
        """Get appropriate temperature for capability"""
        
        temperature_map = {
            "coding": 0.1,
            "reasoning": 0.3,
            "research": 0.2,
            "general": 0.7,
            "fact_checking": 0.2
        }
        
        return temperature_map.get(capability, 0.7)

    def _get_max_tokens_for_capability(self, capability: str) -> int:
        """Get appropriate max tokens for capability"""
        
        token_map = {
            "coding": 3000,
            "reasoning": 2000,
            "research": 4000,
            "general": 1000,
            "fact_checking": 1500
        }
        
        return token_map.get(capability, 1000)

    def get_startup_summary(self) -> Dict[str, Any]:
        """Get summary of startup model discovery"""
        
        if not self.startup_complete:
            return {"status": "not_complete"}
        
        return {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "total_hf_models": len(self.model_snapshot.all_models) if self.model_snapshot else 0,
            "total_local_models": sum(len(models) for models in self.provider_models.values()),
            "capabilities_mapped": len(self.integrated_models),
            "best_models": {
                cap: {"name": model.name, "provider": model.provider}
                for cap, model in self.integrated_models.items()
            },
            "provider_stats": self.provider_models
        }

# Global instance
startup_model_manager = StartupModelManager()
