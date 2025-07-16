#!/usr/bin/env python3
"""
Hugging Face Model Discovery Service
Dynamically discovers and categorizes the best models for each capability
"""

import asyncio
import logging
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a discovered model"""
    name: str
    provider: str  # 'ollama', 'openrouter', 'huggingface'
    capabilities: List[str]
    performance_score: float
    size_gb: Optional[float] = None
    context_length: Optional[int] = None
    is_free: bool = True
    downloads: int = 0
    likes: int = 0
    last_updated: Optional[str] = None
    description: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class ModelSnapshot:
    """Snapshot of best models by capability"""
    timestamp: str
    best_models: Dict[str, ModelInfo]  # capability -> best model
    all_models: List[ModelInfo]
    provider_stats: Dict[str, int]
    total_models: int

class HuggingFaceModelDiscovery:
    """Discovers models from Hugging Face Hub"""
    
    def __init__(self):
        self.base_url = "https://huggingface.co/api"
        self.session = None
        
        # Model categories and their HF tags
        self.capability_tags = {
            "coding": ["code-generation", "code", "programming", "python", "javascript"],
            "reasoning": ["reasoning", "logic", "math", "problem-solving"],
            "general": ["text-generation", "conversational", "chat"],
            "research": ["question-answering", "summarization", "analysis"],
            "multimodal": ["multimodal", "vision", "image-text"],
            "embedding": ["sentence-similarity", "feature-extraction", "embeddings"],
            "fast_response": ["efficient", "small", "lightweight"],
            "fact_checking": ["fact-checking", "verification", "truthfulness"]
        }
        
        # Performance indicators (higher is better)
        self.performance_weights = {
            "downloads": 0.3,
            "likes": 0.2,
            "recent_activity": 0.2,
            "model_size": 0.1,  # Smaller can be better for efficiency
            "context_length": 0.2
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def discover_models_by_capability(self, capability: str, limit: int = 20) -> List[ModelInfo]:
        """Discover models for a specific capability"""
        
        if capability not in self.capability_tags:
            logger.warning(f"Unknown capability: {capability}")
            return []
        
        tags = self.capability_tags[capability]
        models = []
        
        for tag in tags:
            try:
                tag_models = await self._search_models_by_tag(tag, limit=limit//len(tags))
                models.extend(tag_models)
            except Exception as e:
                logger.warning(f"Failed to search tag '{tag}': {e}")
        
        # Remove duplicates and sort by performance
        unique_models = {m.name: m for m in models}
        sorted_models = sorted(
            unique_models.values(), 
            key=lambda m: m.performance_score, 
            reverse=True
        )
        
        return sorted_models[:limit]

    async def _search_models_by_tag(self, tag: str, limit: int = 10) -> List[ModelInfo]:
        """Search models by a specific tag"""
        
        try:
            url = f"{self.base_url}/models"
            params = {
                "filter": tag,
                "sort": "downloads",
                "direction": -1,
                "limit": limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data:
                        model_info = await self._parse_model_data(model_data, tag)
                        if model_info:
                            models.append(model_info)
                    
                    return models
                else:
                    logger.error(f"HF API error for tag '{tag}': {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching tag '{tag}': {e}")
            return []

    async def _parse_model_data(self, model_data: Dict, primary_tag: str) -> Optional[ModelInfo]:
        """Parse model data from HF API"""
        
        try:
            name = model_data.get("id", "")
            if not name:
                return None
            
            # Calculate performance score
            downloads = model_data.get("downloads", 0)
            likes = model_data.get("likes", 0)
            
            # Estimate performance based on popularity and recency
            performance_score = (
                (downloads / 1000000) * self.performance_weights["downloads"] +
                (likes / 1000) * self.performance_weights["likes"] +
                self._calculate_recency_score(model_data.get("lastModified")) * self.performance_weights["recent_activity"]
            )
            
            # Determine capabilities from tags
            tags = model_data.get("tags", [])
            capabilities = self._determine_capabilities(tags, primary_tag)
            
            return ModelInfo(
                name=name,
                provider="huggingface",
                capabilities=capabilities,
                performance_score=performance_score,
                downloads=downloads,
                likes=likes,
                last_updated=model_data.get("lastModified"),
                description=model_data.get("description", "")[:200],
                tags=tags[:10]  # Limit tags
            )
            
        except Exception as e:
            logger.warning(f"Error parsing model data: {e}")
            return None

    def _calculate_recency_score(self, last_modified: Optional[str]) -> float:
        """Calculate recency score (0-1, higher for more recent)"""
        
        if not last_modified:
            return 0.0
        
        try:
            modified_date = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
            days_ago = (datetime.now().astimezone() - modified_date).days
            
            # Score decreases with age, 0.9 for recent, 0.1 for very old
            if days_ago <= 30:
                return 0.9
            elif days_ago <= 90:
                return 0.7
            elif days_ago <= 365:
                return 0.5
            else:
                return 0.1
                
        except Exception:
            return 0.0

    def _determine_capabilities(self, tags: List[str], primary_tag: str) -> List[str]:
        """Determine model capabilities from tags"""
        
        capabilities = []
        
        # Add primary capability
        for capability, capability_tags in self.capability_tags.items():
            if primary_tag in capability_tags:
                capabilities.append(capability)
                break
        
        # Add secondary capabilities
        for capability, capability_tags in self.capability_tags.items():
            if any(tag in tags for tag in capability_tags):
                if capability not in capabilities:
                    capabilities.append(capability)
        
        return capabilities or ["general"]

class ModelDiscoveryService:
    """Main service for model discovery and caching"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "model_snapshot.json"
        self.cache_duration = timedelta(hours=6)  # Refresh every 6 hours
        
        self.hf_discovery = HuggingFaceModelDiscovery()
        self.current_snapshot: Optional[ModelSnapshot] = None

    async def get_model_snapshot(self, force_refresh: bool = False) -> ModelSnapshot:
        """Get current model snapshot, refreshing if needed"""
        
        if not force_refresh and self._is_cache_valid():
            logger.info("Using cached model snapshot")
            return await self._load_cached_snapshot()
        
        logger.info("Creating new model snapshot from Hugging Face...")
        snapshot = await self._create_new_snapshot()
        await self._save_snapshot(snapshot)
        
        self.current_snapshot = snapshot
        return snapshot

    async def get_best_model_for_capability(self, capability: str) -> Optional[ModelInfo]:
        """Get the best model for a specific capability"""
        
        snapshot = await self.get_model_snapshot()
        return snapshot.best_models.get(capability)

    async def get_available_models_for_capability(self, capability: str, limit: int = 5) -> List[ModelInfo]:
        """Get available models for a capability"""
        
        snapshot = await self.get_model_snapshot()
        
        # Filter models by capability
        matching_models = [
            model for model in snapshot.all_models
            if capability in model.capabilities
        ]
        
        # Sort by performance score
        matching_models.sort(key=lambda m: m.performance_score, reverse=True)
        
        return matching_models[:limit]

    async def _create_new_snapshot(self) -> ModelSnapshot:
        """Create a new model snapshot"""
        
        all_models = []
        best_models = {}
        provider_stats = {"huggingface": 0}
        
        async with self.hf_discovery:
            # Discover models for each capability
            for capability in self.hf_discovery.capability_tags.keys():
                logger.info(f"Discovering models for capability: {capability}")
                
                try:
                    capability_models = await self.hf_discovery.discover_models_by_capability(
                        capability, limit=10
                    )
                    
                    if capability_models:
                        # Best model for this capability
                        best_models[capability] = capability_models[0]
                        
                        # Add to all models
                        all_models.extend(capability_models)
                        provider_stats["huggingface"] += len(capability_models)
                        
                        logger.info(f"Found {len(capability_models)} models for {capability}")
                    else:
                        logger.warning(f"No models found for capability: {capability}")
                        
                except Exception as e:
                    logger.error(f"Error discovering models for {capability}: {e}")
        
        # Remove duplicates from all_models
        unique_models = {m.name: m for m in all_models}
        all_models = list(unique_models.values())
        
        snapshot = ModelSnapshot(
            timestamp=datetime.now().isoformat(),
            best_models=best_models,
            all_models=all_models,
            provider_stats=provider_stats,
            total_models=len(all_models)
        )
        
        logger.info(f"Created model snapshot with {len(all_models)} models")
        return snapshot

    def _is_cache_valid(self) -> bool:
        """Check if cached snapshot is still valid"""
        
        if not self.cache_file.exists():
            return False
        
        try:
            cache_time = datetime.fromtimestamp(self.cache_file.stat().st_mtime)
            return datetime.now() - cache_time < self.cache_duration
        except Exception:
            return False

    async def _load_cached_snapshot(self) -> ModelSnapshot:
        """Load snapshot from cache"""
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            # Convert dict data back to ModelSnapshot
            best_models = {
                k: ModelInfo(**v) for k, v in data["best_models"].items()
            }
            all_models = [ModelInfo(**m) for m in data["all_models"]]
            
            return ModelSnapshot(
                timestamp=data["timestamp"],
                best_models=best_models,
                all_models=all_models,
                provider_stats=data["provider_stats"],
                total_models=data["total_models"]
            )
            
        except Exception as e:
            logger.error(f"Error loading cached snapshot: {e}")
            # Fall back to creating new snapshot
            return await self._create_new_snapshot()

    async def _save_snapshot(self, snapshot: ModelSnapshot) -> None:
        """Save snapshot to cache"""
        
        try:
            # Convert to serializable format
            data = {
                "timestamp": snapshot.timestamp,
                "best_models": {k: asdict(v) for k, v in snapshot.best_models.items()},
                "all_models": [asdict(m) for m in snapshot.all_models],
                "provider_stats": snapshot.provider_stats,
                "total_models": snapshot.total_models
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved model snapshot to {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

# Global instance
model_discovery_service = ModelDiscoveryService()

async def integrate_with_local_providers(snapshot: ModelSnapshot,
                                       ollama_models: List[str],
                                       openrouter_models: List[str]) -> Dict[str, ModelInfo]:
    """Integrate HF discoveries with local provider models"""

    integrated_models = {}

    # Map local models to capabilities based on HF knowledge
    for capability, hf_model in snapshot.best_models.items():

        # Check if we have a similar model locally
        local_model = None

        # Check Ollama first (prefer local)
        for ollama_model in ollama_models:
            if _models_are_similar(hf_model.name, ollama_model):
                local_model = ModelInfo(
                    name=ollama_model,
                    provider="ollama",
                    capabilities=[capability],
                    performance_score=hf_model.performance_score * 1.1,  # Prefer local
                    is_free=True,
                    description=f"Local Ollama model similar to {hf_model.name}"
                )
                break

        # Check OpenRouter if no Ollama match
        if not local_model:
            for or_model in openrouter_models:
                if _models_are_similar(hf_model.name, or_model):
                    local_model = ModelInfo(
                        name=or_model,
                        provider="openrouter",
                        capabilities=[capability],
                        performance_score=hf_model.performance_score,
                        is_free="free" in or_model.lower(),
                        description=f"OpenRouter model similar to {hf_model.name}"
                    )
                    break

        # Use HF model as fallback
        if not local_model:
            local_model = hf_model

        integrated_models[capability] = local_model

    return integrated_models

def _models_are_similar(hf_name: str, local_name: str) -> bool:
    """Check if models are similar based on name patterns"""

    # Extract key terms from model names
    hf_terms = set(hf_name.lower().replace("-", " ").replace("_", " ").split())
    local_terms = set(local_name.lower().replace("-", " ").replace("_", " ").split())

    # Common model families
    model_families = {
        "deepseek", "qwen", "llama", "phi", "gemma", "mistral", "coder", "chat"
    }

    # Check for family matches
    hf_families = hf_terms & model_families
    local_families = local_terms & model_families

    return bool(hf_families & local_families)
