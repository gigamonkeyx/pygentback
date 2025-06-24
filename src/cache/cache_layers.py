#!/usr/bin/env python3
"""
Multi-Layer Caching System

Comprehensive caching layers for database queries, API responses, and model inference
results with intelligent cache invalidation and performance optimization.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from functools import wraps

from .redis_manager import redis_manager

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration for different layers"""
    # Database cache settings
    db_cache_ttl_seconds: int = 300  # 5 minutes
    db_cache_prefix: str = "db:"
    
    # API cache settings
    api_cache_ttl_seconds: int = 60  # 1 minute
    api_cache_prefix: str = "api:"
    
    # Model inference cache settings
    model_cache_ttl_seconds: int = 3600  # 1 hour
    model_cache_prefix: str = "model:"
    
    # Performance cache settings
    perf_cache_ttl_seconds: int = 30  # 30 seconds
    perf_cache_prefix: str = "perf:"
    
    # Cache invalidation settings
    enable_cache_invalidation: bool = True
    max_cache_size_mb: int = 512  # 512MB max cache size
    
    # Compression settings
    enable_compression: bool = True
    compression_threshold_bytes: int = 1024  # Compress if > 1KB


class CacheManager:
    """Multi-layer cache manager with Redis backend"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.is_initialized = False
        
        # Performance metrics
        self.cache_stats = {
            "db": {"hits": 0, "misses": 0, "sets": 0, "invalidations": 0},
            "api": {"hits": 0, "misses": 0, "sets": 0, "invalidations": 0},
            "model": {"hits": 0, "misses": 0, "sets": 0, "invalidations": 0},
            "perf": {"hits": 0, "misses": 0, "sets": 0, "invalidations": 0}
        }
        
        # Cache invalidation tracking
        self.invalidation_patterns = {}
        self.cache_dependencies = {}
    
    async def initialize(self) -> bool:
        """Initialize cache manager"""
        try:
            logger.info("Initializing multi-layer cache manager...")
            
            if not redis_manager.is_initialized:
                await redis_manager.initialize()
            
            self.is_initialized = True
            logger.info("Multi-layer cache manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            return False
    
    # Database Caching Layer
    async def cache_db_query(self, query_key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Cache database query result"""
        try:
            cache_key = f"{self.config.db_cache_prefix}{query_key}"
            ttl_seconds = ttl or self.config.db_cache_ttl_seconds
            
            success = await redis_manager.set(cache_key, data, ttl=ttl_seconds)
            
            if success:
                self.cache_stats["db"]["sets"] += 1
                logger.debug(f"Database query cached: {query_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache database query {query_key}: {e}")
            return False
    
    async def get_cached_db_query(self, query_key: str) -> Optional[Any]:
        """Get cached database query result"""
        try:
            cache_key = f"{self.config.db_cache_prefix}{query_key}"
            data = await redis_manager.get(cache_key)
            
            if data is not None:
                self.cache_stats["db"]["hits"] += 1
                logger.debug(f"Database cache hit: {query_key}")
                return data
            else:
                self.cache_stats["db"]["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached database query {query_key}: {e}")
            self.cache_stats["db"]["misses"] += 1
            return None
    
    async def invalidate_db_cache(self, pattern: str) -> int:
        """Invalidate database cache entries matching pattern"""
        try:
            # Get all keys matching pattern
            search_pattern = f"{self.config.db_cache_prefix}{pattern}"
            
            # Use Redis SCAN to find matching keys
            keys_to_delete = []
            async for key in redis_manager.redis_client.scan_iter(match=search_pattern):
                keys_to_delete.append(key)
            
            # Delete matching keys
            if keys_to_delete:
                deleted_count = await redis_manager.redis_client.delete(*keys_to_delete)
                self.cache_stats["db"]["invalidations"] += deleted_count
                logger.info(f"Invalidated {deleted_count} database cache entries")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to invalidate database cache pattern {pattern}: {e}")
            return 0
    
    # API Response Caching Layer
    async def cache_api_response(self, endpoint: str, params: Dict[str, Any], 
                                response_data: Any, ttl: Optional[int] = None) -> bool:
        """Cache API response"""
        try:
            # Create cache key from endpoint and parameters
            cache_key = self._generate_api_cache_key(endpoint, params)
            ttl_seconds = ttl or self.config.api_cache_ttl_seconds
            
            success = await redis_manager.set(cache_key, response_data, ttl=ttl_seconds)
            
            if success:
                self.cache_stats["api"]["sets"] += 1
                logger.debug(f"API response cached: {endpoint}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache API response {endpoint}: {e}")
            return False
    
    async def get_cached_api_response(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached API response"""
        try:
            cache_key = self._generate_api_cache_key(endpoint, params)
            data = await redis_manager.get(cache_key)
            
            if data is not None:
                self.cache_stats["api"]["hits"] += 1
                logger.debug(f"API cache hit: {endpoint}")
                return data
            else:
                self.cache_stats["api"]["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached API response {endpoint}: {e}")
            self.cache_stats["api"]["misses"] += 1
            return None
    
    def _generate_api_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for API endpoint and parameters"""
        # Sort parameters for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"{self.config.api_cache_prefix}{endpoint}:{param_hash}"
    
    # Model Inference Caching Layer
    async def cache_model_inference(self, model_name: str, input_hash: str, 
                                  result: Any, ttl: Optional[int] = None) -> bool:
        """Cache model inference result"""
        try:
            cache_key = f"{self.config.model_cache_prefix}{model_name}:{input_hash}"
            ttl_seconds = ttl or self.config.model_cache_ttl_seconds
            
            success = await redis_manager.set(cache_key, result, ttl=ttl_seconds)
            
            if success:
                self.cache_stats["model"]["sets"] += 1
                logger.debug(f"Model inference cached: {model_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache model inference {model_name}: {e}")
            return False
    
    async def get_cached_model_inference(self, model_name: str, input_hash: str) -> Optional[Any]:
        """Get cached model inference result"""
        try:
            cache_key = f"{self.config.model_cache_prefix}{model_name}:{input_hash}"
            data = await redis_manager.get(cache_key)
            
            if data is not None:
                self.cache_stats["model"]["hits"] += 1
                logger.debug(f"Model inference cache hit: {model_name}")
                return data
            else:
                self.cache_stats["model"]["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached model inference {model_name}: {e}")
            self.cache_stats["model"]["misses"] += 1
            return None
    
    def generate_input_hash(self, input_data: Any) -> str:
        """Generate hash for model input data"""
        try:
            # Convert input to JSON string and hash
            if isinstance(input_data, (dict, list)):
                input_str = json.dumps(input_data, sort_keys=True)
            else:
                input_str = str(input_data)
            
            return hashlib.sha256(input_str.encode()).hexdigest()[:16]  # First 16 chars
            
        except Exception as e:
            logger.error(f"Failed to generate input hash: {e}")
            return str(hash(str(input_data)))[:16]
    
    # Performance Metrics Caching
    async def cache_performance_metric(self, metric_name: str, value: Any, 
                                     ttl: Optional[int] = None) -> bool:
        """Cache performance metric"""
        try:
            cache_key = f"{self.config.perf_cache_prefix}{metric_name}"
            ttl_seconds = ttl or self.config.perf_cache_ttl_seconds
            
            success = await redis_manager.set(cache_key, value, ttl=ttl_seconds)
            
            if success:
                self.cache_stats["perf"]["sets"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache performance metric {metric_name}: {e}")
            return False
    
    async def get_cached_performance_metric(self, metric_name: str) -> Optional[Any]:
        """Get cached performance metric"""
        try:
            cache_key = f"{self.config.perf_cache_prefix}{metric_name}"
            data = await redis_manager.get(cache_key)
            
            if data is not None:
                self.cache_stats["perf"]["hits"] += 1
                return data
            else:
                self.cache_stats["perf"]["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached performance metric {metric_name}: {e}")
            self.cache_stats["perf"]["misses"] += 1
            return None
    
    # Cache Decorators
    def cache_db_result(self, key_func: Callable = None, ttl: int = None):
        """Decorator for caching database query results"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                cached_result = await self.get_cached_db_query(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                if result is not None:
                    await self.cache_db_query(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def cache_api_result(self, ttl: int = None):
        """Decorator for caching API responses"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                endpoint = func.__name__
                params = {"args": args, "kwargs": kwargs}
                
                # Try to get from cache
                cached_result = await self.get_cached_api_response(endpoint, params)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                if result is not None:
                    await self.cache_api_response(endpoint, params, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def cache_model_result(self, model_name: str, ttl: int = None):
        """Decorator for caching model inference results"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate input hash
                input_data = {"args": args, "kwargs": kwargs}
                input_hash = self.generate_input_hash(input_data)
                
                # Try to get from cache
                cached_result = await self.get_cached_model_inference(model_name, input_hash)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                if result is not None:
                    await self.cache_model_inference(model_name, input_hash, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    # Cache Management
    async def clear_cache_layer(self, layer: str) -> int:
        """Clear entire cache layer"""
        try:
            prefix_map = {
                "db": self.config.db_cache_prefix,
                "api": self.config.api_cache_prefix,
                "model": self.config.model_cache_prefix,
                "perf": self.config.perf_cache_prefix
            }
            
            if layer not in prefix_map:
                return 0
            
            prefix = prefix_map[layer]
            pattern = f"{prefix}*"
            
            # Find and delete all keys with prefix
            keys_to_delete = []
            async for key in redis_manager.redis_client.scan_iter(match=pattern):
                keys_to_delete.append(key)
            
            if keys_to_delete:
                deleted_count = await redis_manager.redis_client.delete(*keys_to_delete)
                logger.info(f"Cleared {deleted_count} entries from {layer} cache")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to clear {layer} cache: {e}")
            return 0
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            # Calculate hit rates
            stats = {}
            for layer, layer_stats in self.cache_stats.items():
                total_requests = layer_stats["hits"] + layer_stats["misses"]
                hit_rate = (layer_stats["hits"] / total_requests * 100) if total_requests > 0 else 0

                stats[layer] = {
                    **layer_stats,
                    "hit_rate_percent": round(hit_rate, 2),
                    "total_requests": total_requests
                }

            # Get Redis memory info
            redis_info = await redis_manager.redis_client.info("memory")

            return {
                "cache_layers": stats,
                "redis_memory": {
                    "used_memory_mb": round(redis_info.get("used_memory", 0) / 1024 / 1024, 2),
                    "used_memory_peak_mb": round(redis_info.get("used_memory_peak", 0) / 1024 / 1024, 2),
                    "memory_fragmentation_ratio": redis_info.get("mem_fragmentation_ratio", 0)
                },
                "config": {
                    "db_cache_ttl_seconds": self.config.db_cache_ttl_seconds,
                    "api_cache_ttl_seconds": self.config.api_cache_ttl_seconds,
                    "model_cache_ttl_seconds": self.config.model_cache_ttl_seconds,
                    "max_cache_size_mb": self.config.max_cache_size_mb
                }
            }

        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {"error": str(e)}

    # Cache Invalidation and TTL Management
    async def setup_cache_invalidation_patterns(self, patterns: Dict[str, List[str]]):
        """Setup cache invalidation patterns for automatic cleanup"""
        try:
            self.invalidation_patterns = patterns

            # Store patterns in Redis for persistence
            await redis_manager.set("cache_invalidation_patterns", patterns, ttl=86400)  # 24 hours

            logger.info(f"Setup {len(patterns)} cache invalidation patterns")

        except Exception as e:
            logger.error(f"Failed to setup cache invalidation patterns: {e}")

    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags"""
        try:
            total_invalidated = 0

            for tag in tags:
                # Find all cache keys with this tag
                for prefix in [self.config.db_cache_prefix, self.config.api_cache_prefix,
                              self.config.model_cache_prefix]:
                    pattern = f"{prefix}*{tag}*"

                    keys_to_delete = []
                    async for key in redis_manager.redis_client.scan_iter(match=pattern):
                        keys_to_delete.append(key)

                    if keys_to_delete:
                        deleted_count = await redis_manager.redis_client.delete(*keys_to_delete)
                        total_invalidated += deleted_count

            logger.info(f"Invalidated {total_invalidated} cache entries by tags: {tags}")
            return total_invalidated

        except Exception as e:
            logger.error(f"Failed to invalidate cache by tags {tags}: {e}")
            return 0

    async def extend_cache_ttl(self, key_pattern: str, additional_seconds: int) -> int:
        """Extend TTL for cache keys matching pattern"""
        try:
            extended_count = 0

            # Find matching keys
            async for key in redis_manager.redis_client.scan_iter(match=key_pattern):
                # Get current TTL
                current_ttl = await redis_manager.redis_client.ttl(key)

                if current_ttl > 0:  # Key has TTL set
                    new_ttl = current_ttl + additional_seconds
                    await redis_manager.redis_client.expire(key, new_ttl)
                    extended_count += 1

            logger.info(f"Extended TTL for {extended_count} cache keys matching {key_pattern}")
            return extended_count

        except Exception as e:
            logger.error(f"Failed to extend cache TTL for pattern {key_pattern}: {e}")
            return 0

    async def get_cache_key_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a cache key"""
        try:
            # Check if key exists
            exists = await redis_manager.redis_client.exists(key)
            if not exists:
                return None

            # Get TTL
            ttl = await redis_manager.redis_client.ttl(key)

            # Get memory usage
            memory_usage = await redis_manager.redis_client.memory_usage(key)

            # Get type
            key_type = await redis_manager.redis_client.type(key)

            return {
                "key": key,
                "exists": True,
                "ttl_seconds": ttl,
                "memory_usage_bytes": memory_usage,
                "type": key_type.decode() if isinstance(key_type, bytes) else key_type,
                "expires_at": datetime.utcnow() + timedelta(seconds=ttl) if ttl > 0 else None
            }

        except Exception as e:
            logger.error(f"Failed to get cache key info for {key}: {e}")
            return None

    async def optimize_cache_memory(self) -> Dict[str, Any]:
        """Optimize cache memory usage"""
        try:
            optimization_results = {
                "actions_taken": [],
                "memory_freed_mb": 0,
                "keys_removed": 0
            }

            # Get memory info
            redis_info = await redis_manager.redis_client.info("memory")
            used_memory_mb = redis_info.get("used_memory", 0) / 1024 / 1024

            if used_memory_mb > self.config.max_cache_size_mb * 0.9:  # 90% of max
                # Remove expired keys
                expired_removed = await self._remove_expired_keys()
                optimization_results["keys_removed"] += expired_removed
                optimization_results["actions_taken"].append(f"Removed {expired_removed} expired keys")

                # Clear least important cache layer (performance cache)
                perf_cleared = await self.clear_cache_layer("perf")
                optimization_results["keys_removed"] += perf_cleared
                optimization_results["actions_taken"].append(f"Cleared {perf_cleared} performance cache entries")

                # Get new memory usage
                new_redis_info = await redis_manager.redis_client.info("memory")
                new_used_memory_mb = new_redis_info.get("used_memory", 0) / 1024 / 1024
                optimization_results["memory_freed_mb"] = round(used_memory_mb - new_used_memory_mb, 2)

            return optimization_results

        except Exception as e:
            logger.error(f"Cache memory optimization failed: {e}")
            return {"error": str(e)}

    async def _remove_expired_keys(self) -> int:
        """Remove expired keys (Redis handles this automatically, but we can force it)"""
        try:
            # This is mainly for monitoring - Redis handles expiration automatically
            # We can scan for keys with very low TTL and remove them proactively
            removed_count = 0

            for prefix in [self.config.db_cache_prefix, self.config.api_cache_prefix,
                          self.config.model_cache_prefix, self.config.perf_cache_prefix]:
                pattern = f"{prefix}*"

                async for key in redis_manager.redis_client.scan_iter(match=pattern):
                    ttl = await redis_manager.redis_client.ttl(key)

                    # Remove keys expiring in next 10 seconds
                    if 0 < ttl <= 10:
                        await redis_manager.redis_client.delete(key)
                        removed_count += 1

            return removed_count

        except Exception as e:
            logger.error(f"Failed to remove expired keys: {e}")
            return 0


# Global cache manager instance
cache_manager = CacheManager()
