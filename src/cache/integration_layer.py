#!/usr/bin/env python3
"""
Redis Integration Layer

Integrates Redis caching with production database and GPU optimization systems.
Provides unified caching strategy across all PyGent Factory components.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from contextlib import asynccontextmanager

from .redis_manager import redis_manager
from .cache_layers import cache_manager
from .session_manager import session_manager
from .rate_limiter import rate_limiter

# Import database and GPU systems
try:
    from ..database.production_manager import db_manager
    from ..core.gpu_optimization import gpu_optimizer
    from ..core.ollama_gpu_integration import ollama_gpu_manager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from database.production_manager import db_manager
    from core.gpu_optimization import gpu_optimizer
    from core.ollama_gpu_integration import ollama_gpu_manager

logger = logging.getLogger(__name__)


class IntegratedCacheSystem:
    """Unified cache system integrating Redis with database and GPU systems"""
    
    def __init__(self):
        self.is_initialized = False
        self.performance_metrics = {
            "cache_enabled_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "gpu_cache_hits": 0,
            "gpu_cache_misses": 0,
            "session_operations": 0,
            "rate_limit_checks": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize integrated cache system"""
        try:
            logger.info("Initializing integrated cache system...")
            
            # Initialize Redis manager
            if not redis_manager.is_initialized:
                await redis_manager.initialize()
            
            # Initialize cache layers
            if not cache_manager.is_initialized:
                await cache_manager.initialize()
            
            # Initialize session manager
            if not session_manager.is_initialized:
                await session_manager.initialize()
            
            # Initialize rate limiter
            if not rate_limiter.is_initialized:
                await rate_limiter.initialize()
            
            self.is_initialized = True
            logger.info("Integrated cache system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize integrated cache system: {e}")
            return False
    
    # Database Integration Methods
    @asynccontextmanager
    async def cached_db_session(self, cache_key: Optional[str] = None):
        """Database session with automatic caching"""
        async with db_manager.get_session() as session:
            # Wrap session with caching capabilities
            original_execute = session.execute
            
            async def cached_execute(statement, parameters=None, **kwargs):
                # Generate cache key if not provided
                if cache_key:
                    query_key = f"{cache_key}:{hash(str(statement))}"
                    
                    # Try cache first
                    cached_result = await cache_manager.get_cached_db_query(query_key)
                    if cached_result is not None:
                        self.performance_metrics["cache_hits"] += 1
                        return cached_result
                
                # Execute query
                result = await original_execute(statement, parameters, **kwargs)
                
                # Cache result if cache_key provided
                if cache_key and result:
                    await cache_manager.cache_db_query(query_key, result)
                    self.performance_metrics["cache_enabled_queries"] += 1
                
                if cache_key:
                    self.performance_metrics["cache_misses"] += 1
                
                return result
            
            # Replace execute method
            session.execute = cached_execute
            yield session
    
    async def cached_db_query(self, query_func: Callable, cache_key: str, 
                            ttl: Optional[int] = None, *args, **kwargs) -> Any:
        """Execute database query with caching"""
        try:
            # Try cache first
            cached_result = await cache_manager.get_cached_db_query(cache_key)
            if cached_result is not None:
                self.performance_metrics["cache_hits"] += 1
                return cached_result
            
            # Execute query
            result = await query_func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                await cache_manager.cache_db_query(cache_key, result, ttl)
                self.performance_metrics["cache_enabled_queries"] += 1
            
            self.performance_metrics["cache_misses"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Cached database query failed: {e}")
            # Fallback to direct execution
            return await query_func(*args, **kwargs)
    
    async def invalidate_db_cache_for_table(self, table_name: str) -> int:
        """Invalidate all cache entries for a specific database table"""
        try:
            pattern = f"*{table_name}*"
            return await cache_manager.invalidate_db_cache(pattern)
        except Exception as e:
            logger.error(f"Failed to invalidate cache for table {table_name}: {e}")
            return 0
    
    # GPU Integration Methods
    async def cached_gpu_inference(self, model_name: str, input_data: Any, 
                                 inference_func: Callable, ttl: Optional[int] = None) -> Any:
        """GPU inference with caching"""
        try:
            # Generate input hash
            input_hash = cache_manager.generate_input_hash(input_data)
            
            # Try cache first
            cached_result = await cache_manager.get_cached_model_inference(model_name, input_hash)
            if cached_result is not None:
                self.performance_metrics["gpu_cache_hits"] += 1
                logger.debug(f"GPU inference cache hit for {model_name}")
                return cached_result
            
            # Execute inference with GPU optimization
            with gpu_optimizer.optimized_inference(model_name):
                result = await inference_func(input_data)
            
            # Cache result
            if result is not None:
                await cache_manager.cache_model_inference(model_name, input_hash, result, ttl)
            
            self.performance_metrics["gpu_cache_misses"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Cached GPU inference failed for {model_name}: {e}")
            # Fallback to direct execution
            return await inference_func(input_data)
    
    async def cached_ollama_generation(self, prompt: str, model: Optional[str] = None, 
                                     **kwargs) -> Dict[str, Any]:
        """Ollama generation with caching"""
        try:
            # Generate cache key
            cache_data = {"prompt": prompt, "model": model, "kwargs": kwargs}
            input_hash = cache_manager.generate_input_hash(cache_data)
            model_name = model or "default"
            
            # Try cache first
            cached_result = await cache_manager.get_cached_model_inference(model_name, input_hash)
            if cached_result is not None:
                self.performance_metrics["gpu_cache_hits"] += 1
                return cached_result
            
            # Execute generation
            result = await ollama_gpu_manager.generate(prompt, model, **kwargs)
            
            # Cache result
            if result:
                await cache_manager.cache_model_inference(model_name, input_hash, result)
            
            self.performance_metrics["gpu_cache_misses"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Cached Ollama generation failed: {e}")
            # Fallback to direct execution
            return await ollama_gpu_manager.generate(prompt, model, **kwargs)
    
    # Session Management Integration
    async def create_user_session(self, user_id: str, ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None) -> Optional[str]:
        """Create user session with rate limiting"""
        try:
            # Check rate limit for session creation
            rate_result = await rate_limiter.check_rate_limit("api_auth", user_id)
            self.performance_metrics["rate_limit_checks"] += 1
            
            if not rate_result.allowed:
                logger.warning(f"Session creation rate limited for user {user_id}")
                return None
            
            # Create session
            session_data = await session_manager.create_session(
                user_id, ip_address, user_agent
            )
            
            if session_data:
                self.performance_metrics["session_operations"] += 1
                return session_data.session_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create user session: {e}")
            return None
    
    async def validate_session_with_cache(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session with performance caching"""
        try:
            session_data = await session_manager.get_session(session_id)
            
            if session_data:
                self.performance_metrics["session_operations"] += 1
                return {
                    "user_id": session_data.user_id,
                    "session_id": session_data.session_id,
                    "created_at": session_data.created_at.isoformat(),
                    "last_accessed": session_data.last_accessed.isoformat(),
                    "data": session_data.data
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return None
    
    # Performance Optimization Methods
    async def warm_cache(self, cache_keys: List[str]) -> Dict[str, bool]:
        """Warm up cache with frequently accessed data"""
        try:
            results = {}
            
            for cache_key in cache_keys:
                try:
                    # Check if key exists in cache
                    exists = await redis_manager.exists(cache_key)
                    results[cache_key] = exists
                    
                    if not exists:
                        logger.debug(f"Cache key {cache_key} not found for warming")
                        
                except Exception as e:
                    logger.error(f"Failed to warm cache key {cache_key}: {e}")
                    results[cache_key] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
            return {}
    
    async def optimize_cache_performance(self) -> Dict[str, Any]:
        """Optimize cache performance based on usage patterns"""
        try:
            optimization_results = {}
            
            # Get cache statistics
            cache_stats = await cache_manager.get_cache_statistics()
            optimization_results["cache_stats"] = cache_stats
            
            # Optimize based on hit rates
            for layer, stats in cache_stats.get("cache_layers", {}).items():
                hit_rate = stats.get("hit_rate_percent", 0)
                
                if hit_rate < 50:  # Low hit rate
                    # Consider increasing TTL or cache size
                    optimization_results[f"{layer}_optimization"] = "increase_ttl"
                elif hit_rate > 90:  # Very high hit rate
                    # Consider decreasing TTL to save memory
                    optimization_results[f"{layer}_optimization"] = "decrease_ttl"
                else:
                    optimization_results[f"{layer}_optimization"] = "optimal"
            
            # Memory optimization
            redis_memory = cache_stats.get("redis_memory", {})
            used_memory_mb = redis_memory.get("used_memory_mb", 0)
            
            if used_memory_mb > 400:  # Approaching 512MB limit
                # Clear least used cache layers
                await cache_manager.clear_cache_layer("perf")
                optimization_results["memory_optimization"] = "cleared_performance_cache"
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Cache performance optimization failed: {e}")
            return {"error": str(e)}
    
    # Monitoring and Health Methods
    async def get_integrated_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of integrated cache system"""
        try:
            health_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "healthy",
                "components": {}
            }
            
            # Redis health
            redis_health = await redis_manager.health_check()
            health_status["components"]["redis"] = redis_health
            
            # Cache layers health
            cache_stats = await cache_manager.get_cache_statistics()
            health_status["components"]["cache_layers"] = {
                "status": "healthy" if cache_stats else "unhealthy",
                "statistics": cache_stats
            }
            
            # Session manager health
            session_stats = session_manager.get_session_stats()
            health_status["components"]["sessions"] = {
                "status": "healthy",
                "statistics": session_stats
            }
            
            # Rate limiter health
            rate_limit_stats = rate_limiter.get_statistics()
            health_status["components"]["rate_limiter"] = {
                "status": "healthy",
                "statistics": rate_limit_stats
            }
            
            # Integration performance
            health_status["components"]["integration"] = {
                "status": "healthy",
                "metrics": self.performance_metrics
            }
            
            # Determine overall status
            component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
            if "unhealthy" in component_statuses:
                health_status["overall_status"] = "unhealthy"
            elif "degraded" in component_statuses:
                health_status["overall_status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Integrated health check failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            # Calculate cache efficiency
            total_cache_requests = self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]
            cache_hit_rate = (self.performance_metrics["cache_hits"] / total_cache_requests * 100) if total_cache_requests > 0 else 0
            
            total_gpu_requests = self.performance_metrics["gpu_cache_hits"] + self.performance_metrics["gpu_cache_misses"]
            gpu_cache_hit_rate = (self.performance_metrics["gpu_cache_hits"] / total_gpu_requests * 100) if total_gpu_requests > 0 else 0
            
            return {
                "cache_performance": {
                    "overall_hit_rate_percent": round(cache_hit_rate, 2),
                    "gpu_cache_hit_rate_percent": round(gpu_cache_hit_rate, 2),
                    "total_cache_requests": total_cache_requests,
                    "total_gpu_requests": total_gpu_requests
                },
                "integration_metrics": self.performance_metrics,
                "redis_performance": await redis_manager.health_check(),
                "recommendations": self._generate_performance_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Performance summary failed: {e}")
            return {"error": str(e)}
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Cache hit rate recommendations
        total_requests = self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]
        if total_requests > 0:
            hit_rate = self.performance_metrics["cache_hits"] / total_requests
            
            if hit_rate < 0.5:
                recommendations.append("Consider increasing cache TTL values to improve hit rates")
            elif hit_rate > 0.9:
                recommendations.append("Excellent cache performance - consider expanding cache coverage")
        
        # GPU cache recommendations
        gpu_requests = self.performance_metrics["gpu_cache_hits"] + self.performance_metrics["gpu_cache_misses"]
        if gpu_requests > 0:
            gpu_hit_rate = self.performance_metrics["gpu_cache_hits"] / gpu_requests
            
            if gpu_hit_rate < 0.3:
                recommendations.append("GPU inference caching could be improved - check input variability")
        
        # Session recommendations
        if self.performance_metrics["session_operations"] > 1000:
            recommendations.append("High session activity - consider session cleanup optimization")
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup integrated cache system"""
        try:
            self.is_initialized = False
            
            # Cleanup individual components
            await session_manager.cleanup()
            await redis_manager.cleanup()
            
            logger.info("Integrated cache system cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during integrated cache cleanup: {e}")


# Global integrated cache system instance
integrated_cache = IntegratedCacheSystem()
