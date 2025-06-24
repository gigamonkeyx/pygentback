#!/usr/bin/env python3
"""
Redis Monitoring and Management API

Comprehensive Redis monitoring, cache management, and performance analytics
API endpoints for PyGent Factory production deployment.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

try:
    from ..cache.redis_manager import redis_manager
    from ..cache.cache_layers import cache_manager
    from ..cache.session_manager import session_manager
    from ..cache.rate_limiter import rate_limiter
    from ..cache.integration_layer import integrated_cache
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from cache.redis_manager import redis_manager
    from cache.cache_layers import cache_manager
    from cache.session_manager import session_manager
    from cache.rate_limiter import rate_limiter
    from cache.integration_layer import integrated_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/redis", tags=["Redis Monitoring"])


class RedisStatusResponse(BaseModel):
    """Redis status response model"""
    redis_available: bool
    redis_version: str
    connected_clients: int
    used_memory_mb: float
    used_memory_peak_mb: float
    memory_fragmentation_ratio: float
    uptime_seconds: float
    cache_hit_rate: float
    total_commands_processed: int


class CacheOperationRequest(BaseModel):
    """Cache operation request model"""
    layer: Optional[str] = None
    key_pattern: Optional[str] = None
    ttl_seconds: Optional[int] = None
    tags: Optional[List[str]] = None


class SessionRequest(BaseModel):
    """Session management request model"""
    user_id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@router.get("/status", response_model=RedisStatusResponse)
async def get_redis_status():
    """Get comprehensive Redis status and metrics"""
    try:
        if not redis_manager.is_initialized:
            raise HTTPException(status_code=503, detail="Redis not initialized")
        
        # Get Redis health check
        health_data = await redis_manager.health_check()
        
        if health_data.get("status") != "healthy":
            raise HTTPException(status_code=503, detail="Redis unhealthy")
        
        # Get additional Redis info
        redis_info = await redis_manager.redis_client.info()
        
        # Calculate cache hit rate
        cache_stats = await cache_manager.get_cache_statistics()
        overall_hit_rate = 0.0
        
        if cache_stats.get("cache_layers"):
            total_hits = sum(layer.get("hits", 0) for layer in cache_stats["cache_layers"].values())
            total_requests = sum(layer.get("total_requests", 0) for layer in cache_stats["cache_layers"].values())
            overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return RedisStatusResponse(
            redis_available=True,
            redis_version=redis_info.get("redis_version", "unknown"),
            connected_clients=redis_info.get("connected_clients", 0),
            used_memory_mb=round(redis_info.get("used_memory", 0) / 1024 / 1024, 2),
            used_memory_peak_mb=round(redis_info.get("used_memory_peak", 0) / 1024 / 1024, 2),
            memory_fragmentation_ratio=redis_info.get("mem_fragmentation_ratio", 0),
            uptime_seconds=redis_info.get("uptime_in_seconds", 0),
            cache_hit_rate=round(overall_hit_rate, 2),
            total_commands_processed=redis_info.get("total_commands_processed", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get Redis status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Redis status: {str(e)}")


@router.get("/health")
async def redis_health_check():
    """Comprehensive Redis and cache system health check"""
    try:
        return await integrated_cache.get_integrated_health_status()
        
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "error",
            "error": str(e)
        }


@router.get("/performance")
async def get_redis_performance():
    """Get detailed Redis and cache performance metrics"""
    try:
        if not integrated_cache.is_initialized:
            raise HTTPException(status_code=503, detail="Integrated cache not initialized")
        
        return await integrated_cache.get_performance_summary()
        
    except Exception as e:
        logger.error(f"Failed to get Redis performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.get("/cache/statistics")
async def get_cache_statistics():
    """Get comprehensive cache layer statistics"""
    try:
        return await cache_manager.get_cache_statistics()
        
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}")


@router.post("/cache/clear")
async def clear_cache(request: CacheOperationRequest, background_tasks: BackgroundTasks):
    """Clear cache entries by layer or pattern"""
    try:
        if not cache_manager.is_initialized:
            raise HTTPException(status_code=503, detail="Cache manager not initialized")
        
        cleared_count = 0
        
        if request.layer:
            # Clear specific cache layer
            cleared_count = await cache_manager.clear_cache_layer(request.layer)
            message = f"Cleared {cleared_count} entries from {request.layer} cache"
        elif request.key_pattern:
            # Clear by pattern (implement pattern-based clearing)
            # This would need to be implemented in cache_manager
            message = f"Pattern-based clearing not yet implemented"
        elif request.tags:
            # Clear by tags
            cleared_count = await cache_manager.invalidate_by_tags(request.tags)
            message = f"Cleared {cleared_count} entries with tags: {request.tags}"
        else:
            raise HTTPException(status_code=400, detail="Must specify layer, pattern, or tags")
        
        return {
            "status": "success",
            "message": message,
            "cleared_count": cleared_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.post("/cache/optimize")
async def optimize_cache(background_tasks: BackgroundTasks):
    """Optimize cache memory usage and performance"""
    try:
        if not cache_manager.is_initialized:
            raise HTTPException(status_code=503, detail="Cache manager not initialized")
        
        # Run optimization in background
        background_tasks.add_task(cache_manager.optimize_cache_memory)
        
        # Also run integrated cache optimization
        optimization_results = await integrated_cache.optimize_cache_performance()
        
        return {
            "status": "success",
            "message": "Cache optimization initiated",
            "optimization_results": optimization_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to optimize cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize cache: {str(e)}")


@router.get("/cache/keys/{key}")
async def get_cache_key_info(key: str):
    """Get detailed information about a specific cache key"""
    try:
        key_info = await cache_manager.get_cache_key_info(key)
        
        if key_info is None:
            raise HTTPException(status_code=404, detail=f"Cache key '{key}' not found")
        
        return key_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cache key info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get key info: {str(e)}")


@router.post("/cache/ttl/extend")
async def extend_cache_ttl(request: CacheOperationRequest):
    """Extend TTL for cache keys matching pattern"""
    try:
        if not request.key_pattern or not request.ttl_seconds:
            raise HTTPException(status_code=400, detail="Must specify key_pattern and ttl_seconds")
        
        extended_count = await cache_manager.extend_cache_ttl(
            request.key_pattern, 
            request.ttl_seconds
        )
        
        return {
            "status": "success",
            "message": f"Extended TTL for {extended_count} keys",
            "extended_count": extended_count,
            "pattern": request.key_pattern,
            "additional_seconds": request.ttl_seconds,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extend cache TTL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extend TTL: {str(e)}")


@router.get("/sessions/statistics")
async def get_session_statistics():
    """Get session management statistics"""
    try:
        return session_manager.get_session_stats()
        
    except Exception as e:
        logger.error(f"Failed to get session statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session statistics: {str(e)}")


@router.post("/sessions/create")
async def create_session(request: SessionRequest):
    """Create a new user session"""
    try:
        session_id = await integrated_cache.create_user_session(
            request.user_id,
            request.ip_address,
            request.user_agent
        )
        
        if session_id:
            return {
                "status": "success",
                "session_id": session_id,
                "user_id": request.user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=429, detail="Session creation rate limited")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    try:
        session_data = await integrated_cache.validate_session_with_cache(session_id)
        
        if session_data:
            return session_data
        else:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    try:
        success = await session_manager.delete_session(session_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Session {session_id} deleted",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.get("/rate-limits/statistics")
async def get_rate_limit_statistics():
    """Get rate limiting statistics"""
    try:
        return rate_limiter.get_statistics()
        
    except Exception as e:
        logger.error(f"Failed to get rate limit statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get rate limit statistics: {str(e)}")


@router.get("/rate-limits/{rule_name}/{identifier}")
async def get_rate_limit_status(rule_name: str, identifier: str):
    """Get current rate limit status for identifier"""
    try:
        status = await rate_limiter.get_rate_limit_status(rule_name, identifier)
        
        if status:
            return {
                "rule_name": rule_name,
                "identifier": identifier,
                "allowed": status.allowed,
                "remaining_requests": status.remaining_requests,
                "reset_time": status.reset_time.isoformat(),
                "current_usage": status.current_usage,
                "limit": status.limit
            }
        else:
            raise HTTPException(status_code=404, detail=f"Rate limit rule '{rule_name}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get rate limit status: {str(e)}")


@router.post("/rate-limits/{rule_name}/{identifier}/reset")
async def reset_rate_limit(rule_name: str, identifier: str):
    """Reset rate limit for specific identifier"""
    try:
        success = await rate_limiter.reset_rate_limit(rule_name, identifier)
        
        if success:
            return {
                "status": "success",
                "message": f"Rate limit reset for {rule_name}:{identifier}",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Rate limit rule '{rule_name}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset rate limit: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset rate limit: {str(e)}")


@router.get("/memory/usage")
async def get_memory_usage():
    """Get detailed Redis memory usage information"""
    try:
        redis_info = await redis_manager.redis_client.info("memory")
        
        return {
            "used_memory_mb": round(redis_info.get("used_memory", 0) / 1024 / 1024, 2),
            "used_memory_peak_mb": round(redis_info.get("used_memory_peak", 0) / 1024 / 1024, 2),
            "used_memory_rss_mb": round(redis_info.get("used_memory_rss", 0) / 1024 / 1024, 2),
            "used_memory_dataset_mb": round(redis_info.get("used_memory_dataset", 0) / 1024 / 1024, 2),
            "memory_fragmentation_ratio": redis_info.get("mem_fragmentation_ratio", 0),
            "memory_efficiency_percent": round((1 - redis_info.get("mem_fragmentation_ratio", 1)) * 100, 2),
            "total_system_memory_mb": round(redis_info.get("total_system_memory", 0) / 1024 / 1024, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory usage: {str(e)}")


@router.post("/benchmark")
async def run_redis_benchmark():
    """Run Redis performance benchmark"""
    try:
        if not redis_manager.is_initialized:
            raise HTTPException(status_code=503, detail="Redis not initialized")
        
        benchmark_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "tests": {}
        }
        
        # Test 1: SET performance
        start_time = datetime.utcnow()
        for i in range(100):
            await redis_manager.set(f"benchmark_set_{i}", f"value_{i}", ttl=60)
        set_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Test 2: GET performance
        start_time = datetime.utcnow()
        for i in range(100):
            await redis_manager.get(f"benchmark_set_{i}")
        get_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Test 3: DELETE performance
        start_time = datetime.utcnow()
        for i in range(100):
            await redis_manager.delete(f"benchmark_set_{i}")
        delete_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        benchmark_results["tests"] = {
            "set_100_keys_ms": round(set_time, 2),
            "get_100_keys_ms": round(get_time, 2),
            "delete_100_keys_ms": round(delete_time, 2),
            "total_time_ms": round(set_time + get_time + delete_time, 2),
            "operations_per_second": round(300 / ((set_time + get_time + delete_time) / 1000), 2)
        }
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Redis benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.post("/initialize")
async def initialize_redis_system():
    """Initialize the complete Redis caching system"""
    try:
        success = await integrated_cache.initialize()
        
        if success:
            return {
                "status": "success",
                "message": "Redis caching system initialized successfully",
                "components": {
                    "redis_manager": redis_manager.is_initialized,
                    "cache_manager": cache_manager.is_initialized,
                    "session_manager": session_manager.is_initialized,
                    "rate_limiter": rate_limiter.is_initialized,
                    "integrated_cache": integrated_cache.is_initialized
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize Redis system")
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis system: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.post("/shutdown")
async def shutdown_redis_system():
    """Gracefully shutdown Redis caching system"""
    try:
        await integrated_cache.cleanup()
        
        return {
            "status": "success",
            "message": "Redis caching system shutdown completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to shutdown Redis system: {e}")
        raise HTTPException(status_code=500, detail=f"Shutdown failed: {str(e)}")
