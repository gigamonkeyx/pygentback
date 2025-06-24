"""
Real Memory/Cache Client

Production-grade memory and caching system replacing all mock memory operations.
Supports both Redis and in-memory implementations with persistence.
"""

import asyncio
import logging
import json
import pickle
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import threading

# Handle redis.asyncio import gracefully for compatibility
try:
    import redis.asyncio as redis
    REDIS_ASYNC_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_ASYNC_AVAILABLE = False

logger = logging.getLogger(__name__)


class RealMemoryClient:
    """
    Production memory/cache client with real storage operations.
    
    Features:
    - Redis integration for distributed caching
    - In-memory fallback for development
    - TTL support for automatic expiration
    - Serialization for complex objects
    - Atomic operations and transactions
    """
    
    def __init__(self, redis_url: str = None, use_redis: bool = True):
        self.redis_url = redis_url or "redis://localhost:6379"
        self.use_redis = use_redis and REDIS_ASYNC_AVAILABLE
        self.redis_client: Optional[Any] = None
        self.local_cache: Dict[str, Any] = {}
        self.local_cache_ttl: Dict[str, datetime] = {}
        self.is_connected = False
        self._lock = threading.RLock()
        
    async def connect(self) -> bool:
        """Establish real memory/cache connection."""
        try:
            if self.use_redis and REDIS_ASYNC_AVAILABLE:
                # Try Redis connection
                try:
                    self.redis_client = redis.from_url(
                        self.redis_url,
                        encoding="utf-8",
                        decode_responses=True,
                        socket_timeout=5,
                        socket_connect_timeout=5
                    )
                    
                    # Test Redis connection
                    await self.redis_client.ping()
                    logger.info(f"Connected to Redis: {self.redis_url}")
                    self.is_connected = True
                    return True
                    
                except Exception as e:
                    logger.warning(f"Redis connection failed: {e}, falling back to in-memory cache")
                    self.use_redis = False
            
            # Use in-memory cache
            self.local_cache = {}
            self.local_cache_ttl = {}
            logger.info("Using in-memory cache (development mode)")
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Memory client connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Close memory/cache connections."""
        if self.redis_client:
            await self.redis_client.close()
        
        self.local_cache.clear()
        self.local_cache_ttl.clear()
        self.is_connected = False
        logger.info("Memory client disconnected")
    
    async def store(self, key: str, value: Any, ttl: int = None) -> bool:
        """Store value with optional TTL."""
        if not self.is_connected:
            raise ConnectionError("Memory client not connected")
        
        try:
            if self.use_redis and self.redis_client:
                # Redis storage
                serialized_value = json.dumps(value) if not isinstance(value, str) else value
                
                if ttl:
                    await self.redis_client.setex(key, ttl, serialized_value)
                else:
                    await self.redis_client.set(key, serialized_value)
                
            else:
                # Local cache storage
                with self._lock:
                    self.local_cache[key] = value
                    if ttl:
                        self.local_cache_ttl[key] = datetime.utcnow() + timedelta(seconds=ttl)
                    elif key in self.local_cache_ttl:
                        del self.local_cache_ttl[key]
            
            logger.debug(f"Stored key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store key {key}: {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value by key."""
        if not self.is_connected:
            raise ConnectionError("Memory client not connected")
        
        try:
            if self.use_redis and self.redis_client:
                # Redis retrieval
                value = await self.redis_client.get(key)
                if value is None:
                    return None
                
                # Try to deserialize JSON
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
                
            else:
                # Local cache retrieval
                with self._lock:
                    # Check TTL
                    if key in self.local_cache_ttl:
                        if datetime.utcnow() > self.local_cache_ttl[key]:
                            # Expired
                            del self.local_cache[key]
                            del self.local_cache_ttl[key]
                            return None
                    
                    return self.local_cache.get(key)
                    
        except Exception as e:
            logger.error(f"Failed to retrieve key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
        if not self.is_connected:
            raise ConnectionError("Memory client not connected")
        
        try:
            if self.use_redis and self.redis_client:
                # Redis deletion
                result = await self.redis_client.delete(key)
                return result > 0
                
            else:
                # Local cache deletion
                with self._lock:
                    if key in self.local_cache:
                        del self.local_cache[key]
                    if key in self.local_cache_ttl:
                        del self.local_cache_ttl[key]
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.is_connected:
            raise ConnectionError("Memory client not connected")
        
        try:
            if self.use_redis and self.redis_client:
                return await self.redis_client.exists(key) > 0
            else:
                with self._lock:
                    # Check TTL
                    if key in self.local_cache_ttl:
                        if datetime.utcnow() > self.local_cache_ttl[key]:
                            # Expired
                            if key in self.local_cache:
                                del self.local_cache[key]
                            del self.local_cache_ttl[key]
                            return False
                    
                    return key in self.local_cache
                    
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False
    
    async def get_keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        if not self.is_connected:
            raise ConnectionError("Memory client not connected")
        
        try:
            if self.use_redis and self.redis_client:
                return await self.redis_client.keys(pattern)
            else:
                with self._lock:
                    # Clean expired keys first
                    expired_keys = []
                    for key, expiry in self.local_cache_ttl.items():
                        if datetime.utcnow() > expiry:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        if key in self.local_cache:
                            del self.local_cache[key]
                        del self.local_cache_ttl[key]
                    
                    # Return keys (simple pattern matching)
                    if pattern == "*":
                        return list(self.local_cache.keys())
                    else:
                        # Basic pattern matching
                        import fnmatch
                        return [key for key in self.local_cache.keys() if fnmatch.fnmatch(key, pattern)]
                        
        except Exception as e:
            logger.error(f"Failed to get keys with pattern {pattern}: {e}")
            return []
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric value."""
        if not self.is_connected:
            raise ConnectionError("Memory client not connected")
        
        try:
            if self.use_redis and self.redis_client:
                return await self.redis_client.incrby(key, amount)
            else:
                with self._lock:
                    current = self.local_cache.get(key, 0)
                    if isinstance(current, (int, float)):
                        new_value = int(current) + amount
                        self.local_cache[key] = new_value
                        return new_value
                    else:
                        # Initialize as number
                        self.local_cache[key] = amount
                        return amount
                        
        except Exception as e:
            logger.error(f"Failed to increment key {key}: {e}")
            return None
    
    async def store_agent_state(self, agent_id: str, state_data: Dict[str, Any]) -> bool:
        """Store agent state information."""
        key = f"agent_state:{agent_id}"
        return await self.store(key, state_data, ttl=3600)  # 1 hour TTL
    
    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve agent state information."""
        key = f"agent_state:{agent_id}"
        return await self.retrieve(key)
    
    async def store_task_cache(self, task_id: str, cache_data: Dict[str, Any], ttl: int = 1800) -> bool:
        """Store task-related cache data."""
        key = f"task_cache:{task_id}"
        return await self.store(key, cache_data, ttl=ttl)
    
    async def get_task_cache(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task cache data."""
        key = f"task_cache:{task_id}"
        return await self.retrieve(key)
    
    async def store_system_metrics(self, metrics: Dict[str, float]) -> bool:
        """Store system metrics with timestamp."""
        timestamp = datetime.utcnow().isoformat()
        key = f"metrics:{timestamp}"
        return await self.store(key, metrics, ttl=86400)  # 24 hours TTL
    
    async def get_recent_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get recent system metrics."""
        try:
            keys = await self.get_keys("metrics:*")
            
            # Sort by timestamp and get recent ones
            recent_keys = []
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            for key in keys:
                try:
                    timestamp_str = key.split(":", 1)[1]
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp >= cutoff_time:
                        recent_keys.append((timestamp, key))
                except (ValueError, IndexError):
                    continue
            
            # Sort by timestamp
            recent_keys.sort(key=lambda x: x[0])
            
            # Retrieve metrics
            metrics_list = []
            for timestamp, key in recent_keys:
                metrics = await self.retrieve(key)
                if metrics:
                    metrics["timestamp"] = timestamp.isoformat()
                    metrics_list.append(metrics)
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Failed to get recent metrics: {e}")
            return []


class MemoryIntegrationAdapter:
    """
    Adapter to integrate real memory client with existing orchestration system.
    """
    
    def __init__(self, memory_client: RealMemoryClient):
        self.memory_client = memory_client
    
    async def execute_memory_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute real memory request (replaces mock implementation)."""
        try:
            operation = request.get("operation", "")
            
            if operation == "store":
                key = request.get("key", "")
                value = request.get("value", "")
                ttl = request.get("ttl")
                
                success = await self.memory_client.store(key, value, ttl)
                
                return {
                    "status": "success" if success else "error",
                    "message": f"Stored: {key}" if success else f"Failed to store: {key}",
                    "key": key
                }
            
            elif operation == "retrieve":
                key = request.get("key", "")
                
                value = await self.memory_client.retrieve(key)
                
                return {
                    "status": "success" if value is not None else "error",
                    "key": key,
                    "value": value,
                    "found": value is not None
                }
            
            elif operation == "delete":
                key = request.get("key", "")
                
                success = await self.memory_client.delete(key)
                
                return {
                    "status": "success" if success else "error",
                    "message": f"Deleted: {key}" if success else f"Failed to delete: {key}",
                    "key": key
                }
            
            elif operation == "exists":
                key = request.get("key", "")
                
                exists = await self.memory_client.exists(key)
                
                return {
                    "status": "success",
                    "key": key,
                    "exists": exists
                }
            
            elif operation == "get_keys":
                pattern = request.get("pattern", "*")
                
                keys = await self.memory_client.get_keys(pattern)
                
                return {
                    "status": "success",
                    "keys": keys,
                    "count": len(keys)
                }
            
            elif operation == "increment":
                key = request.get("key", "")
                amount = request.get("amount", 1)
                
                new_value = await self.memory_client.increment(key, amount)
                
                return {
                    "status": "success" if new_value is not None else "error",
                    "key": key,
                    "new_value": new_value
                }
            
            else:
                return {
                    "status": "error",
                    "error": f"Unknown memory operation: {operation}"
                }
                
        except Exception as e:
            logger.error(f"Memory request execution failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


# Integration Configuration
MEMORY_CONFIG = {
    "redis_url": "redis://localhost:6379",
    "use_redis": True,  # Set to False for development/testing
    "fallback_to_local": True
}


async def create_real_memory_client() -> RealMemoryClient:
    """Factory function to create real memory client."""
    client = RealMemoryClient(
        MEMORY_CONFIG["redis_url"],
        MEMORY_CONFIG["use_redis"]
    )
    
    success = await client.connect()
    if not success:
        raise ConnectionError("Failed to connect to memory/cache system")
    
    return client