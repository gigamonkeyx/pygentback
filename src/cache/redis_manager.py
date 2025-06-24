#!/usr/bin/env python3
"""
Production Redis Cache Manager

Enterprise-grade Redis caching and session management for PyGent Factory.
Provides high-performance caching, session storage, and rate limiting.
"""

import asyncio
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

logger = logging.getLogger(__name__)


class RedisManager:
    """Production Redis manager with connection pooling and error handling"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or self._get_redis_url()
        self.redis_client = None
        self.connection_pool = None
        self.is_initialized = False
        
        # Configuration
        self.pool_size = int(os.getenv("REDIS_POOL_SIZE", "10"))
        self.timeout = int(os.getenv("REDIS_TIMEOUT", "5"))
        self.retry_on_timeout = True
        self.health_check_interval = 30
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        self.cache_deletes = 0
        self.start_time = datetime.utcnow()
    
    def _get_redis_url(self) -> str:
        """Get Redis URL from environment"""
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            return redis_url
        
        # Construct from components
        host = os.getenv("REDIS_HOST", "localhost")
        port = os.getenv("REDIS_PORT", "6379")
        db = os.getenv("REDIS_DB", "0")
        password = os.getenv("REDIS_PASSWORD", "")
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"
    
    async def initialize(self) -> bool:
        """Initialize Redis connection pool"""
        try:
            logger.info("Initializing Redis cache manager...")
            
            # Create connection pool
            self.connection_pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.pool_size,
                retry_on_timeout=self.retry_on_timeout,
                socket_timeout=self.timeout,
                socket_connect_timeout=self.timeout,
                health_check_interval=self.health_check_interval
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=False  # Handle encoding manually for flexibility
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self.is_initialized = True
            logger.info("Redis cache manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis manager: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive Redis health check"""
        try:
            if not self.is_initialized:
                return {"status": "not_initialized"}
            
            start_time = datetime.utcnow()
            
            # Test basic operations
            test_key = "health_check_test"
            await self.redis_client.set(test_key, "test_value", ex=10)
            test_value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Get Redis info
            info = await self.redis_client.info()
            
            # Calculate uptime
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Calculate cache hit rate
            total_operations = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_operations * 100) if total_operations > 0 else 0
            
            return {
                "status": "healthy" if test_value == b"test_value" else "unhealthy",
                "response_time_ms": round(response_time * 1000, 2),
                "uptime_seconds": round(uptime, 2),
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory_human"),
                "cache_statistics": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "sets": self.cache_sets,
                    "deletes": self.cache_deletes,
                    "hit_rate_percent": round(hit_rate, 2)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  serialize: bool = True) -> bool:
        """Set a value in Redis cache"""
        try:
            if not self.is_initialized:
                return False
            
            # Serialize value if needed
            if serialize:
                if isinstance(value, (dict, list)):
                    serialized_value = json.dumps(value).encode('utf-8')
                else:
                    serialized_value = pickle.dumps(value)
            else:
                serialized_value = value
            
            # Set with optional TTL
            if ttl:
                await self.redis_client.setex(key, ttl, serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)
            
            self.cache_sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    async def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """Get a value from Redis cache"""
        try:
            if not self.is_initialized:
                return None
            
            value = await self.redis_client.get(key)
            
            if value is None:
                self.cache_misses += 1
                return None
            
            self.cache_hits += 1
            
            # Deserialize value if needed
            if deserialize:
                try:
                    # Try JSON first
                    return json.loads(value.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    try:
                        # Fall back to pickle
                        return pickle.loads(value)
                    except Exception:
                        # Return raw bytes if deserialization fails
                        return value
            else:
                return value
                
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            self.cache_misses += 1
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key from Redis cache"""
        try:
            if not self.is_initialized:
                return False
            
            result = await self.redis_client.delete(key)
            self.cache_deletes += 1
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis"""
        try:
            if not self.is_initialized:
                return False
            
            return await self.redis_client.exists(key) > 0
            
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for an existing key"""
        try:
            if not self.is_initialized:
                return False
            
            return await self.redis_client.expire(key, ttl)
            
        except Exception as e:
            logger.error(f"Failed to set TTL for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a numeric value"""
        try:
            if not self.is_initialized:
                return None
            
            return await self.redis_client.incrby(key, amount)
            
        except Exception as e:
            logger.error(f"Failed to increment key {key}: {e}")
            return None
    
    async def set_hash(self, key: str, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set a hash in Redis"""
        try:
            if not self.is_initialized:
                return False
            
            # Serialize hash values
            serialized_mapping = {}
            for field, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized_mapping[field] = json.dumps(value)
                else:
                    serialized_mapping[field] = str(value)
            
            await self.redis_client.hset(key, mapping=serialized_mapping)
            
            if ttl:
                await self.redis_client.expire(key, ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set hash {key}: {e}")
            return False
    
    async def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a hash from Redis"""
        try:
            if not self.is_initialized:
                return None
            
            hash_data = await self.redis_client.hgetall(key)
            
            if not hash_data:
                return None
            
            # Deserialize hash values
            result = {}
            for field, value in hash_data.items():
                field_str = field.decode('utf-8') if isinstance(field, bytes) else field
                value_str = value.decode('utf-8') if isinstance(value, bytes) else value
                
                try:
                    # Try to parse as JSON
                    result[field_str] = json.loads(value_str)
                except json.JSONDecodeError:
                    # Keep as string
                    result[field_str] = value_str
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get hash {key}: {e}")
            return None
    
    async def add_to_set(self, key: str, *values: Any) -> bool:
        """Add values to a Redis set"""
        try:
            if not self.is_initialized:
                return False

            # Serialize values
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(str(value))

            await self.redis_client.sadd(key, *serialized_values)
            return True

        except Exception as e:
            logger.error(f"Failed to add to set {key}: {e}")
            return False

    async def get_set_members(self, key: str) -> List[Any]:
        """Get all members of a Redis set"""
        try:
            if not self.is_initialized:
                return []

            members = await self.redis_client.smembers(key)

            # Deserialize members
            result = []
            for member in members:
                member_str = member.decode('utf-8') if isinstance(member, bytes) else member
                try:
                    result.append(json.loads(member_str))
                except json.JSONDecodeError:
                    result.append(member_str)

            return result

        except Exception as e:
            logger.error(f"Failed to get set members {key}: {e}")
            return []

    async def push_to_list(self, key: str, *values: Any, direction: str = "left") -> bool:
        """Push values to a Redis list"""
        try:
            if not self.is_initialized:
                return False

            # Serialize values
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(str(value))

            if direction.lower() == "left":
                await self.redis_client.lpush(key, *serialized_values)
            else:
                await self.redis_client.rpush(key, *serialized_values)

            return True

        except Exception as e:
            logger.error(f"Failed to push to list {key}: {e}")
            return False

    async def pop_from_list(self, key: str, direction: str = "left") -> Optional[Any]:
        """Pop value from a Redis list"""
        try:
            if not self.is_initialized:
                return None

            if direction.lower() == "left":
                value = await self.redis_client.lpop(key)
            else:
                value = await self.redis_client.rpop(key)

            if value is None:
                return None

            # Deserialize value
            value_str = value.decode('utf-8') if isinstance(value, bytes) else value
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                return value_str

        except Exception as e:
            logger.error(f"Failed to pop from list {key}: {e}")
            return None

    async def get_list_range(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range of values from a Redis list"""
        try:
            if not self.is_initialized:
                return []

            values = await self.redis_client.lrange(key, start, end)

            # Deserialize values
            result = []
            for value in values:
                value_str = value.decode('utf-8') if isinstance(value, bytes) else value
                try:
                    result.append(json.loads(value_str))
                except json.JSONDecodeError:
                    result.append(value_str)

            return result

        except Exception as e:
            logger.error(f"Failed to get list range {key}: {e}")
            return []

    async def publish(self, channel: str, message: Any) -> bool:
        """Publish message to Redis channel"""
        try:
            if not self.is_initialized:
                return False

            # Serialize message
            if isinstance(message, (dict, list)):
                serialized_message = json.dumps(message)
            else:
                serialized_message = str(message)

            await self.redis_client.publish(channel, serialized_message)
            return True

        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            return False

    async def subscribe(self, *channels: str):
        """Subscribe to Redis channels"""
        try:
            if not self.is_initialized:
                return None

            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(*channels)
            return pubsub

        except Exception as e:
            logger.error(f"Failed to subscribe to channels {channels}: {e}")
            return None

    async def execute_pipeline(self, commands: List[Dict[str, Any]]) -> List[Any]:
        """Execute multiple Redis commands in a pipeline"""
        try:
            if not self.is_initialized:
                return []

            pipe = self.redis_client.pipeline()

            for command in commands:
                cmd = command.get("command")
                args = command.get("args", [])
                kwargs = command.get("kwargs", {})

                if hasattr(pipe, cmd):
                    getattr(pipe, cmd)(*args, **kwargs)

            results = await pipe.execute()
            return results

        except Exception as e:
            logger.error(f"Failed to execute pipeline: {e}")
            return []

    # Alias methods for compatibility with test expectations
    async def set_data(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Alias for set method"""
        return await self.set(key, value, ttl)

    async def get_data(self, key: str) -> Optional[Any]:
        """Alias for get method"""
        return await self.get(key)

    async def delete_data(self, key: str) -> bool:
        """Alias for delete method"""
        return await self.delete(key)

    async def get_list(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Alias for get_list_range method"""
        return await self.get_list_range(key, start, end)

    async def cleanup(self):
        """Cleanup Redis connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis client closed")

            if self.connection_pool:
                await self.connection_pool.disconnect()
                logger.info("Redis connection pool closed")

            self.is_initialized = False
            logger.info("Redis manager cleanup completed")

        except Exception as e:
            logger.error(f"Error during Redis cleanup: {e}")


# Global Redis manager instance
redis_manager = RedisManager()

async def initialize_redis():
    """Initialize the global Redis manager"""
    if not redis_manager.is_initialized:
        success = await redis_manager.initialize()
        if not success:
            logger.error("Failed to initialize Redis manager")
            return False
    return True

async def ensure_redis_initialized():
    """Ensure Redis is initialized, initialize if not"""
    if not redis_manager.is_initialized:
        return await initialize_redis()
    return True
