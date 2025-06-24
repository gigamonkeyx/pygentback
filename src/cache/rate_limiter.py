#!/usr/bin/env python3
"""
Redis Rate Limiting System

Production-ready rate limiting with Redis backend for PyGent Factory.
Supports multiple algorithms, user-based limits, and performance optimization.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .redis_manager import redis_manager

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    name: str
    requests_per_window: int
    window_seconds: int
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    burst_allowance: int = 0  # Additional requests allowed in burst
    key_prefix: str = "rate_limit:"


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining_requests: int
    reset_time: datetime
    retry_after_seconds: int
    current_usage: int
    limit: int


class RateLimiter:
    """Redis-based rate limiter with multiple algorithms"""
    
    def __init__(self):
        self.is_initialized = False
        self.rules: Dict[str, RateLimitRule] = {}
        
        # Performance metrics
        self.rate_limit_checks = 0
        self.rate_limit_blocks = 0
        self.rate_limit_allows = 0
        
        # Default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default rate limiting rules"""
        self.rules = {
            "api_general": RateLimitRule(
                name="api_general",
                requests_per_window=1000,
                window_seconds=3600,  # 1 hour
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            ),
            "api_auth": RateLimitRule(
                name="api_auth",
                requests_per_window=10,
                window_seconds=300,  # 5 minutes
                algorithm=RateLimitAlgorithm.FIXED_WINDOW
            ),
            "model_inference": RateLimitRule(
                name="model_inference",
                requests_per_window=100,
                window_seconds=3600,  # 1 hour
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                burst_allowance=20
            ),
            "file_upload": RateLimitRule(
                name="file_upload",
                requests_per_window=50,
                window_seconds=3600,  # 1 hour
                algorithm=RateLimitAlgorithm.LEAKY_BUCKET
            ),
            "websocket": RateLimitRule(
                name="websocket",
                requests_per_window=1000,
                window_seconds=60,  # 1 minute
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
        }
    
    async def initialize(self) -> bool:
        """Initialize rate limiter"""
        try:
            logger.info("Initializing Redis rate limiter...")
            
            if not redis_manager.is_initialized:
                await redis_manager.initialize()
            
            self.is_initialized = True
            logger.info("Redis rate limiter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {e}")
            return False
    
    async def check_rate_limit(self, rule_name: str, identifier: str, 
                             cost: int = 1) -> RateLimitResult:
        """Check if request is within rate limit"""
        try:
            self.rate_limit_checks += 1
            
            if rule_name not in self.rules:
                logger.warning(f"Unknown rate limit rule: {rule_name}")
                return RateLimitResult(
                    allowed=True,
                    remaining_requests=999999,
                    reset_time=datetime.utcnow() + timedelta(hours=1),
                    retry_after_seconds=0,
                    current_usage=0,
                    limit=999999
                )
            
            rule = self.rules[rule_name]
            
            # Choose algorithm
            if rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                result = await self._sliding_window_check(rule, identifier, cost)
            elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                result = await self._fixed_window_check(rule, identifier, cost)
            elif rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                result = await self._token_bucket_check(rule, identifier, cost)
            elif rule.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                result = await self._leaky_bucket_check(rule, identifier, cost)
            else:
                result = await self._sliding_window_check(rule, identifier, cost)
            
            if result.allowed:
                self.rate_limit_allows += 1
            else:
                self.rate_limit_blocks += 1
                logger.warning(f"Rate limit exceeded for {rule_name}:{identifier}")
            
            return result
            
        except Exception as e:
            logger.error(f"Rate limit check failed for {rule_name}:{identifier}: {e}")
            # Fail open - allow request if rate limiter fails
            return RateLimitResult(
                allowed=True,
                remaining_requests=0,
                reset_time=datetime.utcnow() + timedelta(hours=1),
                retry_after_seconds=0,
                current_usage=0,
                limit=0
            )
    
    async def _sliding_window_check(self, rule: RateLimitRule, identifier: str, 
                                  cost: int) -> RateLimitResult:
        """Sliding window rate limiting algorithm"""
        try:
            current_time = time.time()
            window_start = current_time - rule.window_seconds
            
            # Redis key for this identifier
            key = f"{rule.key_prefix}{rule.name}:{identifier}"
            
            # Use Redis pipeline for atomic operations
            pipe = redis_manager.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, rule.window_seconds + 1)
            
            results = await pipe.execute()
            current_count = results[1] + cost  # Count after adding current request
            
            # Check if within limit
            allowed = current_count <= rule.requests_per_window
            remaining = max(0, rule.requests_per_window - current_count)
            
            # Calculate reset time (end of current window)
            reset_time = datetime.fromtimestamp(current_time + rule.window_seconds)
            retry_after = rule.window_seconds if not allowed else 0
            
            return RateLimitResult(
                allowed=allowed,
                remaining_requests=remaining,
                reset_time=reset_time,
                retry_after_seconds=retry_after,
                current_usage=current_count,
                limit=rule.requests_per_window
            )
            
        except Exception as e:
            logger.error(f"Sliding window check failed: {e}")
            raise
    
    async def _fixed_window_check(self, rule: RateLimitRule, identifier: str, 
                                cost: int) -> RateLimitResult:
        """Fixed window rate limiting algorithm"""
        try:
            current_time = time.time()
            window_start = int(current_time // rule.window_seconds) * rule.window_seconds
            
            # Redis key for this window
            key = f"{rule.key_prefix}{rule.name}:{identifier}:{window_start}"
            
            # Increment counter
            current_count = await redis_manager.redis_client.incr(key)
            
            # Set expiration on first increment
            if current_count == 1:
                await redis_manager.redis_client.expire(key, rule.window_seconds)
            
            # Check if within limit
            allowed = current_count <= rule.requests_per_window
            remaining = max(0, rule.requests_per_window - current_count)
            
            # Calculate reset time (end of current window)
            window_end = window_start + rule.window_seconds
            reset_time = datetime.fromtimestamp(window_end)
            retry_after = int(window_end - current_time) if not allowed else 0
            
            return RateLimitResult(
                allowed=allowed,
                remaining_requests=remaining,
                reset_time=reset_time,
                retry_after_seconds=retry_after,
                current_usage=current_count,
                limit=rule.requests_per_window
            )
            
        except Exception as e:
            logger.error(f"Fixed window check failed: {e}")
            raise
    
    async def _token_bucket_check(self, rule: RateLimitRule, identifier: str, 
                                cost: int) -> RateLimitResult:
        """Token bucket rate limiting algorithm"""
        try:
            current_time = time.time()
            key = f"{rule.key_prefix}{rule.name}:{identifier}"
            
            # Get current bucket state
            bucket_data = await redis_manager.get_hash(key)
            
            if bucket_data:
                tokens = float(bucket_data.get("tokens", rule.requests_per_window))
                last_refill = float(bucket_data.get("last_refill", current_time))
            else:
                tokens = float(rule.requests_per_window)
                last_refill = current_time
            
            # Calculate tokens to add based on time elapsed
            time_elapsed = current_time - last_refill
            tokens_to_add = (time_elapsed / rule.window_seconds) * rule.requests_per_window
            
            # Add tokens (up to bucket capacity + burst allowance)
            max_tokens = rule.requests_per_window + rule.burst_allowance
            tokens = min(max_tokens, tokens + tokens_to_add)
            
            # Check if enough tokens for this request
            allowed = tokens >= cost
            
            if allowed:
                tokens -= cost
            
            # Update bucket state
            await redis_manager.set_hash(key, {
                "tokens": str(tokens),
                "last_refill": str(current_time)
            }, ttl=rule.window_seconds * 2)
            
            # Calculate when bucket will be full again
            time_to_full = (max_tokens - tokens) * rule.window_seconds / rule.requests_per_window
            reset_time = datetime.fromtimestamp(current_time + time_to_full)
            
            return RateLimitResult(
                allowed=allowed,
                remaining_requests=int(tokens),
                reset_time=reset_time,
                retry_after_seconds=int(time_to_full) if not allowed else 0,
                current_usage=rule.requests_per_window - int(tokens),
                limit=rule.requests_per_window
            )
            
        except Exception as e:
            logger.error(f"Token bucket check failed: {e}")
            raise
    
    async def _leaky_bucket_check(self, rule: RateLimitRule, identifier: str, 
                                cost: int) -> RateLimitResult:
        """Leaky bucket rate limiting algorithm"""
        try:
            current_time = time.time()
            key = f"{rule.key_prefix}{rule.name}:{identifier}"
            
            # Get current bucket state
            bucket_data = await redis_manager.get_hash(key)
            
            if bucket_data:
                volume = float(bucket_data.get("volume", 0))
                last_leak = float(bucket_data.get("last_leak", current_time))
            else:
                volume = 0.0
                last_leak = current_time
            
            # Calculate volume leaked since last check
            time_elapsed = current_time - last_leak
            leak_rate = rule.requests_per_window / rule.window_seconds
            volume_leaked = time_elapsed * leak_rate
            
            # Update volume (can't go below 0)
            volume = max(0, volume - volume_leaked)
            
            # Check if adding this request would overflow
            new_volume = volume + cost
            allowed = new_volume <= rule.requests_per_window
            
            if allowed:
                volume = new_volume
            
            # Update bucket state
            await redis_manager.set_hash(key, {
                "volume": str(volume),
                "last_leak": str(current_time)
            }, ttl=rule.window_seconds * 2)
            
            # Calculate when bucket will be empty
            time_to_empty = volume / leak_rate if leak_rate > 0 else 0
            reset_time = datetime.fromtimestamp(current_time + time_to_empty)
            
            return RateLimitResult(
                allowed=allowed,
                remaining_requests=max(0, rule.requests_per_window - int(volume)),
                reset_time=reset_time,
                retry_after_seconds=int(time_to_empty) if not allowed else 0,
                current_usage=int(volume),
                limit=rule.requests_per_window
            )
            
        except Exception as e:
            logger.error(f"Leaky bucket check failed: {e}")
            raise
    
    async def reset_rate_limit(self, rule_name: str, identifier: str) -> bool:
        """Reset rate limit for specific identifier"""
        try:
            if rule_name not in self.rules:
                return False
            
            rule = self.rules[rule_name]
            pattern = f"{rule.key_prefix}{rule.name}:{identifier}*"
            
            # Find and delete all keys for this identifier
            keys_to_delete = []
            async for key in redis_manager.redis_client.scan_iter(match=pattern):
                keys_to_delete.append(key)
            
            if keys_to_delete:
                await redis_manager.redis_client.delete(*keys_to_delete)
                logger.info(f"Reset rate limit for {rule_name}:{identifier}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset rate limit for {rule_name}:{identifier}: {e}")
            return False
    
    async def get_rate_limit_status(self, rule_name: str, identifier: str) -> Optional[RateLimitResult]:
        """Get current rate limit status without incrementing"""
        try:
            if rule_name not in self.rules:
                return None
            
            rule = self.rules[rule_name]
            
            # Check current status without incrementing
            if rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                current_time = time.time()
                window_start = current_time - rule.window_seconds
                key = f"{rule.key_prefix}{rule.name}:{identifier}"
                
                # Count current requests in window
                current_count = await redis_manager.redis_client.zcount(key, window_start, current_time)
                
                remaining = max(0, rule.requests_per_window - current_count)
                reset_time = datetime.fromtimestamp(current_time + rule.window_seconds)
                
                return RateLimitResult(
                    allowed=current_count < rule.requests_per_window,
                    remaining_requests=remaining,
                    reset_time=reset_time,
                    retry_after_seconds=0,
                    current_usage=current_count,
                    limit=rule.requests_per_window
                )
            
            # For other algorithms, would need similar non-incrementing checks
            return None
            
        except Exception as e:
            logger.error(f"Failed to get rate limit status for {rule_name}:{identifier}: {e}")
            return None
    
    def add_rule(self, rule: RateLimitRule):
        """Add or update a rate limiting rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added rate limit rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rate limiting rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed rate limit rule: {rule_name}")
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        total_checks = self.rate_limit_checks
        block_rate = (self.rate_limit_blocks / total_checks * 100) if total_checks > 0 else 0
        
        return {
            "total_checks": self.rate_limit_checks,
            "total_blocks": self.rate_limit_blocks,
            "total_allows": self.rate_limit_allows,
            "block_rate_percent": round(block_rate, 2),
            "rules_configured": len(self.rules),
            "rules": {name: {
                "requests_per_window": rule.requests_per_window,
                "window_seconds": rule.window_seconds,
                "algorithm": rule.algorithm.value,
                "burst_allowance": rule.burst_allowance
            } for name, rule in self.rules.items()}
        }


# Global rate limiter instance
rate_limiter = RateLimiter()
