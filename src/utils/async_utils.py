"""
Async Utilities

This module provides utilities for async programming including batching,
retry mechanisms, timeouts, rate limiting, and circuit breakers.
"""

import asyncio
import time
import random
from typing import List, Any, Callable, Optional, Union, TypeVar, Awaitable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Configuration for retry mechanism"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, fixed


class AsyncBatch:
    """Utility for batching async operations"""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.items: List[Any] = []
        self.last_flush = time.time()
        self._lock = asyncio.Lock()
    
    async def add(self, item: Any) -> Optional[List[Any]]:
        """Add item to batch, returns batch if ready to flush"""
        async with self._lock:
            self.items.append(item)
            
            # Check if batch is ready
            if (len(self.items) >= self.batch_size or 
                time.time() - self.last_flush >= self.max_wait_time):
                return await self._flush()
            
            return None
    
    async def flush(self) -> List[Any]:
        """Force flush current batch"""
        async with self._lock:
            return await self._flush()
    
    async def _flush(self) -> List[Any]:
        """Internal flush method"""
        if not self.items:
            return []
        
        batch = self.items.copy()
        self.items.clear()
        self.last_flush = time.time()
        return batch
    
    async def process_with_handler(self, handler: Callable[[List[Any]], Awaitable[Any]]):
        """Process items with a batch handler"""
        while True:
            # Wait for items or timeout
            await asyncio.sleep(0.1)
            
            # Check if batch is ready
            current_time = time.time()
            should_flush = (
                len(self.items) >= self.batch_size or
                (self.items and current_time - self.last_flush >= self.max_wait_time)
            )
            
            if should_flush:
                batch = await self.flush()
                if batch:
                    try:
                        await handler(batch)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")


class AsyncRetry:
    """Async retry mechanism with configurable strategies"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    async def execute(self, 
                     func: Callable[..., Awaitable[T]], 
                     *args, 
                     **kwargs) -> T:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    # Last attempt, re-raise
                    raise e
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}"
                )
                
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.config.backoff_strategy == "exponential":
            delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        elif self.config.backoff_strategy == "linear":
            delay = self.config.base_delay * (attempt + 1)
        else:  # fixed
            delay = self.config.base_delay
        
        # Apply max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay


class AsyncTimeout:
    """Async timeout utility"""
    
    def __init__(self, timeout: float, error_message: str = "Operation timed out"):
        self.timeout = timeout
        self.error_message = error_message
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.task = asyncio.current_task()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass
    
    async def execute(self, coro: Awaitable[T]) -> T:
        """Execute coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(self.error_message)


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from bucket"""
        async with self._lock:
            now = time.time()
            
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1):
        """Wait until tokens are available"""
        while not await self.acquire(tokens):
            # Calculate wait time
            wait_time = tokens / self.rate
            await asyncio.sleep(min(wait_time, 0.1))


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            # Check if we should attempt reset
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.expected_exception as e:
            await self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return False
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    async def _on_success(self):
        """Handle successful execution"""
        async with self._lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker reset to CLOSED state")
    
    async def _on_failure(self):
        """Handle failed execution"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if (self.failure_count >= self.failure_threshold and 
                self.state == CircuitBreakerState.CLOSED):
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning("Circuit breaker returned to OPEN from HALF_OPEN")


class AsyncPool:
    """Pool for managing async resources"""
    
    def __init__(self, 
                 factory: Callable[[], Awaitable[T]], 
                 max_size: int = 10,
                 min_size: int = 1):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.created_count = 0
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the pool with minimum resources"""
        for _ in range(self.min_size):
            resource = await self.factory()
            await self.pool.put(resource)
            self.created_count += 1
    
    async def acquire(self) -> T:
        """Acquire a resource from the pool"""
        try:
            # Try to get existing resource
            return self.pool.get_nowait()
        except asyncio.QueueEmpty:
            # Create new resource if under limit
            async with self._lock:
                if self.created_count < self.max_size:
                    resource = await self.factory()
                    self.created_count += 1
                    return resource
            
            # Wait for available resource
            return await self.pool.get()
    
    async def release(self, resource: T):
        """Release a resource back to the pool"""
        try:
            self.pool.put_nowait(resource)
        except asyncio.QueueFull:
            # Pool is full, discard resource
            pass
    
    async def close(self):
        """Close all resources in the pool"""
        while not self.pool.empty():
            try:
                resource = self.pool.get_nowait()
                if hasattr(resource, 'close'):
                    await resource.close()
            except asyncio.QueueEmpty:
                break


def retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry logic to async functions"""
    retry_handler = AsyncRetry(config)
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await retry_handler.execute(func, *args, **kwargs)
        return wrapper
    
    return decorator


def timeout(seconds: float, error_message: str = "Operation timed out"):
    """Decorator for adding timeout to async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            timeout_handler = AsyncTimeout(seconds, error_message)
            return await timeout_handler.execute(func(*args, **kwargs))
        return wrapper
    
    return decorator


async def gather_with_concurrency(tasks: List[Awaitable[T]],
                                 max_concurrency: int = 10) -> List[T]:
    """Execute tasks with limited concurrency"""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_task(task):
        async with semaphore:
            return await task

    limited_tasks = [limited_task(task) for task in tasks]
    return await asyncio.gather(*limited_tasks)


async def run_with_timeout(coro: Awaitable[T], timeout: float) -> T:
    """
    Run a coroutine with a timeout

    Args:
        coro: The coroutine to run
        timeout: Timeout in seconds

    Returns:
        The result of the coroutine

    Raises:
        asyncio.TimeoutError: If the operation times out
    """
    return await asyncio.wait_for(coro, timeout=timeout)
