"""
Integration Utilities

Common utility functions and helper classes for integration components,
validation, logging, and operational support.
"""

import logging
import asyncio
import time
import hashlib
import json
import inspect
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)


@dataclass
class OperationResult:
    """Result of an operation with metadata"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RetryConfig:
    """Configuration for retry operations"""
    
    def __init__(self, max_attempts: int = 3, delay_seconds: float = 1.0, 
                 backoff_multiplier: float = 2.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay


class ValidationError(Exception):
    """Custom validation error"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


def retry_async(config: RetryConfig = None):
    """Decorator for async functions with retry logic"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = config.delay_seconds
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        break
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    await asyncio.sleep(delay)
                    delay = min(delay * config.backoff_multiplier, config.max_delay)
            
            logger.error(f"All {config.max_attempts} attempts failed for {func.__name__}")
            raise last_exception
        
        return wrapper
    return decorator


def retry_sync(config: RetryConfig = None):
    """Decorator for sync functions with retry logic"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = config.delay_seconds
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        break
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    time.sleep(delay)
                    delay = min(delay * config.backoff_multiplier, config.max_delay)
            
            logger.error(f"All {config.max_attempts} attempts failed for {func.__name__}")
            raise last_exception
        
        return wrapper
    return decorator


def measure_time(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            logger.debug(f"{func.__name__} completed in {duration:.2f}ms")
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"{func.__name__} failed after {duration:.2f}ms: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            logger.debug(f"{func.__name__} completed in {duration:.2f}ms")
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"{func.__name__} failed after {duration:.2f}ms: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


async def execute_with_timeout(coro, timeout_seconds: float) -> OperationResult:
    """Execute coroutine with timeout"""
    start_time = time.time()
    
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        duration = (time.time() - start_time) * 1000
        
        return OperationResult(
            success=True,
            result=result,
            duration_ms=duration
        )
    
    except asyncio.TimeoutError:
        duration = (time.time() - start_time) * 1000
        return OperationResult(
            success=False,
            error=f"Operation timed out after {timeout_seconds}s",
            duration_ms=duration
        )
    
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        return OperationResult(
            success=False,
            error=str(e),
            duration_ms=duration
        )


def safe_execute(func: Callable, *args, **kwargs) -> OperationResult:
    """Safely execute a function and return result with error handling"""
    start_time = time.time()
    
    try:
        if asyncio.iscoroutinefunction(func):
            raise ValueError("Use safe_execute_async for coroutines")
        
        result = func(*args, **kwargs)
        duration = (time.time() - start_time) * 1000
        
        return OperationResult(
            success=True,
            result=result,
            duration_ms=duration
        )
    
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        error_details = f"{type(e).__name__}: {str(e)}"
        
        return OperationResult(
            success=False,
            error=error_details,
            duration_ms=duration,
            metadata={'traceback': traceback.format_exc()}
        )


async def safe_execute_async(coro) -> OperationResult:
    """Safely execute a coroutine and return result with error handling"""
    start_time = time.time()
    
    try:
        result = await coro
        duration = (time.time() - start_time) * 1000
        
        return OperationResult(
            success=True,
            result=result,
            duration_ms=duration
        )
    
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        error_details = f"{type(e).__name__}: {str(e)}"
        
        return OperationResult(
            success=False,
            error=error_details,
            duration_ms=duration,
            metadata={'traceback': traceback.format_exc()}
        )


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """Validate that required fields are present in data"""
    missing_fields = []
    
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    return missing_fields


def validate_field_types(data: Dict[str, Any], type_specs: Dict[str, Type]) -> List[str]:
    """Validate field types in data"""
    type_errors = []
    
    for field, expected_type in type_specs.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], expected_type):
                type_errors.append(
                    f"Field '{field}' expected {expected_type.__name__}, "
                    f"got {type(data[field]).__name__}"
                )
    
    return type_errors


def validate_data_structure(data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate data structure against schema"""
    errors = []
    
    # Check required fields
    required_fields = schema.get('required', [])
    missing_fields = validate_required_fields(data, required_fields)
    errors.extend([f"Missing required field: {field}" for field in missing_fields])
    
    # Check field types
    field_types = schema.get('types', {})
    type_errors = validate_field_types(data, field_types)
    errors.extend(type_errors)
    
    # Check field constraints
    constraints = schema.get('constraints', {})
    for field, constraint in constraints.items():
        if field in data and data[field] is not None:
            value = data[field]
            
            # Min/max constraints for numbers
            if isinstance(value, (int, float)):
                if 'min' in constraint and value < constraint['min']:
                    errors.append(f"Field '{field}' value {value} below minimum {constraint['min']}")
                if 'max' in constraint and value > constraint['max']:
                    errors.append(f"Field '{field}' value {value} above maximum {constraint['max']}")
            
            # Length constraints for strings/lists
            if isinstance(value, (str, list)):
                if 'min_length' in constraint and len(value) < constraint['min_length']:
                    errors.append(f"Field '{field}' length {len(value)} below minimum {constraint['min_length']}")
                if 'max_length' in constraint and len(value) > constraint['max_length']:
                    errors.append(f"Field '{field}' length {len(value)} above maximum {constraint['max_length']}")
            
            # Choice constraints
            if 'choices' in constraint and value not in constraint['choices']:
                errors.append(f"Field '{field}' value '{value}' not in allowed choices {constraint['choices']}")
    
    return errors


def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a unique ID"""
    timestamp = str(int(time.time() * 1000))
    hash_input = f"{timestamp}_{time.time()}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:length]
    
    if prefix:
        return f"{prefix}_{hash_value}"
    return hash_value


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionary"""
    def _flatten(obj, parent_key=""):
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                items.extend(_flatten(value, new_key).items())
        else:
            return {parent_key: obj}
        
        return dict(items)
    
    return _flatten(data)


def unflatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Unflatten dictionary with separator-based keys"""
    result = {}
    
    for key, value in data.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def serialize_for_json(obj: Any) -> Any:
    """Serialize object for JSON encoding"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, timedelta):
        return obj.total_seconds()
    elif hasattr(obj, '__dict__'):
        return asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    else:
        return obj


def get_function_signature(func: Callable) -> Dict[str, Any]:
    """Get function signature information"""
    sig = inspect.signature(func)
    
    return {
        'name': func.__name__,
        'parameters': {
            name: {
                'type': param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'Any',
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty
            }
            for name, param in sig.parameters.items()
        },
        'return_type': sig.return_annotation.__name__ if sig.return_annotation != inspect.Signature.empty else 'Any',
        'is_async': asyncio.iscoroutinefunction(func)
    }


def create_logger(name: str, level: str = "INFO", 
                 format_string: str = None) -> logging.Logger:
    """Create a configured logger"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e


class RateLimiter:
    """Rate limiter implementation"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def is_allowed(self) -> bool:
        """Check if call is allowed under rate limit"""
        now = time.time()
        
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        # Check if under limit
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        return False
    
    def wait_time(self) -> float:
        """Get time to wait before next call is allowed"""
        if not self.calls:
            return 0.0
        
        oldest_call = min(self.calls)
        return max(0.0, self.time_window - (time.time() - oldest_call))


def batch_process(items: List[Any], batch_size: int, 
                 processor: Callable[[List[Any]], Any]) -> List[Any]:
    """Process items in batches"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_result = processor(batch)
        results.append(batch_result)
    
    return results


async def batch_process_async(items: List[Any], batch_size: int,
                            processor: Callable[[List[Any]], Any]) -> List[Any]:
    """Process items in batches asynchronously"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        if asyncio.iscoroutinefunction(processor):
            batch_result = await processor(batch)
        else:
            batch_result = processor(batch)
        
        results.append(batch_result)
    
    return results
