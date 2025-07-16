"""
Utilities Module

This module provides common utilities and helper functions used throughout
the PyGent Factory system, including logging, performance monitoring,
data processing, and system utilities.
"""

# Import all utility components
from .embedding import (
    EmbeddingService, EmbeddingResult, EmbeddingCache,
    OpenAIEmbeddingProvider, SentenceTransformerProvider,
    get_embedding_service
)
from .utf8_logger import (
    get_logger, get_pygent_logger, configure_utf8_logging,
    UTF8StreamHandler, UTF8FileHandler, RIPEROmegaFormatter
)
from .performance import (
    PerformanceMonitor, Timer, ProfilerContext,
    MemoryTracker, SystemMetrics
)
from .data import (
    DataProcessor, TextProcessor, JSONProcessor,
    DataValidator, SchemaValidator
)
from .async_utils import (
    AsyncBatch, AsyncRetry, AsyncTimeout,
    RateLimiter, CircuitBreaker
)
from .system import (
    SystemInfo, ResourceMonitor, HealthChecker,
    ProcessManager, FileWatcher
)

# Export all utilities
__all__ = [
    # Embedding utilities
    "EmbeddingService",
    "EmbeddingResult", 
    "EmbeddingCache",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    "get_embedding_service",
    
    # UTF-8 Logging utilities
    "get_logger",
    "get_pygent_logger",
    "configure_utf8_logging",
    "UTF8StreamHandler",
    "UTF8FileHandler",
    "RIPEROmegaFormatter",
    
    # Performance utilities
    "PerformanceMonitor",
    "Timer",
    "ProfilerContext",
    "MemoryTracker",
    "SystemMetrics",
    
    # Data processing utilities
    "DataProcessor",
    "TextProcessor",
    "JSONProcessor",
    "DataValidator",
    "SchemaValidator",
    
    # Async utilities
    "AsyncBatch",
    "AsyncRetry",
    "AsyncTimeout",
    "RateLimiter",
    "CircuitBreaker",
    
    # System utilities
    "SystemInfo",
    "ResourceMonitor",
    "HealthChecker",
    "ProcessManager",
    "FileWatcher"
]
