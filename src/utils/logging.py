"""
Logging Utilities

This module provides advanced logging capabilities including structured logging,
performance logging, and centralized logger configuration.
"""

import logging
import logging.handlers
import json
import time
import traceback
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import sys


@dataclass
class LoggerConfig:
    """Configuration for logger setup"""
    name: str = "pygent_factory"
    level: str = "INFO"
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_structured: bool = False
    correlation_id_header: str = "X-Correlation-ID"


class StructuredLogger:
    """Structured logger that outputs JSON formatted logs"""
    
    def __init__(self, name: str, config: Optional[LoggerConfig] = None):
        self.name = name
        self.config = config or LoggerConfig()
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup structured logger"""
        self.logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = StructuredFormatter()
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.file_path:
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.file_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal log method"""
        extra = {
            'structured_data': kwargs,
            'timestamp': datetime.utcnow().isoformat(),
            'logger_name': self.name
        }
        self.logger.log(level, message, extra=extra)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logs"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add structured data if available
        if hasattr(record, 'structured_data'):
            log_data.update(record.structured_data)
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """Logger specialized for performance monitoring"""
    
    def __init__(self, name: str = "performance"):
        self.logger = StructuredLogger(f"{name}.performance")
    
    def log_operation(self, 
                     operation: str,
                     duration_ms: float,
                     success: bool = True,
                     **metadata):
        """Log operation performance"""
        self.logger.info(
            f"Operation completed: {operation}",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **metadata
        )
    
    def log_request(self,
                   method: str,
                   path: str,
                   status_code: int,
                   duration_ms: float,
                   **metadata):
        """Log HTTP request performance"""
        self.logger.info(
            f"Request completed: {method} {path}",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            **metadata
        )
    
    def log_query(self,
                 query_type: str,
                 duration_ms: float,
                 result_count: int,
                 **metadata):
        """Log database query performance"""
        self.logger.info(
            f"Query completed: {query_type}",
            query_type=query_type,
            duration_ms=duration_ms,
            result_count=result_count,
            **metadata
        )


class ContextualLogger:
    """Logger that maintains context across operations"""
    
    def __init__(self, base_logger: StructuredLogger):
        self.base_logger = base_logger
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set context that will be included in all log messages"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context"""
        self.context.clear()
    
    def info(self, message: str, **kwargs):
        """Log info with context"""
        combined_kwargs = {**self.context, **kwargs}
        self.base_logger.info(message, **combined_kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning with context"""
        combined_kwargs = {**self.context, **kwargs}
        self.base_logger.warning(message, **combined_kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error with context"""
        combined_kwargs = {**self.context, **kwargs}
        self.base_logger.error(message, **combined_kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug with context"""
        combined_kwargs = {**self.context, **kwargs}
        self.base_logger.debug(message, **combined_kwargs)


def setup_logging(config: Optional[LoggerConfig] = None) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        config: Logger configuration
        
    Returns:
        logging.Logger: Configured root logger
    """
    config = config or LoggerConfig()
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    if config.enable_structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(config.format_string)
    
    # Console handler
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.file_path:
        # Ensure directory exists
        Path(config.file_path).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str, 
               structured: bool = False,
               config: Optional[LoggerConfig] = None) -> Union[logging.Logger, StructuredLogger]:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        structured: Whether to use structured logging
        config: Logger configuration
        
    Returns:
        Logger instance
    """
    if structured:
        return StructuredLogger(name, config)
    else:
        return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(self.__class__.__name__)
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return self._logger
    
    def log_method_call(self, method_name: str, **kwargs):
        """Log method call with parameters"""
        self._logger.debug(
            f"Calling {self.__class__.__name__}.{method_name}",
            extra={'method_params': kwargs}
        )
    
    def log_method_result(self, method_name: str, result: Any, duration_ms: float):
        """Log method result and duration"""
        self._logger.debug(
            f"Completed {self.__class__.__name__}.{method_name}",
            extra={
                'duration_ms': duration_ms,
                'result_type': type(result).__name__
            }
        )


# Global logger instances
_loggers: Dict[str, Union[logging.Logger, StructuredLogger]] = {}


def get_cached_logger(name: str, **kwargs) -> Union[logging.Logger, StructuredLogger]:
    """Get cached logger instance"""
    if name not in _loggers:
        _loggers[name] = get_logger(name, **kwargs)
    return _loggers[name]
