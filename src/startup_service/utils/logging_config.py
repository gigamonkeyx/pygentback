"""
Logging Configuration for PyGent Factory Startup Service
Structured logging with JSON output for production monitoring.
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from typing import Dict, Any

import structlog
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    json_logs: bool = None,
    log_file: str = None
) -> None:
    """
    Configure structured logging for the startup service.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to use JSON format (auto-detected if None)
        log_file: Optional log file path
    """
    
    # Auto-detect JSON logging based on environment
    if json_logs is None:
        json_logs = os.getenv("LOG_FORMAT", "").lower() == "json" or os.getenv("ENVIRONMENT", "").lower() == "production"
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    handlers = []
    
    # Console handler
    if json_logs:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        )
        console_handler.setFormatter(console_formatter)
    else:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True
        )
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
    
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "pathname": "%(pathname)s", "lineno": %(lineno)d}'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        format="%(message)s"
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    
    # Create startup service logger
    logger = structlog.get_logger("startup_service")
    logger.info(
        "Logging configured",
        level=level,
        json_format=json_logs,
        log_file=log_file,
        timestamp=datetime.utcnow().isoformat()
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class StartupServiceLogger:
    """Custom logger for startup service operations."""
    
    def __init__(self, component: str):
        self.logger = get_logger(f"startup_service.{component}")
        self.component = component
    
    def info(self, message: str, **kwargs):
        """Log info message with component context."""
        self.logger.info(message, component=self.component, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with component context."""
        self.logger.warning(message, component=self.component, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with component context."""
        self.logger.error(message, component=self.component, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with component context."""
        self.logger.debug(message, component=self.component, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with component context."""
        self.logger.critical(message, component=self.component, **kwargs)
    
    def startup_event(self, event: str, status: str, **kwargs):
        """Log startup-specific events."""
        self.logger.info(
            f"Startup event: {event}",
            component=self.component,
            event=event,
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def service_event(self, service: str, action: str, status: str, **kwargs):
        """Log service-specific events."""
        self.logger.info(
            f"Service {action}: {service}",
            component=self.component,
            service=service,
            action=action,
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def performance_metric(self, metric: str, value: float, unit: str = "", **kwargs):
        """Log performance metrics."""
        self.logger.info(
            f"Performance metric: {metric}",
            component=self.component,
            metric=metric,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )


# Pre-configured loggers for common components
database_logger = StartupServiceLogger("database")
orchestrator_logger = StartupServiceLogger("orchestrator")
config_logger = StartupServiceLogger("config")
websocket_logger = StartupServiceLogger("websocket")
api_logger = StartupServiceLogger("api")
security_logger = StartupServiceLogger("security")


def log_startup_progress(phase: str, step: str, status: str, details: Dict[str, Any] = None):
    """Log startup progress with standardized format."""
    logger = get_logger("startup_service.progress")
    logger.info(
        f"Startup progress: {phase} - {step}",
        phase=phase,
        step=step,
        status=status,
        details=details or {},
        timestamp=datetime.utcnow().isoformat()
    )


def log_service_operation(service: str, operation: str, status: str, duration: float = None, details: Dict[str, Any] = None):
    """Log service operations with standardized format."""
    logger = get_logger("startup_service.operations")
    logger.info(
        f"Service operation: {service} {operation}",
        service=service,
        operation=operation,
        status=status,
        duration_seconds=duration,
        details=details or {},
        timestamp=datetime.utcnow().isoformat()
    )


def log_error_with_context(error: Exception, context: Dict[str, Any] = None):
    """Log errors with full context information."""
    logger = get_logger("startup_service.errors")
    logger.error(
        f"Error occurred: {str(error)}",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {},
        timestamp=datetime.utcnow().isoformat(),
        exc_info=True
    )
