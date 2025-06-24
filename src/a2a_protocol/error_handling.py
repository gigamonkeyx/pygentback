#!/usr/bin/env python3
"""
A2A Error Handling

Implements comprehensive error handling using standard JSON-RPC errors plus A2A-specific error codes
according to Google A2A specification.
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class A2AErrorCode(Enum):
    """A2A-specific error codes according to specification"""
    
    # Standard JSON-RPC errors (-32768 to -32000)
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # A2A-specific errors (-32001 to -32006)
    TASK_NOT_FOUND_ERROR = -32001
    UNSUPPORTED_OPERATION_ERROR = -32002
    RESOURCE_UNAVAILABLE_ERROR = -32003
    STREAMING_NOT_SUPPORTED_ERROR = -32004
    AUTHENTICATION_ERROR = -32005
    AUTHORIZATION_ERROR = -32006
    
    # Extended A2A errors (custom range -32100 to -32199)
    AGENT_NOT_FOUND_ERROR = -32100
    AGENT_UNAVAILABLE_ERROR = -32101
    CAPABILITY_NOT_SUPPORTED_ERROR = -32102
    RATE_LIMIT_EXCEEDED_ERROR = -32103
    QUOTA_EXCEEDED_ERROR = -32104
    VALIDATION_ERROR = -32105
    CONFIGURATION_ERROR = -32106
    NETWORK_ERROR = -32107
    TIMEOUT_ERROR = -32108
    DEPENDENCY_ERROR = -32109


@dataclass
class A2AErrorDetail:
    """Detailed error information"""
    code: str
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    constraint: Optional[str] = None


@dataclass
class A2AError:
    """A2A Error structure according to specification"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Ensure error code is valid"""
        if not isinstance(self.code, int):
            raise ValueError("Error code must be an integer")
        
        if not self.message:
            raise ValueError("Error message cannot be empty")


@dataclass
class A2AErrorContext:
    """Context information for error tracking"""
    request_id: Optional[str] = None
    method: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class A2AErrorHandler:
    """Comprehensive A2A error handler"""
    
    def __init__(self, include_stack_traces: bool = False, log_errors: bool = True):
        self.include_stack_traces = include_stack_traces
        self.log_errors = log_errors
        self.error_stats: Dict[int, int] = {}
        
        # Error message templates
        self.error_messages = {
            A2AErrorCode.PARSE_ERROR: "Parse error",
            A2AErrorCode.INVALID_REQUEST: "Invalid Request",
            A2AErrorCode.METHOD_NOT_FOUND: "Method not found",
            A2AErrorCode.INVALID_PARAMS: "Invalid params",
            A2AErrorCode.INTERNAL_ERROR: "Internal error",
            A2AErrorCode.TASK_NOT_FOUND_ERROR: "Task not found",
            A2AErrorCode.UNSUPPORTED_OPERATION_ERROR: "Unsupported operation",
            A2AErrorCode.RESOURCE_UNAVAILABLE_ERROR: "Resource unavailable",
            A2AErrorCode.STREAMING_NOT_SUPPORTED_ERROR: "Streaming not supported",
            A2AErrorCode.AUTHENTICATION_ERROR: "Authentication required",
            A2AErrorCode.AUTHORIZATION_ERROR: "Authorization failed",
            A2AErrorCode.AGENT_NOT_FOUND_ERROR: "Agent not found",
            A2AErrorCode.AGENT_UNAVAILABLE_ERROR: "Agent unavailable",
            A2AErrorCode.CAPABILITY_NOT_SUPPORTED_ERROR: "Capability not supported",
            A2AErrorCode.RATE_LIMIT_EXCEEDED_ERROR: "Rate limit exceeded",
            A2AErrorCode.QUOTA_EXCEEDED_ERROR: "Quota exceeded",
            A2AErrorCode.VALIDATION_ERROR: "Validation error",
            A2AErrorCode.CONFIGURATION_ERROR: "Configuration error",
            A2AErrorCode.NETWORK_ERROR: "Network error",
            A2AErrorCode.TIMEOUT_ERROR: "Timeout error",
            A2AErrorCode.DEPENDENCY_ERROR: "Dependency error"
        }
    
    def create_error(self, error_code: A2AErrorCode, message: Optional[str] = None,
                    details: Optional[List[A2AErrorDetail]] = None,
                    context: Optional[A2AErrorContext] = None,
                    exception: Optional[Exception] = None) -> A2AError:
        """Create a standardized A2A error"""
        
        # Use default message if not provided
        if not message:
            message = self.error_messages.get(error_code, "Unknown error")
        
        # Build error data
        error_data = {}
        
        # Add context information
        if context:
            error_data["context"] = {
                "request_id": context.request_id,
                "method": context.method,
                "agent_id": context.agent_id,
                "task_id": context.task_id,
                "user_id": context.user_id,
                "timestamp": context.timestamp,
                "trace_id": context.trace_id,
                "span_id": context.span_id
            }
        
        # Add detailed error information
        if details:
            error_data["details"] = [
                {
                    "code": detail.code,
                    "message": detail.message,
                    "field": detail.field,
                    "value": detail.value,
                    "constraint": detail.constraint
                }
                for detail in details
            ]
        
        # Add exception information
        if exception:
            error_data["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception)
            }
            
            if self.include_stack_traces:
                error_data["exception"]["stack_trace"] = traceback.format_exc()
        
        # Create error
        error = A2AError(
            code=error_code.value,
            message=message,
            data=error_data if error_data else None
        )
        
        # Log error
        if self.log_errors:
            self._log_error(error, context, exception)
        
        # Update statistics
        self.error_stats[error_code.value] = self.error_stats.get(error_code.value, 0) + 1
        
        return error
    
    def _log_error(self, error: A2AError, context: Optional[A2AErrorContext], 
                  exception: Optional[Exception]):
        """Log error with appropriate level"""
        log_message = f"A2A Error {error.code}: {error.message}"
        
        if context:
            log_context = []
            if context.request_id:
                log_context.append(f"request_id={context.request_id}")
            if context.method:
                log_context.append(f"method={context.method}")
            if context.agent_id:
                log_context.append(f"agent_id={context.agent_id}")
            if context.task_id:
                log_context.append(f"task_id={context.task_id}")
            
            if log_context:
                log_message += f" [{', '.join(log_context)}]"
        
        # Determine log level based on error code
        if error.code in [-32700, -32600, -32601, -32602]:  # Client errors
            logger.warning(log_message, exc_info=exception)
        elif error.code == -32603:  # Internal error
            logger.error(log_message, exc_info=exception)
        elif -32006 <= error.code <= -32001:  # A2A-specific errors
            logger.warning(log_message, exc_info=exception)
        else:  # Extended errors
            logger.error(log_message, exc_info=exception)
    
    def create_task_not_found_error(self, task_id: str, context: Optional[A2AErrorContext] = None) -> A2AError:
        """Create task not found error"""
        return self.create_error(
            A2AErrorCode.TASK_NOT_FOUND_ERROR,
            f"Task '{task_id}' not found",
            context=context
        )
    
    def create_agent_not_found_error(self, agent_id: str, context: Optional[A2AErrorContext] = None) -> A2AError:
        """Create agent not found error"""
        return self.create_error(
            A2AErrorCode.AGENT_NOT_FOUND_ERROR,
            f"Agent '{agent_id}' not found",
            context=context
        )
    
    def create_authentication_error(self, reason: str = "Authentication required", 
                                  context: Optional[A2AErrorContext] = None) -> A2AError:
        """Create authentication error"""
        return self.create_error(
            A2AErrorCode.AUTHENTICATION_ERROR,
            reason,
            context=context
        )
    
    def create_authorization_error(self, reason: str = "Insufficient permissions", 
                                 context: Optional[A2AErrorContext] = None) -> A2AError:
        """Create authorization error"""
        return self.create_error(
            A2AErrorCode.AUTHORIZATION_ERROR,
            reason,
            context=context
        )
    
    def create_validation_error(self, field: str, value: Any, constraint: str,
                              context: Optional[A2AErrorContext] = None) -> A2AError:
        """Create validation error"""
        details = [A2AErrorDetail(
            code="VALIDATION_FAILED",
            message=f"Validation failed for field '{field}'",
            field=field,
            value=value,
            constraint=constraint
        )]
        
        return self.create_error(
            A2AErrorCode.VALIDATION_ERROR,
            f"Validation failed for field '{field}': {constraint}",
            details=details,
            context=context
        )
    
    def create_rate_limit_error(self, limit: int, window: str, 
                              context: Optional[A2AErrorContext] = None) -> A2AError:
        """Create rate limit exceeded error"""
        return self.create_error(
            A2AErrorCode.RATE_LIMIT_EXCEEDED_ERROR,
            f"Rate limit exceeded: {limit} requests per {window}",
            context=context
        )
    
    def create_timeout_error(self, operation: str, timeout_seconds: float,
                           context: Optional[A2AErrorContext] = None) -> A2AError:
        """Create timeout error"""
        return self.create_error(
            A2AErrorCode.TIMEOUT_ERROR,
            f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            context=context
        )
    
    def create_capability_not_supported_error(self, capability: str, agent_id: str,
                                            context: Optional[A2AErrorContext] = None) -> A2AError:
        """Create capability not supported error"""
        return self.create_error(
            A2AErrorCode.CAPABILITY_NOT_SUPPORTED_ERROR,
            f"Capability '{capability}' not supported by agent '{agent_id}'",
            context=context
        )
    
    def wrap_exception(self, exception: Exception, context: Optional[A2AErrorContext] = None) -> A2AError:
        """Wrap a generic exception into an A2A error"""
        if isinstance(exception, A2AException):
            return exception.to_error()
        
        # Map common exceptions to appropriate error codes
        if isinstance(exception, ValueError):
            error_code = A2AErrorCode.INVALID_PARAMS
        elif isinstance(exception, KeyError):
            error_code = A2AErrorCode.TASK_NOT_FOUND_ERROR
        elif isinstance(exception, PermissionError):
            error_code = A2AErrorCode.AUTHORIZATION_ERROR
        elif isinstance(exception, TimeoutError):
            error_code = A2AErrorCode.TIMEOUT_ERROR
        elif isinstance(exception, ConnectionError):
            error_code = A2AErrorCode.NETWORK_ERROR
        else:
            error_code = A2AErrorCode.INTERNAL_ERROR
        
        return self.create_error(
            error_code,
            str(exception),
            context=context,
            exception=exception
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = sum(self.error_stats.values())
        
        stats = {
            "total_errors": total_errors,
            "error_counts": {},
            "error_rates": {}
        }
        
        for code, count in self.error_stats.items():
            error_name = None
            for error_code in A2AErrorCode:
                if error_code.value == code:
                    error_name = error_code.name
                    break
            
            key = error_name or f"ERROR_{code}"
            stats["error_counts"][key] = count
            
            if total_errors > 0:
                stats["error_rates"][key] = count / total_errors
        
        return stats
    
    def clear_statistics(self):
        """Clear error statistics"""
        self.error_stats.clear()

    def handle_error(self, error: A2AError) -> A2AError:
        """Handle an A2A error (for testing compatibility)"""
        # Just return the error as-is for testing
        return error

    def handle_transport_error(self, error: 'A2ATransportError') -> 'A2ATransportError':
        """Handle a transport error (for testing compatibility)"""
        # Just return the error as-is for testing
        return error


class A2AException(Exception):
    """Base exception for A2A protocol errors"""
    
    def __init__(self, error_code: A2AErrorCode, message: str, 
                 details: Optional[List[A2AErrorDetail]] = None,
                 context: Optional[A2AErrorContext] = None):
        self.error_code = error_code
        self.message = message
        self.details = details
        self.context = context
        super().__init__(message)
    
    def to_error(self) -> A2AError:
        """Convert exception to A2A error"""
        error_handler = A2AErrorHandler()
        return error_handler.create_error(
            self.error_code,
            self.message,
            self.details,
            self.context,
            self
        )


class TaskNotFoundException(A2AException):
    """Task not found exception"""
    
    def __init__(self, task_id: str, context: Optional[A2AErrorContext] = None):
        super().__init__(
            A2AErrorCode.TASK_NOT_FOUND_ERROR,
            f"Task '{task_id}' not found",
            context=context
        )


class AgentNotFoundException(A2AException):
    """Agent not found exception"""
    
    def __init__(self, agent_id: str, context: Optional[A2AErrorContext] = None):
        super().__init__(
            A2AErrorCode.AGENT_NOT_FOUND_ERROR,
            f"Agent '{agent_id}' not found",
            context=context
        )


class AuthenticationException(A2AException):
    """Authentication exception"""
    
    def __init__(self, reason: str = "Authentication required", context: Optional[A2AErrorContext] = None):
        super().__init__(
            A2AErrorCode.AUTHENTICATION_ERROR,
            reason,
            context=context
        )


class AuthorizationException(A2AException):
    """Authorization exception"""
    
    def __init__(self, reason: str = "Insufficient permissions", context: Optional[A2AErrorContext] = None):
        super().__init__(
            A2AErrorCode.AUTHORIZATION_ERROR,
            reason,
            context=context
        )


class UnsupportedOperationException(A2AException):
    """Unsupported operation exception"""
    
    def __init__(self, operation: str, context: Optional[A2AErrorContext] = None):
        super().__init__(
            A2AErrorCode.UNSUPPORTED_OPERATION_ERROR,
            f"Operation '{operation}' is not supported",
            context=context
        )


class StreamingNotSupportedException(A2AException):
    """Streaming not supported exception"""

    def __init__(self, context: Optional[A2AErrorContext] = None):
        super().__init__(
            A2AErrorCode.STREAMING_NOT_SUPPORTED_ERROR,
            "Streaming is not supported",
            context=context
        )


class A2ATransportError(Exception):
    """A2A Transport Error for JSON-RPC style errors"""

    def __init__(self, code: int, message: str, data: Optional[Any] = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"A2A Transport Error {code}: {message}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON-RPC response"""
        error_dict = {
            "code": self.code,
            "message": self.message
        }
        if self.data is not None:
            error_dict["data"] = self.data
        return error_dict


# Global error handler instance
error_handler = A2AErrorHandler()
