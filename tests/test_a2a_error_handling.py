#!/usr/bin/env python3
"""
Test A2A Error Handling

Tests the A2A-compliant error handling implementation according to Google A2A specification.
"""

import pytest
from unittest.mock import Mock, patch

# Import the A2A error handling components
try:
    from src.a2a_protocol.error_handling import (
        A2AErrorHandler, A2AError, A2AErrorCode, A2AErrorContext, A2AErrorDetail,
        A2AException, TaskNotFoundException, AgentNotFoundException,
        AuthenticationException, AuthorizationException, UnsupportedOperationException,
        StreamingNotSupportedException
    )
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AErrorHandler:
    """Test A2A Error Handler"""
    
    def setup_method(self):
        """Setup test environment"""
        self.error_handler = A2AErrorHandler(include_stack_traces=True, log_errors=False)
    
    def test_create_basic_error(self):
        """Test creating basic error"""
        error = self.error_handler.create_error(
            A2AErrorCode.TASK_NOT_FOUND_ERROR,
            "Task not found"
        )
        
        assert error.code == -32001
        assert error.message == "Task not found"
        assert error.data is None
    
    def test_create_error_with_context(self):
        """Test creating error with context"""
        context = A2AErrorContext(
            request_id="req-123",
            method="tasks/get",
            agent_id="agent-456",
            task_id="task-789",
            user_id="user-abc"
        )
        
        error = self.error_handler.create_error(
            A2AErrorCode.TASK_NOT_FOUND_ERROR,
            "Task not found",
            context=context
        )
        
        assert error.code == -32001
        assert error.message == "Task not found"
        assert error.data is not None
        assert error.data["context"]["request_id"] == "req-123"
        assert error.data["context"]["method"] == "tasks/get"
        assert error.data["context"]["agent_id"] == "agent-456"
        assert error.data["context"]["task_id"] == "task-789"
        assert error.data["context"]["user_id"] == "user-abc"
    
    def test_create_error_with_details(self):
        """Test creating error with detailed information"""
        details = [
            A2AErrorDetail(
                code="VALIDATION_FAILED",
                message="Field validation failed",
                field="email",
                value="invalid-email",
                constraint="must be valid email format"
            )
        ]
        
        error = self.error_handler.create_error(
            A2AErrorCode.VALIDATION_ERROR,
            "Validation failed",
            details=details
        )
        
        assert error.code == -32105
        assert error.message == "Validation failed"
        assert error.data is not None
        assert len(error.data["details"]) == 1
        assert error.data["details"][0]["code"] == "VALIDATION_FAILED"
        assert error.data["details"][0]["field"] == "email"
        assert error.data["details"][0]["value"] == "invalid-email"
    
    def test_create_error_with_exception(self):
        """Test creating error with exception information"""
        exception = ValueError("Invalid input value")
        
        error = self.error_handler.create_error(
            A2AErrorCode.INVALID_PARAMS,
            "Invalid parameters",
            exception=exception
        )
        
        assert error.code == -32602
        assert error.message == "Invalid parameters"
        assert error.data is not None
        assert error.data["exception"]["type"] == "ValueError"
        assert error.data["exception"]["message"] == "Invalid input value"
        assert "stack_trace" in error.data["exception"]
    
    def test_create_task_not_found_error(self):
        """Test creating task not found error"""
        error = self.error_handler.create_task_not_found_error("task-123")
        
        assert error.code == -32001
        assert "task-123" in error.message
    
    def test_create_agent_not_found_error(self):
        """Test creating agent not found error"""
        error = self.error_handler.create_agent_not_found_error("agent-456")
        
        assert error.code == -32100
        assert "agent-456" in error.message
    
    def test_create_authentication_error(self):
        """Test creating authentication error"""
        error = self.error_handler.create_authentication_error("Invalid token")
        
        assert error.code == -32005
        assert error.message == "Invalid token"
    
    def test_create_authorization_error(self):
        """Test creating authorization error"""
        error = self.error_handler.create_authorization_error("Access denied")
        
        assert error.code == -32006
        assert error.message == "Access denied"
    
    def test_create_validation_error(self):
        """Test creating validation error"""
        error = self.error_handler.create_validation_error(
            "username", "ab", "must be at least 3 characters"
        )
        
        assert error.code == -32105
        assert "username" in error.message
        assert error.data is not None
        assert len(error.data["details"]) == 1
        assert error.data["details"][0]["field"] == "username"
        assert error.data["details"][0]["value"] == "ab"
    
    def test_create_rate_limit_error(self):
        """Test creating rate limit error"""
        error = self.error_handler.create_rate_limit_error(100, "minute")
        
        assert error.code == -32103
        assert "100 requests per minute" in error.message
    
    def test_create_timeout_error(self):
        """Test creating timeout error"""
        error = self.error_handler.create_timeout_error("task_execution", 30.0)
        
        assert error.code == -32108
        assert "task_execution" in error.message
        assert "30.0 seconds" in error.message
    
    def test_create_capability_not_supported_error(self):
        """Test creating capability not supported error"""
        error = self.error_handler.create_capability_not_supported_error("streaming", "agent-123")
        
        assert error.code == -32102
        assert "streaming" in error.message
        assert "agent-123" in error.message
    
    def test_wrap_exception_value_error(self):
        """Test wrapping ValueError"""
        exception = ValueError("Invalid value provided")
        error = self.error_handler.wrap_exception(exception)
        
        assert error.code == -32602  # INVALID_PARAMS
        assert error.message == "Invalid value provided"
    
    def test_wrap_exception_key_error(self):
        """Test wrapping KeyError"""
        exception = KeyError("missing_key")
        error = self.error_handler.wrap_exception(exception)
        
        assert error.code == -32001  # TASK_NOT_FOUND_ERROR
    
    def test_wrap_exception_permission_error(self):
        """Test wrapping PermissionError"""
        exception = PermissionError("Access denied")
        error = self.error_handler.wrap_exception(exception)
        
        assert error.code == -32006  # AUTHORIZATION_ERROR
    
    def test_wrap_exception_timeout_error(self):
        """Test wrapping TimeoutError"""
        exception = TimeoutError("Operation timed out")
        error = self.error_handler.wrap_exception(exception)
        
        assert error.code == -32108  # TIMEOUT_ERROR
    
    def test_wrap_exception_connection_error(self):
        """Test wrapping ConnectionError"""
        exception = ConnectionError("Connection failed")
        error = self.error_handler.wrap_exception(exception)
        
        assert error.code == -32107  # NETWORK_ERROR
    
    def test_wrap_exception_generic(self):
        """Test wrapping generic exception"""
        exception = RuntimeError("Something went wrong")
        error = self.error_handler.wrap_exception(exception)
        
        assert error.code == -32603  # INTERNAL_ERROR
    
    def test_error_statistics(self):
        """Test error statistics tracking"""
        # Create several errors
        self.error_handler.create_error(A2AErrorCode.TASK_NOT_FOUND_ERROR, "Test 1")
        self.error_handler.create_error(A2AErrorCode.TASK_NOT_FOUND_ERROR, "Test 2")
        self.error_handler.create_error(A2AErrorCode.AUTHENTICATION_ERROR, "Test 3")
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert stats["error_counts"]["TASK_NOT_FOUND_ERROR"] == 2
        assert stats["error_counts"]["AUTHENTICATION_ERROR"] == 1
        assert stats["error_rates"]["TASK_NOT_FOUND_ERROR"] == 2/3
        assert stats["error_rates"]["AUTHENTICATION_ERROR"] == 1/3
    
    def test_clear_statistics(self):
        """Test clearing error statistics"""
        # Create an error
        self.error_handler.create_error(A2AErrorCode.TASK_NOT_FOUND_ERROR, "Test")
        
        # Verify statistics exist
        stats = self.error_handler.get_error_statistics()
        assert stats["total_errors"] == 1
        
        # Clear statistics
        self.error_handler.clear_statistics()
        
        # Verify statistics are cleared
        stats = self.error_handler.get_error_statistics()
        assert stats["total_errors"] == 0


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AExceptions:
    """Test A2A Exception classes"""
    
    def test_task_not_found_exception(self):
        """Test TaskNotFoundException"""
        context = A2AErrorContext(task_id="task-123")
        exception = TaskNotFoundException("task-123", context)
        
        assert exception.error_code == A2AErrorCode.TASK_NOT_FOUND_ERROR
        assert "task-123" in exception.message
        assert exception.context == context
        
        # Test conversion to error
        error = exception.to_error()
        assert error.code == -32001
        assert "task-123" in error.message
    
    def test_agent_not_found_exception(self):
        """Test AgentNotFoundException"""
        context = A2AErrorContext(agent_id="agent-456")
        exception = AgentNotFoundException("agent-456", context)
        
        assert exception.error_code == A2AErrorCode.AGENT_NOT_FOUND_ERROR
        assert "agent-456" in exception.message
        assert exception.context == context
    
    def test_authentication_exception(self):
        """Test AuthenticationException"""
        exception = AuthenticationException("Invalid credentials")
        
        assert exception.error_code == A2AErrorCode.AUTHENTICATION_ERROR
        assert exception.message == "Invalid credentials"
    
    def test_authorization_exception(self):
        """Test AuthorizationException"""
        exception = AuthorizationException("Access denied")
        
        assert exception.error_code == A2AErrorCode.AUTHORIZATION_ERROR
        assert exception.message == "Access denied"
    
    def test_unsupported_operation_exception(self):
        """Test UnsupportedOperationException"""
        exception = UnsupportedOperationException("streaming")
        
        assert exception.error_code == A2AErrorCode.UNSUPPORTED_OPERATION_ERROR
        assert "streaming" in exception.message
    
    def test_streaming_not_supported_exception(self):
        """Test StreamingNotSupportedException"""
        exception = StreamingNotSupportedException()
        
        assert exception.error_code == A2AErrorCode.STREAMING_NOT_SUPPORTED_ERROR
        assert "Streaming is not supported" in exception.message


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AErrorStructures:
    """Test A2A Error data structures"""
    
    def test_a2a_error_creation(self):
        """Test A2AError creation"""
        error = A2AError(
            code=-32001,
            message="Task not found",
            data={"task_id": "task-123"}
        )
        
        assert error.code == -32001
        assert error.message == "Task not found"
        assert error.data["task_id"] == "task-123"
    
    def test_a2a_error_validation(self):
        """Test A2AError validation"""
        # Test invalid code type
        with pytest.raises(ValueError, match="Error code must be an integer"):
            A2AError(code="invalid", message="Test")
        
        # Test empty message
        with pytest.raises(ValueError, match="Error message cannot be empty"):
            A2AError(code=-32001, message="")
    
    def test_a2a_error_context(self):
        """Test A2AErrorContext"""
        context = A2AErrorContext(
            request_id="req-123",
            method="tasks/get",
            agent_id="agent-456",
            task_id="task-789",
            user_id="user-abc",
            trace_id="trace-def",
            span_id="span-ghi"
        )
        
        assert context.request_id == "req-123"
        assert context.method == "tasks/get"
        assert context.agent_id == "agent-456"
        assert context.task_id == "task-789"
        assert context.user_id == "user-abc"
        assert context.trace_id == "trace-def"
        assert context.span_id == "span-ghi"
        assert context.timestamp is not None
    
    def test_a2a_error_detail(self):
        """Test A2AErrorDetail"""
        detail = A2AErrorDetail(
            code="VALIDATION_FAILED",
            message="Field validation failed",
            field="email",
            value="invalid@",
            constraint="must be valid email"
        )
        
        assert detail.code == "VALIDATION_FAILED"
        assert detail.message == "Field validation failed"
        assert detail.field == "email"
        assert detail.value == "invalid@"
        assert detail.constraint == "must be valid email"


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AErrorCodes:
    """Test A2A Error Codes"""
    
    def test_standard_jsonrpc_error_codes(self):
        """Test standard JSON-RPC error codes"""
        assert A2AErrorCode.PARSE_ERROR.value == -32700
        assert A2AErrorCode.INVALID_REQUEST.value == -32600
        assert A2AErrorCode.METHOD_NOT_FOUND.value == -32601
        assert A2AErrorCode.INVALID_PARAMS.value == -32602
        assert A2AErrorCode.INTERNAL_ERROR.value == -32603
    
    def test_a2a_specific_error_codes(self):
        """Test A2A-specific error codes"""
        assert A2AErrorCode.TASK_NOT_FOUND_ERROR.value == -32001
        assert A2AErrorCode.UNSUPPORTED_OPERATION_ERROR.value == -32002
        assert A2AErrorCode.RESOURCE_UNAVAILABLE_ERROR.value == -32003
        assert A2AErrorCode.STREAMING_NOT_SUPPORTED_ERROR.value == -32004
        assert A2AErrorCode.AUTHENTICATION_ERROR.value == -32005
        assert A2AErrorCode.AUTHORIZATION_ERROR.value == -32006
    
    def test_extended_error_codes(self):
        """Test extended A2A error codes"""
        assert A2AErrorCode.AGENT_NOT_FOUND_ERROR.value == -32100
        assert A2AErrorCode.AGENT_UNAVAILABLE_ERROR.value == -32101
        assert A2AErrorCode.CAPABILITY_NOT_SUPPORTED_ERROR.value == -32102
        assert A2AErrorCode.RATE_LIMIT_EXCEEDED_ERROR.value == -32103
        assert A2AErrorCode.QUOTA_EXCEEDED_ERROR.value == -32104
        assert A2AErrorCode.VALIDATION_ERROR.value == -32105
        assert A2AErrorCode.CONFIGURATION_ERROR.value == -32106
        assert A2AErrorCode.NETWORK_ERROR.value == -32107
        assert A2AErrorCode.TIMEOUT_ERROR.value == -32108
        assert A2AErrorCode.DEPENDENCY_ERROR.value == -32109


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
