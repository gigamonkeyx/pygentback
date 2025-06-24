#!/usr/bin/env python3
"""
Test A2A Protocol Transport Layer

Tests the A2A-compliant transport layer implementation according to Google A2A specification.
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import the A2A transport components
try:
    from src.a2a_protocol.transport import A2ATransportLayer, A2ATransportError
    from src.a2a_protocol.streaming import A2AStreamingManager, A2AStreamingHandler
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2ATransportLayer:
    """Test A2A Transport Layer"""
    
    def setup_method(self):
        """Setup test environment"""
        self.transport = A2ATransportLayer("http://localhost:8000")
        self.app = FastAPI()
        self.transport.setup_fastapi_routes(self.app)
        self.client = TestClient(self.app)
    
    def test_jsonrpc_request_validation(self):
        """Test JSON-RPC request validation"""
        # Valid request
        valid_request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {"message": {"role": "user", "parts": []}},
            "id": "test-123"
        }
        assert self.transport._validate_jsonrpc_request(valid_request) == True
        
        # Invalid requests
        invalid_requests = [
            {},  # Empty
            {"method": "test"},  # Missing jsonrpc
            {"jsonrpc": "1.0", "method": "test"},  # Wrong version
            {"jsonrpc": "2.0"},  # Missing method
            {"jsonrpc": "2.0", "method": ""},  # Empty method
            {"jsonrpc": "2.0", "method": "test", "params": "invalid"},  # Invalid params
        ]
        
        for invalid_request in invalid_requests:
            assert self.transport._validate_jsonrpc_request(invalid_request) == False
    
    @pytest.mark.asyncio
    async def test_message_send_method(self):
        """Test message/send method"""
        params = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello, world!"}]
            },
            "taskId": "test-task-123"
        }
        
        result = await self.transport._handle_message_send(params, None)
        
        # Verify response structure
        assert "id" in result
        assert "contextId" in result
        assert "status" in result
        assert "history" in result
        assert "artifacts" in result
        assert "kind" in result
        
        # Verify task ID
        assert result["id"] == "test-task-123"
        
        # Verify status
        assert result["status"]["state"] == "submitted"
        assert "timestamp" in result["status"]
        
        # Verify history includes message
        assert len(result["history"]) == 1
        assert result["history"][0] == params["message"]
    
    @pytest.mark.asyncio
    async def test_message_send_invalid_params(self):
        """Test message/send with invalid parameters"""
        # Missing message
        with pytest.raises(A2ATransportError) as exc_info:
            await self.transport._handle_message_send({}, None)
        assert exc_info.value.code == -32602
        
        # Invalid message structure
        with pytest.raises(A2ATransportError) as exc_info:
            await self.transport._handle_message_send({"message": "invalid"}, None)
        assert exc_info.value.code == -32602
    
    @pytest.mark.asyncio
    async def test_tasks_get_method(self):
        """Test tasks/get method"""
        params = {"id": "test-task-123"}
        
        result = await self.transport._handle_tasks_get(params, None)
        
        # Verify response structure
        assert "id" in result
        assert "contextId" in result
        assert "status" in result
        assert "history" in result
        assert "artifacts" in result
        assert "kind" in result
        
        # Verify task ID
        assert result["id"] == "test-task-123"
    
    @pytest.mark.asyncio
    async def test_tasks_cancel_method(self):
        """Test tasks/cancel method"""
        params = {"id": "test-task-123"}
        
        result = await self.transport._handle_tasks_cancel(params, None)
        
        # Verify response structure
        assert "id" in result
        assert "status" in result
        assert "kind" in result
        
        # Verify task ID and status
        assert result["id"] == "test-task-123"
        assert result["status"]["state"] == "canceled"
    
    @pytest.mark.asyncio
    async def test_push_notification_config_methods(self):
        """Test push notification configuration methods"""
        # Test set config
        set_params = {
            "taskId": "test-task-123",
            "config": {"url": "https://example.com/webhook", "token": "test-token"}
        }
        
        set_result = await self.transport._handle_push_notification_config_set(set_params, None)
        assert set_result["taskId"] == "test-task-123"
        assert set_result["success"] == True
        
        # Test get config
        get_params = {"taskId": "test-task-123"}
        
        get_result = await self.transport._handle_push_notification_config_get(get_params, None)
        assert get_result["taskId"] == "test-task-123"
        assert "config" in get_result
    
    @pytest.mark.asyncio
    async def test_process_jsonrpc_request(self):
        """Test complete JSON-RPC request processing"""
        request_data = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Test message"}]
                }
            },
            "id": "test-123"
        }
        
        response = await self.transport.process_jsonrpc_request(request_data)
        
        # Verify response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-123"
        assert "result" in response
        assert "error" not in response
        
        # Verify result
        result = response["result"]
        assert "id" in result
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_process_jsonrpc_request_invalid_method(self):
        """Test JSON-RPC request with invalid method"""
        request_data = {
            "jsonrpc": "2.0",
            "method": "invalid/method",
            "params": {},
            "id": "test-123"
        }
        
        response = await self.transport.process_jsonrpc_request(request_data)
        
        # Verify error response
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-123"
        assert "error" in response
        assert "result" not in response
        
        # Verify error details
        error = response["error"]
        assert error["code"] == -32601
        assert "Method not found" in error["message"]
    
    def test_fastapi_integration(self):
        """Test FastAPI integration"""
        # Test valid JSON-RPC request
        request_data = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Test message"}]
                }
            },
            "id": "test-123"
        }
        
        response = self.client.post("/a2a/v1", json=request_data)
        
        # Verify HTTP response
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Verify JSON-RPC response
        response_data = response.json()
        assert response_data["jsonrpc"] == "2.0"
        assert response_data["id"] == "test-123"
        assert "result" in response_data
    
    def test_fastapi_cors_handling(self):
        """Test CORS handling"""
        # Test OPTIONS request
        response = self.client.options("/a2a/v1")
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_fastapi_invalid_json(self):
        """Test FastAPI handling of invalid JSON"""
        response = self.client.post(
            "/a2a/v1",
            content="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 400
        response_data = response.json()
        assert response_data["error"]["code"] == -32700
        assert "Parse error" in response_data["error"]["message"]


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AStreamingManager:
    """Test A2A Streaming Manager"""
    
    def setup_method(self):
        """Setup test environment"""
        self.streaming_manager = A2AStreamingManager()
    
    @pytest.mark.asyncio
    async def test_create_and_close_stream(self):
        """Test stream creation and closure"""
        task_id = "test-task-123"
        
        # Create stream
        stream_id = await self.streaming_manager.create_stream(task_id)
        
        # Verify stream creation
        assert stream_id in self.streaming_manager.active_streams
        assert task_id in self.streaming_manager.task_streams
        assert stream_id in self.streaming_manager.task_streams[task_id]
        
        # Close stream
        await self.streaming_manager.close_stream(stream_id)
        
        # Verify stream closure
        assert stream_id not in self.streaming_manager.active_streams
        assert task_id not in self.streaming_manager.task_streams
    
    @pytest.mark.asyncio
    async def test_send_task_status_update(self):
        """Test sending task status updates"""
        task_id = "test-task-123"
        stream_id = await self.streaming_manager.create_stream(task_id)
        
        # Send status update
        status = {
            "state": "working",
            "timestamp": "2024-01-01T00:00:00Z",
            "contextId": "test-context-123"
        }
        
        await self.streaming_manager.send_task_status_update(task_id, status)
        
        # Verify event was queued
        queue = self.streaming_manager.active_streams[stream_id]
        assert not queue.empty()
        
        # Get event
        event = await queue.get()
        assert event.result["taskId"] == task_id
        assert event.result["status"]["state"] == "working"
        assert event.result["kind"] == "status-update"
    
    @pytest.mark.asyncio
    async def test_send_artifact_update(self):
        """Test sending artifact updates"""
        task_id = "test-task-123"
        stream_id = await self.streaming_manager.create_stream(task_id)
        
        # Send artifact update
        artifact = {
            "artifactId": "test-artifact-123",
            "name": "response",
            "parts": [{"kind": "text", "text": "Test response"}]
        }
        
        await self.streaming_manager.send_artifact_update(task_id, artifact, append=False, last_chunk=True)
        
        # Verify event was queued
        queue = self.streaming_manager.active_streams[stream_id]
        assert not queue.empty()
        
        # Get event
        event = await queue.get()
        assert event.result["taskId"] == task_id
        assert event.result["artifact"]["artifactId"] == "test-artifact-123"
        assert event.result["kind"] == "artifact-update"
        assert event.result["lastChunk"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
