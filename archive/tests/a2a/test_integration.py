import pytest
from fastapi.testclient import TestClient

from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

# Use sync client for testing since we have sync endpoints
def test_agent_card_endpoint(client):
    """Test agent card endpoint"""
    response = client.get("/a2a/v1/.well-known/agent.json")
    assert response.status_code == 200
    
    data = response.json()
    assert data["kind"] == "agent"
    assert data["apiVersion"] == "a2a/v1"
    assert "metadata" in data
    assert "spec" in data

def test_message_send(client):
    """Test message send endpoint"""
    message_data = {
        "contextId": "test-context",
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Hello, agent!"
                }
            ]
        }
    }
    
    response = client.post("/a2a/v1/message/send", json=message_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["kind"] == "task"
    assert "id" in data
    assert data["contextId"] == "test-context"

def test_task_retrieval(client):
    """Test task retrieval"""
    # First send a message to create a task
    message_data = {
        "contextId": "test-context",
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Test message"}]
        }
    }
    
    send_response = client.post("/a2a/v1/message/send", json=message_data)
    task_data = send_response.json()
    task_id = task_data["id"]
    
    # Wait a moment for processing
    import time
    time.sleep(0.2)
    
    # Retrieve the task
    get_response = client.get(f"/a2a/v1/tasks/{task_id}")
    assert get_response.status_code == 200
    
    retrieved_task = get_response.json()
    assert retrieved_task["id"] == task_id
    assert retrieved_task["contextId"] == "test-context"

def test_specific_agent_card_endpoint(client):
    """Test specific agent card endpoint"""
    response = client.get("/a2a/v1/agents/test-agent/card")
    assert response.status_code == 200
    
    data = response.json()
    assert data["kind"] == "agent"
    assert data["apiVersion"] == "a2a/v1"

def test_message_stream_creation(client):
    """Test message stream endpoint"""
    message_data = {
        "contextId": "test-stream-context",
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "text", 
                    "text": "Stream this message"
                }
            ]
        }
    }
    
    response = client.post("/a2a/v1/message/stream", json=message_data)
    # Stream endpoints may return different status codes
    assert response.status_code in [200, 202]
