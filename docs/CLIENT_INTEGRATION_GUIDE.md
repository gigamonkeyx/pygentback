# 🔌 A2A Client Integration Guide

## **Quick Start Integration**

This guide helps you integrate your application with the PyGent Factory A2A Multi-Agent System.

---

## 🚀 **Getting Started**

### **1. Prerequisites**
- A2A server running on accessible endpoint
- API credentials (if authentication enabled)
- HTTP client library in your preferred language

### **2. Basic Integration Steps**

1. **Discover Agents**: Query available agents and capabilities
2. **Send Tasks**: Create tasks for agents to process
3. **Monitor Progress**: Check task status and retrieve results
4. **Handle Errors**: Implement proper error handling

---

## 📋 **Integration Checklist**

### **✅ Pre-Integration**
- [ ] A2A server endpoint accessible
- [ ] Network connectivity verified
- [ ] Authentication credentials obtained
- [ ] Rate limiting understood
- [ ] Error handling strategy defined

### **✅ Basic Integration**
- [ ] Agent discovery implemented
- [ ] Task creation working
- [ ] Task status polling implemented
- [ ] Result retrieval working
- [ ] Error handling implemented

### **✅ Production Ready**
- [ ] Connection pooling configured
- [ ] Retry logic implemented
- [ ] Timeout handling added
- [ ] Logging and monitoring setup
- [ ] Load testing completed

---

## 🔧 **Implementation Patterns**

### **Pattern 1: Simple Request-Response**
```python
async def simple_search(query: str) -> dict:
    """Simple synchronous-style search"""
    
    # 1. Send task
    task_response = await send_task({
        "role": "user",
        "parts": [{"type": "text", "text": f"Search for {query}"}]
    })
    
    task_id = task_response["result"]["id"]
    
    # 2. Poll for completion
    while True:
        status = await get_task_status(task_id)
        if status["status"]["state"] == "completed":
            return status["artifacts"]
        elif status["status"]["state"] == "failed":
            raise Exception("Task failed")
        
        await asyncio.sleep(1)
```

### **Pattern 2: Async with Callbacks**
```python
class A2AAsyncClient:
    def __init__(self):
        self.callbacks = {}
    
    async def send_task_async(self, message: dict, callback: callable):
        """Send task with callback for completion"""
        
        task_response = await self.send_task(message)
        task_id = task_response["result"]["id"]
        
        # Store callback
        self.callbacks[task_id] = callback
        
        # Start monitoring in background
        asyncio.create_task(self._monitor_task(task_id))
        
        return task_id
    
    async def _monitor_task(self, task_id: str):
        """Background task monitoring"""
        while True:
            status = await self.get_task_status(task_id)
            state = status["status"]["state"]
            
            if state in ["completed", "failed", "canceled"]:
                callback = self.callbacks.pop(task_id, None)
                if callback:
                    await callback(task_id, status)
                break
            
            await asyncio.sleep(1)
```

### **Pattern 3: Batch Processing**
```python
async def batch_process(queries: list) -> list:
    """Process multiple queries concurrently"""
    
    # Send all tasks
    tasks = []
    for query in queries:
        task_response = await send_task({
            "role": "user",
            "parts": [{"type": "text", "text": query}]
        })
        tasks.append(task_response["result"]["id"])
    
    # Wait for all completions
    results = []
    for task_id in tasks:
        while True:
            status = await get_task_status(task_id)
            if status["status"]["state"] == "completed":
                results.append(status["artifacts"])
                break
            elif status["status"]["state"] == "failed":
                results.append({"error": "Task failed"})
                break
            
            await asyncio.sleep(0.5)
    
    return results
```

---

## 🛡️ **Error Handling Best Practices**

### **1. Network Error Handling**
```python
import aiohttp
from aiohttp import ClientTimeout, ClientError

async def robust_request(url: str, data: dict, max_retries: int = 3):
    """Robust request with retry logic"""
    
    timeout = ClientTimeout(total=30)
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        response.raise_for_status()
        
        except (ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    
    raise Exception(f"Failed after {max_retries} attempts")
```

### **2. Task Error Handling**
```python
async def handle_task_errors(task_id: str) -> dict:
    """Handle various task error scenarios"""
    
    try:
        status = await get_task_status(task_id)
        state = status["status"]["state"]
        
        if state == "completed":
            return status["artifacts"]
        elif state == "failed":
            error_msg = status.get("status", {}).get("message", "Unknown error")
            raise TaskFailedException(f"Task failed: {error_msg}")
        elif state == "canceled":
            raise TaskCanceledException("Task was canceled")
        elif state == "input-required":
            raise TaskInputRequiredException("Task requires additional input")
        else:
            raise TaskUnknownStateException(f"Unknown task state: {state}")
    
    except aiohttp.ClientError as e:
        raise NetworkException(f"Network error: {e}")
    except Exception as e:
        raise UnexpectedErrorException(f"Unexpected error: {e}")
```

---

## ⚡ **Performance Optimization**

### **1. Connection Pooling**
```python
class OptimizedA2AClient:
    def __init__(self, base_url: str, max_connections: int = 100):
        self.base_url = base_url
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def close(self):
        await self.session.close()
```

### **2. Request Batching**
```python
class BatchingA2AClient:
    def __init__(self, batch_size: int = 10, batch_timeout: float = 1.0):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests = []
        self.batch_timer = None
    
    async def send_request(self, request: dict) -> dict:
        """Add request to batch"""
        future = asyncio.Future()
        self.pending_requests.append((request, future))
        
        if len(self.pending_requests) >= self.batch_size:
            await self._flush_batch()
        elif self.batch_timer is None:
            self.batch_timer = asyncio.create_task(
                self._batch_timeout_handler()
            )
        
        return await future
    
    async def _flush_batch(self):
        """Send batched requests"""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests[:]
        self.pending_requests.clear()
        
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        # Send batch request
        batch_request = [req for req, _ in batch]
        responses = await self._send_batch(batch_request)
        
        # Resolve futures
        for (_, future), response in zip(batch, responses):
            future.set_result(response)
```

---

## 📊 **Monitoring and Observability**

### **1. Request Logging**
```python
import logging
import time

logger = logging.getLogger(__name__)

async def logged_request(method: str, params: dict) -> dict:
    """Request with comprehensive logging"""
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"[{request_id}] Starting {method} request", extra={
        "request_id": request_id,
        "method": method,
        "params": params
    })
    
    try:
        response = await send_jsonrpc_request(method, params)
        duration = time.time() - start_time
        
        logger.info(f"[{request_id}] Request completed in {duration:.3f}s", extra={
            "request_id": request_id,
            "duration": duration,
            "success": True
        })
        
        return response
    
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error(f"[{request_id}] Request failed after {duration:.3f}s: {e}", extra={
            "request_id": request_id,
            "duration": duration,
            "error": str(e),
            "success": False
        })
        
        raise
```

### **2. Metrics Collection**
```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
REQUEST_COUNT = Counter('a2a_requests_total', 'Total A2A requests', ['method', 'status'])
REQUEST_DURATION = Histogram('a2a_request_duration_seconds', 'A2A request duration')
ACTIVE_TASKS = Gauge('a2a_active_tasks', 'Number of active tasks')

async def monitored_request(method: str, params: dict) -> dict:
    """Request with metrics collection"""
    
    with REQUEST_DURATION.time():
        try:
            response = await send_jsonrpc_request(method, params)
            REQUEST_COUNT.labels(method=method, status='success').inc()
            
            if method == 'tasks/send':
                ACTIVE_TASKS.inc()
            elif method == 'tasks/get' and response.get('result', {}).get('status', {}).get('state') in ['completed', 'failed', 'canceled']:
                ACTIVE_TASKS.dec()
            
            return response
        
        except Exception as e:
            REQUEST_COUNT.labels(method=method, status='error').inc()
            raise
```

---

## 🔒 **Security Best Practices**

### **1. Secure Configuration**
```python
import os
from cryptography.fernet import Fernet

class SecureA2AClient:
    def __init__(self):
        # Load from environment
        self.api_key = os.getenv('A2A_API_KEY')
        self.base_url = os.getenv('A2A_BASE_URL')
        
        # Encrypt sensitive data
        self.cipher = Fernet(os.getenv('A2A_ENCRYPTION_KEY').encode())
        
        if not all([self.api_key, self.base_url]):
            raise ValueError("Missing required environment variables")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data before transmission"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt received sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
```

### **2. Input Validation**
```python
from typing import Dict, Any
import re

def validate_task_input(message: Dict[str, Any]) -> bool:
    """Validate task input for security"""
    
    # Check required fields
    if not isinstance(message, dict):
        raise ValueError("Message must be a dictionary")
    
    if 'role' not in message or 'parts' not in message:
        raise ValueError("Message must have 'role' and 'parts' fields")
    
    # Validate role
    if message['role'] not in ['user', 'agent']:
        raise ValueError("Role must be 'user' or 'agent'")
    
    # Validate parts
    if not isinstance(message['parts'], list):
        raise ValueError("Parts must be a list")
    
    for part in message['parts']:
        if not isinstance(part, dict) or 'type' not in part:
            raise ValueError("Each part must have a 'type' field")
        
        if part['type'] == 'text' and 'text' in part:
            # Sanitize text input
            text = part['text']
            if len(text) > 10000:  # Limit text length
                raise ValueError("Text too long (max 10000 characters)")
            
            # Check for suspicious patterns
            if re.search(r'<script|javascript:|data:', text, re.IGNORECASE):
                raise ValueError("Potentially malicious content detected")
    
    return True
```

---

## 🧪 **Testing Your Integration**

### **1. Unit Tests**
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_successful_task_creation():
    """Test successful task creation"""
    
    mock_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "id": "test-task-id",
            "status": {"state": "submitted"}
        }
    }
    
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aenter__.return_value.status = 200
        
        client = A2AClient()
        result = await client.send_task({
            "role": "user",
            "parts": [{"type": "text", "text": "test query"}]
        })
        
        assert result["result"]["id"] == "test-task-id"
        assert result["result"]["status"]["state"] == "submitted"
```

### **2. Integration Tests**
```python
@pytest.mark.asyncio
async def test_end_to_end_document_search():
    """Test complete document search workflow"""
    
    client = A2AClient("http://localhost:8080")
    
    # Send search task
    task_response = await client.send_task({
        "role": "user",
        "parts": [{"type": "text", "text": "Search for machine learning papers"}]
    })
    
    task_id = task_response["result"]["id"]
    assert task_id is not None
    
    # Wait for completion
    max_wait = 30
    for _ in range(max_wait):
        status = await client.get_task(task_id)
        state = status["result"]["status"]["state"]
        
        if state == "completed":
            artifacts = status["result"]["artifacts"]
            assert len(artifacts) > 0
            assert artifacts[0]["name"] == "search_results"
            break
        elif state == "failed":
            pytest.fail("Task failed")
        
        await asyncio.sleep(1)
    else:
        pytest.fail("Task did not complete within timeout")
```

---

## 📚 **Additional Resources**

- **API Reference**: [A2A_API_DOCUMENTATION.md](./A2A_API_DOCUMENTATION.md)
- **Example Applications**: [examples/](../examples/)
- **SDK Documentation**: [sdks/](../sdks/)
- **Troubleshooting**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- **Performance Guide**: [PERFORMANCE_GUIDE.md](./PERFORMANCE_GUIDE.md)
