# 📡 A2A Multi-Agent System API Documentation

## **Overview**

The PyGent Factory A2A (Agent-to-Agent) Multi-Agent System provides a production-ready implementation of Google's A2A protocol for agent communication and coordination.

**Base URL**: `http://localhost:8080` (Development) | `https://api.timpayne.net` (Production)  
**Protocol**: JSON-RPC 2.0  
**Content-Type**: `application/json`

---

## 🔗 **Core Endpoints**

### **1. Agent Discovery**
Discover available agents and their capabilities.

```http
GET /.well-known/agent.json
```

**Response:**
```json
{
  "name": "ProductionResearchAgent",
  "description": "PyGent Factory Research Agent",
  "version": "1.0.0",
  "url": "http://localhost:8080",
  "provider": {
    "organization": "PyGent Factory",
    "url": "https://github.com/gigamonkeyx/pygentback"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateTransitionHistory": true
  },
  "skills": [
    {
      "id": "document_search",
      "name": "Document Search",
      "description": "Search and retrieve documents from various sources",
      "tags": ["search", "documents", "retrieval"],
      "examples": [
        "Search for papers about quantum computing",
        "Find documents related to machine learning"
      ]
    }
  ]
}
```

### **2. Health Check**
Monitor system health and status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-22T20:30:00.000Z",
  "agents_registered": 2,
  "tasks_active": 5,
  "uptime": 3600,
  "version": "1.0.0"
}
```

### **3. List Agents**
Get all registered agents.

```http
GET /agents
```

**Response:**
```json
[
  {
    "agent_id": "8502a47a-aef7-41b9-93bc-fd256bfed1d5",
    "name": "ProductionResearchAgent",
    "type": "AgentType.RESEARCH",
    "url": "http://localhost:8080/agents/8502a47a-aef7-41b9-93bc-fd256bfed1d5",
    "status": "AgentStatus.IDLE"
  }
]
```

---

## 🔄 **JSON-RPC Methods**

### **1. tasks/send**
Create and send a task to an agent.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "Search for documents about machine learning"
        }
      ],
      "metadata": {
        "priority": "high",
        "timeout": 30
      }
    },
    "sessionId": "optional-session-id"
  },
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "id": "b89de568-1234-5678-9abc-def012345678",
    "sessionId": "session-uuid",
    "status": {
      "state": "submitted",
      "timestamp": "2025-06-22T20:30:00.000Z"
    },
    "artifacts": [],
    "metadata": {
      "agent_url": "http://localhost:8080/agents/research"
    }
  }
}
```

### **2. tasks/get**
Retrieve task status and results.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/get",
  "params": {
    "id": "b89de568-1234-5678-9abc-def012345678"
  },
  "id": 2
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "id": "b89de568-1234-5678-9abc-def012345678",
    "sessionId": "session-uuid",
    "status": {
      "state": "completed",
      "timestamp": "2025-06-22T20:30:05.000Z"
    },
    "artifacts": [
      {
        "name": "search_results",
        "description": "Document search results",
        "parts": [
          {
            "type": "text",
            "text": "{\"search_method\": \"database_search\", \"total_found\": 5, \"documents\": [...]}"
          }
        ],
        "metadata": {
          "agent_id": "8502a47a-aef7-41b9-93bc-fd256bfed1d5",
          "agent_name": "ProductionResearchAgent"
        }
      }
    ],
    "history": [
      {
        "role": "user",
        "parts": [{"type": "text", "text": "Search for documents about machine learning"}]
      }
    ]
  }
}
```

### **3. tasks/cancel**
Cancel a running task.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/cancel",
  "params": {
    "id": "b89de568-1234-5678-9abc-def012345678"
  },
  "id": 3
}
```

---

## 📊 **Task States**

| State | Description |
|-------|-------------|
| `submitted` | Task has been created and queued |
| `working` | Agent is actively processing the task |
| `input-required` | Task requires additional input from user |
| `completed` | Task has been completed successfully |
| `canceled` | Task was canceled by user request |
| `failed` | Task failed due to an error |

---

## 🛡️ **Error Handling**

### **Standard JSON-RPC Errors**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "details": "Missing required parameter: message"
    }
  }
}
```

### **Common Error Codes**

| Code | Message | Description |
|------|---------|-------------|
| -32700 | Parse error | Invalid JSON |
| -32600 | Invalid Request | Invalid JSON-RPC request |
| -32601 | Method not found | Method does not exist |
| -32602 | Invalid params | Invalid method parameters |
| -32603 | Internal error | Server internal error |

---

## 🔐 **Authentication**

### **Bearer Token Authentication**
```http
Authorization: Bearer your-api-token
```

### **API Key Authentication**
```http
X-API-Key: your-api-key
```

---

## 📈 **Rate Limiting**

- **Default Limit**: 100 requests per minute
- **Burst Limit**: 200 requests per minute
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

---

## 🌐 **CORS Configuration**

**Allowed Origins**: 
- `https://api.timpayne.net`
- `https://timpayne.net`
- `http://localhost:3000` (Development)

**Allowed Methods**: `GET`, `POST`, `OPTIONS`  
**Allowed Headers**: `Content-Type`, `Authorization`, `X-API-Key`

---

## 📝 **Usage Examples**

### **Python Client Example**
```python
import aiohttp
import asyncio
import json

async def search_documents(query: str):
    async with aiohttp.ClientSession() as session:
        # Send task
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": f"Search for documents about {query}"}]
                }
            },
            "id": 1
        }
        
        async with session.post("http://localhost:8080/", json=request) as response:
            result = await response.json()
            task_id = result["result"]["id"]
        
        # Wait and get results
        await asyncio.sleep(2)
        
        get_request = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
            "id": 2
        }
        
        async with session.post("http://localhost:8080/", json=get_request) as response:
            result = await response.json()
            return result["result"]

# Usage
results = asyncio.run(search_documents("machine learning"))
print(json.dumps(results, indent=2))
```

### **JavaScript Client Example**
```javascript
class A2AClient {
    constructor(baseUrl = 'http://localhost:8080') {
        this.baseUrl = baseUrl;
    }
    
    async sendTask(message) {
        const request = {
            jsonrpc: '2.0',
            method: 'tasks/send',
            params: { message },
            id: Date.now()
        };
        
        const response = await fetch(this.baseUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request)
        });
        
        return await response.json();
    }
    
    async getTask(taskId) {
        const request = {
            jsonrpc: '2.0',
            method: 'tasks/get',
            params: { id: taskId },
            id: Date.now()
        };
        
        const response = await fetch(this.baseUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request)
        });
        
        return await response.json();
    }
}

// Usage
const client = new A2AClient();
const task = await client.sendTask({
    role: 'user',
    parts: [{ type: 'text', text: 'Search for AI research papers' }]
});
console.log('Task created:', task.result.id);
```

---

## 🔧 **SDK and Libraries**

### **Official SDKs**
- **Python**: `pip install pygent-a2a-client`
- **JavaScript**: `npm install @pygent/a2a-client`
- **Go**: `go get github.com/pygent-factory/a2a-go-client`

### **Community Libraries**
- **Java**: `com.pygent:a2a-java-client`
- **C#**: `PyGent.A2A.Client`
- **Rust**: `pygent-a2a-client`

---

## 📞 **Support**

- **Documentation**: https://docs.timpayne.net/a2a
- **API Status**: https://status.timpayne.net
- **GitHub**: https://github.com/gigamonkeyx/pygentback
- **Issues**: https://github.com/gigamonkeyx/pygentback/issues
