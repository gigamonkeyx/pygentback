# A2A Protocol Methods

## Overview

This document provides detailed specifications for all A2A Protocol v0.2.1 methods, communication patterns, and interaction models.

## Communication Patterns

### 1. Synchronous Request/Response

**Pattern**: Direct message exchange with immediate or polling-based responses.

**Flow**:
```typescript
// Client sends message
POST /a2a/v1
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Process this request"}],
      "messageId": "uuid-1"
    }
  }
}

// Server responds with completed task or working status
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "id": "task-uuid",
    "status": {"state": "completed"},
    "artifacts": [/* results */]
  }
}
```

**Use Cases**:
- Simple queries and fast computations
- Stateless operations
- Immediate response scenarios
- Resource lookups

### 2. Streaming with Server-Sent Events

**Pattern**: Real-time updates via persistent HTTP connection.

**Flow**:
```typescript
// Client initiates stream
POST /a2a/v1
{
  "method": "message/stream",
  "params": { 
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Generate a long report"}]
    }
  }
}

// Server responds with SSE stream
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"jsonrpc": "2.0", "id": 1, "result": {"kind": "status-update", "status": {"state": "working"}}}

data: {"jsonrpc": "2.0", "id": 1, "result": {"kind": "artifact-update", "artifact": {"content": "partial result"}}}

data: {"jsonrpc": "2.0", "id": 1, "result": {"kind": "status-update", "status": {"state": "completed"}, "final": true}}
```

**Use Cases**:
- Long-running tasks with incremental results
- Real-time monitoring and progress updates
- Live collaboration scenarios
- Streaming data processing

### 3. Asynchronous Push Notifications

**Pattern**: Webhook-based notifications for disconnected scenarios.

**Setup Flow**:
```typescript
// Client configures webhook
{
  "method": "tasks/pushNotificationConfig/set",
  "params": {
    "taskId": "task-uuid",
    "pushNotificationConfig": {
      "url": "https://client.com/webhook",
      "token": "validation-token",
      "authentication": {
        "schemes": ["Bearer"],
        "credentials": "optional-auth-data"
      }
    }
  }
}
```

**Notification Flow**:
```typescript
// Server POSTs to webhook when task completes
POST https://client.com/webhook
Authorization: Bearer <server-jwt>
X-A2A-Notification-Token: validation-token
Content-Type: application/json

{
  "id": "task-uuid",
  "status": {"state": "completed"},
  "artifacts": [{"type": "result", "content": "final output"}]
}
```

**Use Cases**:
- Very long-running tasks (hours/days)
- Mobile and offline clients
- System integration scenarios
- Batch processing workflows

## Core RPC Methods

### message/send

**Purpose**: Send a message to create a new task or continue an existing conversation.

**Parameters**:
```typescript
interface MessageSendParams {
  message: Message;
  contextId?: string;
}
```

**Returns**: `Task` or `Message`

**Examples**:
```typescript
// Create new task
{
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Analyze this data"}]
    }
  }
}

// Continue existing conversation
{
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Can you elaborate on point 3?"}]
    },
    "contextId": "existing-context-id"
  }
}
```

### message/stream

**Purpose**: Send a message with real-time streaming updates.

**Parameters**: Same as `message/send`

**Returns**: Server-Sent Events stream

**Stream Events**:
- `status-update`: Task status changes
- `artifact-update`: Incremental results
- `message-update`: Conversation updates

### tasks/get

**Purpose**: Retrieve task status and results for polling workflows.

**Parameters**:
```typescript
interface TaskQueryParams {
  taskId: string;
}
```

**Returns**: `Task`

**Example**:
```typescript
{
  "method": "tasks/get",
  "params": {
    "taskId": "task-uuid-123"
  }
}
```

### tasks/cancel

**Purpose**: Request task cancellation (not always successful).

**Parameters**:
```typescript
interface TaskIdParams {
  taskId: string;
}
```

**Returns**: `Task`

**Note**: Cancellation is a request, not a guarantee. Check returned task status.

### tasks/pushNotificationConfig/set

**Purpose**: Configure webhook notifications for a task.

**Parameters**:
```typescript
interface TaskPushNotificationConfig {
  taskId: string;
  pushNotificationConfig: {
    url: string;
    token?: string;
    authentication?: {
      schemes: string[];
      credentials?: string;
    };
  };
}
```

**Returns**: `TaskPushNotificationConfig`

### tasks/pushNotificationConfig/get

**Purpose**: Retrieve current webhook configuration.

**Parameters**: `TaskIdParams`

**Returns**: `TaskPushNotificationConfig`

### tasks/resubscribe

**Purpose**: Reconnect to an SSE stream after disconnection.

**Parameters**: `TaskIdParams`

**Returns**: Server-Sent Events stream

**Use Case**: Resume streaming after network interruption

## Agent Discovery Endpoints

### Public Agent Card

**Endpoint**: `GET /.well-known/agent.json`

**Authentication**: None required

**Returns**: Basic `AgentCard`

**Purpose**: Public service discovery

**Example Response**:
```json
{
  "name": "PyGent Factory Agent",
  "description": "Self-improving multi-modal AI agent",
  "url": "https://pygent.example.com/a2a/v1",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true
  },
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["text/plain", "application/json"],
  "skills": ["analysis", "generation", "evolution"]
}
```

### Authenticated Extended Card

**Endpoint**: `GET /agent/authenticatedExtendedCard`

**Authentication**: Required

**Returns**: Detailed `AgentCard` with additional capabilities

**Purpose**: Authenticated clients get more detailed information

## Error Handling

### Error Response Format

```typescript
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32600,
    "message": "Invalid Request",
    "data": {
      "details": "Additional error information"
    }
  }
}
```

### Standard Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32700 | Parse error | Invalid JSON |
| -32600 | Invalid Request | Invalid JSON-RPC |
| -32601 | Method not found | Unknown method |
| -32602 | Invalid params | Invalid parameters |
| -32603 | Internal error | Server error |
| -32000 | Server error | Implementation error |

### A2A-Specific Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32001 | Task not found | Unknown task ID |
| -32002 | Authentication required | Missing auth |
| -32003 | Authorization failed | Insufficient permissions |
| -32004 | Rate limit exceeded | Too many requests |
| -32005 | Resource unavailable | Temporary unavailability |

## Implementation Guidelines

### Client Implementation

1. **Error Handling**: Implement comprehensive error handling for all error codes
2. **Retry Logic**: Use exponential backoff for retryable errors
3. **Connection Management**: Handle SSE disconnections gracefully
4. **Authentication**: Implement token refresh for OAuth flows

### Server Implementation

1. **Validation**: Validate all input parameters strictly
2. **Security**: Implement proper authentication and authorization
3. **Resource Management**: Enforce rate limits and resource constraints
4. **Monitoring**: Log all requests and responses for debugging

### Performance Considerations

1. **Connection Pooling**: Reuse HTTP connections when possible
2. **Streaming Efficiency**: Use efficient SSE implementation
3. **Memory Management**: Handle large files and datasets efficiently
4. **Caching**: Cache agent cards and other frequently accessed data

## Integration with PyGent Factory

### Existing Infrastructure Usage

- **WebSocket to SSE**: Convert WebSocket infrastructure to SSE
- **Agent Registry**: Extend for agent card generation
- **MCP Integration**: Map MCP tools to A2A skills
- **Security Framework**: Integrate with existing authentication

### Implementation Strategy

1. Implement core method handlers
2. Add agent card generation
3. Create SSE streaming infrastructure
4. Integrate webhook notification system
5. Add comprehensive error handling

## Related Documents

- [A2A Protocol Architecture](A2A_PROTOCOL_ARCHITECTURE.md)
- [A2A Protocol Security](A2A_PROTOCOL_SECURITY.md)
- [A2A Protocol Implementation Guide](A2A_PROTOCOL_IMPLEMENTATION_GUIDE.md)
- [A2A Protocol Technical Specification](A2A_PROTOCOL_TECHNICAL_SPEC.md)
