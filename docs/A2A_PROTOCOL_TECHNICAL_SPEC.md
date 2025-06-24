# A2A Protocol Technical Specification

## Core RPC Methods

### Message Methods

| Method | Purpose | Input | Output | Notes |
|--------|---------|--------|--------|-------|
| `message/send` | Send message, create/continue task | MessageSendParams | Task or Message | Synchronous or polling |
| `message/stream` | Send message with real-time updates | MessageSendParams | SSE Stream | Requires streaming capability |

### Task Management Methods

| Method | Purpose | Input | Output | Notes |
|--------|---------|--------|--------|-------|
| `tasks/get` | Retrieve task status and results | TaskQueryParams | Task | For polling workflows |
| `tasks/cancel` | Request task cancellation | TaskIdParams | Task | Not always successful |
| `tasks/resubscribe` | Reconnect to SSE stream | TaskIdParams | SSE Stream | Resume after disconnection |

### Push Notification Methods

| Method | Purpose | Input | Output | Notes |
|--------|---------|--------|--------|-------|
| `tasks/pushNotificationConfig/set` | Configure webhook notifications | TaskPushNotificationConfig | TaskPushNotificationConfig | Requires push capability |
| `tasks/pushNotificationConfig/get` | Retrieve webhook configuration | TaskIdParams | TaskPushNotificationConfig | Configuration inspection |

## Agent Discovery Endpoints

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/.well-known/agent.json` | GET | Public agent card | No |
| `/agent/authenticatedExtendedCard` | GET | Detailed agent card | Yes |

## Complete Type Definitions

### Task Status
```typescript
interface TaskStatus {
  state: "working" | "completed" | "failed" | "cancelled" | "auth-required";
  message?: Message;
  progress?: number;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}
```

### Artifact Types
```typescript
interface Artifact {
  id: string;
  name?: string;
  mimeType: string;
  bytes?: string;      // Base64 encoded
  url?: string;        // External reference
  metadata?: Record<string, any>;
}
```

### Message Parts
```typescript
type Part = TextPart | FilePart | DataPart;

interface TextPart {
  kind: "text";
  text: string;
  metadata?: Record<string, any>;
}

interface FilePart {
  kind: "file";
  file: {
    mimeType: string;
    bytes?: string;    // Base64 encoded
    url?: string;      // External reference
    name?: string;
  };
  metadata?: Record<string, any>;
}

interface DataPart {
  kind: "data";
  data: Record<string, any>;
  metadata?: Record<string, any>;
}
```

### Agent Capabilities
```typescript
interface AgentCapabilities {
  streaming?: boolean;           // SSE support
  pushNotifications?: boolean;   // Webhook support
  contextPreservation?: boolean; // Context management
  extensions?: AgentExtension[]; // Protocol extensions
}

interface AgentSkill {
  name: string;
  description: string;
  inputModes?: string[];   // Accepted media types
  outputModes?: string[];  // Produced media types
  parameters?: Record<string, any>; // Skill configuration
}
```

### Security Schemes
```typescript
interface SecurityScheme {
  type: "http" | "oauth2" | "apiKey";
  scheme?: string;        // For HTTP auth
  bearerFormat?: string;  // JWT, etc.
  flows?: OAuth2Flows;    // For OAuth2
  in?: "query" | "header" | "cookie"; // For API keys
  name?: string;          // Parameter name
}

interface OAuth2Flows {
  authorizationCode?: {
    authorizationUrl: string;
    tokenUrl: string;
    refreshUrl?: string;
    scopes: Record<string, string>;
  };
  clientCredentials?: {
    tokenUrl: string;
    refreshUrl?: string;
    scopes: Record<string, string>;
  };
}
```

## Error Handling

### Standard JSON-RPC Errors (-32700 to -32603)
- Parse errors, invalid requests, method not found, invalid params, internal errors

### A2A-Specific Errors (-32001 to -32099)
- **TaskNotFoundError (-32001)**: Task ID doesn't exist or expired
- **TaskNotCancelableError (-32002)**: Task in non-cancelable state
- **PushNotificationNotSupportedError (-32003)**: Feature not available
- **UnsupportedOperationError (-32004)**: Operation not supported
- **ContentTypeNotSupportedError (-32005)**: Media type incompatible
- **InvalidAgentResponseError (-32006)**: Malformed agent response

### Error Response Structure
```typescript
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32001,
    "message": "Task not found",
    "data": {
      "taskId": "missing-task-uuid",
      "details": "Task may have expired or been purged",
      "retryAfter": 300
    }
  }
}
```

## Streaming Architecture

### Server-Sent Events
```typescript
// SSE Event Types
type StreamingResponse = 
  | TaskStatusUpdateEvent
  | TaskArtifactUpdateEvent
  | Message
  | Task;

interface TaskArtifactUpdateEvent {
  taskId: string;
  kind: "artifact-update";
  artifact: Artifact;
  append?: boolean;      // Append vs replace
  lastChunk?: boolean;   // Final chunk indicator
}
```

### Stream Management
1. Client initiates with `message/stream`
2. Server responds with `200 OK` + `text/event-stream`
3. Server sends incremental updates as SSE events
4. Server closes connection with `final: true` status update
5. Client can reconnect with `tasks/resubscribe`

## Extension Mechanism

```typescript
interface AgentExtension {
  uri: string;           // Extension identifier
  description?: string;  // Human-readable description
  required?: boolean;    // Client compatibility requirement
  params?: Record<string, any>; // Extension configuration
}
```

### Extension Categories
- Protocol enhancements (new methods/objects)
- Content type extensions (new Part types)
- Authentication extensions (new security schemes)
- Behavioral extensions (new task states/transitions)

## Implementation Guidelines

### Client Implementation
- Cache public agent cards with appropriate TTL
- Implement exponential backoff for retries
- Handle network partitions gracefully
- Limit concurrent streaming connections

### Server Implementation
- Use persistent storage for task state
- Implement task TTL and cleanup policies
- Validate all inputs against schemas
- Use secure random generators for IDs

### Security Implementation
- Implement proper CORS policies
- Audit all authentication/authorization decisions
- Use connection pooling for outbound requests
- Optimize JSON serialization/deserialization
