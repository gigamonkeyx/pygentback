# A2A Protocol - Core Data Models and Types

## Task Lifecycle Management

The `Task` object is the fundamental unit of work in A2A, with a sophisticated state machine:

```typescript
interface Task {
  id: string;                    // Server-generated UUID
  contextId: string;             // Logical grouping identifier
  status: TaskStatus;            // Current state + metadata
  history?: Message[];           // Optional conversation history
  artifacts?: Artifact[];        // Generated outputs
  metadata?: Record<string, any>; // Extension data
  kind: "task";                  // Type discriminator
}

enum TaskState {
  Submitted = "submitted",       // Acknowledged, not started
  Working = "working",           // Active processing
  InputRequired = "input-required", // Waiting for user input
  Completed = "completed",       // Successfully finished
  Canceled = "canceled",         // User/system canceled
  Failed = "failed",            // Error during processing
  Rejected = "rejected",        // Agent rejected request
  AuthRequired = "auth-required", // Needs additional auth
  Unknown = "unknown"           // State indeterminate
}
```

## Message & Part Type System

The protocol uses a flexible content model based on typed parts:

```typescript
type Part = TextPart | FilePart | DataPart;

interface TextPart {
  kind: "text";
  text: string;
  metadata?: Record<string, any>;
}

interface FilePart {
  kind: "file";
  file: FileWithBytes | FileWithUri;
  metadata?: Record<string, any>;
}

interface DataPart {
  kind: "data";
  mimeType: string;
  data: string;  // Base64 encoded
  metadata?: Record<string, any>;
}
```

## Agent Cards Structure

```typescript
interface AgentCard {
  apiVersion: string;
  kind: "agent";
  metadata: {
    name: string;
    displayName?: string;
    description?: string;
    version?: string;
    tags?: string[];
  };
  spec: {
    endpoints: {
      a2a: string;      // A2A protocol endpoint
      mcp?: string;     // Optional MCP endpoint
    };
    capabilities: string[];
    authentication?: AuthConfig;
  };
}
```

## Error Handling

A2A uses JSON-RPC 2.0 error format:

```typescript
interface ErrorResponse {
  jsonrpc: "2.0";
  error: {
    code: number;
    message: string;
    data?: any;
  };
  id: string | number | null;
}
```

## Implementation Guidelines

1. **Type Safety**: Use strict typing for all data models
2. **Validation**: Validate all incoming data against schemas
3. **Error Handling**: Implement comprehensive error responses
4. **Extensibility**: Support metadata fields for future extensions
5. **Performance**: Optimize for high-throughput scenarios
