# A2A Protocol Architecture

## Overview

This document provides a detailed technical analysis of the Google Agent2Agent (A2A) Protocol v0.2.1 architecture, design principles, and core data models.

## Executive Summary

The A2A protocol represents a significant advancement in agent-to-agent communication standards, providing a robust, enterprise-ready framework for enabling seamless collaboration between independent AI agents.

**Key Characteristics:**
- Built on established web standards (HTTP, JSON-RPC 2.0, Server-Sent Events)
- Emphasizes security, scalability, and enterprise integration
- Supports multiple interaction patterns (sync, async, streaming, push notifications)
- Maintains opaque agent execution while ensuring interoperability
- Clear separation between agent capabilities (MCP) and communication (A2A)

## Core Design Principles

The A2A protocol is built on six fundamental design principles:

### 1. Simplicity
- Reuses existing, well-understood standards
- HTTP/HTTPS for transport
- JSON-RPC 2.0 for method invocation
- Server-Sent Events for streaming

### 2. Enterprise Ready
- Comprehensive authentication and authorization
- Built-in security, privacy, and monitoring
- Tracing and observability support
- Production deployment considerations

### 3. Async First
- Designed for long-running tasks
- Human-in-the-loop interaction support
- Non-blocking communication patterns
- Task lifecycle management

### 4. Modality Agnostic
- Text, audio, video support
- Structured data handling
- File attachment capabilities
- Embedded UI components

### 5. Opaque Execution
- Agents collaborate without sharing internals
- No exposure of internal thoughts or plans
- Tool implementations remain private
- Focus on inputs and outputs

### 6. Interoperability
- Bridges communication between disparate systems
- Standard protocol enabling ecosystem growth
- Compatible with existing agent frameworks
- Future-proof extensibility

## Protocol Stack Architecture

```
┌─────────────────────────────────────────────────────┐
│                Application Layer                     │
│  Agent Logic, Skills, Business Rules               │
├─────────────────────────────────────────────────────┤
│                A2A Protocol Layer                    │
│  Tasks, Messages, Artifacts, Streaming, Push       │
├─────────────────────────────────────────────────────┤
│              JSON-RPC 2.0 Layer                     │
│  Request/Response Structure, Error Handling        │
├─────────────────────────────────────────────────────┤
│               Transport Layer                        │
│  HTTP/HTTPS, Server-Sent Events, WebHooks          │
├─────────────────────────────────────────────────────┤
│               Security Layer                         │
│  TLS, OAuth 2.0, API Keys, Authorization           │
└─────────────────────────────────────────────────────┘
```

### Layer Responsibilities

**Application Layer**:
- Agent-specific business logic
- Skill implementations
- Domain-specific processing

**A2A Protocol Layer**:
- Task lifecycle management
- Message routing and processing
- Artifact handling
- Streaming coordination

**JSON-RPC 2.0 Layer**:
- Method invocation structure
- Parameter validation
- Error handling and reporting
- Request/response correlation

**Transport Layer**:
- HTTP/HTTPS communication
- Server-Sent Events streaming
- WebHook notifications
- Connection management

**Security Layer**:
- TLS encryption
- Authentication mechanisms
- Authorization policies
- Certificate management

## Core Data Models

### Task Lifecycle Management

The `Task` object is the fundamental unit of work in A2A:

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
```

### Task State Machine

```typescript
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

**Key Features**:
- Comprehensive coverage of real-world scenarios
- Pause states for human-in-the-loop workflows
- Explicit error states for robust error handling
- Clear progression through task lifecycle

### Message and Part Type System

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
  data: Record<string, any>;
  metadata?: Record<string, any>;
}
```

**Benefits**:
- Rich, multimodal communication support
- Type safety with discriminated unions
- Extensible metadata system
- Developer-friendly API design

### Agent Discovery System

The Agent Card serves as A2A's service discovery mechanism:

```typescript
interface AgentCard {
  name: string;                          // Human-readable name
  description: string;                   // Capability description
  url: string;                          // Service endpoint
  version: string;                      // Implementation version
  capabilities: AgentCapabilities;       // Optional features
  securitySchemes?: SecurityScheme[];    // Auth requirements
  security?: SecurityRequirement[];     // Required auth schemes
  defaultInputModes: string[];          // Accepted media types
  defaultOutputModes: string[];         // Produced media types
  skills: AgentSkill[];                 // Available capabilities
  supportsAuthenticatedExtendedCard?: boolean; // Extended card support
}
```

**Key Capabilities**:
- Comprehensive service metadata
- Dynamic discovery and capability negotiation
- Secure communication setup
- Fine-grained feature detection
- Separation of capabilities from skills

## Protocol Extension Points

### Custom Capabilities
- Agent-specific features beyond core A2A
- Extensible capability negotiation
- Backward compatibility maintenance

### Metadata Extensions
- Custom metadata on tasks, messages, and parts
- Domain-specific information transport
- Tool-specific data handling

### Security Scheme Extensions
- Custom authentication mechanisms
- Domain-specific authorization
- Enterprise security integration

## Implementation Considerations

### Performance Optimization
- Connection pooling and reuse
- Efficient streaming implementation
- Resource management for long-running tasks
- Caching strategies for agent cards

### Scalability Patterns
- Horizontal scaling of agent endpoints
- Load balancing across agent instances
- Distributed task routing
- State management in clustered deployments

### Error Resilience
- Comprehensive error handling
- Graceful degradation strategies
- Retry mechanisms with backoff
- Circuit breaker patterns

## Integration with PyGent Factory

### Existing Infrastructure Alignment
- Leverage WebSocket infrastructure for SSE
- Extend agent registry for card generation
- Utilize MCP integration for tool capabilities
- Build on existing security framework

### Implementation Strategy
- Incremental protocol adoption
- Backward compatibility maintenance
- Gradual feature enablement
- Comprehensive testing approach

## Next Steps

1. Implement core protocol handlers
2. Create agent card generation system
3. Build message routing infrastructure
4. Integrate with existing orchestration
5. Develop comprehensive testing suite

## Related Documents

- [A2A Protocol Methods](A2A_PROTOCOL_METHODS.md)
- [A2A Protocol Security](A2A_PROTOCOL_SECURITY.md)
- [A2A Protocol Implementation Guide](A2A_PROTOCOL_IMPLEMENTATION_GUIDE.md)
- [A2A Protocol Technical Specification](A2A_PROTOCOL_TECHNICAL_SPEC.md)
