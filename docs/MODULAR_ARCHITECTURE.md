# PyGent Factory - Modular Architecture Plan

## Current State Analysis
The current implementation has grown into large, monolithic files that could benefit from modularization. This document outlines the plan to break down the system into smaller, focused modules.

## Modularization Strategy

### 1. Core Module Breakdown

#### 1.1 Agent Core (`src/core/agent/`)
- `base.py` - BaseAgent abstract class
- `message.py` - AgentMessage and related classes
- `capability.py` - AgentCapability definitions
- `config.py` - AgentConfig and settings
- `status.py` - Agent status management
- `lifecycle.py` - Agent lifecycle management

#### 1.2 Agent Factory (`src/core/factory/`)
- `registry.py` - AgentRegistry implementation
- `factory.py` - AgentFactory core logic
- `builder.py` - Agent building utilities
- `validator.py` - Agent validation logic

#### 1.3 Message System (`src/core/messaging/`)
- `bus.py` - MessageBus implementation
- `handler.py` - Message handlers
- `router.py` - Message routing logic
- `queue.py` - Message queue management
- `priority.py` - Priority handling

#### 1.4 Capability System (`src/core/capabilities/`)
- `manager.py` - CapabilityManager
- `definition.py` - CapabilityDefinition
- `executor.py` - CapabilityExecutor
- `validator.py` - CapabilityValidator
- `registry.py` - CapabilityRegistry

### 2. Database Module Breakdown

#### 2.1 Database Core (`src/database/`)
- `connection.py` - Connection management (keep as is)
- `models/` - Split models by domain
  - `agent.py` - Agent-related models
  - `memory.py` - Memory-related models
  - `document.py` - Document-related models
  - `mcp.py` - MCP-related models
  - `evaluation.py` - Evaluation-related models
- `repositories/` - Data access layer
  - `agent_repository.py`
  - `memory_repository.py`
  - `document_repository.py`

### 3. Storage Module Breakdown

#### 3.1 Vector Storage (`src/storage/vector/`)
- `base.py` - Abstract vector store interface
- `postgresql.py` - PostgreSQL implementation
- `chromadb.py` - ChromaDB implementation
- `faiss.py` - FAISS implementation
- `manager.py` - VectorStoreManager

#### 3.2 File Storage (`src/storage/files/`)
- `local.py` - Local file storage
- `s3.py` - S3 storage implementation
- `manager.py` - File storage manager

### 4. Memory Module Breakdown

#### 4.1 Memory Core (`src/memory/`)
- `manager.py` - MemoryManager (simplified)
- `space.py` - MemorySpace implementation
- `entry.py` - MemoryEntry and related classes
- `types.py` - Memory type definitions
- `consolidation.py` - Memory consolidation logic
- `retrieval.py` - Memory retrieval logic

### 5. MCP Module Breakdown

#### 5.1 MCP Core (`src/mcp/`)
- `server/` - Server management
  - `registry.py` - Server registry
  - `manager.py` - Server manager
  - `config.py` - Server configuration
  - `lifecycle.py` - Server lifecycle
- `tools/` - Tool management
  - `registry.py` - Tool registry
  - `executor.py` - Tool executor
  - `discovery.py` - Tool discovery
- `client/` - MCP client implementations

### 6. RAG Module Breakdown

#### 6.1 Document Processing (`src/rag/processing/`)
- `extractor.py` - Text extraction
- `chunker.py` - Text chunking
- `processor.py` - Main document processor
- `formats/` - Format-specific processors
  - `pdf.py`
  - `docx.py`
  - `markdown.py`
  - `html.py`

#### 6.2 Retrieval (`src/rag/retrieval/`)
- `system.py` - Main retrieval system
- `strategies/` - Retrieval strategies
  - `semantic.py`
  - `hybrid.py`
  - `contextual.py`
  - `adaptive.py`
- `scoring.py` - Relevance scoring
- `ranking.py` - Result ranking

### 7. Communication Module Breakdown

#### 7.1 Protocols (`src/communication/protocols/`)
- `base.py` - Abstract protocol interface
- `internal.py` - Internal protocol
- `mcp.py` - MCP protocol
- `http.py` - HTTP protocol
- `websocket.py` - WebSocket protocol
- `manager.py` - Protocol manager

### 8. API Module Breakdown

#### 8.1 API Core (`src/api/`)
- `main.py` - FastAPI app (simplified)
- `dependencies.py` - Dependency injection
- `middleware.py` - Custom middleware
- `exceptions.py` - Exception handlers
- `routes/` - Keep existing route structure
- `schemas/` - Pydantic schemas
  - `agent.py`
  - `memory.py`
  - `mcp.py`
  - `rag.py`

### 9. Security Module Breakdown

#### 9.1 Security Core (`src/security/`)
- `auth/` - Authentication
  - `jwt.py` - JWT handling
  - `api_key.py` - API key management
  - `password.py` - Password management
- `authorization/` - Authorization
  - `rbac.py` - Role-based access control
  - `permissions.py` - Permission definitions
- `middleware.py` - Security middleware
- `rate_limiting.py` - Rate limiting

### 10. Utilities Module Breakdown

#### 10.1 Utils (`src/utils/`)
- `embedding/` - Embedding utilities
  - `service.py` - Main embedding service
  - `providers/` - Provider implementations
    - `openai.py`
    - `sentence_transformers.py`
  - `cache.py` - Embedding cache
- `logging.py` - Logging utilities
- `validation.py` - Validation utilities
- `serialization.py` - Serialization utilities

## Implementation Plan

### Phase 1: Core Modules (Steps 36-45)
1. Break down agent core into focused modules
2. Modularize message system
3. Split capability system
4. Refactor database models by domain

### Phase 2: Storage & Memory (Steps 46-55)
1. Modularize vector storage implementations
2. Break down memory management
3. Create repository pattern for data access

### Phase 3: MCP & RAG (Steps 56-65)
1. Split MCP into server/tool/client modules
2. Modularize RAG processing and retrieval
3. Create strategy pattern implementations

### Phase 4: Communication & API (Steps 66-75)
1. Break down protocol implementations
2. Create focused API schemas
3. Modularize security components

### Phase 5: Integration & Testing (Steps 76-85)
1. Update imports and dependencies
2. Create module-specific tests
3. Integration testing

## Benefits of Modularization

1. **Maintainability**: Smaller, focused files are easier to understand and modify
2. **Testability**: Individual modules can be tested in isolation
3. **Reusability**: Modules can be reused across different parts of the system
4. **Scalability**: New implementations can be added without modifying existing code
5. **Team Development**: Different team members can work on different modules
6. **Code Quality**: Smaller modules encourage better design patterns

## Migration Strategy

1. **Backward Compatibility**: Maintain existing imports through `__init__.py` files
2. **Gradual Migration**: Move functionality module by module
3. **Testing**: Ensure all tests pass after each module migration
4. **Documentation**: Update documentation to reflect new structure

## File Structure After Modularization

```
src/
├── core/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── message.py
│   │   ├── capability.py
│   │   └── ...
│   ├── factory/
│   ├── messaging/
│   └── capabilities/
├── database/
│   ├── models/
│   └── repositories/
├── storage/
│   ├── vector/
│   └── files/
├── memory/
├── mcp/
│   ├── server/
│   ├── tools/
│   └── client/
├── rag/
│   ├── processing/
│   └── retrieval/
├── communication/
│   └── protocols/
├── api/
│   ├── routes/
│   └── schemas/
├── security/
│   ├── auth/
│   └── authorization/
├── utils/
│   └── embedding/
└── agents/
    ├── research/
    ├── code/
    └── conversation/
```

This modular approach will make the codebase much more maintainable and allow for easier extension and testing.
