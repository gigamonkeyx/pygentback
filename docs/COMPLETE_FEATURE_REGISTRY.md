# PyGent Factory - Complete Feature Registry

**Generated**: 2025-06-10T04:22:20.860417
**Total Features**: 533

This document provides a comprehensive overview of all features in the PyGent Factory project.
It is automatically generated and should not be manually edited.

## Api Endpoint

**Count**: 103

### DELETE /documents/{document_id}

- **Status**: active
- **File**: `src\api\routes\rag.py`
- **Description**: FastAPI DELETE endpoint: /documents/{document_id}
- **Last Modified**: 2025-06-02 09:07:34
- **API Route**: `/documents/{document_id}`

### DELETE /models/{model_name}

- **Status**: active
- **File**: `src\api\routes\ollama.py`
- **Description**: FastAPI DELETE endpoint: /models/{model_name}
- **Last Modified**: 2025-06-02 22:15:36
- **API Route**: `/models/{model_name}`

### DELETE /research-analysis/{workflow_id}

- **Status**: active
- **File**: `src\api\routes\workflows.py`
- **Description**: FastAPI DELETE endpoint: /research-analysis/{workflow_id}
- **Last Modified**: 2025-06-03 01:12:14
- **API Route**: `/research-analysis/{workflow_id}`

### DELETE /servers/{server_id}

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI DELETE endpoint: /servers/{server_id}
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/servers/{server_id}`

### DELETE /token/{provider}

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI DELETE endpoint: /token/{provider}
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/token/{provider}`

### DELETE /{agent_id}

- **Status**: active
- **File**: `src\api\routes\agents.py`
- **Description**: FastAPI DELETE endpoint: /{agent_id}
- **Last Modified**: 2025-06-08 22:50:22
- **API Route**: `/{agent_id}`

### DELETE /{agent_id}/memories/{memory_id}

- **Status**: active
- **File**: `src\api\routes\memory.py`
- **Description**: FastAPI DELETE endpoint: /{agent_id}/memories/{memory_id}
- **Last Modified**: 2025-06-02 09:00:26
- **API Route**: `/{agent_id}/memories/{memory_id}`

### DELETE /{model_name}

- **Status**: active
- **File**: `src\api\routes\models.py`
- **Description**: FastAPI DELETE endpoint: /{model_name}
- **Last Modified**: 2025-06-02 18:08:53
- **API Route**: `/{model_name}`

### GET /

- **Status**: active
- **File**: `src\api\routes\models.py`
- **Description**: FastAPI GET endpoint: /
- **Last Modified**: 2025-06-02 18:08:53
- **API Route**: `/`

### GET /backend/documents

- **Status**: active
- **File**: `src\api\routes\documentation_enhanced.py`
- **Description**: FastAPI GET endpoint: /backend/documents
- **Last Modified**: 2025-06-08 22:30:22
- **API Route**: `/backend/documents`

### GET /by-type/{feature_type}

- **Status**: active
- **File**: `src\api\routes\features.py`
- **Description**: FastAPI GET endpoint: /by-type/{feature_type}
- **Last Modified**: 2025-06-09 21:20:24
- **API Route**: `/by-type/{feature_type}`

### GET /callback/{provider}

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI GET endpoint: /callback/{provider}
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/callback/{provider}`

### GET /categories

- **Status**: active
- **File**: `src\api\routes\documentation.py`
- **Description**: FastAPI GET endpoint: /categories
- **Last Modified**: 2025-06-09 10:20:05
- **API Route**: `/categories`

### GET /cloudflare/authorize

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI GET endpoint: /cloudflare/authorize
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/cloudflare/authorize`

### GET /discovery/servers

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /discovery/servers
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/discovery/servers`

### GET /discovery/status

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /discovery/status
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/discovery/status`

### GET /documentation

- **Status**: active
- **File**: `src\api\routes\features.py`
- **Description**: FastAPI GET endpoint: /documentation
- **Last Modified**: 2025-06-09 21:20:24
- **API Route**: `/documentation`

### GET /documents/{document_id}

- **Status**: active
- **File**: `src\api\routes\documentation_enhanced.py`
- **Description**: FastAPI GET endpoint: /documents/{document_id}
- **Last Modified**: 2025-06-08 22:30:22
- **API Route**: `/documents/{document_id}`

### GET /documents/{document_id}/similar

- **Status**: active
- **File**: `src\api\routes\rag.py`
- **Description**: FastAPI GET endpoint: /documents/{document_id}/similar
- **Last Modified**: 2025-06-02 09:07:34
- **API Route**: `/documents/{document_id}/similar`

### GET /file/{file_path:path}

- **Status**: active
- **File**: `src\api\routes\documentation.py`
- **Description**: FastAPI GET endpoint: /file/{file_path:path}
- **Last Modified**: 2025-06-09 10:20:05
- **API Route**: `/file/{file_path:path}`

### GET /files

- **Status**: active
- **File**: `src\api\routes\documentation.py`
- **Description**: FastAPI GET endpoint: /files
- **Last Modified**: 2025-06-09 10:20:05
- **API Route**: `/files`

### GET /github/authorize

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI GET endpoint: /github/authorize
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/github/authorize`

### GET /global/stats

- **Status**: active
- **File**: `src\api\routes\memory.py`
- **Description**: FastAPI GET endpoint: /global/stats
- **Last Modified**: 2025-06-02 09:00:26
- **API Route**: `/global/stats`

### GET /health

- **Status**: active
- **File**: `src\api\routes\ollama.py`
- **Description**: FastAPI GET endpoint: /health
- **Last Modified**: 2025-06-02 22:15:36
- **API Route**: `/health`

### GET /health/agents

- **Status**: active
- **File**: `src\api\routes\health.py`
- **Description**: FastAPI GET endpoint: /health/agents
- **Last Modified**: 2025-06-09 17:07:50
- **API Route**: `/health/agents`

### GET /health/database

- **Status**: active
- **File**: `src\api\routes\health.py`
- **Description**: FastAPI GET endpoint: /health/database
- **Last Modified**: 2025-06-09 17:07:50
- **API Route**: `/health/database`

### GET /health/mcp

- **Status**: active
- **File**: `src\api\routes\health.py`
- **Description**: FastAPI GET endpoint: /health/mcp
- **Last Modified**: 2025-06-09 17:07:50
- **API Route**: `/health/mcp`

### GET /health/memory

- **Status**: active
- **File**: `src\api\routes\health.py`
- **Description**: FastAPI GET endpoint: /health/memory
- **Last Modified**: 2025-06-09 17:07:50
- **API Route**: `/health/memory`

### GET /health/ollama

- **Status**: active
- **File**: `src\api\routes\health.py`
- **Description**: FastAPI GET endpoint: /health/ollama
- **Last Modified**: 2025-06-09 17:07:50
- **API Route**: `/health/ollama`

### GET /health/rag

- **Status**: active
- **File**: `src\api\routes\health.py`
- **Description**: FastAPI GET endpoint: /health/rag
- **Last Modified**: 2025-06-09 17:07:50
- **API Route**: `/health/rag`

### GET /health/summary

- **Status**: active
- **File**: `src\api\routes\agents.py`
- **Description**: FastAPI GET endpoint: /health/summary
- **Last Modified**: 2025-06-08 22:50:22
- **API Route**: `/health/summary`

### GET /marketplace/categories

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /marketplace/categories
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/marketplace/categories`

### GET /marketplace/featured

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /marketplace/featured
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/marketplace/featured`

### GET /marketplace/popular

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /marketplace/popular
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/marketplace/popular`

### GET /marketplace/search

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /marketplace/search
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/marketplace/search`

### GET /me

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI GET endpoint: /me
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/me`

### GET /metrics

- **Status**: active
- **File**: `src\api\routes\ollama.py`
- **Description**: FastAPI GET endpoint: /metrics
- **Last Modified**: 2025-06-02 22:15:36
- **API Route**: `/metrics`

### GET /models

- **Status**: active
- **File**: `src\api\routes\ollama.py`
- **Description**: FastAPI GET endpoint: /models
- **Last Modified**: 2025-06-02 22:15:36
- **API Route**: `/models`

### GET /persistent

- **Status**: active
- **File**: `src\api\routes\documentation.py`
- **Description**: FastAPI GET endpoint: /persistent
- **Last Modified**: 2025-06-09 10:20:05
- **API Route**: `/persistent`

### GET /persistent/{doc_id}

- **Status**: active
- **File**: `src\api\routes\documentation.py`
- **Description**: FastAPI GET endpoint: /persistent/{doc_id}
- **Last Modified**: 2025-06-09 10:20:05
- **API Route**: `/persistent/{doc_id}`

### GET /providers

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI GET endpoint: /providers
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/providers`

### GET /research-analysis/active

- **Status**: active
- **File**: `src\api\routes\workflows.py`
- **Description**: FastAPI GET endpoint: /research-analysis/active
- **Last Modified**: 2025-06-03 01:12:14
- **API Route**: `/research-analysis/active`

### GET /research-analysis/{workflow_id}/export/{format}

- **Status**: active
- **File**: `src\api\routes\workflows.py`
- **Description**: FastAPI GET endpoint: /research-analysis/{workflow_id}/export/{format}
- **Last Modified**: 2025-06-03 01:12:14
- **API Route**: `/research-analysis/{workflow_id}/export/{format}`

### GET /research-analysis/{workflow_id}/result

- **Status**: active
- **File**: `src\api\routes\workflows.py`
- **Description**: FastAPI GET endpoint: /research-analysis/{workflow_id}/result
- **Last Modified**: 2025-06-03 01:12:14
- **API Route**: `/research-analysis/{workflow_id}/result`

### GET /research-analysis/{workflow_id}/status

- **Status**: active
- **File**: `src\api\routes\workflows.py`
- **Description**: FastAPI GET endpoint: /research-analysis/{workflow_id}/status
- **Last Modified**: 2025-06-03 01:12:14
- **API Route**: `/research-analysis/{workflow_id}/status`

### GET /research-analysis/{workflow_id}/stream

- **Status**: active
- **File**: `src\api\routes\workflows.py`
- **Description**: FastAPI GET endpoint: /research-analysis/{workflow_id}/stream
- **Last Modified**: 2025-06-03 01:12:14
- **API Route**: `/research-analysis/{workflow_id}/stream`

### GET /research-sessions

- **Status**: active
- **File**: `src\api\routes\documentation.py`
- **Description**: FastAPI GET endpoint: /research-sessions
- **Last Modified**: 2025-06-09 10:20:05
- **API Route**: `/research-sessions`

### GET /search

- **Status**: active
- **File**: `src\api\routes\features.py`
- **Description**: FastAPI GET endpoint: /search
- **Last Modified**: 2025-06-09 21:20:24
- **API Route**: `/search`

### GET /servers

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /servers
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/servers`

### GET /servers/install/status

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /servers/install/status
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/servers/install/status`

### GET /servers/install/{server_name}/status

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /servers/install/{server_name}/status
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/servers/install/{server_name}/status`

### GET /servers/{server_id}

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /servers/{server_id}
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/servers/{server_id}`

### GET /session

- **Status**: active
- **File**: `src\api\routes\documentation_enhanced.py`
- **Description**: FastAPI GET endpoint: /session
- **Last Modified**: 2025-06-08 22:30:22
- **API Route**: `/session`

### GET /stats

- **Status**: active
- **File**: `src\api\routes\rag.py`
- **Description**: FastAPI GET endpoint: /stats
- **Last Modified**: 2025-06-02 09:07:34
- **API Route**: `/stats`

### GET /stats/summary

- **Status**: active
- **File**: `src\api\routes\models.py`
- **Description**: FastAPI GET endpoint: /stats/summary
- **Last Modified**: 2025-06-02 18:08:53
- **API Route**: `/stats/summary`

### GET /status

- **Status**: active
- **File**: `src\api\routes\ollama.py`
- **Description**: FastAPI GET endpoint: /status
- **Last Modified**: 2025-06-02 22:15:36
- **API Route**: `/status`

### GET /statuses

- **Status**: active
- **File**: `src\api\routes\features.py`
- **Description**: FastAPI GET endpoint: /statuses
- **Last Modified**: 2025-06-09 21:20:24
- **API Route**: `/statuses`

### GET /strategies

- **Status**: active
- **File**: `src\api\routes\rag.py`
- **Description**: FastAPI GET endpoint: /strategies
- **Last Modified**: 2025-06-02 09:07:34
- **API Route**: `/strategies`

### GET /supported-formats

- **Status**: active
- **File**: `src\api\routes\rag.py`
- **Description**: FastAPI GET endpoint: /supported-formats
- **Last Modified**: 2025-06-02 09:07:34
- **API Route**: `/supported-formats`

### GET /token/{provider}

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI GET endpoint: /token/{provider}
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/token/{provider}`

### GET /tools

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /tools
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/tools`

### GET /tools/{tool_name}

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI GET endpoint: /tools/{tool_name}
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/tools/{tool_name}`

### GET /types

- **Status**: active
- **File**: `src\api\routes\memory.py`
- **Description**: FastAPI GET endpoint: /types
- **Last Modified**: 2025-06-02 09:00:26
- **API Route**: `/types`

### GET /types/available

- **Status**: active
- **File**: `src\api\routes\agents.py`
- **Description**: FastAPI GET endpoint: /types/available
- **Last Modified**: 2025-06-08 22:50:22
- **API Route**: `/types/available`

### GET /ws-test

- **Status**: active
- **File**: `src\api\routes\websocket.py`
- **Description**: FastAPI GET endpoint: /ws-test
- **Last Modified**: 2025-06-08 15:13:35
- **API Route**: `/ws-test`

### GET /{agent_id}

- **Status**: active
- **File**: `src\api\routes\agents.py`
- **Description**: FastAPI GET endpoint: /{agent_id}
- **Last Modified**: 2025-06-08 22:50:22
- **API Route**: `/{agent_id}`

### GET /{agent_id}/capabilities

- **Status**: active
- **File**: `src\api\routes\agents.py`
- **Description**: FastAPI GET endpoint: /{agent_id}/capabilities
- **Last Modified**: 2025-06-08 22:50:22
- **API Route**: `/{agent_id}/capabilities`

### GET /{agent_id}/memories/{memory_id}

- **Status**: active
- **File**: `src\api\routes\memory.py`
- **Description**: FastAPI GET endpoint: /{agent_id}/memories/{memory_id}
- **Last Modified**: 2025-06-02 09:00:26
- **API Route**: `/{agent_id}/memories/{memory_id}`

### GET /{agent_id}/stats

- **Status**: active
- **File**: `src\api\routes\memory.py`
- **Description**: FastAPI GET endpoint: /{agent_id}/stats
- **Last Modified**: 2025-06-02 09:00:26
- **API Route**: `/{agent_id}/stats`

### GET /{agent_id}/status

- **Status**: active
- **File**: `src\api\routes\agents.py`
- **Description**: FastAPI GET endpoint: /{agent_id}/status
- **Last Modified**: 2025-06-08 22:50:22
- **API Route**: `/{agent_id}/status`

### GET /{model_name}

- **Status**: active
- **File**: `src\api\routes\models.py`
- **Description**: FastAPI GET endpoint: /{model_name}
- **Last Modified**: 2025-06-02 18:08:53
- **API Route**: `/{model_name}`

### POST /

- **Status**: active
- **File**: `src\api\routes\models.py`
- **Description**: FastAPI POST endpoint: /
- **Last Modified**: 2025-06-02 18:08:53
- **API Route**: `/`

### POST /authorize/{provider}

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI POST endpoint: /authorize/{provider}
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/authorize/{provider}`

### POST /create

- **Status**: active
- **File**: `src\api\routes\documentation_enhanced.py`
- **Description**: FastAPI POST endpoint: /create
- **Last Modified**: 2025-06-08 22:30:22
- **API Route**: `/create`

### POST /discover

- **Status**: active
- **File**: `src\api\routes\features.py`
- **Description**: FastAPI POST endpoint: /discover
- **Last Modified**: 2025-06-09 21:20:24
- **API Route**: `/discover`

### POST /documents/text

- **Status**: active
- **File**: `src\api\routes\rag.py`
- **Description**: FastAPI POST endpoint: /documents/text
- **Last Modified**: 2025-06-02 09:07:34
- **API Route**: `/documents/text`

### POST /documents/upload

- **Status**: active
- **File**: `src\api\routes\rag.py`
- **Description**: FastAPI POST endpoint: /documents/upload
- **Last Modified**: 2025-06-02 09:07:34
- **API Route**: `/documents/upload`

### POST /login

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI POST endpoint: /login
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/login`

### POST /logout

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI POST endpoint: /logout
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/logout`

### POST /models/pull

- **Status**: active
- **File**: `src\api\routes\ollama.py`
- **Description**: FastAPI POST endpoint: /models/pull
- **Last Modified**: 2025-06-02 22:15:36
- **API Route**: `/models/pull`

### POST /recommend

- **Status**: active
- **File**: `src\api\routes\models.py`
- **Description**: FastAPI POST endpoint: /recommend
- **Last Modified**: 2025-06-02 18:08:53
- **API Route**: `/recommend`

### POST /refresh/{provider}

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI POST endpoint: /refresh/{provider}
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/refresh/{provider}`

### POST /register

- **Status**: active
- **File**: `src\api\routes\auth.py`
- **Description**: FastAPI POST endpoint: /register
- **Last Modified**: 2025-06-09 10:37:31
- **API Route**: `/register`

### POST /research-analysis

- **Status**: active
- **File**: `src\api\routes\workflows.py`
- **Description**: FastAPI POST endpoint: /research-analysis
- **Last Modified**: 2025-06-03 01:12:14
- **API Route**: `/research-analysis`

### POST /research-session

- **Status**: active
- **File**: `src\api\routes\documentation.py`
- **Description**: FastAPI POST endpoint: /research-session
- **Last Modified**: 2025-06-09 10:20:05
- **API Route**: `/research-session`

### POST /research/generate

- **Status**: active
- **File**: `src\api\routes\documentation_enhanced.py`
- **Description**: FastAPI POST endpoint: /research/generate
- **Last Modified**: 2025-06-08 22:30:22
- **API Route**: `/research/generate`

### POST /retrieve

- **Status**: active
- **File**: `src\api\routes\rag.py`
- **Description**: FastAPI POST endpoint: /retrieve
- **Last Modified**: 2025-06-02 09:07:34
- **API Route**: `/retrieve`

### POST /servers

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI POST endpoint: /servers
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/servers`

### POST /servers/install

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI POST endpoint: /servers/install
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/servers/install`

### POST /servers/{server_id}/restart

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI POST endpoint: /servers/{server_id}/restart
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/servers/{server_id}/restart`

### POST /servers/{server_id}/start

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI POST endpoint: /servers/{server_id}/start
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/servers/{server_id}/start`

### POST /servers/{server_id}/stop

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI POST endpoint: /servers/{server_id}/stop
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/servers/{server_id}/stop`

### POST /session/clear

- **Status**: active
- **File**: `src\api\routes\documentation_enhanced.py`
- **Description**: FastAPI POST endpoint: /session/clear
- **Last Modified**: 2025-06-08 22:30:22
- **API Route**: `/session/clear`

### POST /tools/call

- **Status**: active
- **File**: `src\api\routes\mcp.py`
- **Description**: FastAPI POST endpoint: /tools/call
- **Last Modified**: 2025-06-08 01:55:38
- **API Route**: `/tools/call`

### POST /workflow

- **Status**: active
- **File**: `src\api\routes\documentation.py`
- **Description**: FastAPI POST endpoint: /workflow
- **Last Modified**: 2025-06-09 10:20:05
- **API Route**: `/workflow`

### POST /{agent_id}/consolidate

- **Status**: active
- **File**: `src\api\routes\memory.py`
- **Description**: FastAPI POST endpoint: /{agent_id}/consolidate
- **Last Modified**: 2025-06-02 09:00:26
- **API Route**: `/{agent_id}/consolidate`

### POST /{agent_id}/execute

- **Status**: active
- **File**: `src\api\routes\agents.py`
- **Description**: FastAPI POST endpoint: /{agent_id}/execute
- **Last Modified**: 2025-06-08 22:50:22
- **API Route**: `/{agent_id}/execute`

### POST /{agent_id}/message

- **Status**: active
- **File**: `src\api\routes\agents.py`
- **Description**: FastAPI POST endpoint: /{agent_id}/message
- **Last Modified**: 2025-06-08 22:50:22
- **API Route**: `/{agent_id}/message`

### POST /{agent_id}/search

- **Status**: active
- **File**: `src\api\routes\memory.py`
- **Description**: FastAPI POST endpoint: /{agent_id}/search
- **Last Modified**: 2025-06-02 09:00:26
- **API Route**: `/{agent_id}/search`

### POST /{agent_id}/store

- **Status**: active
- **File**: `src\api\routes\memory.py`
- **Description**: FastAPI POST endpoint: /{agent_id}/store
- **Last Modified**: 2025-06-02 09:00:26
- **API Route**: `/{agent_id}/store`

### POST /{model_name}/rate

- **Status**: active
- **File**: `src\api\routes\models.py`
- **Description**: FastAPI POST endpoint: /{model_name}/rate
- **Last Modified**: 2025-06-02 18:08:53
- **API Route**: `/{model_name}/rate`

### PUT /documents/{document_id}

- **Status**: active
- **File**: `src\api\routes\documentation_enhanced.py`
- **Description**: FastAPI PUT endpoint: /documents/{document_id}
- **Last Modified**: 2025-06-08 22:30:22
- **API Route**: `/documents/{document_id}`

### PUT /update/{doc_id}

- **Status**: active
- **File**: `src\api\routes\documentation.py`
- **Description**: FastAPI PUT endpoint: /update/{doc_id}
- **Last Modified**: 2025-06-09 10:20:05
- **API Route**: `/update/{doc_id}`

## Cloudflare Worker

**Count**: 17

### Cloudflare Worker: ai-gateway

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\ai-gateway`
- **Description**: Cloudflare Worker MCP server: ai-gateway
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\ai-gateway\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: auditlogs

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\auditlogs`
- **Description**: Cloudflare Worker MCP server: auditlogs
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\auditlogs\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: autorag

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\autorag`
- **Description**: Cloudflare Worker MCP server: autorag
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\autorag\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: browser-rendering

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\browser-rendering`
- **Description**: Cloudflare Worker MCP server: browser-rendering
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\browser-rendering\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: cloudflare-one-casb

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\cloudflare-one-casb`
- **Description**: Cloudflare Worker MCP server: cloudflare-one-casb
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\cloudflare-one-casb\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: demo-day

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\demo-day`
- **Description**: Cloudflare Worker MCP server: demo-day
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\demo-day\wrangler.json`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: dex-analysis

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\dex-analysis`
- **Description**: Cloudflare Worker MCP server: dex-analysis
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\dex-analysis\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: dns-analytics

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\dns-analytics`
- **Description**: Cloudflare Worker MCP server: dns-analytics
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\dns-analytics\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: docs-autorag

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\docs-autorag`
- **Description**: Cloudflare Worker MCP server: docs-autorag
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\docs-autorag\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: docs-vectorize

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\docs-vectorize`
- **Description**: Cloudflare Worker MCP server: docs-vectorize
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\docs-vectorize\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: graphql

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\graphql`
- **Description**: Cloudflare Worker MCP server: graphql
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\graphql\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: logpush

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\logpush`
- **Description**: Cloudflare Worker MCP server: logpush
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\logpush\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: radar

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\radar`
- **Description**: Cloudflare Worker MCP server: radar
- **Last Modified**: 2025-06-07 23:04:13
- **Config**: `mcp-server-cloudflare\apps\radar\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: sandbox-container

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\sandbox-container`
- **Description**: Cloudflare Worker MCP server: sandbox-container
- **Last Modified**: 2025-06-07 20:53:52
- **Config**: `mcp-server-cloudflare\apps\sandbox-container\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: workers-bindings

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\workers-bindings`
- **Description**: Cloudflare Worker MCP server: workers-bindings
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\workers-bindings\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: workers-builds

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\workers-builds`
- **Description**: Cloudflare Worker MCP server: workers-builds
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\workers-builds\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

### Cloudflare Worker: workers-observability

- **Status**: active
- **File**: `mcp-server-cloudflare\apps\workers-observability`
- **Description**: Cloudflare Worker MCP server: workers-observability
- **Last Modified**: 2025-06-07 20:53:48
- **Config**: `mcp-server-cloudflare\apps\workers-observability\wrangler.jsonc`
- **Dependencies**: cloudflare, wrangler

## Database Model

**Count**: 1

### Database Model: documentation

- **Status**: active
- **File**: `src\api\models\documentation.py`
- **Description**: Database model definition: documentation
- **Last Modified**: 2025-06-08 22:30:22
- **Dependencies**: sqlalchemy

## Configuration

**Count**: 40

### Configuration: .env

- **Status**: active
- **File**: `.env`
- **Description**: Configuration file: .env
- **Last Modified**: 2025-06-04 15:56:01

### Configuration: .env.example

- **Status**: active
- **File**: `.env.example`
- **Description**: Configuration file: .env.example
- **Last Modified**: 2025-06-02 00:37:57

### Configuration: .env.local

- **Status**: active
- **File**: `.env.local`
- **Description**: Configuration file: .env.local
- **Last Modified**: 2025-06-08 01:55:38

### Configuration: Dockerfile

- **Status**: active
- **File**: `Dockerfile`
- **Description**: Configuration file: Dockerfile
- **Last Modified**: 2025-06-01 23:00:11

### Configuration: automated_validation_report.json

- **Status**: active
- **File**: `automated_validation_report.json`
- **Description**: Configuration file: automated_validation_report.json
- **Last Modified**: 2025-05-29 02:11:16

### Configuration: cloudflare_auth.env

- **Status**: active
- **File**: `cloudflare_auth.env`
- **Description**: Configuration file: cloudflare_auth.env
- **Last Modified**: 2025-06-08 01:55:40

### Configuration: cloudflare_auth.env.example

- **Status**: active
- **File**: `cloudflare_auth.env.example`
- **Description**: Configuration file: cloudflare_auth.env.example
- **Last Modified**: 2025-06-08 01:55:38

### Configuration: cloudflared-config.yml

- **Status**: active
- **File**: `cloudflared-config.yml`
- **Description**: Configuration file: cloudflared-config.yml
- **Last Modified**: 2025-06-04 21:01:01

### Configuration: coding_agent_evolution_report.json

- **Status**: active
- **File**: `coding_agent_evolution_report.json`
- **Description**: Configuration file: coding_agent_evolution_report.json
- **Last Modified**: 2025-06-07 19:55:51

### Configuration: docker-compose.yml

- **Status**: active
- **File**: `docker-compose.yml`
- **Description**: Configuration file: docker-compose.yml
- **Last Modified**: 2025-06-02 00:34:08

### Configuration: feature_health_analysis.json

- **Status**: active
- **File**: `feature_health_analysis.json`
- **Description**: Configuration file: feature_health_analysis.json
- **Last Modified**: 2025-06-09 21:22:01

### Configuration: feature_registry.json

- **Status**: active
- **File**: `feature_registry.json`
- **Description**: Configuration file: feature_registry.json
- **Last Modified**: 2025-06-09 21:22:01

### Configuration: final_validation_report.json

- **Status**: active
- **File**: `final_validation_report.json`
- **Description**: Configuration file: final_validation_report.json
- **Last Modified**: 2025-05-29 02:19:39

### Configuration: frontend_connectivity_diagnosis.json

- **Status**: active
- **File**: `frontend_connectivity_diagnosis.json`
- **Description**: Configuration file: frontend_connectivity_diagnosis.json
- **Last Modified**: 2025-06-03 22:58:00

### Configuration: gpu_usage_analysis.json

- **Status**: active
- **File**: `gpu_usage_analysis.json`
- **Description**: Configuration file: gpu_usage_analysis.json
- **Last Modified**: 2025-06-06 00:42:26

### Configuration: mcp_functionality_test_report.json

- **Status**: active
- **File**: `mcp_functionality_test_report.json`
- **Description**: Configuration file: mcp_functionality_test_report.json
- **Last Modified**: 2025-06-07 19:57:23

### Configuration: mcp_server_configs.json

- **Status**: active
- **File**: `mcp_server_configs.json`
- **Description**: Configuration file: mcp_server_configs.json
- **Last Modified**: 2025-06-09 19:42:15

### Configuration: mcp_tool_analysis.json

- **Status**: active
- **File**: `mcp_tool_analysis.json`
- **Description**: Configuration file: mcp_tool_analysis.json
- **Last Modified**: 2025-06-05 23:42:03

### Configuration: mcp_tool_discovery_results.json

- **Status**: active
- **File**: `mcp_tool_discovery_results.json`
- **Description**: Configuration file: mcp_tool_discovery_results.json
- **Last Modified**: 2025-06-06 00:01:48

### Configuration: memory_system_analysis.json

- **Status**: active
- **File**: `memory_system_analysis.json`
- **Description**: Configuration file: memory_system_analysis.json
- **Last Modified**: 2025-06-06 00:34:12

### Configuration: memory_system_gpu_test_report.json

- **Status**: active
- **File**: `memory_system_gpu_test_report.json`
- **Description**: Configuration file: memory_system_gpu_test_report.json
- **Last Modified**: 2025-06-06 00:50:29

### Configuration: memory_system_performance_report.json

- **Status**: active
- **File**: `memory_system_performance_report.json`
- **Description**: Configuration file: memory_system_performance_report.json
- **Last Modified**: 2025-06-06 00:49:16

### Configuration: memory_system_verification.json

- **Status**: active
- **File**: `memory_system_verification.json`
- **Description**: Configuration file: memory_system_verification.json
- **Last Modified**: 2025-06-06 00:38:30

### Configuration: oauth.env

- **Status**: active
- **File**: `oauth.env`
- **Description**: Configuration file: oauth.env
- **Last Modified**: 2025-06-08 01:55:38

### Configuration: oauth.env.example

- **Status**: active
- **File**: `oauth.env.example`
- **Description**: Configuration file: oauth.env.example
- **Last Modified**: 2025-06-08 01:55:38

### Configuration: overnight_collection_results_20250529_0833.json

- **Status**: active
- **File**: `overnight_collection_results_20250529_0833.json`
- **Description**: Configuration file: overnight_collection_results_20250529_0833.json
- **Last Modified**: 2025-05-29 08:33:37

### Configuration: package-lock.json

- **Status**: active
- **File**: `package-lock.json`
- **Description**: Configuration file: package-lock.json
- **Last Modified**: 2025-06-08 11:12:13

### Configuration: package.json

- **Status**: active
- **File**: `package.json`
- **Description**: Configuration file: package.json
- **Last Modified**: 2025-06-08 11:24:56

### Configuration: paper_url_check_results.json

- **Status**: active
- **File**: `paper_url_check_results.json`
- **Description**: Configuration file: paper_url_check_results.json
- **Last Modified**: 2025-05-29 02:12:32

### Configuration: requirements.txt

- **Status**: active
- **File**: `requirements.txt`
- **Description**: Configuration file: requirements.txt
- **Last Modified**: 2025-06-08 18:03:24

### Configuration: research_analysis_validation_report.json

- **Status**: active
- **File**: `research_analysis_validation_report.json`
- **Description**: Configuration file: research_analysis_validation_report.json
- **Last Modified**: 2025-06-03 01:41:31

### Configuration: research_results.json

- **Status**: active
- **File**: `research_results.json`
- **Description**: Configuration file: research_results.json
- **Last Modified**: 2025-05-29 01:20:03

### Configuration: research_validation_report.json

- **Status**: active
- **File**: `research_validation_report.json`
- **Description**: Configuration file: research_validation_report.json
- **Last Modified**: 2025-05-29 02:06:14

### Configuration: simple-tunnel.yml

- **Status**: active
- **File**: `simple-tunnel.yml`
- **Description**: Configuration file: simple-tunnel.yml
- **Last Modified**: 2025-06-04 21:03:03

### Configuration: test_workflow.json

- **Status**: active
- **File**: `test_workflow.json`
- **Description**: Configuration file: test_workflow.json
- **Last Modified**: 2025-06-04 19:18:54

### Configuration: test_workflow_simple.json

- **Status**: active
- **File**: `test_workflow_simple.json`
- **Description**: Configuration file: test_workflow_simple.json
- **Last Modified**: 2025-06-04 19:35:39

### Configuration: tool_discovery_database_test_report.json

- **Status**: active
- **File**: `tool_discovery_database_test_report.json`
- **Description**: Configuration file: tool_discovery_database_test_report.json
- **Last Modified**: 2025-06-06 00:13:53

### Configuration: ui_test_report.json

- **Status**: active
- **File**: `ui_test_report.json`
- **Description**: Configuration file: ui_test_report.json
- **Last Modified**: 2025-06-02 12:48:34

### Configuration: validation_config.json

- **Status**: active
- **File**: `validation_config.json`
- **Description**: Configuration file: validation_config.json
- **Last Modified**: 2025-05-29 02:09:30

### Configuration: validation_history.json

- **Status**: active
- **File**: `validation_history.json`
- **Description**: Configuration file: validation_history.json
- **Last Modified**: 2025-05-29 02:10:01

## Utility Script

**Count**: 136

### Utility Script: analyze_cloudflare_mcp_servers

- **Status**: active
- **File**: `analyze_cloudflare_mcp_servers.py`
- **Description**: Utility/analysis script: analyze_cloudflare_mcp_servers
- **Last Modified**: 2025-06-09 20:10:45

### Utility Script: analyze_mcp_server_languages

- **Status**: active
- **File**: `analyze_mcp_server_languages.py`
- **Description**: Utility/analysis script: analyze_mcp_server_languages
- **Last Modified**: 2025-06-09 20:06:39

### Utility Script: analyze_memory_system

- **Status**: active
- **File**: `analyze_memory_system.py`
- **Description**: Utility/analysis script: analyze_memory_system
- **Last Modified**: 2025-06-06 00:30:22

### Utility Script: check_mcp_servers

- **Status**: active
- **File**: `check_mcp_servers.py`
- **Description**: Utility/analysis script: check_mcp_servers
- **Last Modified**: 2025-06-05 22:29:42

### Utility Script: check_mcp_servers_fixed

- **Status**: active
- **File**: `check_mcp_servers_fixed.py`
- **Description**: Utility/analysis script: check_mcp_servers_fixed
- **Last Modified**: 2025-06-05 11:29:09

### Utility Script: comprehensive_memory_test

- **Status**: active
- **File**: `comprehensive_memory_test.py`
- **Description**: Utility/analysis script: comprehensive_memory_test
- **Last Modified**: 2025-06-06 00:33:52

### Utility Script: comprehensive_test

- **Status**: active
- **File**: `comprehensive_test.py`
- **Description**: Utility/analysis script: comprehensive_test
- **Last Modified**: 2025-06-07 11:22:36

### Utility Script: comprehensive_validation_report

- **Status**: active
- **File**: `comprehensive_validation_report.py`
- **Description**: Utility/analysis script: comprehensive_validation_report
- **Last Modified**: 2025-06-07 11:22:36

### Utility Script: context7_demo

- **Status**: active
- **File**: `context7_demo.py`
- **Description**: Utility/analysis script: context7_demo
- **Last Modified**: 2025-06-05 11:29:11

### Utility Script: context7_http_wrapper

- **Status**: active
- **File**: `context7_http_wrapper.py`
- **Description**: Utility/analysis script: context7_http_wrapper
- **Last Modified**: 2025-06-05 11:29:11

### Utility Script: context7_mcp_client

- **Status**: active
- **File**: `context7_mcp_client.py`
- **Description**: Utility/analysis script: context7_mcp_client
- **Last Modified**: 2025-06-05 11:29:11

### Utility Script: context7_sdk_test

- **Status**: active
- **File**: `context7_sdk_test.py`
- **Description**: Utility/analysis script: context7_sdk_test
- **Last Modified**: 2025-06-05 11:41:51

### Utility Script: context7_simple_test

- **Status**: active
- **File**: `context7_simple_test.py`
- **Description**: Utility/analysis script: context7_simple_test
- **Last Modified**: 2025-06-05 11:29:11

### Utility Script: create_mcp_tables

- **Status**: active
- **File**: `create_mcp_tables.py`
- **Description**: Utility/analysis script: create_mcp_tables
- **Last Modified**: 2025-06-06 00:30:22

### Utility Script: create_printable_workflow

- **Status**: active
- **File**: `create_printable_workflow.py`
- **Description**: Utility/analysis script: create_printable_workflow
- **Last Modified**: 2025-06-08 15:13:35

### Utility Script: debug_document_loading

- **Status**: active
- **File**: `debug_document_loading.py`
- **Description**: Utility/analysis script: debug_document_loading
- **Last Modified**: 2025-06-09 17:07:50

### Utility Script: debug_mcp_ui

- **Status**: active
- **File**: `debug_mcp_ui.py`
- **Description**: Utility/analysis script: debug_mcp_ui
- **Last Modified**: 2025-06-09 17:07:50

### Utility Script: debug_settings

- **Status**: active
- **File**: `debug_settings.py`
- **Description**: Utility/analysis script: debug_settings
- **Last Modified**: 2025-06-09 17:28:23

### Utility Script: demo_context7_python

- **Status**: active
- **File**: `demo_context7_python.py`
- **Description**: Utility/analysis script: demo_context7_python
- **Last Modified**: 2025-06-05 11:29:11

### Utility Script: demo_mcp_weapon_system

- **Status**: active
- **File**: `demo_mcp_weapon_system.py`
- **Description**: Utility/analysis script: demo_mcp_weapon_system
- **Last Modified**: 2025-06-07 15:35:22

### Utility Script: doc_server

- **Status**: active
- **File**: `doc_server.py`
- **Description**: Utility/analysis script: doc_server
- **Last Modified**: 2025-06-09 11:16:24

### Utility Script: fastapi_best_practices

- **Status**: active
- **File**: `fastapi_best_practices.py`
- **Description**: Utility/analysis script: fastapi_best_practices
- **Last Modified**: 2025-06-05 22:29:42

### Utility Script: feature_discovery_system

- **Status**: active
- **File**: `feature_discovery_system.py`
- **Description**: Utility/analysis script: feature_discovery_system
- **Last Modified**: 2025-06-09 21:12:35

### Utility Script: feature_workflow_integration

- **Status**: active
- **File**: `feature_workflow_integration.py`
- **Description**: Utility/analysis script: feature_workflow_integration
- **Last Modified**: 2025-06-09 21:20:24

### Utility Script: final_mock_removal_validation

- **Status**: active
- **File**: `final_mock_removal_validation.py`
- **Description**: Utility/analysis script: final_mock_removal_validation
- **Last Modified**: 2025-06-07 11:22:36

### Utility Script: final_syntax_validation

- **Status**: active
- **File**: `final_syntax_validation.py`
- **Description**: Utility/analysis script: final_syntax_validation
- **Last Modified**: 2025-06-07 11:22:36

### Utility Script: final_test

- **Status**: active
- **File**: `final_test.py`
- **Description**: Utility/analysis script: final_test
- **Last Modified**: 2025-06-07 11:22:36

### Utility Script: final_test_battery

- **Status**: active
- **File**: `final_test_battery.py`
- **Description**: Utility/analysis script: final_test_battery
- **Last Modified**: 2025-06-07 11:22:36

### Utility Script: fix_integration_plan

- **Status**: active
- **File**: `fix_integration_plan.py`
- **Description**: Utility/analysis script: fix_integration_plan
- **Last Modified**: 2025-06-06 00:06:12

### Utility Script: fix_mcp_tool_discovery

- **Status**: active
- **File**: `fix_mcp_tool_discovery.py`
- **Description**: Utility/analysis script: fix_mcp_tool_discovery
- **Last Modified**: 2025-06-06 00:06:12

### Utility Script: fix_memory_system

- **Status**: active
- **File**: `fix_memory_system.py`
- **Description**: Utility/analysis script: fix_memory_system
- **Last Modified**: 2025-06-06 00:36:28

### Utility Script: frontend_connectivity_diagnosis

- **Status**: active
- **File**: `frontend_connectivity_diagnosis.py`
- **Description**: Utility/analysis script: frontend_connectivity_diagnosis
- **Last Modified**: 2025-06-03 22:39:52

### Utility Script: generate_workflow_pdf

- **Status**: active
- **File**: `generate_workflow_pdf.py`
- **Description**: Utility/analysis script: generate_workflow_pdf
- **Last Modified**: 2025-06-08 15:13:35

### Utility Script: get_fastapi_best_practices

- **Status**: active
- **File**: `get_fastapi_best_practices.py`
- **Description**: Utility/analysis script: get_fastapi_best_practices
- **Last Modified**: 2025-06-05 22:29:42

### Utility Script: investigate_capabilities_comprehensive

- **Status**: active
- **File**: `investigate_capabilities_comprehensive.py`
- **Description**: Utility/analysis script: investigate_capabilities_comprehensive
- **Last Modified**: 2025-06-05 23:57:46

### Utility Script: investigate_capabilities_simple

- **Status**: active
- **File**: `investigate_capabilities_simple.py`
- **Description**: Utility/analysis script: investigate_capabilities_simple
- **Last Modified**: 2025-06-05 23:57:45

### Utility Script: investigate_live_tools

- **Status**: active
- **File**: `investigate_live_tools.py`
- **Description**: Utility/analysis script: investigate_live_tools
- **Last Modified**: 2025-06-06 00:00:45

### Utility Script: investigate_mcp_capabilities

- **Status**: active
- **File**: `investigate_mcp_capabilities.py`
- **Description**: Utility/analysis script: investigate_mcp_capabilities
- **Last Modified**: 2025-06-05 23:57:45

### Utility Script: investigate_mcp_persistence

- **Status**: active
- **File**: `investigate_mcp_persistence.py`
- **Description**: Utility/analysis script: investigate_mcp_persistence
- **Last Modified**: 2025-06-05 23:57:45

### Utility Script: investigate_mcp_persistence_fixed

- **Status**: active
- **File**: `investigate_mcp_persistence_fixed.py`
- **Description**: Utility/analysis script: investigate_mcp_persistence_fixed
- **Last Modified**: 2025-06-05 23:57:45

### Utility Script: main

- **Status**: active
- **File**: `main.py`
- **Description**: Utility/analysis script: main
- **Last Modified**: 2025-06-07 11:38:02

### Utility Script: mcp_server_python

- **Status**: active
- **File**: `mcp_server_python.py`
- **Description**: Utility/analysis script: mcp_server_python
- **Last Modified**: 2025-06-08 01:55:38

### Utility Script: mcp_servers_reference

- **Status**: active
- **File**: `mcp_servers_reference.py`
- **Description**: Utility/analysis script: mcp_servers_reference
- **Last Modified**: 2025-06-07 20:10:03

### Utility Script: memory_system_analysis

- **Status**: active
- **File**: `memory_system_analysis.py`
- **Description**: Utility/analysis script: memory_system_analysis
- **Last Modified**: 2025-06-06 00:30:22

### Utility Script: memory_system_gpu_test

- **Status**: active
- **File**: `memory_system_gpu_test.py`
- **Description**: Utility/analysis script: memory_system_gpu_test
- **Last Modified**: 2025-06-06 15:19:45

### Utility Script: memory_system_monitor

- **Status**: active
- **File**: `memory_system_monitor.py`
- **Description**: Utility/analysis script: memory_system_monitor
- **Last Modified**: 2025-06-06 15:19:45

### Utility Script: minimal_test

- **Status**: active
- **File**: `minimal_test.py`
- **Description**: Utility/analysis script: minimal_test
- **Last Modified**: 2025-06-04 12:09:22

### Utility Script: oauth_manager

- **Status**: active
- **File**: `oauth_manager.py`
- **Description**: Utility/analysis script: oauth_manager
- **Last Modified**: 2025-06-08 01:55:38

### Utility Script: production_test

- **Status**: active
- **File**: `production_test.py`
- **Description**: Utility/analysis script: production_test
- **Last Modified**: 2025-06-07 11:22:36

### Utility Script: pygent_context7_integration

- **Status**: active
- **File**: `pygent_context7_integration.py`
- **Description**: Utility/analysis script: pygent_context7_integration
- **Last Modified**: 2025-06-05 11:41:51

### Utility Script: quick_api_test

- **Status**: active
- **File**: `quick_api_test.py`
- **Description**: Utility/analysis script: quick_api_test
- **Last Modified**: 2025-06-09 17:07:50

### Utility Script: quick_backend_test

- **Status**: active
- **File**: `quick_backend_test.py`
- **Description**: Utility/analysis script: quick_backend_test
- **Last Modified**: 2025-06-09 17:28:23

### Utility Script: quick_health_check

- **Status**: active
- **File**: `quick_health_check.py`
- **Description**: Utility/analysis script: quick_health_check
- **Last Modified**: 2025-06-07 11:22:36

### Utility Script: quick_mcp_test

- **Status**: active
- **File**: `quick_mcp_test.py`
- **Description**: Utility/analysis script: quick_mcp_test
- **Last Modified**: 2025-06-08 01:55:38

### Utility Script: quick_production_test

- **Status**: active
- **File**: `quick_production_test.py`
- **Description**: Utility/analysis script: quick_production_test
- **Last Modified**: 2025-06-07 11:22:36

### Utility Script: quick_test

- **Status**: active
- **File**: `quick_test.py`
- **Description**: Utility/analysis script: quick_test
- **Last Modified**: 2025-06-04 20:20:40

### Utility Script: register_context7

- **Status**: active
- **File**: `register_context7.py`
- **Description**: Utility/analysis script: register_context7
- **Last Modified**: 2025-06-05 11:29:09

### Utility Script: setup_git_hooks

- **Status**: active
- **File**: `setup_git_hooks.py`
- **Description**: Utility/analysis script: setup_git_hooks
- **Last Modified**: 2025-06-09 21:20:24

### Utility Script: simple_backend

- **Status**: active
- **File**: `simple_backend.py`
- **Description**: Utility/analysis script: simple_backend
- **Last Modified**: 2025-06-08 11:49:02

### Utility Script: simple_coding_agent_test

- **Status**: active
- **File**: `simple_coding_agent_test.py`
- **Description**: Utility/analysis script: simple_coding_agent_test
- **Last Modified**: 2025-06-07 19:55:48

### Utility Script: simple_coding_agent_test_backup

- **Status**: active
- **File**: `simple_coding_agent_test_backup.py`
- **Description**: Utility/analysis script: simple_coding_agent_test_backup
- **Last Modified**: 2025-06-07 19:53:55

### Utility Script: simple_coding_agent_test_fixed

- **Status**: active
- **File**: `simple_coding_agent_test_fixed.py`
- **Description**: Utility/analysis script: simple_coding_agent_test_fixed
- **Last Modified**: 2025-06-07 19:55:48

### Utility Script: simple_context7_test

- **Status**: active
- **File**: `simple_context7_test.py`
- **Description**: Utility/analysis script: simple_context7_test
- **Last Modified**: 2025-06-05 11:29:11

### Utility Script: simple_feature_discovery

- **Status**: active
- **File**: `simple_feature_discovery.py`
- **Description**: Utility/analysis script: simple_feature_discovery
- **Last Modified**: 2025-06-09 21:12:35

### Utility Script: simple_pdf_generator

- **Status**: active
- **File**: `simple_pdf_generator.py`
- **Description**: Utility/analysis script: simple_pdf_generator
- **Last Modified**: 2025-06-08 15:13:35

### Utility Script: simple_server

- **Status**: active
- **File**: `simple_server.py`
- **Description**: Utility/analysis script: simple_server
- **Last Modified**: 2025-06-09 10:57:55

### Utility Script: simple_test

- **Status**: active
- **File**: `simple_test.py`
- **Description**: Utility/analysis script: simple_test
- **Last Modified**: 2025-06-04 19:37:59

### Utility Script: simple_tool_discovery_test

- **Status**: active
- **File**: `simple_tool_discovery_test.py`
- **Description**: Utility/analysis script: simple_tool_discovery_test
- **Last Modified**: 2025-06-06 00:00:45

### Utility Script: start-backend

- **Status**: active
- **File**: `start-backend.py`
- **Description**: Utility/analysis script: start-backend
- **Last Modified**: 2025-06-08 02:33:02

### Utility Script: start_backend_with_real_servers

- **Status**: active
- **File**: `start_backend_with_real_servers.py`
- **Description**: Utility/analysis script: start_backend_with_real_servers
- **Last Modified**: 2025-06-05 11:29:09

### Utility Script: startup_real_mcp_servers

- **Status**: active
- **File**: `startup_real_mcp_servers.py`
- **Description**: Utility/analysis script: startup_real_mcp_servers
- **Last Modified**: 2025-06-05 11:29:09

### Utility Script: test_all_mcp_servers

- **Status**: active
- **File**: `test_all_mcp_servers.py`
- **Description**: Utility/analysis script: test_all_mcp_servers
- **Last Modified**: 2025-06-08 01:55:38

### Utility Script: test_api_connection

- **Status**: active
- **File**: `test_api_connection.py`
- **Description**: Utility/analysis script: test_api_connection
- **Last Modified**: 2025-06-03 22:55:11

### Utility Script: test_api_fixes

- **Status**: active
- **File**: `test_api_fixes.py`
- **Description**: Utility/analysis script: test_api_fixes
- **Last Modified**: 2025-06-09 17:56:32

### Utility Script: test_authentication_flow

- **Status**: active
- **File**: `test_authentication_flow.py`
- **Description**: Utility/analysis script: test_authentication_flow
- **Last Modified**: 2025-06-09 10:41:29

### Utility Script: test_authentication_flow_fixed

- **Status**: active
- **File**: `test_authentication_flow_fixed.py`
- **Description**: Utility/analysis script: test_authentication_flow_fixed
- **Last Modified**: 2025-06-09 10:41:29

### Utility Script: test_coding_agent_evolution

- **Status**: active
- **File**: `test_coding_agent_evolution.py`
- **Description**: Utility/analysis script: test_coding_agent_evolution
- **Last Modified**: 2025-06-07 21:06:11

### Utility Script: test_complete_deployment

- **Status**: active
- **File**: `test_complete_deployment.py`
- **Description**: Utility/analysis script: test_complete_deployment
- **Last Modified**: 2025-06-03 23:18:39

### Utility Script: test_context7

- **Status**: active
- **File**: `test_context7.py`
- **Description**: Utility/analysis script: test_context7
- **Last Modified**: 2025-06-05 11:29:10

### Utility Script: test_context7_simple

- **Status**: active
- **File**: `test_context7_simple.py`
- **Description**: Utility/analysis script: test_context7_simple
- **Last Modified**: 2025-06-05 11:29:11

### Utility Script: test_core_orchestrator

- **Status**: active
- **File**: `test_core_orchestrator.py`
- **Description**: Utility/analysis script: test_core_orchestrator
- **Last Modified**: 2025-06-04 12:31:13

### Utility Script: test_darwinian_a2a_phase1

- **Status**: active
- **File**: `test_darwinian_a2a_phase1.py`
- **Description**: Utility/analysis script: test_darwinian_a2a_phase1
- **Last Modified**: 2025-06-06 15:19:45

### Utility Script: test_darwinian_a2a_phase1_8

- **Status**: active
- **File**: `test_darwinian_a2a_phase1_8.py`
- **Description**: Utility/analysis script: test_darwinian_a2a_phase1_8
- **Last Modified**: 2025-06-06 15:19:45

### Utility Script: test_darwinian_a2a_phase2_1

- **Status**: active
- **File**: `test_darwinian_a2a_phase2_1.py`
- **Description**: Utility/analysis script: test_darwinian_a2a_phase2_1
- **Last Modified**: 2025-06-06 15:19:45

### Utility Script: test_darwinian_a2a_phase2_2

- **Status**: active
- **File**: `test_darwinian_a2a_phase2_2.py`
- **Description**: Utility/analysis script: test_darwinian_a2a_phase2_2
- **Last Modified**: 2025-06-06 15:19:45

### Utility Script: test_database_connection

- **Status**: active
- **File**: `test_database_connection.py`
- **Description**: Utility/analysis script: test_database_connection
- **Last Modified**: 2025-06-05 23:57:45

### Utility Script: test_distributed_genetic_algorithm_comprehensive

- **Status**: active
- **File**: `test_distributed_genetic_algorithm_comprehensive.py`
- **Description**: Utility/analysis script: test_distributed_genetic_algorithm_comprehensive
- **Last Modified**: 2025-06-06 15:19:45

### Utility Script: test_distributed_genetic_algorithm_phase2_1

- **Status**: active
- **File**: `test_distributed_genetic_algorithm_phase2_1.py`
- **Description**: Utility/analysis script: test_distributed_genetic_algorithm_phase2_1
- **Last Modified**: 2025-06-06 15:19:45

### Utility Script: test_doc_server

- **Status**: active
- **File**: `test_doc_server.py`
- **Description**: Utility/analysis script: test_doc_server
- **Last Modified**: 2025-06-09 10:57:55

### Utility Script: test_documentation_api

- **Status**: active
- **File**: `test_documentation_api.py`
- **Description**: Utility/analysis script: test_documentation_api
- **Last Modified**: 2025-06-09 10:26:43

### Utility Script: test_documentation_import

- **Status**: active
- **File**: `test_documentation_import.py`
- **Description**: Utility/analysis script: test_documentation_import
- **Last Modified**: 2025-06-04 12:08:31

### Utility Script: test_documentation_ui

- **Status**: active
- **File**: `test_documentation_ui.py`
- **Description**: Utility/analysis script: test_documentation_ui
- **Last Modified**: 2025-06-09 17:07:50

### Utility Script: test_endpoint_fixes

- **Status**: active
- **File**: `test_endpoint_fixes.py`
- **Description**: Utility/analysis script: test_endpoint_fixes
- **Last Modified**: 2025-06-09 17:28:23

### Utility Script: test_enhanced_registry

- **Status**: active
- **File**: `test_enhanced_registry.py`
- **Description**: Utility/analysis script: test_enhanced_registry
- **Last Modified**: 2025-06-06 00:00:45

### Utility Script: test_evolution_effectiveness

- **Status**: active
- **File**: `test_evolution_effectiveness.py`
- **Description**: Utility/analysis script: test_evolution_effectiveness
- **Last Modified**: 2025-06-07 15:35:22

### Utility Script: test_evolution_effectiveness_corrected

- **Status**: active
- **File**: `test_evolution_effectiveness_corrected.py`
- **Description**: Utility/analysis script: test_evolution_effectiveness_corrected
- **Last Modified**: 2025-06-07 15:35:22

### Utility Script: test_fixed_orchestrator

- **Status**: active
- **File**: `test_fixed_orchestrator.py`
- **Description**: Utility/analysis script: test_fixed_orchestrator
- **Last Modified**: 2025-06-04 12:22:59

### Utility Script: test_full_documentation_flow

- **Status**: active
- **File**: `test_full_documentation_flow.py`
- **Description**: Utility/analysis script: test_full_documentation_flow
- **Last Modified**: 2025-06-09 12:08:21

### Utility Script: test_health

- **Status**: active
- **File**: `test_health.py`
- **Description**: Utility/analysis script: test_health
- **Last Modified**: 2025-06-04 19:51:29

### Utility Script: test_intelligent_docs

- **Status**: active
- **File**: `test_intelligent_docs.py`
- **Description**: Utility/analysis script: test_intelligent_docs
- **Last Modified**: 2025-06-04 13:45:44

### Utility Script: test_mcp_api

- **Status**: active
- **File**: `test_mcp_api.py`
- **Description**: Utility/analysis script: test_mcp_api
- **Last Modified**: 2025-06-05 11:29:09

### Utility Script: test_mcp_functionality

- **Status**: active
- **File**: `test_mcp_functionality.py`
- **Description**: Utility/analysis script: test_mcp_functionality
- **Last Modified**: 2025-06-07 20:10:03

### Utility Script: test_mcp_installation

- **Status**: active
- **File**: `test_mcp_installation.py`
- **Description**: Utility/analysis script: test_mcp_installation
- **Last Modified**: 2025-06-02 20:26:30

### Utility Script: test_mcp_simple

- **Status**: active
- **File**: `test_mcp_simple.py`
- **Description**: Utility/analysis script: test_mcp_simple
- **Last Modified**: 2025-06-08 01:55:38

### Utility Script: test_mcp_weapon_advanced

- **Status**: active
- **File**: `test_mcp_weapon_advanced.py`
- **Description**: Utility/analysis script: test_mcp_weapon_advanced
- **Last Modified**: 2025-06-07 15:35:22

### Utility Script: test_mcp_weapon_integration

- **Status**: active
- **File**: `test_mcp_weapon_integration.py`
- **Description**: Utility/analysis script: test_mcp_weapon_integration
- **Last Modified**: 2025-06-07 15:35:22

### Utility Script: test_mcp_weapon_simple

- **Status**: active
- **File**: `test_mcp_weapon_simple.py`
- **Description**: Utility/analysis script: test_mcp_weapon_simple
- **Last Modified**: 2025-06-07 15:35:22

### Utility Script: test_mcp_weapon_simple_integration

- **Status**: active
- **File**: `test_mcp_weapon_simple_integration.py`
- **Description**: Utility/analysis script: test_mcp_weapon_simple_integration
- **Last Modified**: 2025-06-07 15:35:22

### Utility Script: test_mermaid_basic

- **Status**: active
- **File**: `test_mermaid_basic.py`
- **Description**: Utility/analysis script: test_mermaid_basic
- **Last Modified**: 2025-06-04 13:57:35

### Utility Script: test_mock_removal_validation

- **Status**: active
- **File**: `test_mock_removal_validation.py`
- **Description**: Utility/analysis script: test_mock_removal_validation
- **Last Modified**: 2025-06-06 22:01:40

### Utility Script: test_oauth_integration

- **Status**: active
- **File**: `test_oauth_integration.py`
- **Description**: Utility/analysis script: test_oauth_integration
- **Last Modified**: 2025-06-08 22:38:17

### Utility Script: test_ollama

- **Status**: active
- **File**: `test_ollama.py`
- **Description**: Utility/analysis script: test_ollama
- **Last Modified**: 2025-06-04 20:16:02

### Utility Script: test_orchestrator_imports

- **Status**: active
- **File**: `test_orchestrator_imports.py`
- **Description**: Utility/analysis script: test_orchestrator_imports
- **Last Modified**: 2025-06-04 12:09:58

### Utility Script: test_orchestrator_integration

- **Status**: active
- **File**: `test_orchestrator_integration.py`
- **Description**: Utility/analysis script: test_orchestrator_integration
- **Last Modified**: 2025-06-04 12:15:50

### Utility Script: test_original_problem

- **Status**: active
- **File**: `test_original_problem.py`
- **Description**: Utility/analysis script: test_original_problem
- **Last Modified**: 2025-06-04 12:17:51

### Utility Script: test_production_websocket

- **Status**: active
- **File**: `test_production_websocket.py`
- **Description**: Utility/analysis script: test_production_websocket
- **Last Modified**: 2025-06-03 22:37:33

### Utility Script: test_public_documentation

- **Status**: active
- **File**: `test_public_documentation.py`
- **Description**: Utility/analysis script: test_public_documentation
- **Last Modified**: 2025-06-09 12:08:21

### Utility Script: test_python_mcp_servers

- **Status**: active
- **File**: `test_python_mcp_servers.py`
- **Description**: Utility/analysis script: test_python_mcp_servers
- **Last Modified**: 2025-06-08 01:55:40

### Utility Script: test_real_gpu_usage

- **Status**: active
- **File**: `test_real_gpu_usage.py`
- **Description**: Utility/analysis script: test_real_gpu_usage
- **Last Modified**: 2025-06-06 15:19:45

### Utility Script: test_research_analysis_workflow

- **Status**: active
- **File**: `test_research_analysis_workflow.py`
- **Description**: Utility/analysis script: test_research_analysis_workflow
- **Last Modified**: 2025-06-03 01:34:45

### Utility Script: test_status

- **Status**: active
- **File**: `test_status.py`
- **Description**: Utility/analysis script: test_status
- **Last Modified**: 2025-06-04 19:50:02

### Utility Script: test_tool_discovery_database

- **Status**: active
- **File**: `test_tool_discovery_database.py`
- **Description**: Utility/analysis script: test_tool_discovery_database
- **Last Modified**: 2025-06-06 00:30:22

### Utility Script: test_tool_discovery_database_fixed

- **Status**: active
- **File**: `test_tool_discovery_database_fixed.py`
- **Description**: Utility/analysis script: test_tool_discovery_database_fixed
- **Last Modified**: 2025-06-06 00:30:22

### Utility Script: test_two_phase_evolution

- **Status**: active
- **File**: `test_two_phase_evolution.py`
- **Description**: Utility/analysis script: test_two_phase_evolution
- **Last Modified**: 2025-06-07 15:35:22

### Utility Script: test_user_services_integration

- **Status**: active
- **File**: `test_user_services_integration.py`
- **Description**: Utility/analysis script: test_user_services_integration
- **Last Modified**: 2025-06-08 22:51:31

### Utility Script: test_websocket

- **Status**: active
- **File**: `test_websocket.py`
- **Description**: Utility/analysis script: test_websocket
- **Last Modified**: 2025-06-08 11:28:23

### Utility Script: test_websocket_connection

- **Status**: active
- **File**: `test_websocket_connection.py`
- **Description**: Utility/analysis script: test_websocket_connection
- **Last Modified**: 2025-06-03 22:57:40

### Utility Script: test_workflow_api

- **Status**: active
- **File**: `test_workflow_api.py`
- **Description**: Utility/analysis script: test_workflow_api
- **Last Modified**: 2025-06-03 08:10:02

### Utility Script: test_workflows_import

- **Status**: active
- **File**: `test_workflows_import.py`
- **Description**: Utility/analysis script: test_workflows_import
- **Last Modified**: 2025-06-03 02:48:00

### Utility Script: trigger_deployment

- **Status**: active
- **File**: `trigger_deployment.py`
- **Description**: Utility/analysis script: trigger_deployment
- **Last Modified**: 2025-06-03 23:19:51

### Utility Script: update_mcp_servers

- **Status**: active
- **File**: `update_mcp_servers.py`
- **Description**: Utility/analysis script: update_mcp_servers
- **Last Modified**: 2025-06-05 11:29:09

### Utility Script: update_mcp_servers_api

- **Status**: active
- **File**: `update_mcp_servers_api.py`
- **Description**: Utility/analysis script: update_mcp_servers_api
- **Last Modified**: 2025-06-05 23:57:45

### Utility Script: validate_mcp_servers

- **Status**: active
- **File**: `validate_mcp_servers.py`
- **Description**: Utility/analysis script: validate_mcp_servers
- **Last Modified**: 2025-06-08 01:55:38

### Utility Script: validate_mcp_servers_fixed

- **Status**: active
- **File**: `validate_mcp_servers_fixed.py`
- **Description**: Utility/analysis script: validate_mcp_servers_fixed
- **Last Modified**: 2025-06-08 01:55:38

### Utility Script: validate_research_analysis_implementation

- **Status**: active
- **File**: `validate_research_analysis_implementation.py`
- **Description**: Utility/analysis script: validate_research_analysis_implementation
- **Last Modified**: 2025-06-03 01:40:31

### Utility Script: verify_memory_fixes

- **Status**: active
- **File**: `verify_memory_fixes.py`
- **Description**: Utility/analysis script: verify_memory_fixes
- **Last Modified**: 2025-06-06 00:37:45

## Documentation

**Count**: 149

### Documentation: A2A_ARCHITECTURE

- **Status**: active
- **File**: `docs\A2A_ARCHITECTURE.md`
- **Description**: Documentation file: A2A_ARCHITECTURE.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_COMMUNICATION_ANALYSIS

- **Status**: active
- **File**: `A2A_COMMUNICATION_ANALYSIS.md`
- **Description**: Documentation file: A2A_COMMUNICATION_ANALYSIS.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_DATA_MODELS

- **Status**: active
- **File**: `docs\A2A_DATA_MODELS.md`
- **Description**: Documentation file: A2A_DATA_MODELS.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_DGM_ADVANCED_FEATURES

- **Status**: active
- **File**: `docs\A2A_DGM_ADVANCED_FEATURES.md`
- **Description**: Documentation file: A2A_DGM_ADVANCED_FEATURES.md
- **Last Modified**: 2025-06-08 20:16:03

### Documentation: A2A_DGM_COMPREHENSIVE_REFINEMENT_PLAN

- **Status**: active
- **File**: `docs\A2A_DGM_COMPREHENSIVE_REFINEMENT_PLAN.md`
- **Description**: Documentation file: A2A_DGM_COMPREHENSIVE_REFINEMENT_PLAN.md
- **Last Modified**: 2025-06-09 09:41:44

### Documentation: A2A_DGM_DOCUMENTATION_INDEX

- **Status**: active
- **File**: `docs\A2A_DGM_DOCUMENTATION_INDEX.md`
- **Description**: Documentation file: A2A_DGM_DOCUMENTATION_INDEX.md
- **Last Modified**: 2025-06-08 20:31:58

### Documentation: A2A_DGM_IMPLEMENTATION_COMPLETE

- **Status**: active
- **File**: `docs\A2A_DGM_IMPLEMENTATION_COMPLETE.md`
- **Description**: Documentation file: A2A_DGM_IMPLEMENTATION_COMPLETE.md
- **Last Modified**: 2025-06-08 20:16:03

### Documentation: A2A_DGM_IMPLEMENTATION_ROADMAP

- **Status**: active
- **File**: `docs\A2A_DGM_IMPLEMENTATION_ROADMAP.md`
- **Description**: Documentation file: A2A_DGM_IMPLEMENTATION_ROADMAP.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_DGM_INTEGRATION_STRATEGY

- **Status**: active
- **File**: `docs\A2A_DGM_INTEGRATION_STRATEGY.md`
- **Description**: Documentation file: A2A_DGM_INTEGRATION_STRATEGY.md
- **Last Modified**: 2025-06-09 09:41:44

### Documentation: A2A_DGM_RISK_ASSESSMENT

- **Status**: active
- **File**: `docs\A2A_DGM_RISK_ASSESSMENT.md`
- **Description**: Documentation file: A2A_DGM_RISK_ASSESSMENT.md
- **Last Modified**: 2025-06-08 17:16:51

### Documentation: A2A_DGM_RISK_MITIGATION

- **Status**: active
- **File**: `docs\A2A_DGM_RISK_MITIGATION.md`
- **Description**: Documentation file: A2A_DGM_RISK_MITIGATION.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_INTEGRATION_PHASE_1

- **Status**: active
- **File**: `docs\A2A_INTEGRATION_PHASE_1.md`
- **Description**: Documentation file: A2A_INTEGRATION_PHASE_1.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_PROTOCOL_ARCHITECTURE

- **Status**: active
- **File**: `docs\A2A_PROTOCOL_ARCHITECTURE.md`
- **Description**: Documentation file: A2A_PROTOCOL_ARCHITECTURE.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_PROTOCOL_IMPLEMENTATION_GUIDE

- **Status**: active
- **File**: `docs\A2A_PROTOCOL_IMPLEMENTATION_GUIDE.md`
- **Description**: Documentation file: A2A_PROTOCOL_IMPLEMENTATION_GUIDE.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_PROTOCOL_METHODS

- **Status**: active
- **File**: `docs\A2A_PROTOCOL_METHODS.md`
- **Description**: Documentation file: A2A_PROTOCOL_METHODS.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_PROTOCOL_OVERVIEW

- **Status**: active
- **File**: `docs\A2A_PROTOCOL_OVERVIEW.md`
- **Description**: Documentation file: A2A_PROTOCOL_OVERVIEW.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_PROTOCOL_SECURITY

- **Status**: active
- **File**: `docs\A2A_PROTOCOL_SECURITY.md`
- **Description**: Documentation file: A2A_PROTOCOL_SECURITY.md
- **Last Modified**: 2025-06-09 09:41:44

### Documentation: A2A_PROTOCOL_TECHNICAL_SPEC

- **Status**: active
- **File**: `docs\A2A_PROTOCOL_TECHNICAL_SPEC.md`
- **Description**: Documentation file: A2A_PROTOCOL_TECHNICAL_SPEC.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_SECURITY

- **Status**: active
- **File**: `docs\A2A_SECURITY.md`
- **Description**: Documentation file: A2A_SECURITY.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: A2A_SECURITY_AUTHENTICATION

- **Status**: active
- **File**: `docs\A2A_SECURITY_AUTHENTICATION.md`
- **Description**: Documentation file: A2A_SECURITY_AUTHENTICATION.md
- **Last Modified**: 2025-06-08 20:31:58

### Documentation: A2A_SECURITY_OVERVIEW

- **Status**: active
- **File**: `docs\A2A_SECURITY_OVERVIEW.md`
- **Description**: Documentation file: A2A_SECURITY_OVERVIEW.md
- **Last Modified**: 2025-06-08 20:31:58

### Documentation: ADVANCED_PERFORMANCE_ANALYSIS

- **Status**: active
- **File**: `ADVANCED_PERFORMANCE_ANALYSIS.md`
- **Description**: Documentation file: ADVANCED_PERFORMANCE_ANALYSIS.md
- **Last Modified**: 2025-05-28 09:11:29

### Documentation: AGENT_WORKFLOW_DIAGRAM

- **Status**: active
- **File**: `AGENT_WORKFLOW_DIAGRAM.md`
- **Description**: Documentation file: AGENT_WORKFLOW_DIAGRAM.md
- **Last Modified**: 2025-06-08 15:13:35

### Documentation: ARCHITECTURE

- **Status**: active
- **File**: `ARCHITECTURE.md`
- **Description**: Documentation file: ARCHITECTURE.md
- **Last Modified**: 2025-05-27 22:10:38

### Documentation: ARCHITECTURE_OVERVIEW

- **Status**: active
- **File**: `docs\ARCHITECTURE_OVERVIEW.md`
- **Description**: Documentation file: ARCHITECTURE_OVERVIEW.md
- **Last Modified**: 2025-06-08 20:16:03

### Documentation: CLOUDFLARE_MCP_AUTHENTICATION_GUIDE

- **Status**: active
- **File**: `CLOUDFLARE_MCP_AUTHENTICATION_GUIDE.md`
- **Description**: Documentation file: CLOUDFLARE_MCP_AUTHENTICATION_GUIDE.md
- **Last Modified**: 2025-06-08 01:55:38

### Documentation: CLOUDFLARE_MCP_INTEGRATION_ANALYSIS

- **Status**: active
- **File**: `CLOUDFLARE_MCP_INTEGRATION_ANALYSIS.md`
- **Description**: Documentation file: CLOUDFLARE_MCP_INTEGRATION_ANALYSIS.md
- **Last Modified**: 2025-06-08 01:55:38

### Documentation: CLOUDFLARE_MCP_INTEGRATION_COMPLETE

- **Status**: active
- **File**: `CLOUDFLARE_MCP_INTEGRATION_COMPLETE.md`
- **Description**: Documentation file: CLOUDFLARE_MCP_INTEGRATION_COMPLETE.md
- **Last Modified**: 2025-06-08 01:55:40

### Documentation: CLOUDFLARE_PAGES_SETUP

- **Status**: active
- **File**: `CLOUDFLARE_PAGES_SETUP.md`
- **Description**: Documentation file: CLOUDFLARE_PAGES_SETUP.md
- **Last Modified**: 2025-06-08 13:12:37

### Documentation: COMPLETE_FEATURE_REGISTRY

- **Status**: active
- **File**: `docs\COMPLETE_FEATURE_REGISTRY.md`
- **Description**: Documentation file: COMPLETE_FEATURE_REGISTRY.md
- **Last Modified**: 2025-06-09 21:22:01

### Documentation: CRITICAL_FIXES_REQUIRED

- **Status**: active
- **File**: `CRITICAL_FIXES_REQUIRED.md`
- **Description**: Documentation file: CRITICAL_FIXES_REQUIRED.md
- **Last Modified**: 2025-06-04 17:03:48

### Documentation: CURRENT_STATUS

- **Status**: active
- **File**: `CURRENT_STATUS.md`
- **Description**: Documentation file: CURRENT_STATUS.md
- **Last Modified**: 2025-06-09 12:08:21

### Documentation: DARWINIAN_A2A_IMPLEMENTATION_PLAN

- **Status**: active
- **File**: `DARWINIAN_A2A_IMPLEMENTATION_PLAN.md`
- **Description**: Documentation file: DARWINIAN_A2A_IMPLEMENTATION_PLAN.md
- **Last Modified**: 2025-06-06 15:19:45

### Documentation: DEEP_A2A_PROTOCOL_ANALYSIS

- **Status**: active
- **File**: `DEEP_A2A_PROTOCOL_ANALYSIS.md`
- **Description**: Documentation file: DEEP_A2A_PROTOCOL_ANALYSIS.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: DEEP_RESEARCH_COMPLETION_SUMMARY

- **Status**: active
- **File**: `DEEP_RESEARCH_COMPLETION_SUMMARY.md`
- **Description**: Documentation file: DEEP_RESEARCH_COMPLETION_SUMMARY.md
- **Last Modified**: 2025-06-04 23:14:11

### Documentation: DEEP_RESEARCH_TOT_MCP_ORCHESTRATION

- **Status**: active
- **File**: `DEEP_RESEARCH_TOT_MCP_ORCHESTRATION.md`
- **Description**: Documentation file: DEEP_RESEARCH_TOT_MCP_ORCHESTRATION.md
- **Last Modified**: 2025-06-08 15:13:35

### Documentation: DEEP_SYSTEM_ANALYSIS

- **Status**: active
- **File**: `DEEP_SYSTEM_ANALYSIS.md`
- **Description**: Documentation file: DEEP_SYSTEM_ANALYSIS.md
- **Last Modified**: 2025-06-05 11:29:08

### Documentation: DEEP_SYSTEM_RESEARCH_REPORT_2025

- **Status**: active
- **File**: `docs\DEEP_SYSTEM_RESEARCH_REPORT_2025.md`
- **Description**: Documentation file: DEEP_SYSTEM_RESEARCH_REPORT_2025.md
- **Last Modified**: 2025-06-09 20:14:55

### Documentation: DEPENDENCIES

- **Status**: active
- **File**: `DEPENDENCIES.md`
- **Description**: Documentation file: DEPENDENCIES.md
- **Last Modified**: 2025-05-27 22:11:48

### Documentation: DEPLOYMENT_CHECKLIST

- **Status**: active
- **File**: `DEPLOYMENT_CHECKLIST.md`
- **Description**: Documentation file: DEPLOYMENT_CHECKLIST.md
- **Last Modified**: 2025-06-08 12:16:21

### Documentation: DEPLOYMENT_FIX_STATUS

- **Status**: active
- **File**: `DEPLOYMENT_FIX_STATUS.md`
- **Description**: Documentation file: DEPLOYMENT_FIX_STATUS.md
- **Last Modified**: 2025-06-08 13:52:00

### Documentation: DEPLOYMENT_GUIDE

- **Status**: active
- **File**: `DEPLOYMENT_GUIDE.md`
- **Description**: Documentation file: DEPLOYMENT_GUIDE.md
- **Last Modified**: 2025-06-08 12:16:21

### Documentation: DEPLOYMENT_STATUS_FINAL

- **Status**: active
- **File**: `DEPLOYMENT_STATUS_FINAL.md`
- **Description**: Documentation file: DEPLOYMENT_STATUS_FINAL.md
- **Last Modified**: 2025-06-08 13:12:37

### Documentation: DGM_ARCHITECTURE

- **Status**: active
- **File**: `docs\DGM_ARCHITECTURE.md`
- **Description**: Documentation file: DGM_ARCHITECTURE.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: DGM_COMPONENTS_GUIDE

- **Status**: active
- **File**: `docs\DGM_COMPONENTS_GUIDE.md`
- **Description**: Documentation file: DGM_COMPONENTS_GUIDE.md
- **Last Modified**: 2025-06-08 20:16:03

### Documentation: DGM_CORE_ENGINE_DESIGN

- **Status**: active
- **File**: `docs\DGM_CORE_ENGINE_DESIGN.md`
- **Description**: Documentation file: DGM_CORE_ENGINE_DESIGN.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: DGM_ENGINE_IMPLEMENTATION

- **Status**: active
- **File**: `docs\DGM_ENGINE_IMPLEMENTATION.md`
- **Description**: Documentation file: DGM_ENGINE_IMPLEMENTATION.md
- **Last Modified**: 2025-06-08 20:16:03

### Documentation: DGM_INTEGRATION_PHASE_2

- **Status**: active
- **File**: `docs\DGM_INTEGRATION_PHASE_2.md`
- **Description**: Documentation file: DGM_INTEGRATION_PHASE_2.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: DGM_MODELS_SPECIFICATION

- **Status**: active
- **File**: `docs\DGM_MODELS_SPECIFICATION.md`
- **Description**: Documentation file: DGM_MODELS_SPECIFICATION.md
- **Last Modified**: 2025-06-08 20:16:03

### Documentation: DOCUMENTATION_INTEGRATION_COMPLETE

- **Status**: active
- **File**: `DOCUMENTATION_INTEGRATION_COMPLETE.md`
- **Description**: Documentation file: DOCUMENTATION_INTEGRATION_COMPLETE.md
- **Last Modified**: 2025-06-09 12:08:21

### Documentation: DOCUMENTATION_ORCHESTRATOR_IMPLEMENTATION

- **Status**: active
- **File**: `DOCUMENTATION_ORCHESTRATOR_IMPLEMENTATION.md`
- **Description**: Documentation file: DOCUMENTATION_ORCHESTRATOR_IMPLEMENTATION.md
- **Last Modified**: 2025-06-04 11:43:26

### Documentation: DOCUMENTATION_REORGANIZATION_COMPLETE

- **Status**: active
- **File**: `docs\DOCUMENTATION_REORGANIZATION_COMPLETE.md`
- **Description**: Documentation file: DOCUMENTATION_REORGANIZATION_COMPLETE.md
- **Last Modified**: 2025-06-08 20:31:58

### Documentation: DOCUMENTATION_SPLIT_PLAN

- **Status**: active
- **File**: `docs\DOCUMENTATION_SPLIT_PLAN.md`
- **Description**: Documentation file: DOCUMENTATION_SPLIT_PLAN.md
- **Last Modified**: 2025-06-08 20:31:58

### Documentation: DOCUMENTATION_SYSTEM_INTEGRATION_COMPLETE

- **Status**: active
- **File**: `docs\DOCUMENTATION_SYSTEM_INTEGRATION_COMPLETE.md`
- **Description**: Documentation file: DOCUMENTATION_SYSTEM_INTEGRATION_COMPLETE.md
- **Last Modified**: 2025-06-08 22:30:22

### Documentation: DOCUMENTATION_UI_FINAL_STATUS

- **Status**: active
- **File**: `DOCUMENTATION_UI_FINAL_STATUS.md`
- **Description**: Documentation file: DOCUMENTATION_UI_FINAL_STATUS.md
- **Last Modified**: 2025-06-09 17:07:50

### Documentation: ENHANCED_SEARCH_IMPLEMENTATION

- **Status**: active
- **File**: `ENHANCED_SEARCH_IMPLEMENTATION.md`
- **Description**: Documentation file: ENHANCED_SEARCH_IMPLEMENTATION.md
- **Last Modified**: 2025-06-09 17:07:50

### Documentation: ERROR_ANALYSIS

- **Status**: active
- **File**: `ERROR_ANALYSIS.md`
- **Description**: Documentation file: ERROR_ANALYSIS.md
- **Last Modified**: 2025-06-04 17:02:35

### Documentation: FEATURE_REGISTRY_SYSTEM

- **Status**: active
- **File**: `docs\FEATURE_REGISTRY_SYSTEM.md`
- **Description**: Documentation file: FEATURE_REGISTRY_SYSTEM.md
- **Last Modified**: 2025-06-09 21:20:24

### Documentation: FINAL_PROJECT_SUMMARY

- **Status**: active
- **File**: `FINAL_PROJECT_SUMMARY.md`
- **Description**: Documentation file: FINAL_PROJECT_SUMMARY.md
- **Last Modified**: 2025-05-28 09:06:22

### Documentation: FINAL_RESEARCH_SUMMARY

- **Status**: active
- **File**: `FINAL_RESEARCH_SUMMARY.md`
- **Description**: Documentation file: FINAL_RESEARCH_SUMMARY.md
- **Last Modified**: 2025-05-28 21:29:13

### Documentation: FINAL_TESTING_STATUS

- **Status**: active
- **File**: `FINAL_TESTING_STATUS.md`
- **Description**: Documentation file: FINAL_TESTING_STATUS.md
- **Last Modified**: 2025-05-28 11:29:14

### Documentation: FRONTEND_BACKEND_INTEGRATION_COMPLETE

- **Status**: active
- **File**: `FRONTEND_BACKEND_INTEGRATION_COMPLETE.md`
- **Description**: Documentation file: FRONTEND_BACKEND_INTEGRATION_COMPLETE.md
- **Last Modified**: 2025-06-08 09:20:31

### Documentation: GOOGLE_A2A_PROTOCOL_ANALYSIS

- **Status**: active
- **File**: `GOOGLE_A2A_PROTOCOL_ANALYSIS.md`
- **Description**: Documentation file: GOOGLE_A2A_PROTOCOL_ANALYSIS.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: Guidelines

- **Status**: active
- **File**: `Guidelines.md`
- **Description**: Documentation file: Guidelines.md
- **Last Modified**: 2025-06-06 01:58:29

### Documentation: IMPLEMENTATION_COMPLETE

- **Status**: active
- **File**: `IMPLEMENTATION_COMPLETE.md`
- **Description**: Documentation file: IMPLEMENTATION_COMPLETE.md
- **Last Modified**: 2025-06-02 00:41:39

### Documentation: IMPLEMENTATION_PLAN

- **Status**: active
- **File**: `IMPLEMENTATION_PLAN.md`
- **Description**: Documentation file: IMPLEMENTATION_PLAN.md
- **Last Modified**: 2025-05-27 22:08:14

### Documentation: INSTALLED_PYTHON_SDKS

- **Status**: active
- **File**: `INSTALLED_PYTHON_SDKS.md`
- **Description**: Documentation file: INSTALLED_PYTHON_SDKS.md
- **Last Modified**: 2025-06-05 11:41:51

### Documentation: INTEGRATION_ROADMAP

- **Status**: active
- **File**: `docs\INTEGRATION_ROADMAP.md`
- **Description**: Documentation file: INTEGRATION_ROADMAP.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: LESSONS_LEARNED

- **Status**: active
- **File**: `LESSONS_LEARNED.md`
- **Description**: Documentation file: LESSONS_LEARNED.md
- **Last Modified**: 2025-06-08 01:55:40

### Documentation: MASTER_DOCUMENTATION_INDEX

- **Status**: active
- **File**: `docs\MASTER_DOCUMENTATION_INDEX.md`
- **Description**: Documentation file: MASTER_DOCUMENTATION_INDEX.md
- **Last Modified**: 2025-06-08 20:31:58

### Documentation: MASTER_IMPLEMENTATION_PLAN_INDEX

- **Status**: active
- **File**: `docs\MASTER_IMPLEMENTATION_PLAN_INDEX.md`
- **Description**: Documentation file: MASTER_IMPLEMENTATION_PLAN_INDEX.md
- **Last Modified**: 2025-06-08 20:31:58

### Documentation: MASTER_IMPLEMENTATION_PLAN_PART_1

- **Status**: active
- **File**: `docs\MASTER_IMPLEMENTATION_PLAN_PART_1.md`
- **Description**: Documentation file: MASTER_IMPLEMENTATION_PLAN_PART_1.md
- **Last Modified**: 2025-06-08 18:01:49

### Documentation: MASTER_IMPLEMENTATION_PLAN_PART_2

- **Status**: active
- **File**: `docs\MASTER_IMPLEMENTATION_PLAN_PART_2.md`
- **Description**: Documentation file: MASTER_IMPLEMENTATION_PLAN_PART_2.md
- **Last Modified**: 2025-06-08 18:01:49

### Documentation: MASTER_IMPLEMENTATION_PLAN_PART_3

- **Status**: active
- **File**: `docs\MASTER_IMPLEMENTATION_PLAN_PART_3.md`
- **Description**: Documentation file: MASTER_IMPLEMENTATION_PLAN_PART_3.md
- **Last Modified**: 2025-06-09 09:41:44

### Documentation: MASTER_IMPLEMENTATION_PLAN_PART_4

- **Status**: active
- **File**: `docs\MASTER_IMPLEMENTATION_PLAN_PART_4.md`
- **Description**: Documentation file: MASTER_IMPLEMENTATION_PLAN_PART_4.md
- **Last Modified**: 2025-06-08 18:01:49

### Documentation: MASTER_IMPLEMENTATION_PLAN_PART_5

- **Status**: active
- **File**: `docs\MASTER_IMPLEMENTATION_PLAN_PART_5.md`
- **Description**: Documentation file: MASTER_IMPLEMENTATION_PLAN_PART_5.md
- **Last Modified**: 2025-06-08 18:01:49

### Documentation: MASTER_IMPLEMENTATION_PLAN_PART_6

- **Status**: active
- **File**: `docs\MASTER_IMPLEMENTATION_PLAN_PART_6.md`
- **Description**: Documentation file: MASTER_IMPLEMENTATION_PLAN_PART_6.md
- **Last Modified**: 2025-06-08 18:01:49

### Documentation: MCP_AUTO_DETECTION_SYSTEM

- **Status**: active
- **File**: `MCP_AUTO_DETECTION_SYSTEM.md`
- **Description**: Documentation file: MCP_AUTO_DETECTION_SYSTEM.md
- **Last Modified**: 2025-06-07 20:49:44

### Documentation: MCP_AUTO_DETECTION_SYSTEM_ANALYSIS

- **Status**: active
- **File**: `MCP_AUTO_DETECTION_SYSTEM_ANALYSIS.md`
- **Description**: Documentation file: MCP_AUTO_DETECTION_SYSTEM_ANALYSIS.md
- **Last Modified**: 2025-06-07 20:49:44

### Documentation: MCP_MARKETPLACE_FIXES

- **Status**: active
- **File**: `MCP_MARKETPLACE_FIXES.md`
- **Description**: Documentation file: MCP_MARKETPLACE_FIXES.md
- **Last Modified**: 2025-06-09 17:07:50

### Documentation: MCP_MONITORING_FEATURES_UNLOCKED

- **Status**: active
- **File**: `MCP_MONITORING_FEATURES_UNLOCKED.md`
- **Description**: Documentation file: MCP_MONITORING_FEATURES_UNLOCKED.md
- **Last Modified**: 2025-06-08 01:55:38

### Documentation: MCP_SERVERS

- **Status**: active
- **File**: `MCP_SERVERS.md`
- **Description**: Documentation file: MCP_SERVERS.md
- **Last Modified**: 2025-06-08 01:55:40

### Documentation: MCP_TOOL_DISCOVERY_ANALYSIS

- **Status**: active
- **File**: `MCP_TOOL_DISCOVERY_ANALYSIS.md`
- **Description**: Documentation file: MCP_TOOL_DISCOVERY_ANALYSIS.md
- **Last Modified**: 2025-06-06 00:30:22

### Documentation: MCP_TOOL_DISCOVERY_PROJECT_COMPLETE

- **Status**: active
- **File**: `MCP_TOOL_DISCOVERY_PROJECT_COMPLETE.md`
- **Description**: Documentation file: MCP_TOOL_DISCOVERY_PROJECT_COMPLETE.md
- **Last Modified**: 2025-06-06 00:30:22

### Documentation: MCP_WEAPON_SELECTION_COMPLETE

- **Status**: active
- **File**: `MCP_WEAPON_SELECTION_COMPLETE.md`
- **Description**: Documentation file: MCP_WEAPON_SELECTION_COMPLETE.md
- **Last Modified**: 2025-06-07 15:35:22

### Documentation: MCP_WEAPON_SELECTION_CONTEXT_REFRESH

- **Status**: active
- **File**: `MCP_WEAPON_SELECTION_CONTEXT_REFRESH.md`
- **Description**: Documentation file: MCP_WEAPON_SELECTION_CONTEXT_REFRESH.md
- **Last Modified**: 2025-06-07 15:35:22

### Documentation: MCP_WEAPON_SELECTION_FINAL_STATUS

- **Status**: active
- **File**: `MCP_WEAPON_SELECTION_FINAL_STATUS.md`
- **Description**: Documentation file: MCP_WEAPON_SELECTION_FINAL_STATUS.md
- **Last Modified**: 2025-06-07 15:35:22

### Documentation: MEMORY_SYSTEM_ANALYSIS

- **Status**: active
- **File**: `MEMORY_SYSTEM_ANALYSIS.md`
- **Description**: Documentation file: MEMORY_SYSTEM_ANALYSIS.md
- **Last Modified**: 2025-06-06 00:36:28

### Documentation: MEMORY_SYSTEM_ANALYSIS_REPORT

- **Status**: active
- **File**: `MEMORY_SYSTEM_ANALYSIS_REPORT.md`
- **Description**: Documentation file: MEMORY_SYSTEM_ANALYSIS_REPORT.md
- **Last Modified**: 2025-06-06 00:30:22

### Documentation: MEMORY_SYSTEM_COMPLETE_ANALYSIS

- **Status**: active
- **File**: `MEMORY_SYSTEM_COMPLETE_ANALYSIS.md`
- **Description**: Documentation file: MEMORY_SYSTEM_COMPLETE_ANALYSIS.md
- **Last Modified**: 2025-06-06 15:19:45

### Documentation: MEMORY_SYSTEM_RESEARCH_COMPLETE

- **Status**: active
- **File**: `MEMORY_SYSTEM_RESEARCH_COMPLETE.md`
- **Description**: Documentation file: MEMORY_SYSTEM_RESEARCH_COMPLETE.md
- **Last Modified**: 2025-06-06 15:19:45

### Documentation: MIGRATION_COMPLETE_SUMMARY

- **Status**: active
- **File**: `MIGRATION_COMPLETE_SUMMARY.md`
- **Description**: Documentation file: MIGRATION_COMPLETE_SUMMARY.md
- **Last Modified**: 2025-06-08 12:16:21

### Documentation: MIGRATION_LEARNINGS_FINAL

- **Status**: active
- **File**: `MIGRATION_LEARNINGS_FINAL.md`
- **Description**: Documentation file: MIGRATION_LEARNINGS_FINAL.md
- **Last Modified**: 2025-06-08 13:12:37

### Documentation: MINIMUM_MCP_SERVERS_STATUS

- **Status**: active
- **File**: `MINIMUM_MCP_SERVERS_STATUS.md`
- **Description**: Documentation file: MINIMUM_MCP_SERVERS_STATUS.md
- **Last Modified**: 2025-06-07 21:01:03

### Documentation: MOCK_REMOVAL_FINAL_SUCCESS_REPORT

- **Status**: active
- **File**: `MOCK_REMOVAL_FINAL_SUCCESS_REPORT.md`
- **Description**: Documentation file: MOCK_REMOVAL_FINAL_SUCCESS_REPORT.md
- **Last Modified**: 2025-06-07 11:22:36

### Documentation: MOCK_REMOVAL_IMPACT_ANALYSIS

- **Status**: active
- **File**: `MOCK_REMOVAL_IMPACT_ANALYSIS.md`
- **Description**: Documentation file: MOCK_REMOVAL_IMPACT_ANALYSIS.md
- **Last Modified**: 2025-06-06 22:01:40

### Documentation: MOCK_REMOVAL_PROJECT_COMPLETE_REPORT

- **Status**: active
- **File**: `MOCK_REMOVAL_PROJECT_COMPLETE_REPORT.md`
- **Description**: Documentation file: MOCK_REMOVAL_PROJECT_COMPLETE_REPORT.md
- **Last Modified**: 2025-05-28 17:49:47

### Documentation: MOCK_REMOVAL_PROJECT_FINAL_COMPLETION_REPORT

- **Status**: active
- **File**: `MOCK_REMOVAL_PROJECT_FINAL_COMPLETION_REPORT.md`
- **Description**: Documentation file: MOCK_REMOVAL_PROJECT_FINAL_COMPLETION_REPORT.md
- **Last Modified**: 2025-06-07 11:22:36

### Documentation: MOCK_REMOVAL_PROJECT_FINAL_SUCCESS_REPORT

- **Status**: active
- **File**: `MOCK_REMOVAL_PROJECT_FINAL_SUCCESS_REPORT.md`
- **Description**: Documentation file: MOCK_REMOVAL_PROJECT_FINAL_SUCCESS_REPORT.md
- **Last Modified**: 2025-06-07 15:35:22

### Documentation: MOCK_REMOVAL_SUCCESS_AND_ROOT_CAUSE_ANALYSIS

- **Status**: active
- **File**: `MOCK_REMOVAL_SUCCESS_AND_ROOT_CAUSE_ANALYSIS.md`
- **Description**: Documentation file: MOCK_REMOVAL_SUCCESS_AND_ROOT_CAUSE_ANALYSIS.md
- **Last Modified**: 2025-06-07 15:35:22

### Documentation: MODULARIZATION_PROGRESS

- **Status**: active
- **File**: `MODULARIZATION_PROGRESS.md`
- **Description**: Documentation file: MODULARIZATION_PROGRESS.md
- **Last Modified**: 2025-05-27 23:15:30

### Documentation: MODULARIZATION_PROJECT_COMPLETE

- **Status**: active
- **File**: `MODULARIZATION_PROJECT_COMPLETE.md`
- **Description**: Documentation file: MODULARIZATION_PROJECT_COMPLETE.md
- **Last Modified**: 2025-05-28 09:02:55

### Documentation: MODULAR_ARCHITECTURE

- **Status**: active
- **File**: `MODULAR_ARCHITECTURE.md`
- **Description**: Documentation file: MODULAR_ARCHITECTURE.md
- **Last Modified**: 2025-05-27 23:06:21

### Documentation: MODULAR_IMPLEMENTATION_GUIDE

- **Status**: active
- **File**: `MODULAR_IMPLEMENTATION_GUIDE.md`
- **Description**: Documentation file: MODULAR_IMPLEMENTATION_GUIDE.md
- **Last Modified**: 2025-05-28 09:04:14

### Documentation: OAUTH_INTEGRATION_COMPLETE

- **Status**: active
- **File**: `OAUTH_INTEGRATION_COMPLETE.md`
- **Description**: Documentation file: OAUTH_INTEGRATION_COMPLETE.md
- **Last Modified**: 2025-06-08 01:55:38

### Documentation: ORIGINAL_ANALYSIS

- **Status**: active
- **File**: `ORIGINAL_ANALYSIS.md`
- **Description**: Documentation file: ORIGINAL_ANALYSIS.md
- **Last Modified**: 2025-05-27 22:12:26

### Documentation: OVERNIGHT_COLLECTION_README

- **Status**: active
- **File**: `OVERNIGHT_COLLECTION_README.md`
- **Description**: Documentation file: OVERNIGHT_COLLECTION_README.md
- **Last Modified**: 2025-05-29 02:24:56

### Documentation: PDF_GENERATION_CAPABILITY

- **Status**: active
- **File**: `docs\PDF_GENERATION_CAPABILITY.md`
- **Description**: Documentation file: PDF_GENERATION_CAPABILITY.md
- **Last Modified**: 2025-06-08 17:14:32

### Documentation: PHASE2_MODULARIZATION_COMPLETE

- **Status**: active
- **File**: `PHASE2_MODULARIZATION_COMPLETE.md`
- **Description**: Documentation file: PHASE2_MODULARIZATION_COMPLETE.md
- **Last Modified**: 2025-05-27 23:27:36

### Documentation: PHASE3_COMPLETE

- **Status**: active
- **File**: `PHASE3_COMPLETE.md`
- **Description**: Documentation file: PHASE3_COMPLETE.md
- **Last Modified**: 2025-05-27 23:50:24

### Documentation: PHASE3_MCP_MODULARIZATION_PROGRESS

- **Status**: active
- **File**: `PHASE3_MCP_MODULARIZATION_PROGRESS.md`
- **Description**: Documentation file: PHASE3_MCP_MODULARIZATION_PROGRESS.md
- **Last Modified**: 2025-05-27 23:39:54

### Documentation: PHASE4_COMPLETE

- **Status**: active
- **File**: `PHASE4_COMPLETE.md`
- **Description**: Documentation file: PHASE4_COMPLETE.md
- **Last Modified**: 2025-05-28 00:00:23

### Documentation: PRE_DEPLOYMENT_TEST_SUMMARY

- **Status**: active
- **File**: `PRE_DEPLOYMENT_TEST_SUMMARY.md`
- **Description**: Documentation file: PRE_DEPLOYMENT_TEST_SUMMARY.md
- **Last Modified**: 2025-06-05 11:29:09

### Documentation: PYGENT_FACTORY_COMPLETE_SYSTEM_DOCS

- **Status**: active
- **File**: `PYGENT_FACTORY_COMPLETE_SYSTEM_DOCS.md`
- **Description**: Documentation file: PYGENT_FACTORY_COMPLETE_SYSTEM_DOCS.md
- **Last Modified**: 2025-06-08 01:55:38

### Documentation: PYGENT_FACTORY_UI_IMPLEMENTATION

- **Status**: active
- **File**: `PYGENT_FACTORY_UI_IMPLEMENTATION.md`
- **Description**: Documentation file: PYGENT_FACTORY_UI_IMPLEMENTATION.md
- **Last Modified**: 2025-06-02 00:31:39

### Documentation: PyGent_Factory_Deep_Review_2025

- **Status**: active
- **File**: `PyGent_Factory_Deep_Review_2025.md`
- **Description**: Documentation file: PyGent_Factory_Deep_Review_2025.md
- **Last Modified**: 2025-06-07 21:06:11

### Documentation: Pygent Factory Review

- **Status**: active
- **File**: `Pygent Factory Review.md`
- **Description**: Documentation file: Pygent Factory Review.md
- **Last Modified**: 2025-06-07 21:06:11

### Documentation: QUICK_FIX_GUIDE

- **Status**: active
- **File**: `QUICK_FIX_GUIDE.md`
- **Description**: Documentation file: QUICK_FIX_GUIDE.md
- **Last Modified**: 2025-06-08 13:58:21

### Documentation: QUICK_START

- **Status**: active
- **File**: `QUICK_START.md`
- **Description**: Documentation file: QUICK_START.md
- **Last Modified**: 2025-06-02 00:39:47

### Documentation: README

- **Status**: active
- **File**: `docs\README.md`
- **Description**: Documentation file: README.md
- **Last Modified**: 2025-06-05 11:41:51

### Documentation: REAL_MCP_SERVERS_COMPLETE

- **Status**: active
- **File**: `REAL_MCP_SERVERS_COMPLETE.md`
- **Description**: Documentation file: REAL_MCP_SERVERS_COMPLETE.md
- **Last Modified**: 2025-06-08 01:55:40

### Documentation: RESEARCH_ANALYSIS_WORKFLOW_COMPLETE

- **Status**: active
- **File**: `RESEARCH_ANALYSIS_WORKFLOW_COMPLETE.md`
- **Description**: Documentation file: RESEARCH_ANALYSIS_WORKFLOW_COMPLETE.md
- **Last Modified**: 2025-06-03 01:43:07

### Documentation: RESEARCH_DOCUMENTATION_INTEGRATION

- **Status**: active
- **File**: `docs\RESEARCH_DOCUMENTATION_INTEGRATION.md`
- **Description**: Documentation file: RESEARCH_DOCUMENTATION_INTEGRATION.md
- **Last Modified**: 2025-06-08 22:30:22

### Documentation: RESEARCH_PROJECT_COMPLETE_SUMMARY

- **Status**: active
- **File**: `RESEARCH_PROJECT_COMPLETE_SUMMARY.md`
- **Description**: Documentation file: RESEARCH_PROJECT_COMPLETE_SUMMARY.md
- **Last Modified**: 2025-06-06 15:19:45

### Documentation: RESEARCH_PROJECT_FINAL_SUMMARY

- **Status**: active
- **File**: `RESEARCH_PROJECT_FINAL_SUMMARY.md`
- **Description**: Documentation file: RESEARCH_PROJECT_FINAL_SUMMARY.md
- **Last Modified**: 2025-06-06 15:19:45

### Documentation: SDK_SUMMARY

- **Status**: active
- **File**: `SDK_SUMMARY.md`
- **Description**: Documentation file: SDK_SUMMARY.md
- **Last Modified**: 2025-06-05 11:41:51

### Documentation: SESSION_SUMMARY_CLOUDFLARE_INTEGRATION

- **Status**: active
- **File**: `SESSION_SUMMARY_CLOUDFLARE_INTEGRATION.md`
- **Description**: Documentation file: SESSION_SUMMARY_CLOUDFLARE_INTEGRATION.md
- **Last Modified**: 2025-06-08 01:55:40

### Documentation: STARTUP_SYSTEM_RESEARCH_REPORT_2025

- **Status**: active
- **File**: `docs\STARTUP_SYSTEM_RESEARCH_REPORT_2025.md`
- **Description**: Documentation file: STARTUP_SYSTEM_RESEARCH_REPORT_2025.md
- **Last Modified**: 2025-06-09 19:42:15

### Documentation: SYSTEM_DEPLOYMENT_COMPLETE

- **Status**: active
- **File**: `SYSTEM_DEPLOYMENT_COMPLETE.md`
- **Description**: Documentation file: SYSTEM_DEPLOYMENT_COMPLETE.md
- **Last Modified**: 2025-06-08 09:20:31

### Documentation: SYSTEM_OVERVIEW

- **Status**: active
- **File**: `docs\SYSTEM_OVERVIEW.md`
- **Description**: Documentation file: SYSTEM_OVERVIEW.md
- **Last Modified**: 2025-06-08 20:16:03

### Documentation: TECHNICAL_IMPLEMENTATION_GUIDE

- **Status**: active
- **File**: `TECHNICAL_IMPLEMENTATION_GUIDE.md`
- **Description**: Documentation file: TECHNICAL_IMPLEMENTATION_GUIDE.md
- **Last Modified**: 2025-05-28 17:50:40

### Documentation: TESTING_FRAMEWORK_STATUS

- **Status**: active
- **File**: `TESTING_FRAMEWORK_STATUS.md`
- **Description**: Documentation file: TESTING_FRAMEWORK_STATUS.md
- **Last Modified**: 2025-05-28 10:58:52

### Documentation: TEST_RESULTS_SUMMARY

- **Status**: active
- **File**: `TEST_RESULTS_SUMMARY.md`
- **Description**: Documentation file: TEST_RESULTS_SUMMARY.md
- **Last Modified**: 2025-05-28 17:08:42

### Documentation: TOC_AND_LOADING_FIXES

- **Status**: active
- **File**: `TOC_AND_LOADING_FIXES.md`
- **Description**: Documentation file: TOC_AND_LOADING_FIXES.md
- **Last Modified**: 2025-06-09 17:07:50

### Documentation: TODO_DOCUMENT_SEARCH

- **Status**: active
- **File**: `TODO_DOCUMENT_SEARCH.md`
- **Description**: Documentation file: TODO_DOCUMENT_SEARCH.md
- **Last Modified**: 2025-06-09 17:07:50

### Documentation: TOP_10_MCP_SERVERS_FOR_CODE

- **Status**: active
- **File**: `TOP_10_MCP_SERVERS_FOR_CODE.md`
- **Description**: Documentation file: TOP_10_MCP_SERVERS_FOR_CODE.md
- **Last Modified**: 2025-06-07 20:10:03

### Documentation: TREE OF THOUGHT IMPLEMENTATION PLAN - NO

- **Status**: active
- **File**: `TREE OF THOUGHT IMPLEMENTATION PLAN - NO.md`
- **Description**: Documentation file: TREE OF THOUGHT IMPLEMENTATION PLAN - NO.md
- **Last Modified**: 2025-06-03 13:29:48

### Documentation: WEBSOCKET_FIX_STATUS

- **Status**: active
- **File**: `WEBSOCKET_FIX_STATUS.md`
- **Description**: Documentation file: WEBSOCKET_FIX_STATUS.md
- **Last Modified**: 2025-06-08 02:33:02

### Documentation: WEBSOCKET_LAYER_COMPLETE_ANALYSIS

- **Status**: active
- **File**: `WEBSOCKET_LAYER_COMPLETE_ANALYSIS.md`
- **Description**: Documentation file: WEBSOCKET_LAYER_COMPLETE_ANALYSIS.md
- **Last Modified**: 2025-06-06 15:19:45

### Documentation: WINDOWS_COMPATIBILITY_COMPLETE_SOLUTION

- **Status**: active
- **File**: `WINDOWS_COMPATIBILITY_COMPLETE_SOLUTION.md`
- **Description**: Documentation file: WINDOWS_COMPATIBILITY_COMPLETE_SOLUTION.md
- **Last Modified**: 2025-05-28 21:26:57

### Documentation: cloudflare_pages_setup_guide

- **Status**: active
- **File**: `cloudflare_pages_setup_guide.md`
- **Description**: Documentation file: cloudflare_pages_setup_guide.md
- **Last Modified**: 2025-06-03 23:19:02

### Documentation: errors

- **Status**: active
- **File**: `errors.md`
- **Description**: Documentation file: errors.md
- **Last Modified**: 2025-06-04 16:57:21

### Documentation: errors and some lint warnings

- **Status**: active
- **File**: `errors and some lint warnings.md`
- **Description**: Documentation file: errors and some lint warnings.md
- **Last Modified**: 2025-06-07 11:45:45

### Documentation: python-sdks

- **Status**: active
- **File**: `docs\python-sdks.md`
- **Description**: Documentation file: python-sdks.md
- **Last Modified**: 2025-06-05 11:41:51

### Documentation: requirements

- **Status**: active
- **File**: `requirements.txt`
- **Description**: Documentation file: requirements.txt
- **Last Modified**: 2025-06-08 18:03:24

### Documentation: requirements-dev

- **Status**: active
- **File**: `requirements-dev.txt`
- **Description**: Documentation file: requirements-dev.txt
- **Last Modified**: 2025-06-01 23:01:56

### Documentation: requirements-gpu

- **Status**: active
- **File**: `requirements-gpu.txt`
- **Description**: Documentation file: requirements-gpu.txt
- **Last Modified**: 2025-06-01 23:01:45

### Documentation: requirements-test

- **Status**: active
- **File**: `requirements-test.txt`
- **Description**: Documentation file: requirements-test.txt
- **Last Modified**: 2025-05-28 15:20:01

### Documentation: tunnel-instructions

- **Status**: active
- **File**: `tunnel-instructions.md`
- **Description**: Documentation file: tunnel-instructions.md
- **Last Modified**: 2025-06-04 20:42:30

## Test Suite

**Count**: 87

### Test Suite: comprehensive_memory_test

- **Status**: active
- **File**: `comprehensive_memory_test.py`
- **Description**: Test suite: comprehensive_memory_test.py
- **Last Modified**: 2025-06-06 00:33:52

### Test Suite: comprehensive_test

- **Status**: active
- **File**: `comprehensive_test.py`
- **Description**: Test suite: comprehensive_test.py
- **Last Modified**: 2025-06-07 11:22:36

### Test Suite: context7_sdk_test

- **Status**: active
- **File**: `context7_sdk_test.py`
- **Description**: Test suite: context7_sdk_test.py
- **Last Modified**: 2025-06-05 11:41:51

### Test Suite: context7_simple_test

- **Status**: active
- **File**: `context7_simple_test.py`
- **Description**: Test suite: context7_simple_test.py
- **Last Modified**: 2025-06-05 11:29:11

### Test Suite: final_test

- **Status**: active
- **File**: `final_test.py`
- **Description**: Test suite: final_test.py
- **Last Modified**: 2025-06-07 11:22:36

### Test Suite: memory_system_gpu_test

- **Status**: active
- **File**: `memory_system_gpu_test.py`
- **Description**: Test suite: memory_system_gpu_test.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: minimal_test

- **Status**: active
- **File**: `minimal_test.py`
- **Description**: Test suite: minimal_test.py
- **Last Modified**: 2025-06-04 12:09:22

### Test Suite: production_test

- **Status**: active
- **File**: `production_test.py`
- **Description**: Test suite: production_test.py
- **Last Modified**: 2025-06-07 11:22:36

### Test Suite: quick_api_test

- **Status**: active
- **File**: `quick_api_test.py`
- **Description**: Test suite: quick_api_test.py
- **Last Modified**: 2025-06-09 17:07:50

### Test Suite: quick_backend_test

- **Status**: active
- **File**: `quick_backend_test.py`
- **Description**: Test suite: quick_backend_test.py
- **Last Modified**: 2025-06-09 17:28:23

### Test Suite: quick_mcp_test

- **Status**: active
- **File**: `quick_mcp_test.py`
- **Description**: Test suite: quick_mcp_test.py
- **Last Modified**: 2025-06-08 01:55:38

### Test Suite: quick_production_test

- **Status**: active
- **File**: `quick_production_test.py`
- **Description**: Test suite: quick_production_test.py
- **Last Modified**: 2025-06-07 11:22:36

### Test Suite: quick_test

- **Status**: active
- **File**: `quick_test.py`
- **Description**: Test suite: quick_test.py
- **Last Modified**: 2025-06-04 20:20:40

### Test Suite: simple_coding_agent_test

- **Status**: active
- **File**: `simple_coding_agent_test.py`
- **Description**: Test suite: simple_coding_agent_test.py
- **Last Modified**: 2025-06-07 19:55:48

### Test Suite: simple_context7_test

- **Status**: active
- **File**: `simple_context7_test.py`
- **Description**: Test suite: simple_context7_test.py
- **Last Modified**: 2025-06-05 11:29:11

### Test Suite: simple_test

- **Status**: active
- **File**: `simple_test.py`
- **Description**: Test suite: simple_test.py
- **Last Modified**: 2025-06-04 19:37:59

### Test Suite: simple_tool_discovery_test

- **Status**: active
- **File**: `simple_tool_discovery_test.py`
- **Description**: Test suite: simple_tool_discovery_test.py
- **Last Modified**: 2025-06-06 00:00:45

### Test Suite: test_ai_components_real

- **Status**: active
- **File**: `tests\test_ai_components_real.py`
- **Description**: Test suite: test_ai_components_real.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_all_mcp_servers

- **Status**: active
- **File**: `test_all_mcp_servers.py`
- **Description**: Test suite: test_all_mcp_servers.py
- **Last Modified**: 2025-06-08 01:55:38

### Test Suite: test_api_connection

- **Status**: active
- **File**: `test_api_connection.py`
- **Description**: Test suite: test_api_connection.py
- **Last Modified**: 2025-06-03 22:55:11

### Test Suite: test_api_fixes

- **Status**: active
- **File**: `test_api_fixes.py`
- **Description**: Test suite: test_api_fixes.py
- **Last Modified**: 2025-06-09 17:56:32

### Test Suite: test_authentication_flow

- **Status**: active
- **File**: `test_authentication_flow.py`
- **Description**: Test suite: test_authentication_flow.py
- **Last Modified**: 2025-06-09 10:41:29

### Test Suite: test_authentication_flow_fixed

- **Status**: active
- **File**: `test_authentication_flow_fixed.py`
- **Description**: Test suite: test_authentication_flow_fixed.py
- **Last Modified**: 2025-06-09 10:41:29

### Test Suite: test_basic_functionality

- **Status**: active
- **File**: `tests\test_basic_functionality.py`
- **Description**: Test suite: test_basic_functionality.py
- **Last Modified**: 2025-06-05 11:29:09

### Test Suite: test_coding_agent_evolution

- **Status**: active
- **File**: `test_coding_agent_evolution.py`
- **Description**: Test suite: test_coding_agent_evolution.py
- **Last Modified**: 2025-06-07 21:06:11

### Test Suite: test_complete_deployment

- **Status**: active
- **File**: `test_complete_deployment.py`
- **Description**: Test suite: test_complete_deployment.py
- **Last Modified**: 2025-06-03 23:18:39

### Test Suite: test_context7

- **Status**: active
- **File**: `test_context7.py`
- **Description**: Test suite: test_context7.py
- **Last Modified**: 2025-06-05 11:29:10

### Test Suite: test_context7_simple

- **Status**: active
- **File**: `test_context7_simple.py`
- **Description**: Test suite: test_context7_simple.py
- **Last Modified**: 2025-06-05 11:29:11

### Test Suite: test_core

- **Status**: active
- **File**: `tests\integration\test_core.py`
- **Description**: Test suite: test_core.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_core_orchestrator

- **Status**: active
- **File**: `test_core_orchestrator.py`
- **Description**: Test suite: test_core_orchestrator.py
- **Last Modified**: 2025-06-04 12:31:13

### Test Suite: test_darwinian_a2a_phase1

- **Status**: active
- **File**: `test_darwinian_a2a_phase1.py`
- **Description**: Test suite: test_darwinian_a2a_phase1.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_darwinian_a2a_phase1_8

- **Status**: active
- **File**: `test_darwinian_a2a_phase1_8.py`
- **Description**: Test suite: test_darwinian_a2a_phase1_8.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_darwinian_a2a_phase2_1

- **Status**: active
- **File**: `test_darwinian_a2a_phase2_1.py`
- **Description**: Test suite: test_darwinian_a2a_phase2_1.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_darwinian_a2a_phase2_2

- **Status**: active
- **File**: `test_darwinian_a2a_phase2_2.py`
- **Description**: Test suite: test_darwinian_a2a_phase2_2.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_database_connection

- **Status**: active
- **File**: `test_database_connection.py`
- **Description**: Test suite: test_database_connection.py
- **Last Modified**: 2025-06-05 23:57:45

### Test Suite: test_dgm_engine

- **Status**: active
- **File**: `tests\dgm\test_dgm_engine.py`
- **Description**: Test suite: test_dgm_engine.py
- **Last Modified**: 2025-06-08 18:36:43

### Test Suite: test_dgm_integration

- **Status**: active
- **File**: `tests\dgm\test_dgm_integration.py`
- **Description**: Test suite: test_dgm_integration.py
- **Last Modified**: 2025-06-08 19:09:02

### Test Suite: test_distributed_genetic_algorithm_comprehensive

- **Status**: active
- **File**: `test_distributed_genetic_algorithm_comprehensive.py`
- **Description**: Test suite: test_distributed_genetic_algorithm_comprehensive.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_distributed_genetic_algorithm_phase2_1

- **Status**: active
- **File**: `test_distributed_genetic_algorithm_phase2_1.py`
- **Description**: Test suite: test_distributed_genetic_algorithm_phase2_1.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_doc_server

- **Status**: active
- **File**: `test_doc_server.py`
- **Description**: Test suite: test_doc_server.py
- **Last Modified**: 2025-06-09 10:57:55

### Test Suite: test_documentation_api

- **Status**: active
- **File**: `test_documentation_api.py`
- **Description**: Test suite: test_documentation_api.py
- **Last Modified**: 2025-06-09 10:26:43

### Test Suite: test_documentation_import

- **Status**: active
- **File**: `test_documentation_import.py`
- **Description**: Test suite: test_documentation_import.py
- **Last Modified**: 2025-06-04 12:08:31

### Test Suite: test_documentation_ui

- **Status**: active
- **File**: `test_documentation_ui.py`
- **Description**: Test suite: test_documentation_ui.py
- **Last Modified**: 2025-06-09 17:07:50

### Test Suite: test_endpoint_fixes

- **Status**: active
- **File**: `test_endpoint_fixes.py`
- **Description**: Test suite: test_endpoint_fixes.py
- **Last Modified**: 2025-06-09 17:28:23

### Test Suite: test_enhanced_registry

- **Status**: active
- **File**: `test_enhanced_registry.py`
- **Description**: Test suite: test_enhanced_registry.py
- **Last Modified**: 2025-06-06 00:00:45

### Test Suite: test_evolution_effectiveness

- **Status**: active
- **File**: `test_evolution_effectiveness.py`
- **Description**: Test suite: test_evolution_effectiveness.py
- **Last Modified**: 2025-06-07 15:35:22

### Test Suite: test_evolution_effectiveness_corrected

- **Status**: active
- **File**: `test_evolution_effectiveness_corrected.py`
- **Description**: Test suite: test_evolution_effectiveness_corrected.py
- **Last Modified**: 2025-06-07 15:35:22

### Test Suite: test_fixed_orchestrator

- **Status**: active
- **File**: `test_fixed_orchestrator.py`
- **Description**: Test suite: test_fixed_orchestrator.py
- **Last Modified**: 2025-06-04 12:22:59

### Test Suite: test_full_documentation_flow

- **Status**: active
- **File**: `test_full_documentation_flow.py`
- **Description**: Test suite: test_full_documentation_flow.py
- **Last Modified**: 2025-06-09 12:08:21

### Test Suite: test_full_system_integration

- **Status**: active
- **File**: `tests\integration\test_full_system_integration.py`
- **Description**: Test suite: test_full_system_integration.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_health

- **Status**: active
- **File**: `test_health.py`
- **Description**: Test suite: test_health.py
- **Last Modified**: 2025-06-04 19:51:29

### Test Suite: test_integration

- **Status**: active
- **File**: `tests\protocols\a2a\test_integration.py`
- **Description**: Test suite: test_integration.py
- **Last Modified**: 2025-06-08 18:59:31

### Test Suite: test_intelligent_docs

- **Status**: active
- **File**: `test_intelligent_docs.py`
- **Description**: Test suite: test_intelligent_docs.py
- **Last Modified**: 2025-06-04 13:45:44

### Test Suite: test_mcp_api

- **Status**: active
- **File**: `test_mcp_api.py`
- **Description**: Test suite: test_mcp_api.py
- **Last Modified**: 2025-06-05 11:29:09

### Test Suite: test_mcp_functionality

- **Status**: active
- **File**: `test_mcp_functionality.py`
- **Description**: Test suite: test_mcp_functionality.py
- **Last Modified**: 2025-06-07 20:10:03

### Test Suite: test_mcp_installation

- **Status**: active
- **File**: `test_mcp_installation.py`
- **Description**: Test suite: test_mcp_installation.py
- **Last Modified**: 2025-06-02 20:26:30

### Test Suite: test_mcp_simple

- **Status**: active
- **File**: `test_mcp_simple.py`
- **Description**: Test suite: test_mcp_simple.py
- **Last Modified**: 2025-06-08 01:55:38

### Test Suite: test_mcp_weapon_advanced

- **Status**: active
- **File**: `test_mcp_weapon_advanced.py`
- **Description**: Test suite: test_mcp_weapon_advanced.py
- **Last Modified**: 2025-06-07 15:35:22

### Test Suite: test_mcp_weapon_integration

- **Status**: active
- **File**: `test_mcp_weapon_integration.py`
- **Description**: Test suite: test_mcp_weapon_integration.py
- **Last Modified**: 2025-06-07 15:35:22

### Test Suite: test_mcp_weapon_simple

- **Status**: active
- **File**: `test_mcp_weapon_simple.py`
- **Description**: Test suite: test_mcp_weapon_simple.py
- **Last Modified**: 2025-06-07 15:35:22

### Test Suite: test_mcp_weapon_simple_integration

- **Status**: active
- **File**: `test_mcp_weapon_simple_integration.py`
- **Description**: Test suite: test_mcp_weapon_simple_integration.py
- **Last Modified**: 2025-06-07 15:35:22

### Test Suite: test_mermaid_basic

- **Status**: active
- **File**: `test_mermaid_basic.py`
- **Description**: Test suite: test_mermaid_basic.py
- **Last Modified**: 2025-06-04 13:57:35

### Test Suite: test_mock_removal_validation

- **Status**: active
- **File**: `test_mock_removal_validation.py`
- **Description**: Test suite: test_mock_removal_validation.py
- **Last Modified**: 2025-06-06 22:01:40

### Test Suite: test_oauth_integration

- **Status**: active
- **File**: `test_oauth_integration.py`
- **Description**: Test suite: test_oauth_integration.py
- **Last Modified**: 2025-06-08 22:38:17

### Test Suite: test_ollama

- **Status**: active
- **File**: `test_ollama.py`
- **Description**: Test suite: test_ollama.py
- **Last Modified**: 2025-06-04 20:16:02

### Test Suite: test_orchestrator_imports

- **Status**: active
- **File**: `test_orchestrator_imports.py`
- **Description**: Test suite: test_orchestrator_imports.py
- **Last Modified**: 2025-06-04 12:09:58

### Test Suite: test_orchestrator_integration

- **Status**: active
- **File**: `test_orchestrator_integration.py`
- **Description**: Test suite: test_orchestrator_integration.py
- **Last Modified**: 2025-06-04 12:15:50

### Test Suite: test_original_problem

- **Status**: active
- **File**: `test_original_problem.py`
- **Description**: Test suite: test_original_problem.py
- **Last Modified**: 2025-06-04 12:17:51

### Test Suite: test_production_websocket

- **Status**: active
- **File**: `test_production_websocket.py`
- **Description**: Test suite: test_production_websocket.py
- **Last Modified**: 2025-06-03 22:37:33

### Test Suite: test_public_documentation

- **Status**: active
- **File**: `test_public_documentation.py`
- **Description**: Test suite: test_public_documentation.py
- **Last Modified**: 2025-06-09 12:08:21

### Test Suite: test_python_mcp_servers

- **Status**: active
- **File**: `test_python_mcp_servers.py`
- **Description**: Test suite: test_python_mcp_servers.py
- **Last Modified**: 2025-06-08 01:55:40

### Test Suite: test_real_ai_with_ollama

- **Status**: active
- **File**: `tests\test_real_ai_with_ollama.py`
- **Description**: Test suite: test_real_ai_with_ollama.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_real_gpu_usage

- **Status**: active
- **File**: `test_real_gpu_usage.py`
- **Description**: Test suite: test_real_gpu_usage.py
- **Last Modified**: 2025-06-06 15:19:45

### Test Suite: test_real_workflow_execution

- **Status**: active
- **File**: `tests\test_real_workflow_execution.py`
- **Description**: Test suite: test_real_workflow_execution.py
- **Last Modified**: 2025-05-28 17:36:57

### Test Suite: test_recipe_parser

- **Status**: active
- **File**: `tests\ai\nlp\test_recipe_parser.py`
- **Description**: Test suite: test_recipe_parser.py
- **Last Modified**: 2025-05-28 15:00:03

### Test Suite: test_research_analysis_workflow

- **Status**: active
- **File**: `test_research_analysis_workflow.py`
- **Description**: Test suite: test_research_analysis_workflow.py
- **Last Modified**: 2025-06-03 01:34:45

### Test Suite: test_simple

- **Status**: active
- **File**: `tests\test_simple.py`
- **Description**: Test suite: test_simple.py
- **Last Modified**: 2025-05-28 15:10:40

### Test Suite: test_status

- **Status**: active
- **File**: `test_status.py`
- **Description**: Test suite: test_status.py
- **Last Modified**: 2025-06-04 19:50:02

### Test Suite: test_tool_discovery_database

- **Status**: active
- **File**: `test_tool_discovery_database.py`
- **Description**: Test suite: test_tool_discovery_database.py
- **Last Modified**: 2025-06-06 00:30:22

### Test Suite: test_tool_discovery_database_fixed

- **Status**: active
- **File**: `test_tool_discovery_database_fixed.py`
- **Description**: Test suite: test_tool_discovery_database_fixed.py
- **Last Modified**: 2025-06-06 00:30:22

### Test Suite: test_tot_engine

- **Status**: active
- **File**: `tests\ai\reasoning\test_tot_engine.py`
- **Description**: Test suite: test_tot_engine.py
- **Last Modified**: 2025-06-01 22:16:25

### Test Suite: test_two_phase_evolution

- **Status**: active
- **File**: `test_two_phase_evolution.py`
- **Description**: Test suite: test_two_phase_evolution.py
- **Last Modified**: 2025-06-07 15:35:22

### Test Suite: test_user_services_integration

- **Status**: active
- **File**: `test_user_services_integration.py`
- **Description**: Test suite: test_user_services_integration.py
- **Last Modified**: 2025-06-08 22:51:31

### Test Suite: test_websocket

- **Status**: active
- **File**: `test_websocket.py`
- **Description**: Test suite: test_websocket.py
- **Last Modified**: 2025-06-08 11:28:23

### Test Suite: test_websocket_connection

- **Status**: active
- **File**: `test_websocket_connection.py`
- **Description**: Test suite: test_websocket_connection.py
- **Last Modified**: 2025-06-03 22:57:40

### Test Suite: test_workflow_api

- **Status**: active
- **File**: `test_workflow_api.py`
- **Description**: Test suite: test_workflow_api.py
- **Last Modified**: 2025-06-03 08:10:02

### Test Suite: test_workflows_import

- **Status**: active
- **File**: `test_workflows_import.py`
- **Description**: Test suite: test_workflows_import.py
- **Last Modified**: 2025-06-03 02:48:00
