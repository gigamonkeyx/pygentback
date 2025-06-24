# A2A Protocol Implementation Guide

## Overview

This document provides detailed technical implementation guidance for Phase 1 of the A2A+DGM integration: establishing complete A2A Protocol v0.2.1 compliance in PyGent Factory.

## Phase 1 Implementation Components

### 1.1 Core A2A Protocol Implementation

**File**: `src/protocols/a2a/protocol_handler.py`

```python
class A2AProtocolHandler:
    """Full A2A Protocol v0.2.1 implementation"""
    
    async def handle_message_send(self, params: MessageSendParams) -> Task:
        """Implement message/send method"""
        task_id = generate_uuid()
        
        # Create task for async processing
        task = Task(
            id=task_id,
            status=TaskStatus.RUNNING,
            created_at=datetime.utcnow(),
            message=params.message,
            agent_id=params.agent_id
        )
        
        # Process message asynchronously
        asyncio.create_task(self._process_message_async(task))
        
        return task
    
    async def handle_message_stream(self, params: MessageSendParams) -> SSEStream:
        """Implement streaming with Server-Sent Events"""
        stream = SSEStream(task_id=generate_uuid())
        
        # Start streaming response
        asyncio.create_task(self._stream_response(stream, params))
        
        return stream
    
    async def handle_tasks_get(self, params: TaskQueryParams) -> Task:
        """Implement task retrieval and polling"""
        task = await self.task_registry.get_task(params.task_id)
        
        if not task:
            raise TaskNotFoundException(params.task_id)
            
        return task
    
    async def handle_push_notification_config(self, params: TaskPushNotificationConfig):
        """Implement webhook configuration for async workflows"""
        await self.webhook_manager.configure_webhook(
            task_id=params.task_id,
            webhook_url=params.webhook_url,
            events=params.events
        )
```

**Integration Points**:
- Extend existing `src/orchestration/evolutionary_orchestrator.py` A2A server
- Leverage existing WebSocket infrastructure for SSE streaming
- Connect with agent registry for capability discovery

### 1.2 Agent Card Generation System

**File**: `src/protocols/a2a/agent_card_generator.py`

```python
class AgentCardGenerator:
    """Dynamic agent card generation for service discovery"""
    
    async def generate_public_card(self, agent_id: str) -> AgentCard:
        """Generate public agent card for /.well-known/agent.json"""
        agent = await self.agent_registry.get_agent(agent_id)
        
        return AgentCard(
            name=agent.name,
            description=agent.description,
            url=f"https://{self.domain}/agents/{agent_id}/a2a",
            version="1.0.0",
            capabilities=self._map_capabilities(agent),
            skills=self._generate_skills(agent),
            defaultInputModes=["text/plain", "application/json"],
            defaultOutputModes=["text/plain", "application/json"],
            securitySchemes=self._get_security_schemes()
        )
    
    async def generate_authenticated_card(self, agent_id: str) -> EnhancedAgentCard:
        """Generate detailed card for authenticated endpoints"""
        base_card = await self.generate_public_card(agent_id)
        agent = await self.agent_registry.get_agent(agent_id)
        
        return EnhancedAgentCard(
            **base_card.dict(),
            internal_capabilities=agent.internal_capabilities,
            performance_metrics=await self._get_performance_metrics(agent_id),
            resource_requirements=agent.resource_requirements,
            collaboration_preferences=agent.collaboration_preferences
        )
    
    def _map_capabilities(self, agent) -> List[str]:
        """Map internal agent capabilities to A2A capability strings"""
        capabilities = []
        
        if hasattr(agent, 'mcp_tools'):
            capabilities.extend(agent.mcp_tools)
        
        if hasattr(agent, 'skill_set'):
            capabilities.extend(agent.skill_set)
            
        return capabilities
    
    def _generate_skills(self, agent) -> List[str]:
        """Generate skill descriptions for agent card"""
        skills = []
        
        # Extract from agent configuration
        if hasattr(agent, 'specializations'):
            skills.extend(agent.specializations)
            
        # Add evolution capabilities if available
        if hasattr(agent, 'evolution_enabled') and agent.evolution_enabled:
            skills.append("self-improvement")
            skills.append("code-evolution")
            
        return skills
```

### 1.3 Multi-Modal Message Processing

**File**: `src/protocols/a2a/message_processor.py`

```python
class A2AMessageProcessor:
    """Handle multi-modal A2A message parts"""
    
    async def process_text_part(self, part: TextPart) -> Dict[str, Any]:
        """Process text content using existing LLM infrastructure"""
        response = await self.ollama_backend.generate_response(part.text)
        
        return {
            "type": "text_response",
            "content": response,
            "processing_time": response.processing_time,
            "model_used": response.model_name
        }
    
    async def process_file_part(self, part: FilePart) -> Dict[str, Any]:
        """Process file attachments using MCP tools"""
        # Use existing MCP infrastructure for file processing
        mcp_result = await self.mcp_handler.process_file(
            file_data=part.data,
            mime_type=part.mimeType,
            filename=part.name
        )
        
        return {
            "type": "file_response",
            "content": mcp_result,
            "file_analysis": await self._analyze_file_content(part)
        }
    
    async def process_data_part(self, part: DataPart) -> Dict[str, Any]:
        """Process structured data using agent capabilities"""
        data_type = part.data.get("type", "unknown")
        
        if data_type == "evolution_validation_request":
            return await self._handle_evolution_validation(part.data)
        elif data_type == "capability_query":
            return await self._handle_capability_query(part.data)
        else:
            # Generic data processing
            return await self._process_generic_data(part.data)
    
    async def _handle_evolution_validation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle evolution validation requests from peer agents"""
        modification = CodeModification.from_dict(data["modification"])
        
        # Validate the proposed modification
        validation_result = await self.evolution_validator.validate_modification(
            modification, data.get("context", {})
        )
        
        return {
            "type": "evolution_validation_response",
            "validation_result": asdict(validation_result),
            "peer_feedback": {
                "safety_assessment": validation_result.safety_score,
                "improvement_likelihood": validation_result.improvement_probability,
                "suggestions": validation_result.suggestions
            }
        }
```

## Implementation Strategy

### Integration with Existing Systems

1. **Evolutionary Orchestrator Extension**
   - Add A2A server endpoints to existing orchestrator
   - Maintain backward compatibility with current API
   - Extend WebSocket infrastructure for SSE support

2. **Agent Registry Enhancement**
   - Add agent card generation capabilities
   - Implement capability mapping functions
   - Support both public and authenticated card endpoints

3. **MCP Tool Integration**
   - Leverage existing MCP infrastructure for file processing
   - Map MCP capabilities to A2A skills
   - Maintain existing tool interfaces

### Testing Strategy

1. **Unit Tests**
   - Test each A2A method implementation
   - Validate agent card generation accuracy
   - Test multi-modal message processing

2. **Integration Tests**
   - Test A2A endpoint integration with orchestrator
   - Validate agent discovery workflows
   - Test message routing and processing

3. **End-to-End Tests**
   - Test complete A2A communication flows
   - Validate interoperability with external A2A agents
   - Test streaming and webhook functionality

## Next Steps

1. Implement core A2A protocol handler
2. Create agent card generation system
3. Build multi-modal message processor
4. Integrate with existing orchestrator
5. Implement comprehensive testing suite

## Related Documents

- [A2A Protocol Technical Specification](A2A_PROTOCOL_TECHNICAL_SPEC.md)
- [A2A Protocol Architecture](A2A_PROTOCOL_ARCHITECTURE.md)
- [DGM Core Engine Design](DGM_CORE_ENGINE_DESIGN.md)
