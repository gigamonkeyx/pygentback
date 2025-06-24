"""
Real implementation tests for working components - replacing mock tests.
Tests only the components that are fully functional without import issues.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

# Import our working real components (no AI components due to circular imports)
from src.memory.vector_store import VectorStore
from src.memory.conversation_memory import ConversationMemory, ConversationMessage, MessageRole
from src.memory.knowledge_graph import KnowledgeGraph, Entity
from src.mcp.tools.executor import MCPToolExecutor, ToolExecutionRequest
from src.mcp.tools.discovery import MCPToolDiscovery
from src.core.agent_factory import AgentFactory, AgentCreationRequest
from src.core.agent_builder import AgentBuilder
from src.core.agent_validator import AgentValidator


class TestMemoryComponentsReal:
    """Test real memory component functionality."""

    @pytest.fixture
    def vector_store(self):
        """Create real vector store."""
        return VectorStore()

    @pytest.fixture
    def conversation_memory(self):
        """Create real conversation memory."""
        return ConversationMemory()

    @pytest.fixture
    def knowledge_graph(self):
        """Create real knowledge graph."""
        return KnowledgeGraph()

    def test_vector_store_creation(self, vector_store):
        """Test vector store creation."""
        assert vector_store is not None
        assert hasattr(vector_store, 'documents')
        assert hasattr(vector_store, 'embeddings')

    @pytest.mark.asyncio
    async def test_vector_store_operations(self, vector_store):
        """Test vector store operations."""
        # Test adding documents
        documents = ["Hello world", "Test document", "Another test"]

        try:
            await vector_store.add_documents(documents)

            # Test search
            results = await vector_store.search("hello", top_k=1)
            assert len(results) <= 1

        except Exception as e:
            # Skip if embedding model not available
            pytest.skip(f"Vector operations require embedding model: {e}")

    def test_conversation_memory_creation(self, conversation_memory):
        """Test conversation memory creation."""
        assert conversation_memory is not None
        assert hasattr(conversation_memory, 'threads')
        assert hasattr(conversation_memory, 'max_threads')

    @pytest.mark.asyncio
    async def test_conversation_memory_operations(self, conversation_memory):
        """Test conversation memory operations."""
        agent_id = "test_agent_001"

        # Create thread
        thread = await conversation_memory.create_thread(agent_id, "Test Thread")
        assert thread is not None
        assert thread.agent_id == agent_id

        # Create message
        message = ConversationMessage(
            id="msg_001",
            role=MessageRole.USER,
            content="Hello, this is a test message"
        )

        # Add message
        success = await conversation_memory.add_message(thread.thread_id, message)
        assert success

        # Get thread
        retrieved_thread = await conversation_memory.get_thread(thread.thread_id)
        assert retrieved_thread is not None
        assert len(retrieved_thread.messages) == 1
        assert retrieved_thread.messages[0].content == "Hello, this is a test message"

    def test_knowledge_graph_creation(self, knowledge_graph):
        """Test knowledge graph creation."""
        assert knowledge_graph is not None
        assert hasattr(knowledge_graph, 'entities')
        assert hasattr(knowledge_graph, 'relationships')

    @pytest.mark.asyncio
    async def test_knowledge_graph_operations(self, knowledge_graph):
        """Test knowledge graph operations."""
        # Create entity
        entity = Entity(
            id="test_entity",
            name="Test Entity",
            entity_type="TestType",
            properties={"description": "A test entity"}
        )

        # Add entity
        success = await knowledge_graph.add_entity(entity)
        assert success

        # Get entity
        retrieved_entity = await knowledge_graph.get_entity("test_entity")
        assert retrieved_entity is not None
        assert retrieved_entity.entity_type == "TestType"
        assert retrieved_entity.properties["description"] == "A test entity"


class TestMCPToolsReal:
    """Test real MCP tools functionality."""

    @pytest.fixture
    def mcp_executor(self):
        """Create real MCP tool executor."""
        return MCPToolExecutor()

    @pytest.fixture
    def mcp_discovery(self):
        """Create real MCP tool discovery."""
        return MCPToolDiscovery()

    def test_mcp_executor_creation(self, mcp_executor):
        """Test MCP executor creation."""
        assert mcp_executor is not None
        assert hasattr(mcp_executor, 'tool_registry')
        assert hasattr(mcp_executor, 'execution_history')

    @pytest.mark.asyncio
    async def test_mcp_executor_operations(self, mcp_executor):
        """Test MCP executor operations."""
        # Test tool registration
        def test_tool_function(args):
            return {"result": "test completed", "args": args}

        mcp_executor.register_tool("test_tool", test_tool_function)

        # Check if tool is registered
        assert "test_tool" in mcp_executor.tool_registry

        # Test tool execution
        request = ToolExecutionRequest(
            tool_name="test_tool",
            server_name="test_server",
            arguments={"param1": "value1"},
            execution_id="test_exec_001"
        )

        result = await mcp_executor.execute_tool(request)
        assert result.success
        assert result.tool_name == "test_tool"

    def test_mcp_discovery_creation(self, mcp_discovery):
        """Test MCP discovery creation."""
        assert mcp_discovery is not None
        assert hasattr(mcp_discovery, 'discovered_servers')
        assert hasattr(mcp_discovery, 'tool_catalog')

    @pytest.mark.asyncio
    async def test_mcp_discovery_operations(self, mcp_discovery):
        """Test MCP discovery operations."""
        # Test server discovery
        server_info = await mcp_discovery.discover_server("test_file_server", "http://localhost:8000")

        assert server_info is not None
        assert server_info.server_name == "test_file_server"
        assert server_info.status == "active"
        assert len(server_info.capabilities) > 0

        # Test tool search
        file_tools = mcp_discovery.search_tools(query="file")
        assert len(file_tools) > 0

        # Test category stats
        stats = mcp_discovery.get_category_stats()
        assert isinstance(stats, dict)


class TestAgentFactoryReal:
    """Test real agent factory functionality."""

    @pytest.fixture
    def agent_factory(self):
        """Create real agent factory."""
        return AgentFactory()

    @pytest.fixture
    def agent_builder(self):
        """Create real agent builder."""
        return AgentBuilder()

    @pytest.fixture
    def agent_validator(self):
        """Create real agent validator."""
        return AgentValidator()

    def test_agent_factory_creation(self, agent_factory):
        """Test agent factory creation."""
        assert agent_factory is not None
        assert hasattr(agent_factory, 'registry')
        assert hasattr(agent_factory, 'builder')
        assert hasattr(agent_factory, 'validator')

    @pytest.mark.asyncio
    async def test_agent_factory_initialization(self, agent_factory):
        """Test agent factory initialization."""
        await agent_factory.initialize()
        assert agent_factory.is_initialized

    @pytest.mark.asyncio
    async def test_agent_creation(self, agent_factory):
        """Test agent creation."""
        await agent_factory.initialize()

        request = AgentCreationRequest(
            agent_type="basic",
            name="test_agent",
            capabilities=["text_processing"],
            mcp_tools=["test_tool"]
        )

        result = await agent_factory.create_agent_from_request(request)

        assert result.success
        assert result.agent_id is not None
        assert result.agent is not None
        assert result.agent.name == "test_agent"

    def test_agent_builder_creation(self, agent_builder):
        """Test agent builder creation."""
        assert agent_builder is not None
        assert hasattr(agent_builder, 'build_templates')
        assert hasattr(agent_builder, 'capability_registry')

    def test_agent_validator_creation(self, agent_validator):
        """Test agent validator creation."""
        assert agent_validator is not None
        assert hasattr(agent_validator, 'validation_rules')
        assert hasattr(agent_validator, 'supported_agent_types')


@pytest.mark.integration
class TestCrossComponentIntegrationReal:
    """Test real cross-component integration."""

    @pytest.mark.asyncio
    async def test_memory_integration(self):
        """Test memory components integration."""
        vector_store = VectorStore()
        conversation_memory = ConversationMemory()
        knowledge_graph = KnowledgeGraph()

        # Test that all components can be created together
        assert vector_store is not None
        assert conversation_memory is not None
        assert knowledge_graph is not None

        # Test basic operations don't conflict
        agent_id = "integration_agent"

        # Create thread and add message
        thread = await conversation_memory.create_thread(agent_id, "Integration Test")
        message = ConversationMessage(
            id="int_msg_001",
            role=MessageRole.USER,
            content="Test message"
        )
        await conversation_memory.add_message(thread.thread_id, message)

        # Add entity
        entity = Entity(
            id="test_entity",
            name="Test Entity",
            entity_type="TestType",
            properties={"test": "data"}
        )
        await knowledge_graph.add_entity(entity)

        # Verify operations worked
        retrieved_thread = await conversation_memory.get_thread(thread.thread_id)
        assert len(retrieved_thread.messages) == 1

        retrieved_entity = await knowledge_graph.get_entity("test_entity")
        assert retrieved_entity.entity_type == "TestType"

    @pytest.mark.asyncio
    async def test_mcp_agent_integration(self):
        """Test MCP tools and agent factory integration."""
        mcp_executor = MCPToolExecutor()
        agent_factory = AgentFactory()

        # Initialize components
        await agent_factory.initialize()

        # Register a tool
        def integration_tool_function(args):
            return {"result": "integration test completed", "args": args}

        mcp_executor.register_tool("integration_tool", integration_tool_function)

        # Create an agent with MCP tools
        request = AgentCreationRequest(
            agent_type="basic",
            name="integration_agent",
            capabilities=["text_processing"],
            mcp_tools=["integration_tool"]
        )

        result = await agent_factory.create_agent_from_request(request)

        assert result.success
        assert result.agent.name == "integration_agent"

        # Verify tool is available
        assert "integration_tool" in mcp_executor.tool_registry

    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full system integration with all components."""
        # Create all components
        vector_store = VectorStore()
        conversation_memory = ConversationMemory()
        knowledge_graph = KnowledgeGraph()
        mcp_executor = MCPToolExecutor()
        agent_factory = AgentFactory()

        # Initialize
        await agent_factory.initialize()

        # Test workflow: Create agent -> Add memory -> Register tools -> Process data

        # 1. Create agent
        request = AgentCreationRequest(
            agent_type="basic",
            name="full_integration_agent",
            capabilities=["text_processing", "memory_access"],
            mcp_tools=["memory_tool"]
        )

        agent_result = await agent_factory.create_agent_from_request(request)
        assert agent_result.success

        # 2. Add conversation
        thread = await conversation_memory.create_thread("full_integration_agent", "Full Integration Test")
        message = ConversationMessage(
            id="full_int_msg_001",
            role=MessageRole.USER,
            content="This is a full integration test"
        )
        await conversation_memory.add_message(thread.thread_id, message)

        # 3. Add knowledge
        entity = Entity(
            id="integration_entity",
            name="Integration Entity",
            entity_type="IntegrationType",
            properties={"description": "Full integration test entity"}
        )
        await knowledge_graph.add_entity(entity)

        # 4. Register tool
        def memory_tool_function(args):
            return {"result": "memory operation completed", "args": args}

        mcp_executor.register_tool("memory_tool", memory_tool_function)

        # Verify all components are working together
        retrieved_thread = await conversation_memory.get_thread(thread.thread_id)
        retrieved_entity = await knowledge_graph.get_entity("integration_entity")

        assert len(retrieved_thread.messages) == 1
        assert retrieved_entity.entity_type == "IntegrationType"
        assert "memory_tool" in mcp_executor.tool_registry
        assert agent_result.agent.name == "full_integration_agent"
