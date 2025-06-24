"""
Real AI tests using Ollama local models - NO MORE MOCKS!
Tests actual AI functionality with local language models.
"""

import pytest
import asyncio
import requests
from datetime import datetime
from typing import Dict, Any, Optional

# Import our working real components
from src.memory.vector_store import VectorStore
from src.memory.conversation_memory import ConversationMemory, ConversationMessage, MessageRole
from src.memory.knowledge_graph import KnowledgeGraph, Entity
from src.mcp.tools.executor import MCPToolExecutor, ToolExecutionRequest
from src.core.agent_factory import AgentFactory, AgentCreationRequest


class OllamaClient:
    """Simple Ollama client for testing"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
    
    async def check_availability(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model["name"] for model in models_data.get("models", [])]
                return True
        except Exception:
            pass
        return False
    
    async def generate(self, model: str, prompt: str, max_tokens: int = 100) -> Optional[str]:
        """Generate text using Ollama"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            print(f"Ollama generation failed: {e}")
        return None


@pytest.fixture
async def ollama_client():
    """Create Ollama client fixture"""
    client = OllamaClient()
    is_available = await client.check_availability()
    if not is_available:
        pytest.skip("Ollama is not available")
    return client


@pytest.fixture
def preferred_model(ollama_client):
    """Get preferred model for testing"""
    # Prefer smaller/faster models for testing
    model_preferences = ["deepseek1:latest", "deepseek2:latest", "janus:latest", "codellama:latest"]
    
    for model in model_preferences:
        if model in ollama_client.available_models:
            return model
    
    # Fallback to first available model
    if ollama_client.available_models:
        return ollama_client.available_models[0]
    
    pytest.skip("No Ollama models available")


class TestRealAIWithOllama:
    """Test real AI functionality using Ollama models"""
    
    @pytest.mark.asyncio
    async def test_ollama_basic_generation(self, ollama_client, preferred_model):
        """Test basic text generation with Ollama"""
        prompt = "What is 2+2? Answer briefly."
        
        response = await ollama_client.generate(preferred_model, prompt, max_tokens=50)
        
        assert response is not None
        assert len(response) > 0
        assert "4" in response  # Should contain the answer
        print(f"✅ Ollama generated: {response}")
    
    @pytest.mark.asyncio
    async def test_ollama_code_generation(self, ollama_client, preferred_model):
        """Test code generation with Ollama"""
        prompt = "Write a simple Python function to add two numbers. Just the function, no explanation."
        
        response = await ollama_client.generate(preferred_model, prompt, max_tokens=100)
        
        assert response is not None
        assert "def" in response
        assert "return" in response
        print(f"✅ Ollama generated code: {response}")
    
    @pytest.mark.asyncio
    async def test_memory_with_ai_integration(self, ollama_client, preferred_model):
        """Test memory systems with AI-generated content"""
        # Create memory components
        conversation_memory = ConversationMemory()
        knowledge_graph = KnowledgeGraph()
        
        # Generate AI content
        prompt = "Explain what a knowledge graph is in one sentence."
        ai_response = await ollama_client.generate(preferred_model, prompt, max_tokens=100)
        
        assert ai_response is not None
        
        # Store in conversation memory
        thread = await conversation_memory.create_thread("ai_test_agent", "AI Integration Test")
        
        user_message = ConversationMessage(
            id="user_msg_001",
            role=MessageRole.USER,
            content=prompt
        )
        
        ai_message = ConversationMessage(
            id="ai_msg_001", 
            role=MessageRole.ASSISTANT,
            content=ai_response
        )
        
        await conversation_memory.add_message(thread.thread_id, user_message)
        await conversation_memory.add_message(thread.thread_id, ai_message)
        
        # Store in knowledge graph
        entity = Entity(
            id="ai_knowledge_graph_concept",
            name="AI Knowledge Graph Concept",
            entity_type="Concept",
            properties={
                "definition": ai_response,
                "source": "ai_generated",
                "model": preferred_model
            }
        )
        await knowledge_graph.add_entity(entity)
        
        # Verify storage
        retrieved_thread = await conversation_memory.get_thread(thread.thread_id)
        assert len(retrieved_thread.messages) == 2
        assert retrieved_thread.messages[1].content == ai_response
        
        retrieved_entity = await knowledge_graph.get_entity("ai_knowledge_graph_concept")
        assert retrieved_entity.properties["definition"] == ai_response
        
        print(f"✅ AI-generated content stored in memory systems")
        print(f"   AI Response: {ai_response[:100]}...")
    
    @pytest.mark.asyncio
    async def test_agent_factory_with_ai_capabilities(self, ollama_client, preferred_model):
        """Test agent factory creating AI-capable agents"""
        agent_factory = AgentFactory()
        await agent_factory.initialize()
        
        # Create an AI-capable agent
        request = AgentCreationRequest(
            agent_type="nlp",
            name="ai_powered_agent",
            capabilities=["text_processing", "language_analysis", "ai_generation"],
            mcp_tools=["ollama_client"],
            custom_config={
                "ollama_model": preferred_model,
                "max_tokens": 200,
                "temperature": 0.7
            }
        )
        
        result = await agent_factory.create_agent_from_request(request)
        
        assert result.success
        assert result.agent.name == "ai_powered_agent"
        assert "ai_generation" in result.agent.capabilities
        assert result.agent.config.custom_config["ollama_model"] == preferred_model
        
        print(f"✅ Created AI-powered agent with model: {preferred_model}")
    
    @pytest.mark.asyncio
    async def test_mcp_tool_with_ai_integration(self, ollama_client, preferred_model):
        """Test MCP tools integrated with AI generation"""
        mcp_executor = MCPToolExecutor()
        
        # Register an AI-powered tool
        async def ai_text_processor(args):
            prompt = args.get("prompt", "")
            if not prompt:
                return {"error": "No prompt provided"}
            
            response = await ollama_client.generate(preferred_model, prompt, max_tokens=150)
            
            return {
                "success": True,
                "original_prompt": prompt,
                "ai_response": response,
                "model_used": preferred_model,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        mcp_executor.register_tool("ai_text_processor", ai_text_processor)
        
        # Test the AI tool
        request = ToolExecutionRequest(
            tool_name="ai_text_processor",
            server_name="local_ai_server",
            arguments={"prompt": "Explain machine learning in simple terms."},
            execution_id="ai_test_001"
        )
        
        result = await mcp_executor.execute_tool(request)
        
        assert result.success
        assert result.result["success"] is True
        assert "ai_response" in result.result
        assert result.result["model_used"] == preferred_model
        
        ai_response = result.result["ai_response"]
        assert ai_response is not None
        assert len(ai_response) > 0
        
        print(f"✅ AI-powered MCP tool executed successfully")
        print(f"   AI Response: {ai_response[:100]}...")
    
    @pytest.mark.asyncio
    async def test_full_ai_workflow(self, ollama_client, preferred_model):
        """Test complete AI workflow with all components"""
        # Initialize all components
        agent_factory = AgentFactory()
        conversation_memory = ConversationMemory()
        knowledge_graph = KnowledgeGraph()
        mcp_executor = MCPToolExecutor()
        
        await agent_factory.initialize()
        
        # 1. Create AI agent
        agent_request = AgentCreationRequest(
            agent_type="nlp",
            name="workflow_ai_agent",
            capabilities=["text_processing", "analysis", "ai_generation"],
            mcp_tools=["ai_analyzer"],
            custom_config={"model": preferred_model}
        )
        
        agent_result = await agent_factory.create_agent_from_request(agent_request)
        assert agent_result.success
        
        # 2. Register AI tool
        async def ai_analyzer(args):
            text = args.get("text", "")
            analysis_prompt = f"Analyze this text and provide key insights: {text}"
            
            response = await ollama_client.generate(preferred_model, analysis_prompt, max_tokens=200)
            
            return {
                "analysis": response,
                "text_length": len(text),
                "model": preferred_model
            }
        
        mcp_executor.register_tool("ai_analyzer", ai_analyzer)
        
        # 3. Process text with AI
        test_text = "Artificial intelligence is transforming how we work and live."
        
        tool_request = ToolExecutionRequest(
            tool_name="ai_analyzer",
            server_name="ai_workflow_server",
            arguments={"text": test_text},
            execution_id="workflow_001"
        )
        
        tool_result = await mcp_executor.execute_tool(tool_request)
        assert tool_result.success
        
        ai_analysis = tool_result.result["analysis"]
        
        # 4. Store conversation
        thread = await conversation_memory.create_thread(agent_result.agent_id, "AI Workflow Test")
        
        user_msg = ConversationMessage(
            id="workflow_user_001",
            role=MessageRole.USER,
            content=f"Please analyze: {test_text}"
        )
        
        ai_msg = ConversationMessage(
            id="workflow_ai_001",
            role=MessageRole.ASSISTANT,
            content=ai_analysis
        )
        
        await conversation_memory.add_message(thread.thread_id, user_msg)
        await conversation_memory.add_message(thread.thread_id, ai_msg)
        
        # 5. Store knowledge
        entity = Entity(
            id="ai_workflow_analysis",
            name="AI Workflow Analysis",
            entity_type="Analysis",
            properties={
                "original_text": test_text,
                "ai_analysis": ai_analysis,
                "agent_id": agent_result.agent_id,
                "model": preferred_model
            }
        )
        await knowledge_graph.add_entity(entity)
        
        # 6. Verify complete workflow
        retrieved_thread = await conversation_memory.get_thread(thread.thread_id)
        retrieved_entity = await knowledge_graph.get_entity("ai_workflow_analysis")
        
        assert len(retrieved_thread.messages) == 2
        assert retrieved_entity.properties["ai_analysis"] == ai_analysis
        assert "ai_analyzer" in mcp_executor.tool_registry
        
        print(f"✅ Complete AI workflow executed successfully")
        print(f"   Agent: {agent_result.agent.name}")
        print(f"   Model: {preferred_model}")
        print(f"   Analysis: {ai_analysis[:100]}...")
        
        return {
            "agent": agent_result.agent,
            "conversation": retrieved_thread,
            "knowledge": retrieved_entity,
            "ai_analysis": ai_analysis
        }
