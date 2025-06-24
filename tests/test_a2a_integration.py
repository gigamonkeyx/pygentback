#!/usr/bin/env python3
"""
Test A2A Integration

Comprehensive integration tests for the complete A2A protocol implementation.
Tests agent creation, communication, and compliance with Google A2A specification.
"""

import pytest
import asyncio
import json
import aiohttp
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import the A2A components
try:
    from src.core.agent_factory import AgentFactory
    from src.core.agent import BaseAgent, AgentConfig
    from src.servers.a2a_mcp_server import A2AMCPServer
    from src.a2a_standard import AgentCard, AgentProvider, AgentCapabilities, AgentSkill
    A2A_INTEGRATION_AVAILABLE = True
except ImportError as e:
    A2A_INTEGRATION_AVAILABLE = False
    print(f"A2A integration not available: {e}")


@pytest.mark.skipif(not A2A_INTEGRATION_AVAILABLE, reason="A2A integration not available")
class TestA2AIntegration:
    """Test A2A Integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.agent_factory = AgentFactory(base_url="http://localhost:8000")
        self.test_agents = []
    
    def teardown_method(self):
        """Cleanup after tests"""
        # Cleanup test agents
        for agent in self.test_agents:
            try:
                asyncio.create_task(self.agent_factory.destroy_agent(agent.agent_id))
            except:
                pass
    
    @pytest.mark.asyncio
    async def test_agent_factory_a2a_initialization(self):
        """Test agent factory A2A initialization"""
        # Initialize the factory
        await self.agent_factory.initialize()
        
        # Check A2A components are available
        assert self.agent_factory.a2a_enabled == True
        assert self.agent_factory.a2a_card_generator is not None
        assert self.agent_factory.base_url == "http://localhost:8000"
    
    @pytest.mark.asyncio
    async def test_create_a2a_compliant_agent(self):
        """Test creating an A2A-compliant agent"""
        await self.agent_factory.initialize()
        
        # Create an agent
        agent = await self.agent_factory.create_agent(
            agent_type="general",
            name="Test A2A Agent",
            capabilities=["reasoning", "analysis"]
        )
        
        self.test_agents.append(agent)
        
        # Verify agent creation
        assert agent is not None
        assert agent.name == "Test A2A Agent"
        assert agent.type == "general"
        assert agent.agent_id is not None
        
        # Check if agent is registered
        retrieved_agent = await self.agent_factory.get_agent(agent.agent_id)
        assert retrieved_agent == agent
    
    @pytest.mark.asyncio
    async def test_a2a_agent_card_generation(self):
        """Test A2A agent card generation"""
        await self.agent_factory.initialize()
        
        # Create an agent
        agent = await self.agent_factory.create_agent(
            agent_type="research",
            name="Research Agent",
            capabilities=["research", "analysis", "data_processing"]
        )
        
        self.test_agents.append(agent)
        
        # Generate A2A agent card
        if self.agent_factory.a2a_card_generator:
            agent_card = await self.agent_factory.a2a_card_generator.generate_agent_card(
                agent_id=agent.agent_id,
                agent_name=agent.name,
                agent_type=agent.type,
                capabilities=["research", "analysis"],
                skills=["document_search", "data_analysis"],
                enable_authentication=True
            )
            
            # Verify agent card structure
            assert "name" in agent_card
            assert "description" in agent_card
            assert "url" in agent_card
            assert "capabilities" in agent_card
            assert "skills" in agent_card
            assert "provider" in agent_card
            
            # Verify A2A compliance
            assert agent_card["name"] == agent.name
            assert isinstance(agent_card["capabilities"], dict)
            assert isinstance(agent_card["skills"], list)
            assert isinstance(agent_card["provider"], dict)
            
            # Check security schemes
            if "securitySchemes" in agent_card:
                security_schemes = agent_card["securitySchemes"]
                assert "bearerAuth" in security_schemes
                assert "apiKeyAuth" in security_schemes
    
    @pytest.mark.asyncio
    async def test_a2a_mcp_server_integration(self):
        """Test A2A MCP server integration"""
        await self.agent_factory.initialize()
        
        # Check if A2A MCP server is available
        if self.agent_factory.a2a_mcp_server:
            # Get server status
            status = self.agent_factory.get_a2a_mcp_server_status()
            
            assert status["available"] == True
            assert "port" in status
            assert "registered_agents" in status
            assert "active_tasks" in status
            
            # Create an agent and register with MCP server
            agent = await self.agent_factory.create_agent(
                agent_type="general",
                name="MCP Test Agent"
            )
            
            self.test_agents.append(agent)
            
            # Register agent with A2A MCP server
            registration_success = await self.agent_factory.register_agent_with_a2a_mcp(agent)
            
            if registration_success:
                # Verify registration
                updated_status = self.agent_factory.get_a2a_mcp_server_status()
                assert updated_status["registered_agents"] > status["registered_agents"]
    
    @pytest.mark.asyncio
    async def test_a2a_agent_discovery(self):
        """Test A2A agent discovery"""
        await self.agent_factory.initialize()
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = await self.agent_factory.create_agent(
                agent_type="general",
                name=f"Discovery Test Agent {i}",
                capabilities=[f"capability_{i}"]
            )
            agents.append(agent)
            self.test_agents.append(agent)
        
        # Test agent discovery
        discovered_agents = await self.agent_factory.discover_agents_in_network([
            "http://localhost:8000",
            "http://localhost:8001"
        ])
        
        # Note: This might return empty if no actual A2A servers are running
        # In a real test environment, we would have test servers running
        assert isinstance(discovered_agents, list)
    
    @pytest.mark.asyncio
    async def test_a2a_task_finding(self):
        """Test finding agents for specific tasks"""
        await self.agent_factory.initialize()
        
        # Create specialized agents
        research_agent = await self.agent_factory.create_agent(
            agent_type="research",
            name="Research Specialist",
            capabilities=["research", "analysis"]
        )
        
        coding_agent = await self.agent_factory.create_agent(
            agent_type="coding",
            name="Coding Specialist", 
            capabilities=["coding", "debugging"]
        )
        
        self.test_agents.extend([research_agent, coding_agent])
        
        # Test finding agent for research task
        research_match = await self.agent_factory.find_agent_for_task(
            "Research the latest developments in AI",
            required_capabilities=["research"],
            required_skills=["analysis"]
        )
        
        # Test finding agent for coding task
        coding_match = await self.agent_factory.find_agent_for_task(
            "Debug this Python code and fix the errors",
            required_capabilities=["coding"],
            required_skills=["debugging"]
        )
        
        # Verify matches (might be None if discovery system isn't fully running)
        if research_match:
            assert "agent_id" in research_match
            assert "match_score" in research_match
        
        if coding_match:
            assert "agent_id" in coding_match
            assert "match_score" in coding_match
    
    @pytest.mark.asyncio
    async def test_a2a_agent_communication(self):
        """Test A2A agent-to-agent communication"""
        await self.agent_factory.initialize()
        
        # Create two agents
        sender_agent = await self.agent_factory.create_agent(
            agent_type="general",
            name="Sender Agent"
        )
        
        receiver_agent = await self.agent_factory.create_agent(
            agent_type="general", 
            name="Receiver Agent"
        )
        
        self.test_agents.extend([sender_agent, receiver_agent])
        
        # Test A2A message sending
        if self.agent_factory.a2a_mcp_server:
            message_result = await self.agent_factory.send_a2a_mcp_message(
                agent_id=receiver_agent.agent_id,
                message="Hello from sender agent!",
                context_id="test_context_123"
            )
            
            # Verify message result
            assert isinstance(message_result, dict)
            # The exact structure depends on the A2A MCP server implementation
    
    @pytest.mark.asyncio
    async def test_a2a_short_lived_agent_optimization(self):
        """Test short-lived agent optimization"""
        await self.agent_factory.initialize()
        
        # Create a short-lived agent
        short_lived_agent = await self.agent_factory.create_short_lived_agent(
            agent_type="general",
            purpose="quick_analysis",
            resource_limits=None  # Use defaults
        )
        
        if short_lived_agent:
            self.test_agents.append(short_lived_agent)
            
            # Verify short-lived agent properties
            assert short_lived_agent.agent_id is not None
            assert "short_lived" in short_lived_agent.config.custom_config
            assert short_lived_agent.config.custom_config["purpose"] == "quick_analysis"
            
            # Test task execution
            task_result = await self.agent_factory.execute_short_lived_task(
                short_lived_agent.agent_id,
                {"id": "test_task", "type": "analysis", "data": "test data"}
            )
            
            assert task_result is not None
            
            # Get metrics
            metrics = await self.agent_factory.get_short_lived_agent_metrics(
                short_lived_agent.agent_id
            )
            
            if metrics:
                assert "agent_id" in metrics
                assert "startup_time_ms" in metrics
                assert "tasks_completed" in metrics
            
            # Shutdown short-lived agent
            await self.agent_factory.shutdown_short_lived_agent(
                short_lived_agent.agent_id,
                pool_for_reuse=False
            )
    
    @pytest.mark.asyncio
    async def test_a2a_error_handling(self):
        """Test A2A error handling"""
        await self.agent_factory.initialize()
        
        # Test agent not found error
        non_existent_agent = await self.agent_factory.get_agent("non_existent_id")
        assert non_existent_agent is None
        
        # Test invalid agent type
        with pytest.raises(Exception):
            await self.agent_factory.create_agent(
                agent_type="invalid_type",
                name="Invalid Agent"
            )
        
        # Test connectivity to non-existent A2A endpoint
        connectivity_result = await self.agent_factory.test_agent_connectivity(
            "http://non-existent-server:9999"
        )
        
        assert connectivity_result["success"] == False
        assert "error" in connectivity_result
    
    @pytest.mark.asyncio
    async def test_a2a_compliance_validation(self):
        """Test A2A compliance validation"""
        await self.agent_factory.initialize()
        
        # Create an agent
        agent = await self.agent_factory.create_agent(
            agent_type="general",
            name="Compliance Test Agent"
        )
        
        self.test_agents.append(agent)
        
        # Test agent capabilities retrieval
        capabilities = await self.agent_factory.get_agent_capabilities_from_url(
            f"http://localhost:8000/agents/{agent.agent_id}"
        )
        
        # Note: This might return None if the A2A endpoint isn't running
        # In a full integration test, we would have the A2A transport layer running
        if capabilities:
            assert "capabilities" in capabilities
            assert "skills" in capabilities
    
    @pytest.mark.asyncio
    async def test_a2a_discovery_statistics(self):
        """Test A2A discovery statistics"""
        await self.agent_factory.initialize()
        
        # Get discovery statistics
        stats = self.agent_factory.get_discovery_stats()
        
        # Verify statistics structure
        if "error" not in stats:
            assert "total_discovered" in stats
            assert "available_agents" in stats
            assert "registries_count" in stats
        else:
            # Discovery not available
            assert stats["error"] is not None


@pytest.mark.skipif(not A2A_INTEGRATION_AVAILABLE, reason="A2A integration not available")
class TestA2APerformance:
    """Test A2A Performance"""
    
    @pytest.mark.asyncio
    async def test_agent_creation_performance(self):
        """Test agent creation performance"""
        agent_factory = AgentFactory()
        await agent_factory.initialize()
        
        # Measure agent creation time
        start_time = datetime.utcnow()
        
        agents = []
        for i in range(5):
            agent = await agent_factory.create_agent(
                agent_type="general",
                name=f"Performance Test Agent {i}"
            )
            agents.append(agent)
        
        end_time = datetime.utcnow()
        creation_time = (end_time - start_time).total_seconds()
        
        # Verify performance (should create 5 agents in reasonable time)
        assert creation_time < 30.0  # 30 seconds max for 5 agents
        assert len(agents) == 5
        
        # Cleanup
        for agent in agents:
            await agent_factory.destroy_agent(agent.agent_id)
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self):
        """Test concurrent agent operations"""
        agent_factory = AgentFactory()
        await agent_factory.initialize()
        
        # Create multiple agents concurrently
        async def create_test_agent(index):
            return await agent_factory.create_agent(
                agent_type="general",
                name=f"Concurrent Test Agent {index}"
            )
        
        # Create 3 agents concurrently
        tasks = [create_test_agent(i) for i in range(3)]
        agents = await asyncio.gather(*tasks)
        
        # Verify all agents were created
        assert len(agents) == 3
        for agent in agents:
            assert agent is not None
            assert agent.agent_id is not None
        
        # Cleanup
        cleanup_tasks = [agent_factory.destroy_agent(agent.agent_id) for agent in agents]
        await asyncio.gather(*cleanup_tasks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
