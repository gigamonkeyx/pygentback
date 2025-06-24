#!/usr/bin/env python3
"""
Test A2A Agent Discovery Mechanism

Tests the A2A-compliant agent discovery implementation according to Google A2A specification.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from aiohttp import web, ClientSession
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

# Import the A2A discovery components
try:
    from src.a2a_protocol.discovery import (
        A2AAgentDiscovery, A2ADiscoveryClient, DiscoveredAgent, 
        DiscoveryConfig, CapabilityMatch
    )
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AAgentDiscovery:
    """Test A2A Agent Discovery"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = DiscoveryConfig(
            discovery_timeout_seconds=5,
            verification_interval_minutes=1,
            max_concurrent_discoveries=5
        )
        self.discovery = A2AAgentDiscovery(self.config)
    
    def teardown_method(self):
        """Cleanup after tests"""
        if self.discovery.verification_task:
            self.discovery.verification_task.cancel()
    
    def test_discovery_config(self):
        """Test discovery configuration"""
        config = DiscoveryConfig(
            discovery_timeout_seconds=10,
            verification_interval_minutes=30,
            max_concurrent_discoveries=15
        )
        
        assert config.discovery_timeout_seconds == 10
        assert config.verification_interval_minutes == 30
        assert config.max_concurrent_discoveries == 15
        assert config.cache_duration_minutes == 60  # Default value
    
    def test_discovered_agent_creation(self):
        """Test DiscoveredAgent data structure"""
        agent = DiscoveredAgent(
            agent_id="test-agent-123",
            name="Test Agent",
            description="A test agent for discovery",
            url="http://localhost:8000/a2a/agents/test-agent-123",
            well_known_url="http://localhost:8000/.well-known/agent.json",
            capabilities={"streaming": True, "pushNotifications": False},
            skills=[
                {"id": "research", "name": "Research", "tags": ["search", "analysis"]}
            ],
            provider={"name": "Test Provider", "organization": "Test Org"}
        )
        
        assert agent.agent_id == "test-agent-123"
        assert agent.name == "Test Agent"
        assert agent.capabilities["streaming"] == True
        assert len(agent.skills) == 1
        assert agent.available == True  # Default value
        assert agent.discovered_at is not None
    
    @pytest.mark.asyncio
    async def test_parse_agent_card_individual(self):
        """Test parsing individual agent card"""
        agent_card_data = {
            "name": "Research Agent",
            "description": "Specialized research agent",
            "url": "http://localhost:8000/a2a/agents/research-123",
            "capabilities": {"streaming": True, "pushNotifications": False},
            "skills": [
                {"id": "document_search", "name": "Document Search", "tags": ["search"]}
            ],
            "provider": {"name": "PyGent Factory", "organization": "PyGent Factory"},
            "metadata": {"agent_id": "research-123"}
        }
        
        agent = self.discovery._parse_agent_card(
            agent_card_data,
            "http://localhost:8000/.well-known/agent.json",
            "http://localhost:8000",
            100.0
        )
        
        assert agent is not None
        assert agent.agent_id == "research-123"
        assert agent.name == "Research Agent"
        assert agent.url == "http://localhost:8000/a2a/agents/research-123"
        assert agent.response_time_ms == 100.0
        assert len(agent.skills) == 1
    
    @pytest.mark.asyncio
    async def test_parse_agent_card_list(self):
        """Test parsing agent list response"""
        agent_list_data = {
            "agents": [
                {"id": "agent-1", "name": "Agent 1"},
                {"id": "agent-2", "name": "Agent 2"}
            ],
            "total": 2
        }
        
        agent = self.discovery._parse_agent_card(
            agent_list_data,
            "http://localhost:8000/.well-known/agent.json",
            "http://localhost:8000",
            50.0
        )
        
        assert agent is not None
        assert "service_" in agent.agent_id
        assert "Agent Service" in agent.name
        assert agent.capabilities["multi_agent"] == True
        assert agent.capabilities["agent_count"] == 2
        assert agent.metadata == agent_list_data
    
    def test_capability_matching(self):
        """Test capability matching algorithm"""
        agent = DiscoveredAgent(
            agent_id="test-agent",
            name="Test Agent",
            description="Test agent",
            url="http://localhost:8000/agents/test",
            well_known_url="http://localhost:8000/.well-known/agent.json",
            capabilities={"streaming": True, "pushNotifications": False},
            skills=[
                {"id": "research", "name": "Research", "tags": ["search", "analysis"]},
                {"id": "data_processing", "name": "Data Processing", "tags": ["transform"]}
            ],
            provider={"name": "Test", "organization": "Test"}
        )
        
        # Test perfect match
        score, matching_skills, missing = self.discovery._calculate_capability_match(
            agent, ["streaming"], ["research"]
        )
        assert score == 1.0
        assert "research" in matching_skills
        assert len(missing) == 0
        
        # Test partial match
        score, matching_skills, missing = self.discovery._calculate_capability_match(
            agent, ["streaming", "push_notifications"], ["research", "translation"]
        )
        assert score == 0.5  # 2 out of 4 requirements met
        assert "research" in matching_skills
        assert "push_notifications" in missing
        
        # Test no match
        score, matching_skills, missing = self.discovery._calculate_capability_match(
            agent, ["push_notifications"], ["translation"]
        )
        assert score == 0.0
        assert len(matching_skills) == 0
        assert "push_notifications" in missing
    
    @pytest.mark.asyncio
    async def test_find_agents_by_capability(self):
        """Test finding agents by capability"""
        # Add test agents
        agent1 = DiscoveredAgent(
            agent_id="research-agent",
            name="Research Agent",
            description="Research specialist",
            url="http://localhost:8000/agents/research",
            well_known_url="http://localhost:8000/.well-known/agent.json",
            capabilities={"streaming": True},
            skills=[{"id": "research", "tags": ["search"]}],
            provider={"name": "Test", "organization": "Test"}
        )
        
        agent2 = DiscoveredAgent(
            agent_id="analysis-agent",
            name="Analysis Agent",
            description="Analysis specialist",
            url="http://localhost:8000/agents/analysis",
            well_known_url="http://localhost:8000/.well-known/agent.json",
            capabilities={"pushNotifications": True},
            skills=[{"id": "analysis", "tags": ["statistics"]}],
            provider={"name": "Test", "organization": "Test"}
        )
        
        self.discovery.discovered_agents["research-agent"] = agent1
        self.discovery.discovered_agents["analysis-agent"] = agent2
        
        # Find agents with streaming capability
        matches = await self.discovery.find_agents_by_capability(["streaming"])
        
        assert len(matches) == 1
        assert matches[0].agent_id == "research-agent"
        assert matches[0].match_score == 1.0
        
        # Find agents with research skill
        matches = await self.discovery.find_agents_by_capability([], ["research"])
        
        assert len(matches) == 1
        assert matches[0].agent_id == "research-agent"
        assert "research" in matches[0].matching_skills
    
    def test_discovery_stats(self):
        """Test discovery statistics"""
        # Add test agents
        agent1 = DiscoveredAgent(
            agent_id="agent-1", name="Agent 1", description="Test",
            url="http://test1", well_known_url="http://test1/.well-known/agent.json",
            capabilities={}, skills=[], provider={}, available=True, response_time_ms=100.0
        )
        
        agent2 = DiscoveredAgent(
            agent_id="agent-2", name="Agent 2", description="Test",
            url="http://test2", well_known_url="http://test2/.well-known/agent.json",
            capabilities={}, skills=[], provider={}, available=False, response_time_ms=200.0
        )
        
        self.discovery.discovered_agents["agent-1"] = agent1
        self.discovery.discovered_agents["agent-2"] = agent2
        self.discovery.add_agent_registry("http://registry1")
        self.discovery.add_agent_registry("http://registry2")
        
        stats = self.discovery.get_discovery_stats()
        
        assert stats["total_discovered"] == 2
        assert stats["available_agents"] == 1
        assert stats["unavailable_agents"] == 1
        assert stats["average_response_time_ms"] == 100.0  # Only available agents
        assert stats["registries_count"] == 2


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2ADiscoveryClient:
    """Test A2A Discovery Client"""
    
    @pytest.mark.asyncio
    async def test_extract_capabilities_from_description(self):
        """Test capability extraction from task description"""
        client = A2ADiscoveryClient()
        
        # Test streaming detection
        capabilities = client._extract_capabilities_from_description(
            "I need real-time streaming data analysis"
        )
        assert "streaming" in capabilities
        
        # Test notification detection
        capabilities = client._extract_capabilities_from_description(
            "Send me notifications when the task completes"
        )
        assert "push_notifications" in capabilities
        
        # Test history detection
        capabilities = client._extract_capabilities_from_description(
            "Track the history of all changes"
        )
        assert "state_history" in capabilities
    
    @pytest.mark.asyncio
    async def test_extract_skills_from_description(self):
        """Test skill extraction from task description"""
        client = A2ADiscoveryClient()
        
        # Test research skill detection
        skills = client._extract_skills_from_description(
            "Research the latest papers on machine learning"
        )
        assert "research" in skills
        
        # Test analysis skill detection
        skills = client._extract_skills_from_description(
            "Analyze the data and provide insights"
        )
        assert "analysis" in skills
        
        # Test generation skill detection
        skills = client._extract_skills_from_description(
            "Generate a comprehensive report"
        )
        assert "generation" in skills
        
        # Test multiple skills
        skills = client._extract_skills_from_description(
            "Research the topic, analyze the data, and generate a summary"
        )
        assert "research" in skills
        assert "analysis" in skills
        assert "generation" in skills
        assert "summarization" in skills


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestCapabilityMatch:
    """Test Capability Match data structure"""
    
    def test_capability_match_creation(self):
        """Test CapabilityMatch creation"""
        match = CapabilityMatch(
            agent_id="test-agent",
            agent_name="Test Agent",
            match_score=0.85,
            matching_skills=["research", "analysis"],
            missing_capabilities=["streaming"],
            agent_url="http://localhost:8000/agents/test"
        )
        
        assert match.agent_id == "test-agent"
        assert match.agent_name == "Test Agent"
        assert match.match_score == 0.85
        assert len(match.matching_skills) == 2
        assert "research" in match.matching_skills
        assert "streaming" in match.missing_capabilities
        assert match.agent_url == "http://localhost:8000/agents/test"


# Mock HTTP server for testing
class MockAgentServer:
    """Mock agent server for testing discovery"""
    
    def __init__(self):
        self.agent_cards = {}
    
    def add_agent_card(self, path: str, card_data: dict):
        """Add agent card to mock server"""
        self.agent_cards[path] = card_data
    
    async def handle_request(self, request):
        """Handle HTTP request"""
        path = request.path
        
        if path in self.agent_cards:
            return web.json_response(self.agent_cards[path])
        else:
            return web.Response(status=404)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
