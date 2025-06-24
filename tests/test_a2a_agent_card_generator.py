#!/usr/bin/env python3
"""
Test A2A Agent Card Generator

Tests the A2A-compliant agent card generation according to Google A2A specification.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock
from datetime import datetime

# Import the A2A components
try:
    from src.a2a_protocol.agent_card_generator import (
        A2AAgentCardGenerator, AgentCard, AgentSkill, AgentCapabilities,
        AgentProvider, SecurityScheme
    )
    from src.a2a_protocol.well_known_handler import A2AWellKnownHandler
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AAgentCardGenerator:
    """Test A2A Agent Card Generator"""
    
    def setup_method(self):
        """Setup test environment"""
        self.base_url = "http://localhost:8000"
        self.generator = A2AAgentCardGenerator(self.base_url)
        
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "test-agent-123"
        self.mock_agent.name = "Test Agent"
        self.mock_agent.description = "A test agent for A2A compliance"
        self.mock_agent.capabilities = []
    
    def test_generate_basic_agent_card(self):
        """Test basic agent card generation"""
        agent_card = self.generator.generate_agent_card(
            agent=self.mock_agent,
            agent_type="general"
        )
        
        # Verify required A2A fields
        assert agent_card.name == "Test Agent"
        assert agent_card.url == f"{self.base_url}/a2a/agents/{self.mock_agent.agent_id}"
        assert agent_card.version == "1.0.0"
        assert isinstance(agent_card.provider, AgentProvider)
        assert isinstance(agent_card.capabilities, AgentCapabilities)
        assert isinstance(agent_card.skills, list)
        assert len(agent_card.skills) >= 1  # Should have at least one skill
        
        # Verify security schemes
        assert "bearerAuth" in agent_card.securitySchemes
        assert "apiKeyAuth" in agent_card.securitySchemes
        assert len(agent_card.security) == 2
        
        # Verify provider information
        assert agent_card.provider.name == "PyGent Factory"
        assert agent_card.provider.organization == "PyGent Factory"
        
        # Verify capabilities
        assert agent_card.capabilities.streaming == True
        assert agent_card.capabilities.stateTransitionHistory == True
        
        # Verify metadata
        assert "created" in agent_card.metadata
        assert agent_card.metadata["agent_id"] == self.mock_agent.agent_id
        assert agent_card.metadata["agent_type"] == "general"
    
    def test_generate_agent_card_with_custom_skills(self):
        """Test agent card generation with custom skills"""
        custom_skills = [
            AgentSkill(
                id="custom_skill",
                name="Custom Skill",
                description="A custom skill for testing",
                tags=["custom", "test"],
                examples=["Test this custom skill"]
            )
        ]
        
        agent_card = self.generator.generate_agent_card(
            agent=self.mock_agent,
            agent_type="research",
            custom_skills=custom_skills
        )
        
        # Verify custom skills are included
        skill_ids = [skill.id for skill in agent_card.skills]
        assert "custom_skill" in skill_ids
        
        # Find the custom skill
        custom_skill = next(skill for skill in agent_card.skills if skill.id == "custom_skill")
        assert custom_skill.name == "Custom Skill"
        assert custom_skill.description == "A custom skill for testing"
        assert "custom" in custom_skill.tags
        assert "test" in custom_skill.tags
    
    def test_generate_agent_card_research_type(self):
        """Test agent card generation for research agent type"""
        agent_card = self.generator.generate_agent_card(
            agent=self.mock_agent,
            agent_type="research"
        )
        
        # Verify research-specific skills are included
        skill_ids = [skill.id for skill in agent_card.skills]
        assert "document_search" in skill_ids
        assert "literature_review" in skill_ids
        
        # Verify description includes research specialization
        assert "research" in agent_card.description.lower()
    
    def test_generate_agent_card_analysis_type(self):
        """Test agent card generation for analysis agent type"""
        agent_card = self.generator.generate_agent_card(
            agent=self.mock_agent,
            agent_type="analysis"
        )
        
        # Verify analysis-specific skills are included
        skill_ids = [skill.id for skill in agent_card.skills]
        assert "statistical_analysis" in skill_ids
        assert "data_visualization" in skill_ids
        
        # Verify description includes analysis specialization
        assert "analysis" in agent_card.description.lower()
    
    def test_generate_agent_card_with_agent_capabilities(self):
        """Test agent card generation with agent capabilities"""
        # Add mock capabilities to agent
        mock_capability = Mock()
        mock_capability.name = "Document Search"
        mock_capability.description = "Search through documents"
        self.mock_agent.capabilities = [mock_capability]
        
        agent_card = self.generator.generate_agent_card(
            agent=self.mock_agent,
            agent_type="general"
        )
        
        # Verify capability is converted to skill
        skill_ids = [skill.id for skill in agent_card.skills]
        assert "document_search" in skill_ids
        
        # Find the capability-based skill
        doc_search_skill = next(skill for skill in agent_card.skills if skill.id == "document_search")
        assert doc_search_skill.name == "Document Search"
        assert doc_search_skill.description == "Search through documents"
    
    def test_generate_agent_card_without_authentication(self):
        """Test agent card generation without authentication"""
        agent_card = self.generator.generate_agent_card(
            agent=self.mock_agent,
            agent_type="general",
            enable_authentication=False
        )
        
        # Verify no security schemes when authentication disabled
        assert len(agent_card.securitySchemes) == 0
        assert len(agent_card.security) == 0
        assert agent_card.supportsAuthenticatedExtendedCard == False
    
    def test_generate_agent_card_with_push_notifications(self):
        """Test agent card generation with push notifications enabled"""
        agent_card = self.generator.generate_agent_card(
            agent=self.mock_agent,
            agent_type="general",
            enable_push_notifications=True
        )
        
        # Verify push notifications capability
        assert agent_card.capabilities.pushNotifications == True
    
    def test_to_json_serialization(self):
        """Test agent card JSON serialization"""
        agent_card = self.generator.generate_agent_card(
            agent=self.mock_agent,
            agent_type="general"
        )
        
        # Convert to JSON
        json_data = self.generator.to_json(agent_card)
        
        # Verify JSON structure
        assert isinstance(json_data, dict)
        assert "name" in json_data
        assert "description" in json_data
        assert "url" in json_data
        assert "version" in json_data
        assert "provider" in json_data
        assert "capabilities" in json_data
        assert "skills" in json_data
        assert "securitySchemes" in json_data
        assert "security" in json_data
        
        # Verify JSON is serializable
        json_string = json.dumps(json_data)
        assert isinstance(json_string, str)
        
        # Verify JSON can be parsed back
        parsed_data = json.loads(json_string)
        assert parsed_data["name"] == agent_card.name
    
    def test_well_known_url_content_generation(self):
        """Test well-known URL content generation"""
        agent_card = self.generator.generate_agent_card(
            agent=self.mock_agent,
            agent_type="general"
        )
        
        # Generate well-known URL content
        content = self.generator.generate_well_known_url_content(agent_card)
        
        # Verify content is valid JSON
        parsed_content = json.loads(content)
        assert isinstance(parsed_content, dict)
        assert parsed_content["name"] == agent_card.name
        assert parsed_content["url"] == agent_card.url


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AWellKnownHandler:
    """Test A2A Well-Known URL Handler"""
    
    def setup_method(self):
        """Setup test environment"""
        self.base_url = "http://localhost:8000"
        self.handler = A2AWellKnownHandler(self.base_url)
        self.generator = A2AAgentCardGenerator(self.base_url)
        
        # Create mock agent and agent card
        mock_agent = Mock()
        mock_agent.agent_id = "test-agent-123"
        mock_agent.name = "Test Agent"
        mock_agent.capabilities = []
        
        self.agent_card = self.generator.generate_agent_card(
            agent=mock_agent,
            agent_type="general"
        )
    
    def test_register_agent_card(self):
        """Test agent card registration"""
        agent_id = "test-agent-123"
        
        # Register agent card
        self.handler.register_agent_card(agent_id, self.agent_card)
        
        # Verify registration
        assert agent_id in self.handler.agent_cards
        assert self.handler.agent_cards[agent_id] == self.agent_card
    
    def test_unregister_agent_card(self):
        """Test agent card unregistration"""
        agent_id = "test-agent-123"
        
        # Register then unregister
        self.handler.register_agent_card(agent_id, self.agent_card)
        self.handler.unregister_agent_card(agent_id)
        
        # Verify unregistration
        assert agent_id not in self.handler.agent_cards
    
    def test_get_agent_discovery_info(self):
        """Test agent discovery information"""
        agent_id = "test-agent-123"
        self.handler.register_agent_card(agent_id, self.agent_card)
        
        # Get discovery info
        discovery_info = self.handler.get_agent_discovery_info()
        
        # Verify discovery info structure
        assert "registered_agents" in discovery_info
        assert "total_agents" in discovery_info
        assert "base_url" in discovery_info
        assert "well_known_urls" in discovery_info
        
        # Verify agent is listed
        assert agent_id in discovery_info["registered_agents"]
        assert discovery_info["total_agents"] == 1
        assert discovery_info["base_url"] == self.base_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
