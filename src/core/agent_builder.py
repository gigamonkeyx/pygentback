"""
Agent Builder - Agent Construction and Configuration

This module provides capabilities for building agents with complex configurations,
templates, and validation rules.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .agent import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentTemplate:
    """Template for building agents with predefined configurations."""
    name: str
    agent_type: str
    base_config: Dict[str, Any]
    capabilities: List[str]
    description: str = ""


class AgentBuilder:
    """
    Builder class for constructing agents with complex configurations.
    
    Provides template-based building, validation, and configuration management.
    """
    
    def __init__(self):
        self.build_templates: Dict[str, AgentTemplate] = {}
        self.capability_registry: Dict[str, Dict[str, Any]] = {}
        self._setup_default_templates()
    
    def _setup_default_templates(self):
        """Setup default agent templates."""
        # Basic agent template
        self.build_templates["basic"] = AgentTemplate(
            name="basic",
            agent_type="general",
            base_config={
                "max_tokens": 500,
                "temperature": 0.7,
                "model_name": "llama3.2:3b-instruct-q4_K_M"
            },
            capabilities=["text_generation", "conversation"],
            description="Basic general-purpose agent"
        )
        
        # NLP agent template
        self.build_templates["nlp"] = AgentTemplate(
            name="nlp",
            agent_type="general",
            base_config={
                "max_tokens": 1000,
                "temperature": 0.3,
                "model_name": "llama3.2:3b-instruct-q4_K_M"
            },
            capabilities=["text_processing", "entity_extraction", "classification"],
            description="Natural Language Processing agent"
        )
        
        # Research agent template
        self.build_templates["research"] = AgentTemplate(
            name="research",
            agent_type="research",
            base_config={
                "max_tokens": 2000,
                "temperature": 0.4,
                "search_depth": 3
            },
            capabilities=["research", "analysis", "synthesis"],
            description="Research and analysis agent"
        )
    
    def get_template(self, template_name: str) -> Optional[AgentTemplate]:
        """Get an agent template by name."""
        return self.build_templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.build_templates.keys())
    
    def register_template(self, template: AgentTemplate):
        """Register a new agent template."""
        self.build_templates[template.name] = template
    
    def build_from_template(self, template_name: str, **overrides) -> AgentConfig:
        """
        Build agent configuration from template with optional overrides.
        
        Args:
            template_name: Name of the template to use
            **overrides: Configuration overrides
            
        Returns:
            AgentConfig: Configured agent configuration
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Merge template config with overrides
        config_dict = {**template.base_config, **overrides}
        
        # Create agent config
        return AgentConfig(
            agent_id=overrides.get("agent_id", ""),
            name=overrides.get("name", f"{template.name}_agent"),
            agent_type=template.agent_type,
            enabled_capabilities=template.capabilities,
            custom_config=config_dict
        )
    
    def register_capability(self, name: str, config: Dict[str, Any]):
        """Register a new capability configuration."""
        self.capability_registry[name] = config
    
    def get_capability_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get capability configuration by name."""
        return self.capability_registry.get(name)
