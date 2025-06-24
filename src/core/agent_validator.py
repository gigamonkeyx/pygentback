"""
Agent Validator - Agent Validation and Quality Assurance

This module provides validation capabilities for agents, including configuration
validation, capability verification, and health checks.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from .agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of an agent validation."""
    is_valid: bool
    level: ValidationLevel
    message: str
    component: str = ""
    details: Optional[Dict[str, Any]] = None


class AgentValidator:
    """
    Validator class for agent configurations and implementations.
    
    Provides validation for:
    - Agent configurations
    - Capability requirements
    - Resource availability
    - Health checks
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, callable] = {}
        self.capability_requirements: Dict[str, List[str]] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules."""
        self.validation_rules.update({
            "agent_id_format": self._validate_agent_id_format,
            "name_format": self._validate_name_format,
            "agent_type_valid": self._validate_agent_type,
            "capabilities_valid": self._validate_capabilities,
            "config_completeness": self._validate_config_completeness
        })
        
        # Setup capability requirements
        self.capability_requirements.update({
            "text_generation": ["model_name"],
            "conversation": ["max_tokens", "temperature"],
            "research": ["search_depth"],
            "analysis": ["analysis_type"]
        })
    
    def validate_config(self, config: AgentConfig) -> List[ValidationResult]:
        """
        Validate agent configuration.
        
        Args:
            config: Agent configuration to validate
            
        Returns:
            List[ValidationResult]: List of validation results
        """
        results = []
        
        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                result = rule_func(config)
                if result:
                    results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Validation rule '{rule_name}' failed: {str(e)}",
                    component="validator"
                ))
        
        return results
    
    def validate_agent(self, agent: BaseAgent) -> List[ValidationResult]:
        """
        Validate agent instance.
        
        Args:
            agent: Agent instance to validate
            
        Returns:
            List[ValidationResult]: List of validation results
        """
        results = []
        
        # Validate configuration
        results.extend(self.validate_config(agent.config))
        
        # Validate agent state
        if hasattr(agent, 'status'):
            if not agent.status:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message="Agent status is undefined",
                    component="agent_state"
                ))
        
        # Validate capabilities
        if hasattr(agent, 'enabled_capabilities'):
            for capability in agent.enabled_capabilities:
                cap_result = self._validate_capability_implementation(agent, capability)
                if cap_result:
                    results.append(cap_result)
        
        return results
    
    def _validate_agent_id_format(self, config: AgentConfig) -> Optional[ValidationResult]:
        """Validate agent ID format."""
        if not config.agent_id:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Agent ID is required",
                component="agent_id"
            )
        
        if len(config.agent_id) < 3:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message="Agent ID should be at least 3 characters",
                component="agent_id"
            )
        
        return None
    
    def _validate_name_format(self, config: AgentConfig) -> Optional[ValidationResult]:
        """Validate agent name format."""
        if not config.name:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Agent name is required",
                component="name"
            )
        
        if len(config.name) > 100:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message="Agent name is very long (>100 chars)",
                component="name"
            )
        
        return None
    
    def _validate_agent_type(self, config: AgentConfig) -> Optional[ValidationResult]:
        """Validate agent type."""
        valid_types = ["general", "reasoning", "search", "evolution", "coding", "research", "basic", "nlp"]
        
        if not config.agent_type:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Agent type is required",
                component="agent_type"
            )
        
        if config.agent_type not in valid_types:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Unknown agent type '{config.agent_type}'. Known types: {valid_types}",
                component="agent_type"
            )
        
        return None
    
    def _validate_capabilities(self, config: AgentConfig) -> Optional[ValidationResult]:
        """Validate agent capabilities."""
        if not config.enabled_capabilities:
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="No capabilities enabled",
                component="capabilities"
            )
        
        # Check if capabilities have required config
        missing_reqs = []
        for capability in config.enabled_capabilities:
            if capability in self.capability_requirements:
                required_fields = self.capability_requirements[capability]
                for field in required_fields:
                    if field not in config.custom_config:
                        missing_reqs.append(f"{capability}:{field}")
        
        if missing_reqs:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Missing capability requirements: {missing_reqs}",
                component="capabilities",
                details={"missing_requirements": missing_reqs}
            )
        
        return None
    
    def _validate_config_completeness(self, config: AgentConfig) -> Optional[ValidationResult]:
        """Validate configuration completeness."""
        required_custom_fields = ["model_name", "max_tokens"]
        missing_fields = []
        
        for field in required_custom_fields:
            if field not in config.custom_config:
                missing_fields.append(field)
        
        if missing_fields:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.INFO,
                message=f"Optional config fields missing: {missing_fields}",
                component="config",
                details={"missing_fields": missing_fields}
            )
        
        return None
    
    def _validate_capability_implementation(self, agent: BaseAgent, capability: str) -> Optional[ValidationResult]:
        """Validate that agent implements required capability."""
        # This is a simplified check - in practice would verify actual implementation
        if capability == "text_generation" and not hasattr(agent, 'generate_text'):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Agent does not implement {capability} capability method",
                component="capability_implementation",
                details={"capability": capability}
            )
        
        return None
    
    def register_validation_rule(self, name: str, rule_func: callable):
        """Register a custom validation rule."""
        self.validation_rules[name] = rule_func
    
    def register_capability_requirement(self, capability: str, requirements: List[str]):
        """Register requirements for a capability."""
        self.capability_requirements[capability] = requirements
    
    def is_configuration_valid(self, config: AgentConfig) -> bool:
        """Check if configuration is valid (no errors)."""
        results = self.validate_config(config)
        return not any(r.level == ValidationLevel.ERROR for r in results)
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Get summary of validation results."""
        summary = {
            "total": len(results),
            "errors": sum(1 for r in results if r.level == ValidationLevel.ERROR),
            "warnings": sum(1 for r in results if r.level == ValidationLevel.WARNING),
            "info": sum(1 for r in results if r.level == ValidationLevel.INFO)
        }
        summary["valid"] = summary["errors"] == 0
        return summary
