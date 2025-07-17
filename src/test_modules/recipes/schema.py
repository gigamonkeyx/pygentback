"""
Recipe Schema and Definition System

This module provides the core schema and definition system for Agent + MCP recipes,
enabling standardized recipe creation, validation, and management.
"""

import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

try:
    from ...core.agent.capability import CapabilityType
except ImportError:
    # Fallback for testing
    from enum import Enum
    class CapabilityType(Enum):
        REASONING = "reasoning"
        CODING = "coding"
        RESEARCH = "research"
        ANALYSIS = "analysis"
        COMMUNICATION = "communication"
        TOOL_USE = "tool_use"
try:
    from ...mcp.server.config import MCPServerType
except ImportError:
    # Fallback for testing
    from enum import Enum
    class MCPServerType(Enum):
        FILESYSTEM = "filesystem"
        DATABASE = "database"
        API = "api"
        TOOL = "tool"
        CUSTOM = "custom"


class RecipeCategory(Enum):
    """Categories of Agent + MCP recipes"""
    NLP_PROCESSING = "nlp_processing"
    GRAPHICS_MEDIA = "graphics_media"
    DATABASE_OPS = "database_ops"
    WEB_UI_CREATION = "web_ui_creation"
    DEVELOPMENT = "development"
    DATA_ANALYSIS = "data_analysis"
    AUTOMATION = "automation"
    INTEGRATION = "integration"
    RESEARCH = "research"
    COMMUNICATION = "communication"
    CODING = "coding"
    ACADEMIC_RESEARCH = "academic_research"


class RecipeDifficulty(Enum):
    """Recipe complexity levels"""
    BASIC = "basic"           # Single agent, single MCP tool
    INTERMEDIATE = "intermediate"  # Single agent, multiple tools
    ADVANCED = "advanced"     # Multiple agents, complex workflows
    EXPERT = "expert"         # Multi-agent coordination, advanced patterns


class RecipeStatus(Enum):
    """Recipe validation status"""
    DRAFT = "draft"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class MCPToolRequirement:
    """Defines an MCP tool requirement for a recipe"""
    server_name: str
    tool_name: str
    server_type: MCPServerType
    required_capabilities: List[str] = field(default_factory=list)
    optional_parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    retry_count: int = 3


@dataclass
class AgentRequirement:
    """Defines an agent requirement for a recipe"""
    agent_type: str
    required_capabilities: List[CapabilityType] = field(default_factory=list)
    memory_limit_mb: int = 512
    max_execution_time: int = 300
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecipeStep:
    """Defines a single step in a recipe execution"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    agent_action: str = ""
    mcp_tools: List[str] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)  # Step IDs this depends on
    timeout_seconds: int = 60
    retry_on_failure: bool = True
    critical: bool = True  # If True, recipe fails if this step fails


@dataclass
class RecipeValidationCriteria:
    """Defines validation criteria for a recipe"""
    success_threshold: float = 0.8  # Minimum success rate
    performance_budget_ms: int = 5000  # Maximum execution time
    memory_budget_mb: int = 1024  # Maximum memory usage
    required_outputs: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    custom_validators: List[str] = field(default_factory=list)


@dataclass
class RecipeDefinition:
    """
    Complete definition of an Agent + MCP recipe.
    
    A recipe defines how agents and MCP tools work together to accomplish
    specific tasks, including execution steps, validation criteria, and metadata.
    """
    
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # Classification
    category: RecipeCategory = RecipeCategory.INTEGRATION
    difficulty: RecipeDifficulty = RecipeDifficulty.BASIC
    tags: List[str] = field(default_factory=list)
    
    # Requirements
    agent_requirements: List[AgentRequirement] = field(default_factory=list)
    mcp_requirements: List[MCPToolRequirement] = field(default_factory=list)
    
    # Execution definition
    steps: List[RecipeStep] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)  # Step IDs in order
    parallel_steps: List[List[str]] = field(default_factory=list)  # Groups of parallel steps
    
    # Validation
    validation_criteria: RecipeValidationCriteria = field(default_factory=RecipeValidationCriteria)
    test_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    author: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: RecipeStatus = RecipeStatus.DRAFT
    
    # Performance and usage data
    success_rate: float = 0.0
    average_execution_time_ms: int = 0
    usage_count: int = 0
    last_tested: Optional[datetime] = None
    
    # Additional metadata
    documentation_url: Optional[str] = None
    example_usage: Optional[str] = None
    known_issues: List[str] = field(default_factory=list)
    related_recipes: List[str] = field(default_factory=list)
    
    def add_step(self, step: RecipeStep) -> None:
        """Add a step to the recipe"""
        self.steps.append(step)
        if step.step_id not in self.execution_order:
            self.execution_order.append(step.step_id)
        self.updated_at = datetime.utcnow()
    
    def add_agent_requirement(self, requirement: AgentRequirement) -> None:
        """Add an agent requirement"""
        self.agent_requirements.append(requirement)
        self.updated_at = datetime.utcnow()
    
    def add_mcp_requirement(self, requirement: MCPToolRequirement) -> None:
        """Add an MCP tool requirement"""
        self.mcp_requirements.append(requirement)
        self.updated_at = datetime.utcnow()
    
    def get_required_servers(self) -> List[str]:
        """Get list of required MCP server names"""
        return list(set(req.server_name for req in self.mcp_requirements))
    
    def get_required_tools(self) -> List[str]:
        """Get list of required MCP tool names"""
        return list(set(req.tool_name for req in self.mcp_requirements))
    
    def get_required_agent_types(self) -> List[str]:
        """Get list of required agent types"""
        return list(set(req.agent_type for req in self.agent_requirements))
    
    def validate_structure(self) -> List[str]:
        """Validate recipe structure and return list of issues"""
        issues = []
        
        # Check basic requirements
        if not self.name:
            issues.append("Recipe name is required")
        
        if not self.steps:
            issues.append("Recipe must have at least one step")
        
        if not self.agent_requirements:
            issues.append("Recipe must specify at least one agent requirement")
        
        # Validate step dependencies
        step_ids = {step.step_id for step in self.steps}
        for step in self.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    issues.append(f"Step {step.step_id} depends on non-existent step {dep_id}")
        
        # Validate execution order
        for step_id in self.execution_order:
            if step_id not in step_ids:
                issues.append(f"Execution order references non-existent step {step_id}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recipe to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "tags": self.tags,
            "agent_requirements": [
                {
                    "agent_type": req.agent_type,
                    "required_capabilities": [cap.value for cap in req.required_capabilities],
                    "memory_limit_mb": req.memory_limit_mb,
                    "max_execution_time": req.max_execution_time,
                    "configuration": req.configuration
                }
                for req in self.agent_requirements
            ],
            "mcp_requirements": [
                {
                    "server_name": req.server_name,
                    "tool_name": req.tool_name,
                    "server_type": req.server_type.value,
                    "required_capabilities": req.required_capabilities,
                    "optional_parameters": req.optional_parameters,
                    "timeout_seconds": req.timeout_seconds,
                    "retry_count": req.retry_count
                }
                for req in self.mcp_requirements
            ],
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "description": step.description,
                    "agent_action": step.agent_action,
                    "mcp_tools": step.mcp_tools,
                    "input_data": step.input_data,
                    "expected_output": step.expected_output,
                    "dependencies": step.dependencies,
                    "timeout_seconds": step.timeout_seconds,
                    "retry_on_failure": step.retry_on_failure,
                    "critical": step.critical
                }
                for step in self.steps
            ],
            "execution_order": self.execution_order,
            "parallel_steps": self.parallel_steps,
            "validation_criteria": {
                "success_threshold": self.validation_criteria.success_threshold,
                "performance_budget_ms": self.validation_criteria.performance_budget_ms,
                "memory_budget_mb": self.validation_criteria.memory_budget_mb,
                "required_outputs": self.validation_criteria.required_outputs,
                "quality_metrics": self.validation_criteria.quality_metrics,
                "custom_validators": self.validation_criteria.custom_validators
            },
            "test_scenarios": self.test_scenarios,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "success_rate": self.success_rate,
            "average_execution_time_ms": self.average_execution_time_ms,
            "usage_count": self.usage_count,
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
            "documentation_url": self.documentation_url,
            "example_usage": self.example_usage,
            "known_issues": self.known_issues,
            "related_recipes": self.related_recipes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecipeDefinition':
        """Create recipe from dictionary"""
        # Parse agent requirements
        agent_requirements = []
        for req_data in data.get("agent_requirements", []):
            agent_requirements.append(AgentRequirement(
                agent_type=req_data["agent_type"],
                required_capabilities=[CapabilityType(cap) for cap in req_data.get("required_capabilities", [])],
                memory_limit_mb=req_data.get("memory_limit_mb", 512),
                max_execution_time=req_data.get("max_execution_time", 300),
                configuration=req_data.get("configuration", {})
            ))
        
        # Parse MCP requirements
        mcp_requirements = []
        for req_data in data.get("mcp_requirements", []):
            mcp_requirements.append(MCPToolRequirement(
                server_name=req_data["server_name"],
                tool_name=req_data["tool_name"],
                server_type=MCPServerType(req_data["server_type"]),
                required_capabilities=req_data.get("required_capabilities", []),
                optional_parameters=req_data.get("optional_parameters", {}),
                timeout_seconds=req_data.get("timeout_seconds", 30),
                retry_count=req_data.get("retry_count", 3)
            ))
        
        # Parse steps
        steps = []
        for step_data in data.get("steps", []):
            steps.append(RecipeStep(
                step_id=step_data["step_id"],
                name=step_data["name"],
                description=step_data["description"],
                agent_action=step_data["agent_action"],
                mcp_tools=step_data["mcp_tools"],
                input_data=step_data["input_data"],
                expected_output=step_data.get("expected_output"),
                dependencies=step_data.get("dependencies", []),
                timeout_seconds=step_data.get("timeout_seconds", 60),
                retry_on_failure=step_data.get("retry_on_failure", True),
                critical=step_data.get("critical", True)
            ))
        
        # Parse validation criteria
        validation_data = data.get("validation_criteria", {})
        validation_criteria = RecipeValidationCriteria(
            success_threshold=validation_data.get("success_threshold", 0.8),
            performance_budget_ms=validation_data.get("performance_budget_ms", 5000),
            memory_budget_mb=validation_data.get("memory_budget_mb", 1024),
            required_outputs=validation_data.get("required_outputs", []),
            quality_metrics=validation_data.get("quality_metrics", {}),
            custom_validators=validation_data.get("custom_validators", [])
        )
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            category=RecipeCategory(data.get("category", "integration")),
            difficulty=RecipeDifficulty(data.get("difficulty", "basic")),
            tags=data.get("tags", []),
            agent_requirements=agent_requirements,
            mcp_requirements=mcp_requirements,
            steps=steps,
            execution_order=data.get("execution_order", []),
            parallel_steps=data.get("parallel_steps", []),
            validation_criteria=validation_criteria,
            test_scenarios=data.get("test_scenarios", []),
            author=data.get("author", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            status=RecipeStatus(data.get("status", "draft")),
            success_rate=data.get("success_rate", 0.0),
            average_execution_time_ms=data.get("average_execution_time_ms", 0),
            usage_count=data.get("usage_count", 0),
            last_tested=datetime.fromisoformat(data["last_tested"]) if data.get("last_tested") else None,
            documentation_url=data.get("documentation_url"),
            example_usage=data.get("example_usage"),
            known_issues=data.get("known_issues", []),
            related_recipes=data.get("related_recipes", [])
        )
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save recipe to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'RecipeDefinition':
        """Load recipe from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class RecipeSchema:
    """
    Schema validation and management for recipes.
    
    Provides validation, versioning, and migration capabilities for recipe schemas.
    """
    
    CURRENT_VERSION = "1.0.0"
    
    @staticmethod
    def validate_recipe(recipe: RecipeDefinition) -> List[str]:
        """Validate a recipe against the current schema"""
        issues = []
        
        # Validate structure
        issues.extend(recipe.validate_structure())
        
        # Validate data types and constraints
        if recipe.validation_criteria.success_threshold < 0 or recipe.validation_criteria.success_threshold > 1:
            issues.append("Success threshold must be between 0 and 1")
        
        if recipe.validation_criteria.performance_budget_ms <= 0:
            issues.append("Performance budget must be positive")
        
        if recipe.validation_criteria.memory_budget_mb <= 0:
            issues.append("Memory budget must be positive")
        
        # Validate step timeouts
        for step in recipe.steps:
            if step.timeout_seconds <= 0:
                issues.append(f"Step {step.step_id} timeout must be positive")
        
        return issues
    
    @staticmethod
    def get_schema_version(recipe_data: Dict[str, Any]) -> str:
        """Get schema version from recipe data"""
        return recipe_data.get("schema_version", "1.0.0")
    
    @staticmethod
    def migrate_recipe(recipe_data: Dict[str, Any], target_version: str = None) -> Dict[str, Any]:
        """Migrate recipe data to target schema version"""
        if target_version is None:
            target_version = RecipeSchema.CURRENT_VERSION
        
        current_version = RecipeSchema.get_schema_version(recipe_data)
        
        if current_version == target_version:
            return recipe_data
        
        # Add migration logic here as schema evolves
        migrated_data = recipe_data.copy()
        migrated_data["schema_version"] = target_version
        
        return migrated_data
