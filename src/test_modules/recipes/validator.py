"""
Recipe Validator

This module provides comprehensive validation for Agent + MCP recipes,
ensuring structural integrity, dependency consistency, and execution feasibility.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from .schema import RecipeDefinition, RecipeStep, AgentRequirement, MCPToolRequirement


logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: str  # error, warning, info
    category: str  # structure, dependency, performance, security
    message: str
    location: str  # where the issue was found
    suggestion: Optional[str] = None


class RecipeValidator:
    """
    Comprehensive validator for Agent + MCP recipes.
    
    Validates recipe structure, dependencies, performance constraints,
    and execution feasibility.
    """
    
    def __init__(self):
        # Validation rules configuration
        self.max_steps = 50
        self.max_execution_time = 3600  # 1 hour
        self.max_memory_mb = 8192  # 8GB
        self.max_dependencies_per_step = 10
        self.max_parallel_groups = 5
        
        # Known security risks
        self.security_risk_patterns = [
            "exec", "eval", "subprocess", "os.system", "shell=True",
            "pickle.loads", "yaml.load", "input(", "raw_input("
        ]
        
        # Performance warning thresholds
        self.performance_thresholds = {
            "step_timeout": 300,  # 5 minutes
            "total_timeout": 1800,  # 30 minutes
            "memory_per_agent": 2048,  # 2GB
            "steps_count": 20
        }
    
    def validate_recipe(self, recipe: RecipeDefinition) -> List[ValidationIssue]:
        """
        Perform comprehensive validation of a recipe.
        
        Args:
            recipe: Recipe to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Structural validation
        issues.extend(self._validate_structure(recipe))
        
        # Dependency validation
        issues.extend(self._validate_dependencies(recipe))
        
        # Performance validation
        issues.extend(self._validate_performance(recipe))
        
        # Security validation
        issues.extend(self._validate_security(recipe))
        
        # Consistency validation
        issues.extend(self._validate_consistency(recipe))
        
        # Best practices validation
        issues.extend(self._validate_best_practices(recipe))
        
        return issues
    
    def _validate_structure(self, recipe: RecipeDefinition) -> List[ValidationIssue]:
        """Validate basic recipe structure"""
        issues = []
        
        # Required fields
        if not recipe.name or not recipe.name.strip():
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                message="Recipe name is required",
                location="recipe.name",
                suggestion="Provide a descriptive name for the recipe"
            ))
        
        if not recipe.description or not recipe.description.strip():
            issues.append(ValidationIssue(
                severity="warning",
                category="structure",
                message="Recipe description is missing",
                location="recipe.description",
                suggestion="Add a clear description of what the recipe does"
            ))
        
        # Steps validation
        if not recipe.steps:
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                message="Recipe must have at least one step",
                location="recipe.steps",
                suggestion="Add execution steps to the recipe"
            ))
        
        if len(recipe.steps) > self.max_steps:
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                message=f"Recipe has too many steps ({len(recipe.steps)} > {self.max_steps})",
                location="recipe.steps",
                suggestion="Consider breaking the recipe into smaller sub-recipes"
            ))
        
        # Agent requirements validation
        if not recipe.agent_requirements:
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                message="Recipe must specify at least one agent requirement",
                location="recipe.agent_requirements",
                suggestion="Add agent requirements to define execution context"
            ))
        
        # MCP requirements validation
        if not recipe.mcp_requirements:
            issues.append(ValidationIssue(
                severity="warning",
                category="structure",
                message="Recipe has no MCP tool requirements",
                location="recipe.mcp_requirements",
                suggestion="Consider adding MCP tools to enhance functionality"
            ))
        
        # Step structure validation
        for i, step in enumerate(recipe.steps):
            step_issues = self._validate_step_structure(step, f"step[{i}]")
            issues.extend(step_issues)
        
        return issues
    
    def _validate_step_structure(self, step: RecipeStep, location: str) -> List[ValidationIssue]:
        """Validate individual step structure"""
        issues = []
        
        if not step.name or not step.name.strip():
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                message="Step name is required",
                location=f"{location}.name",
                suggestion="Provide a descriptive name for the step"
            ))
        
        if not step.agent_action or not step.agent_action.strip():
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                message="Step must specify an agent action",
                location=f"{location}.agent_action",
                suggestion="Define what action the agent should perform"
            ))
        
        if step.timeout_seconds <= 0:
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                message="Step timeout must be positive",
                location=f"{location}.timeout_seconds",
                suggestion="Set a reasonable timeout value (e.g., 60 seconds)"
            ))
        
        if step.timeout_seconds > self.performance_thresholds["step_timeout"]:
            issues.append(ValidationIssue(
                severity="warning",
                category="performance",
                message=f"Step timeout is very long ({step.timeout_seconds}s)",
                location=f"{location}.timeout_seconds",
                suggestion="Consider breaking long operations into smaller steps"
            ))
        
        return issues
    
    def _validate_dependencies(self, recipe: RecipeDefinition) -> List[ValidationIssue]:
        """Validate step dependencies"""
        issues = []
        
        # Build step ID map
        step_ids = {step.step_id for step in recipe.steps}
        
        # Check dependency references
        for i, step in enumerate(recipe.steps):
            location = f"step[{i}].dependencies"
            
            # Check for invalid dependency references
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    issues.append(ValidationIssue(
                        severity="error",
                        category="dependency",
                        message=f"Step references non-existent dependency: {dep_id}",
                        location=location,
                        suggestion="Ensure all dependency IDs reference valid steps"
                    ))
            
            # Check for self-dependency
            if step.step_id in step.dependencies:
                issues.append(ValidationIssue(
                    severity="error",
                    category="dependency",
                    message="Step cannot depend on itself",
                    location=location,
                    suggestion="Remove self-reference from dependencies"
                ))
            
            # Check for too many dependencies
            if len(step.dependencies) > self.max_dependencies_per_step:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="dependency",
                    message=f"Step has many dependencies ({len(step.dependencies)})",
                    location=location,
                    suggestion="Consider simplifying step dependencies"
                ))
        
        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies(recipe.steps)
        if circular_deps:
            issues.append(ValidationIssue(
                severity="error",
                category="dependency",
                message=f"Circular dependency detected: {' -> '.join(circular_deps)}",
                location="recipe.steps",
                suggestion="Remove circular dependencies to enable execution"
            ))
        
        return issues
    
    def _validate_performance(self, recipe: RecipeDefinition) -> List[ValidationIssue]:
        """Validate performance constraints"""
        issues = []
        
        # Total execution time
        total_timeout = sum(step.timeout_seconds for step in recipe.steps)
        if total_timeout > self.max_execution_time:
            issues.append(ValidationIssue(
                severity="error",
                category="performance",
                message=f"Total execution time too long ({total_timeout}s > {self.max_execution_time}s)",
                location="recipe.steps",
                suggestion="Reduce step timeouts or break recipe into smaller parts"
            ))
        
        # Memory requirements
        total_memory = sum(req.memory_limit_mb for req in recipe.agent_requirements)
        if total_memory > self.max_memory_mb:
            issues.append(ValidationIssue(
                severity="error",
                category="performance",
                message=f"Memory requirements too high ({total_memory}MB > {self.max_memory_mb}MB)",
                location="recipe.agent_requirements",
                suggestion="Reduce memory limits or optimize agent usage"
            ))
        
        # Performance budget validation
        if recipe.validation_criteria:
            budget = recipe.validation_criteria.performance_budget_ms
            if budget > 0 and total_timeout * 1000 > budget:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="performance",
                    message="Step timeouts exceed performance budget",
                    location="recipe.validation_criteria",
                    suggestion="Align step timeouts with performance budget"
                ))
        
        # Parallel execution validation
        if len(recipe.parallel_steps) > self.max_parallel_groups:
            issues.append(ValidationIssue(
                severity="warning",
                category="performance",
                message=f"Many parallel groups ({len(recipe.parallel_steps)})",
                location="recipe.parallel_steps",
                suggestion="Consider reducing parallel complexity"
            ))
        
        return issues
    
    def _validate_security(self, recipe: RecipeDefinition) -> List[ValidationIssue]:
        """Validate security aspects"""
        issues = []
        
        # Check for security risk patterns in step actions
        for i, step in enumerate(recipe.steps):
            location = f"step[{i}]"
            
            # Check agent action for security risks
            for pattern in self.security_risk_patterns:
                if pattern in step.agent_action.lower():
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="security",
                        message=f"Potential security risk: {pattern} in agent action",
                        location=f"{location}.agent_action",
                        suggestion="Review security implications of this action"
                    ))
            
            # Check input data for security risks
            input_str = str(step.input_data)
            for pattern in self.security_risk_patterns:
                if pattern in input_str.lower():
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="security",
                        message=f"Potential security risk: {pattern} in input data",
                        location=f"{location}.input_data",
                        suggestion="Sanitize input data to prevent security issues"
                    ))
        
        return issues
    
    def _validate_consistency(self, recipe: RecipeDefinition) -> List[ValidationIssue]:
        """Validate internal consistency"""
        issues = []
        
        # Check that MCP tools referenced in steps are defined in requirements
        required_tools = set()
        for step in recipe.steps:
            required_tools.update(step.mcp_tools)
        
        available_tools = set()
        for req in recipe.mcp_requirements:
            available_tools.update(req.required_capabilities)
            available_tools.add(req.tool_name)
        
        missing_tools = required_tools - available_tools
        if missing_tools:
            issues.append(ValidationIssue(
                severity="error",
                category="consistency",
                message=f"Steps reference undefined MCP tools: {', '.join(missing_tools)}",
                location="recipe.steps",
                suggestion="Add MCP requirements for all referenced tools"
            ))
        
        # Check execution order consistency
        if recipe.execution_order:
            step_ids = {step.step_id for step in recipe.steps}
            for step_id in recipe.execution_order:
                if step_id not in step_ids:
                    issues.append(ValidationIssue(
                        severity="error",
                        category="consistency",
                        message=f"Execution order references non-existent step: {step_id}",
                        location="recipe.execution_order",
                        suggestion="Ensure execution order only references valid steps"
                    ))
        
        return issues
    
    def _validate_best_practices(self, recipe: RecipeDefinition) -> List[ValidationIssue]:
        """Validate against best practices"""
        issues = []
        
        # Check for descriptive names
        if len(recipe.name) < 10:
            issues.append(ValidationIssue(
                severity="info",
                category="best_practices",
                message="Recipe name is quite short",
                location="recipe.name",
                suggestion="Consider a more descriptive name"
            ))
        
        # Check for tags
        if not recipe.tags:
            issues.append(ValidationIssue(
                severity="info",
                category="best_practices",
                message="Recipe has no tags",
                location="recipe.tags",
                suggestion="Add tags to improve discoverability"
            ))
        
        # Check for documentation
        if not recipe.documentation_url:
            issues.append(ValidationIssue(
                severity="info",
                category="best_practices",
                message="Recipe has no documentation URL",
                location="recipe.documentation_url",
                suggestion="Add documentation for better maintainability"
            ))
        
        # Check for version
        if recipe.version == "1.0.0":
            issues.append(ValidationIssue(
                severity="info",
                category="best_practices",
                message="Recipe is still at initial version",
                location="recipe.version",
                suggestion="Update version as recipe evolves"
            ))
        
        # Check step descriptions
        for i, step in enumerate(recipe.steps):
            if not step.description or len(step.description) < 10:
                issues.append(ValidationIssue(
                    severity="info",
                    category="best_practices",
                    message="Step has minimal description",
                    location=f"step[{i}].description",
                    suggestion="Add detailed step description"
                ))
        
        return issues
    
    def _detect_circular_dependencies(self, steps: List[RecipeStep]) -> Optional[List[str]]:
        """Detect circular dependencies using DFS"""
        # Build adjacency list
        graph = {}
        for step in steps:
            graph[step.step_id] = step.dependencies
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found cycle, return the cycle path
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            
            if node in visited:
                return None
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                cycle = dfs(neighbor, path + [node])
                if cycle:
                    return cycle
            
            rec_stack.remove(node)
            return None
        
        # Check each node
        for step in steps:
            if step.step_id not in visited:
                cycle = dfs(step.step_id, [])
                if cycle:
                    return cycle
        
        return None
    
    def get_validation_summary(self, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Get summary of validation results"""
        summary = {
            "total_issues": len(issues),
            "errors": len([i for i in issues if i.severity == "error"]),
            "warnings": len([i for i in issues if i.severity == "warning"]),
            "info": len([i for i in issues if i.severity == "info"]),
            "categories": {},
            "is_valid": len([i for i in issues if i.severity == "error"]) == 0
        }
        
        # Count by category
        for issue in issues:
            category = issue.category
            if category not in summary["categories"]:
                summary["categories"][category] = {"errors": 0, "warnings": 0, "info": 0}
            summary["categories"][category][issue.severity] += 1
        
        return summary
