"""
Recipe Generator

This module provides automated generation of Agent + MCP recipe combinations
based on available agents, MCP servers, and intelligent combination strategies.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import itertools
import random

from .schema import (
    RecipeDefinition, RecipeStep, AgentRequirement, MCPToolRequirement,
    RecipeCategory, RecipeDifficulty, RecipeValidationCriteria
)
from ..mcp.discovery import MCPServerInfo
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


logger = logging.getLogger(__name__)


@dataclass
class RecipeTemplate:
    """Template for generating recipes"""
    name_pattern: str
    description_pattern: str
    category: RecipeCategory
    difficulty: RecipeDifficulty
    agent_types: List[str]
    mcp_server_types: List[MCPServerType]
    step_templates: List[Dict[str, Any]]
    validation_criteria: Dict[str, Any]
    tags: List[str]


class RecipeGenerator:
    """
    Automated recipe generator for Agent + MCP combinations.
    
    Generates comprehensive recipe combinations based on available agents,
    MCP servers, and intelligent pairing strategies.
    """
    
    def __init__(self):
        # Recipe templates for different scenarios
        self.recipe_templates = self._initialize_recipe_templates()
        
        # Agent capability mappings
        self.agent_capabilities = {
            "research": [CapabilityType.WEB_SEARCH, CapabilityType.DOCUMENT_ANALYSIS, CapabilityType.MEMORY_RETRIEVAL],
            "code": [CapabilityType.CODE_GENERATION, CapabilityType.CODE_ANALYSIS, CapabilityType.TOOL_EXECUTION],
            "conversation": [CapabilityType.NATURAL_LANGUAGE, CapabilityType.MEMORY_MANAGEMENT],
            "data": [CapabilityType.DATA_PROCESSING, CapabilityType.ANALYSIS, CapabilityType.VISUALIZATION],
            "academic": [CapabilityType.WEB_SEARCH, CapabilityType.DOCUMENT_ANALYSIS, CapabilityType.DATA_PROCESSING, CapabilityType.ANALYSIS]
        }
        
        # MCP server compatibility matrix
        self.server_compatibility = {
            "research": ["brave-search", "web-scraper", "postgres", "elasticsearch"],
            "code": ["github", "filesystem", "docker", "terminal", "vscode", "jupyter"],
            "conversation": ["memory", "postgres", "redis"],
            "data": ["postgres", "mongodb", "elasticsearch", "visualization"],
            "academic": ["arxiv", "pubmed", "scholar", "zotero", "latex", "mendeley", "postgres", "elasticsearch"]
        }
        
        # Common workflow patterns
        self.workflow_patterns = {
            "simple": ["initialize", "execute", "validate"],
            "search_analyze": ["search", "analyze", "summarize", "store"],
            "code_workflow": ["analyze_requirements", "generate_code", "test", "deploy"],
            "data_pipeline": ["extract", "transform", "analyze", "visualize", "store"],
            "research_workflow": ["search", "gather", "analyze", "synthesize", "report"],
            "coding_workflow": ["requirements_analysis", "design", "implementation", "testing", "documentation", "deployment"],
            "academic_workflow": ["literature_search", "data_collection", "analysis", "interpretation", "writing", "peer_review"]
        }
    
    def _initialize_recipe_templates(self) -> List[RecipeTemplate]:
        """Initialize recipe templates for different scenarios"""
        templates = []
        
        # NLP Processing Templates
        templates.append(RecipeTemplate(
            name_pattern="NLP Analysis with {agent_type} and {mcp_server}",
            description_pattern="Perform NLP analysis using {agent_type} agent with {mcp_server} tools",
            category=RecipeCategory.NLP_PROCESSING,
            difficulty=RecipeDifficulty.INTERMEDIATE,
            agent_types=["research", "data"],
            mcp_server_types=[MCPServerType.NLP, MCPServerType.DATABASE],
            step_templates=[
                {"action": "initialize_nlp", "tools": ["tokenize", "analyze"]},
                {"action": "process_text", "tools": ["extract_entities", "sentiment_analysis"]},
                {"action": "store_results", "tools": ["database_insert"]}
            ],
            validation_criteria={"success_threshold": 0.8, "performance_budget_ms": 10000},
            tags=["nlp", "text-processing", "analysis"]
        ))
        
        # Graphics and Media Templates
        templates.append(RecipeTemplate(
            name_pattern="Image Processing with {agent_type} and {mcp_server}",
            description_pattern="Process and analyze images using {agent_type} agent with {mcp_server} tools",
            category=RecipeCategory.GRAPHICS_MEDIA,
            difficulty=RecipeDifficulty.ADVANCED,
            agent_types=["data", "code"],
            mcp_server_types=[MCPServerType.GRAPHICS, MCPServerType.FILESYSTEM],
            step_templates=[
                {"action": "load_image", "tools": ["read_file"]},
                {"action": "process_image", "tools": ["resize", "filter", "analyze"]},
                {"action": "save_results", "tools": ["write_file", "generate_report"]}
            ],
            validation_criteria={"success_threshold": 0.85, "performance_budget_ms": 15000},
            tags=["graphics", "image-processing", "computer-vision"]
        ))
        
        # Database Operations Templates
        templates.append(RecipeTemplate(
            name_pattern="Database Operations with {agent_type} and {mcp_server}",
            description_pattern="Perform database operations using {agent_type} agent with {mcp_server}",
            category=RecipeCategory.DATABASE_OPS,
            difficulty=RecipeDifficulty.INTERMEDIATE,
            agent_types=["data", "research"],
            mcp_server_types=[MCPServerType.DATABASE],
            step_templates=[
                {"action": "connect_database", "tools": ["connect"]},
                {"action": "query_data", "tools": ["query", "select"]},
                {"action": "analyze_results", "tools": ["aggregate", "analyze"]},
                {"action": "store_insights", "tools": ["insert", "update"]}
            ],
            validation_criteria={"success_threshold": 0.9, "performance_budget_ms": 8000},
            tags=["database", "sql", "data-analysis"]
        ))
        
        # Web UI Creation Templates
        templates.append(RecipeTemplate(
            name_pattern="Web UI Creation with {agent_type} and {mcp_server}",
            description_pattern="Create web interfaces using {agent_type} agent with {mcp_server} tools",
            category=RecipeCategory.WEB_UI_CREATION,
            difficulty=RecipeDifficulty.EXPERT,
            agent_types=["code"],
            mcp_server_types=[MCPServerType.WEB_UI, MCPServerType.FILESYSTEM],
            step_templates=[
                {"action": "design_ui", "tools": ["create_component", "style"]},
                {"action": "implement_logic", "tools": ["add_handlers", "connect_data"]},
                {"action": "test_ui", "tools": ["validate", "test_interactions"]},
                {"action": "deploy", "tools": ["build", "deploy"]}
            ],
            validation_criteria={"success_threshold": 0.75, "performance_budget_ms": 20000},
            tags=["web", "ui", "frontend", "development"]
        ))
        
        # Development Workflow Templates
        templates.append(RecipeTemplate(
            name_pattern="Development Workflow with {agent_type} and {mcp_server}",
            description_pattern="Execute development tasks using {agent_type} agent with {mcp_server} tools",
            category=RecipeCategory.DEVELOPMENT,
            difficulty=RecipeDifficulty.ADVANCED,
            agent_types=["code"],
            mcp_server_types=[MCPServerType.VERSION_CONTROL, MCPServerType.FILESYSTEM],
            step_templates=[
                {"action": "analyze_requirements", "tools": ["read_file", "analyze_code"]},
                {"action": "implement_changes", "tools": ["write_file", "modify_code"]},
                {"action": "test_changes", "tools": ["run_tests", "validate"]},
                {"action": "commit_changes", "tools": ["git_add", "git_commit"]}
            ],
            validation_criteria={"success_threshold": 0.85, "performance_budget_ms": 12000},
            tags=["development", "git", "coding", "workflow"]
        ))

        # Coding Templates
        templates.append(RecipeTemplate(
            name_pattern="Code Generation with {agent_type} and {mcp_server}",
            description_pattern="Generate, analyze, and optimize code using {agent_type} agent with {mcp_server} tools",
            category=RecipeCategory.CODING,
            difficulty=RecipeDifficulty.ADVANCED,
            agent_types=["code"],
            mcp_server_types=[MCPServerType.FILESYSTEM, MCPServerType.VERSION_CONTROL],
            step_templates=[
                {"action": "analyze_requirements", "tools": ["read_specification", "parse_requirements"]},
                {"action": "generate_code", "tools": ["code_generation", "template_engine"]},
                {"action": "optimize_code", "tools": ["code_analysis", "performance_optimization"]},
                {"action": "validate_code", "tools": ["syntax_check", "unit_test", "integration_test"]},
                {"action": "document_code", "tools": ["generate_docs", "code_comments"]}
            ],
            validation_criteria={"success_threshold": 0.90, "performance_budget_ms": 15000},
            tags=["coding", "generation", "optimization", "validation"]
        ))

        templates.append(RecipeTemplate(
            name_pattern="Code Review and Analysis with {agent_type} and {mcp_server}",
            description_pattern="Perform comprehensive code review and analysis using {agent_type} agent with {mcp_server} tools",
            category=RecipeCategory.CODING,
            difficulty=RecipeDifficulty.INTERMEDIATE,
            agent_types=["code", "research"],
            mcp_server_types=[MCPServerType.FILESYSTEM, MCPServerType.VERSION_CONTROL],
            step_templates=[
                {"action": "load_codebase", "tools": ["read_files", "parse_structure"]},
                {"action": "analyze_quality", "tools": ["code_metrics", "complexity_analysis", "style_check"]},
                {"action": "detect_issues", "tools": ["bug_detection", "security_scan", "performance_analysis"]},
                {"action": "suggest_improvements", "tools": ["refactoring_suggestions", "best_practices"]},
                {"action": "generate_report", "tools": ["report_generation", "visualization"]}
            ],
            validation_criteria={"success_threshold": 0.85, "performance_budget_ms": 10000},
            tags=["coding", "review", "analysis", "quality"]
        ))

        # Academic Research Templates
        templates.append(RecipeTemplate(
            name_pattern="Literature Review with {agent_type} and {mcp_server}",
            description_pattern="Conduct systematic literature review using {agent_type} agent with {mcp_server} tools",
            category=RecipeCategory.ACADEMIC_RESEARCH,
            difficulty=RecipeDifficulty.EXPERT,
            agent_types=["research", "data"],
            mcp_server_types=[MCPServerType.WEB_SEARCH, MCPServerType.DATABASE, MCPServerType.NLP],
            step_templates=[
                {"action": "define_search_strategy", "tools": ["keyword_extraction", "search_planning"]},
                {"action": "search_literature", "tools": ["academic_search", "database_query", "web_scraping"]},
                {"action": "filter_papers", "tools": ["relevance_scoring", "quality_assessment", "deduplication"]},
                {"action": "extract_information", "tools": ["text_extraction", "data_mining", "citation_analysis"]},
                {"action": "synthesize_findings", "tools": ["content_analysis", "theme_extraction", "summary_generation"]},
                {"action": "generate_review", "tools": ["academic_writing", "citation_formatting", "bibliography"]}
            ],
            validation_criteria={"success_threshold": 0.80, "performance_budget_ms": 30000},
            tags=["academic", "research", "literature", "review", "synthesis"]
        ))

        templates.append(RecipeTemplate(
            name_pattern="Research Data Analysis with {agent_type} and {mcp_server}",
            description_pattern="Analyze research data and generate insights using {agent_type} agent with {mcp_server} tools",
            category=RecipeCategory.ACADEMIC_RESEARCH,
            difficulty=RecipeDifficulty.ADVANCED,
            agent_types=["data", "research"],
            mcp_server_types=[MCPServerType.DATABASE, MCPServerType.GRAPHICS, MCPServerType.NLP],
            step_templates=[
                {"action": "load_dataset", "tools": ["data_import", "format_conversion", "validation"]},
                {"action": "explore_data", "tools": ["descriptive_stats", "data_profiling", "visualization"]},
                {"action": "clean_data", "tools": ["missing_values", "outlier_detection", "normalization"]},
                {"action": "analyze_patterns", "tools": ["statistical_analysis", "correlation", "clustering"]},
                {"action": "test_hypotheses", "tools": ["hypothesis_testing", "significance_tests", "effect_size"]},
                {"action": "generate_insights", "tools": ["interpretation", "visualization", "reporting"]}
            ],
            validation_criteria={"success_threshold": 0.85, "performance_budget_ms": 20000},
            tags=["academic", "research", "data", "analysis", "statistics"]
        ))

        templates.append(RecipeTemplate(
            name_pattern="Academic Paper Writing with {agent_type} and {mcp_server}",
            description_pattern="Assist in academic paper writing and formatting using {agent_type} agent with {mcp_server} tools",
            category=RecipeCategory.ACADEMIC_RESEARCH,
            difficulty=RecipeDifficulty.EXPERT,
            agent_types=["research", "conversation"],
            mcp_server_types=[MCPServerType.NLP, MCPServerType.FILESYSTEM, MCPServerType.WEB_SEARCH],
            step_templates=[
                {"action": "structure_paper", "tools": ["outline_generation", "section_planning", "template_selection"]},
                {"action": "write_sections", "tools": ["content_generation", "academic_writing", "style_adaptation"]},
                {"action": "manage_citations", "tools": ["reference_management", "citation_formatting", "bibliography"]},
                {"action": "review_content", "tools": ["grammar_check", "style_analysis", "coherence_check"]},
                {"action": "format_paper", "tools": ["latex_formatting", "figure_placement", "table_formatting"]},
                {"action": "final_review", "tools": ["plagiarism_check", "submission_prep", "quality_assessment"]}
            ],
            validation_criteria={"success_threshold": 0.75, "performance_budget_ms": 25000},
            tags=["academic", "research", "writing", "paper", "publication"]
        ))
        
        return templates
    
    async def generate_comprehensive_recipes(self, 
                                           agent_types: List[str],
                                           mcp_servers: List[str],
                                           max_recipes_per_combination: int = 3) -> List[RecipeDefinition]:
        """
        Generate comprehensive recipe combinations.
        
        Args:
            agent_types: Available agent types
            mcp_servers: Available MCP server names
            max_recipes_per_combination: Maximum recipes per agent+MCP combination
            
        Returns:
            List of generated recipes
        """
        logger.info(f"Generating recipes for {len(agent_types)} agents and {len(mcp_servers)} MCP servers")
        
        recipes = []
        
        # Generate recipes from templates
        for template in self.recipe_templates:
            template_recipes = await self._generate_from_template(
                template, agent_types, mcp_servers, max_recipes_per_combination
            )
            recipes.extend(template_recipes)
        
        # Generate combinatorial recipes
        combinatorial_recipes = await self._generate_combinatorial_recipes(
            agent_types, mcp_servers, max_recipes_per_combination
        )
        recipes.extend(combinatorial_recipes)
        
        # Generate workflow-based recipes
        workflow_recipes = await self._generate_workflow_recipes(
            agent_types, mcp_servers, max_recipes_per_combination
        )
        recipes.extend(workflow_recipes)
        
        logger.info(f"Generated {len(recipes)} total recipes")
        return recipes
    
    async def generate_recipes_for_servers(self, server_names: List[str]) -> List[RecipeDefinition]:
        """Generate recipes specifically for given MCP servers"""
        recipes = []
        
        # Group servers by category
        server_categories = {}
        for server_name in server_names:
            category = self._infer_server_category(server_name)
            if category not in server_categories:
                server_categories[category] = []
            server_categories[category].append(server_name)
        
        # Generate recipes for each category
        for category, servers in server_categories.items():
            category_recipes = await self._generate_category_recipes(category, servers)
            recipes.extend(category_recipes)
        
        return recipes
    
    async def _generate_from_template(self, 
                                    template: RecipeTemplate,
                                    agent_types: List[str],
                                    mcp_servers: List[str],
                                    max_recipes: int) -> List[RecipeDefinition]:
        """Generate recipes from a specific template"""
        recipes = []
        
        # Filter compatible agents and servers
        compatible_agents = [a for a in agent_types if a in template.agent_types]
        compatible_servers = [s for s in mcp_servers if self._server_matches_template(s, template)]
        
        if not compatible_agents or not compatible_servers:
            return recipes
        
        # Generate combinations
        combinations = list(itertools.product(compatible_agents, compatible_servers))
        random.shuffle(combinations)  # Randomize for variety
        
        for i, (agent_type, server_name) in enumerate(combinations[:max_recipes]):
            recipe = await self._create_recipe_from_template(template, agent_type, server_name)
            if recipe:
                recipes.append(recipe)
        
        return recipes
    
    async def _generate_combinatorial_recipes(self, 
                                            agent_types: List[str],
                                            mcp_servers: List[str],
                                            max_recipes: int) -> List[RecipeDefinition]:
        """Generate recipes using combinatorial approach"""
        recipes = []
        
        # Simple 1:1 combinations
        for agent_type in agent_types:
            compatible_servers = self._get_compatible_servers(agent_type, mcp_servers)
            
            for server_name in compatible_servers[:3]:  # Limit to 3 per agent
                recipe = await self._create_simple_recipe(agent_type, [server_name])
                if recipe:
                    recipes.append(recipe)
        
        # Multi-server combinations for advanced recipes
        for agent_type in agent_types:
            compatible_servers = self._get_compatible_servers(agent_type, mcp_servers)
            
            if len(compatible_servers) >= 2:
                # Create recipes with 2-3 servers
                for server_combo in itertools.combinations(compatible_servers, 2):
                    if len(recipes) >= max_recipes:
                        break
                    
                    recipe = await self._create_multi_server_recipe(agent_type, list(server_combo))
                    if recipe:
                        recipes.append(recipe)
        
        return recipes[:max_recipes]
    
    async def _generate_workflow_recipes(self, 
                                       agent_types: List[str],
                                       mcp_servers: List[str],
                                       max_recipes: int) -> List[RecipeDefinition]:
        """Generate recipes based on common workflow patterns"""
        recipes = []
        
        for workflow_name, steps in self.workflow_patterns.items():
            for agent_type in agent_types:
                compatible_servers = self._get_compatible_servers(agent_type, mcp_servers)
                
                if len(compatible_servers) >= len(steps):
                    recipe = await self._create_workflow_recipe(
                        workflow_name, agent_type, compatible_servers, steps
                    )
                    if recipe:
                        recipes.append(recipe)
                        
                        if len(recipes) >= max_recipes:
                            break
            
            if len(recipes) >= max_recipes:
                break
        
        return recipes
    
    async def _create_recipe_from_template(self, 
                                         template: RecipeTemplate,
                                         agent_type: str,
                                         server_name: str) -> Optional[RecipeDefinition]:
        """Create a recipe from a template"""
        try:
            # Generate recipe name and description
            name = template.name_pattern.format(agent_type=agent_type, mcp_server=server_name)
            description = template.description_pattern.format(agent_type=agent_type, mcp_server=server_name)
            
            # Create recipe
            recipe = RecipeDefinition(
                name=name,
                description=description,
                category=template.category,
                difficulty=template.difficulty,
                tags=template.tags + [agent_type, server_name],
                author="RecipeGenerator"
            )
            
            # Add agent requirement
            agent_req = AgentRequirement(
                agent_type=agent_type,
                required_capabilities=self.agent_capabilities.get(agent_type, []),
                memory_limit_mb=512,
                max_execution_time=300
            )
            recipe.add_agent_requirement(agent_req)
            
            # Add MCP requirements and steps from template
            for i, step_template in enumerate(template.step_templates):
                # Create MCP requirement
                mcp_req = MCPToolRequirement(
                    server_name=server_name,
                    tool_name=step_template["tools"][0] if step_template["tools"] else "default_tool",
                    server_type=self._infer_server_type(server_name),
                    required_capabilities=step_template["tools"]
                )
                recipe.add_mcp_requirement(mcp_req)
                
                # Create recipe step
                step = RecipeStep(
                    name=f"Step {i+1}: {step_template['action']}",
                    description=f"Execute {step_template['action']} using {server_name}",
                    agent_action=step_template["action"],
                    mcp_tools=step_template["tools"],
                    timeout_seconds=60,
                    critical=True
                )
                recipe.add_step(step)
            
            # Set validation criteria
            criteria_data = template.validation_criteria
            recipe.validation_criteria = RecipeValidationCriteria(
                success_threshold=criteria_data.get("success_threshold", 0.8),
                performance_budget_ms=criteria_data.get("performance_budget_ms", 10000),
                memory_budget_mb=criteria_data.get("memory_budget_mb", 1024)
            )
            
            return recipe
        
        except Exception as e:
            logger.error(f"Failed to create recipe from template: {e}")
            return None
    
    async def _create_simple_recipe(self, agent_type: str, server_names: List[str]) -> Optional[RecipeDefinition]:
        """Create a simple recipe with basic workflow"""
        try:
            server_list = ", ".join(server_names)
            recipe = RecipeDefinition(
                name=f"Simple {agent_type.title()} Recipe with {server_list}",
                description=f"Basic workflow using {agent_type} agent with {server_list}",
                category=RecipeCategory.INTEGRATION,
                difficulty=RecipeDifficulty.BASIC,
                tags=["simple", "basic", agent_type] + server_names,
                author="RecipeGenerator"
            )
            
            # Add agent requirement
            agent_req = AgentRequirement(
                agent_type=agent_type,
                required_capabilities=self.agent_capabilities.get(agent_type, []),
                memory_limit_mb=256,
                max_execution_time=180
            )
            recipe.add_agent_requirement(agent_req)
            
            # Add steps for each server
            for i, server_name in enumerate(server_names):
                # Add MCP requirement
                mcp_req = MCPToolRequirement(
                    server_name=server_name,
                    tool_name="execute",
                    server_type=self._infer_server_type(server_name),
                    timeout_seconds=30
                )
                recipe.add_mcp_requirement(mcp_req)
                
                # Add step
                step = RecipeStep(
                    name=f"Execute with {server_name}",
                    description=f"Perform operation using {server_name}",
                    agent_action="execute",
                    mcp_tools=["execute"],
                    timeout_seconds=30
                )
                recipe.add_step(step)
            
            return recipe
        
        except Exception as e:
            logger.error(f"Failed to create simple recipe: {e}")
            return None
    
    async def _create_multi_server_recipe(self, agent_type: str, server_names: List[str]) -> Optional[RecipeDefinition]:
        """Create a recipe using multiple MCP servers"""
        try:
            server_list = ", ".join(server_names)
            recipe = RecipeDefinition(
                name=f"Multi-Server {agent_type.title()} Recipe",
                description=f"Advanced workflow using {agent_type} agent with multiple servers: {server_list}",
                category=RecipeCategory.INTEGRATION,
                difficulty=RecipeDifficulty.ADVANCED,
                tags=["multi-server", "advanced", agent_type] + server_names,
                author="RecipeGenerator"
            )
            
            # Add agent requirement
            agent_req = AgentRequirement(
                agent_type=agent_type,
                required_capabilities=self.agent_capabilities.get(agent_type, []),
                memory_limit_mb=1024,
                max_execution_time=600
            )
            recipe.add_agent_requirement(agent_req)
            
            # Create coordinated workflow
            workflow_steps = [
                ("initialize", "Initialize workflow"),
                ("process", "Process data"),
                ("coordinate", "Coordinate between servers"),
                ("finalize", "Finalize results")
            ]
            
            for i, (action, description) in enumerate(workflow_steps):
                server_name = server_names[i % len(server_names)]
                
                # Add MCP requirement
                mcp_req = MCPToolRequirement(
                    server_name=server_name,
                    tool_name=action,
                    server_type=self._infer_server_type(server_name),
                    timeout_seconds=60
                )
                recipe.add_mcp_requirement(mcp_req)
                
                # Add step
                step = RecipeStep(
                    name=f"{description} ({server_name})",
                    description=f"{description} using {server_name}",
                    agent_action=action,
                    mcp_tools=[action],
                    timeout_seconds=60,
                    dependencies=[recipe.steps[i-1].step_id] if i > 0 else []
                )
                recipe.add_step(step)
            
            return recipe
        
        except Exception as e:
            logger.error(f"Failed to create multi-server recipe: {e}")
            return None
    
    async def _create_workflow_recipe(self, 
                                    workflow_name: str,
                                    agent_type: str,
                                    server_names: List[str],
                                    steps: List[str]) -> Optional[RecipeDefinition]:
        """Create a recipe based on a workflow pattern"""
        try:
            recipe = RecipeDefinition(
                name=f"{workflow_name.title()} Workflow with {agent_type.title()}",
                description=f"Execute {workflow_name} workflow using {agent_type} agent",
                category=self._workflow_to_category(workflow_name),
                difficulty=RecipeDifficulty.INTERMEDIATE,
                tags=["workflow", workflow_name, agent_type],
                author="RecipeGenerator"
            )
            
            # Add agent requirement
            agent_req = AgentRequirement(
                agent_type=agent_type,
                required_capabilities=self.agent_capabilities.get(agent_type, []),
                memory_limit_mb=512,
                max_execution_time=400
            )
            recipe.add_agent_requirement(agent_req)
            
            # Create steps based on workflow pattern
            for i, step_action in enumerate(steps):
                server_name = server_names[i % len(server_names)]
                
                # Add MCP requirement
                mcp_req = MCPToolRequirement(
                    server_name=server_name,
                    tool_name=step_action,
                    server_type=self._infer_server_type(server_name),
                    timeout_seconds=45
                )
                recipe.add_mcp_requirement(mcp_req)
                
                # Add step
                step = RecipeStep(
                    name=f"{step_action.title()} Step",
                    description=f"Execute {step_action} in {workflow_name} workflow",
                    agent_action=step_action,
                    mcp_tools=[step_action],
                    timeout_seconds=45,
                    dependencies=[recipe.steps[i-1].step_id] if i > 0 else []
                )
                recipe.add_step(step)
            
            return recipe
        
        except Exception as e:
            logger.error(f"Failed to create workflow recipe: {e}")
            return None
    
    async def _generate_category_recipes(self, category: str, server_names: List[str]) -> List[RecipeDefinition]:
        """Generate recipes for a specific category of servers"""
        recipes = []
        
        # Get appropriate agent types for category
        agent_types = self._get_category_agents(category)
        
        for agent_type in agent_types:
            for server_name in server_names[:2]:  # Limit to 2 servers per agent
                recipe = await self._create_category_specific_recipe(category, agent_type, server_name)
                if recipe:
                    recipes.append(recipe)
        
        return recipes
    
    async def _create_category_specific_recipe(self, 
                                             category: str,
                                             agent_type: str,
                                             server_name: str) -> Optional[RecipeDefinition]:
        """Create a recipe specific to a category"""
        category_configs = {
            "nlp": {
                "actions": ["tokenize", "analyze", "extract", "classify"],
                "category": RecipeCategory.NLP_PROCESSING
            },
            "graphics": {
                "actions": ["load", "process", "transform", "save"],
                "category": RecipeCategory.GRAPHICS_MEDIA
            },
            "database": {
                "actions": ["connect", "query", "analyze", "store"],
                "category": RecipeCategory.DATABASE_OPS
            },
            "web_ui": {
                "actions": ["design", "implement", "test", "deploy"],
                "category": RecipeCategory.WEB_UI_CREATION
            },
            "development": {
                "actions": ["analyze", "code", "test", "commit"],
                "category": RecipeCategory.DEVELOPMENT
            }
        }
        
        config = category_configs.get(category, {
            "actions": ["initialize", "execute", "validate"],
            "category": RecipeCategory.INTEGRATION
        })
        
        try:
            recipe = RecipeDefinition(
                name=f"{category.title()} Recipe: {agent_type.title()} + {server_name}",
                description=f"Specialized {category} recipe using {agent_type} agent with {server_name}",
                category=config["category"],
                difficulty=RecipeDifficulty.INTERMEDIATE,
                tags=[category, agent_type, server_name, "specialized"],
                author="RecipeGenerator"
            )
            
            # Add agent requirement
            agent_req = AgentRequirement(
                agent_type=agent_type,
                required_capabilities=self.agent_capabilities.get(agent_type, []),
                memory_limit_mb=512,
                max_execution_time=300
            )
            recipe.add_agent_requirement(agent_req)
            
            # Add steps based on category actions
            for action in config["actions"]:
                # Add MCP requirement
                mcp_req = MCPToolRequirement(
                    server_name=server_name,
                    tool_name=action,
                    server_type=self._infer_server_type(server_name),
                    timeout_seconds=60
                )
                recipe.add_mcp_requirement(mcp_req)
                
                # Add step
                step = RecipeStep(
                    name=f"{action.title()} with {server_name}",
                    description=f"Execute {action} using {server_name} in {category} context",
                    agent_action=action,
                    mcp_tools=[action],
                    timeout_seconds=60
                )
                recipe.add_step(step)
            
            return recipe
        
        except Exception as e:
            logger.error(f"Failed to create category-specific recipe: {e}")
            return None
    
    def _server_matches_template(self, server_name: str, template: RecipeTemplate) -> bool:
        """Check if server matches template requirements"""
        server_type = self._infer_server_type(server_name)
        return server_type in template.mcp_server_types
    
    def _get_compatible_servers(self, agent_type: str, mcp_servers: List[str]) -> List[str]:
        """Get MCP servers compatible with agent type"""
        compatible = self.server_compatibility.get(agent_type, [])
        return [s for s in mcp_servers if any(comp in s for comp in compatible)]
    
    def _infer_server_category(self, server_name: str) -> str:
        """Infer category from server name"""
        name_lower = server_name.lower()
        
        if any(keyword in name_lower for keyword in ["nlp", "text", "language", "spacy"]):
            return "nlp"
        elif any(keyword in name_lower for keyword in ["image", "graphics", "opencv", "vision"]):
            return "graphics"
        elif any(keyword in name_lower for keyword in ["database", "sql", "postgres", "mongo"]):
            return "database"
        elif any(keyword in name_lower for keyword in ["ui", "web", "react", "streamlit"]):
            return "web_ui"
        elif any(keyword in name_lower for keyword in ["git", "docker", "terminal", "dev"]):
            return "development"
        else:
            return "general"
    
    def _infer_server_type(self, server_name: str) -> MCPServerType:
        """Infer server type from name"""
        name_lower = server_name.lower()
        
        if "filesystem" in name_lower or "file" in name_lower:
            return MCPServerType.FILESYSTEM
        elif any(keyword in name_lower for keyword in ["database", "postgres", "mongo", "sql"]):
            return MCPServerType.DATABASE
        elif any(keyword in name_lower for keyword in ["search", "web", "brave"]):
            return MCPServerType.WEB_SEARCH
        elif any(keyword in name_lower for keyword in ["git", "github", "version"]):
            return MCPServerType.VERSION_CONTROL
        else:
            return MCPServerType.CUSTOM
    
    def _get_category_agents(self, category: str) -> List[str]:
        """Get appropriate agent types for category"""
        category_agents = {
            "nlp": ["research", "data"],
            "graphics": ["data", "code"],
            "database": ["data", "research"],
            "web_ui": ["code"],
            "development": ["code"],
            "coding": ["code"],
            "academic_research": ["academic", "research", "data"],
            "general": ["research", "code", "conversation", "data"]
        }

        return category_agents.get(category, ["research", "code"])
    
    def _workflow_to_category(self, workflow_name: str) -> RecipeCategory:
        """Map workflow name to recipe category"""
        workflow_categories = {
            "simple": RecipeCategory.INTEGRATION,
            "search_analyze": RecipeCategory.RESEARCH,
            "code_workflow": RecipeCategory.DEVELOPMENT,
            "data_pipeline": RecipeCategory.DATA_ANALYSIS,
            "research_workflow": RecipeCategory.RESEARCH,
            "coding_workflow": RecipeCategory.CODING,
            "academic_workflow": RecipeCategory.ACADEMIC_RESEARCH
        }

        return workflow_categories.get(workflow_name, RecipeCategory.INTEGRATION)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get recipe generation statistics"""
        return {
            "templates": len(self.recipe_templates),
            "agent_types": list(self.agent_capabilities.keys()),
            "workflow_patterns": list(self.workflow_patterns.keys()),
            "server_compatibility": self.server_compatibility
        }
