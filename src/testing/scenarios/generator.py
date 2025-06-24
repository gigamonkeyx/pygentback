"""
Test Scenario Generator

This module provides intelligent generation of test scenarios for Agent + MCP
recipes based on recipe requirements, real-world usage patterns, and edge cases.
"""

import asyncio
import logging
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import itertools

from ..recipes.schema import RecipeDefinition, RecipeCategory, RecipeDifficulty
from ..mcp.discovery import MCPServerInfo


logger = logging.getLogger(__name__)


@dataclass
class ScenarioTemplate:
    """Template for generating test scenarios"""
    name: str
    description: str
    category: str
    difficulty: str
    input_patterns: List[Dict[str, Any]]
    expected_outputs: List[str]
    edge_cases: List[Dict[str, Any]]
    performance_expectations: Dict[str, Any]


@dataclass
class TestScenario:
    """A complete test scenario"""
    scenario_id: str
    name: str
    description: str
    category: str
    input_data: Dict[str, Any]
    expected_outputs: List[str]
    success_criteria: Dict[str, Any]
    step_inputs: Dict[str, Dict[str, Any]]  # Step-specific inputs
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestScenarioGenerator:
    """
    Intelligent test scenario generator for Agent + MCP recipes.
    
    Generates comprehensive test scenarios including normal cases, edge cases,
    performance tests, and error conditions based on recipe characteristics.
    """
    
    def __init__(self):
        # Scenario templates by category
        self.scenario_templates = self._initialize_scenario_templates()
        
        # Data generators for different types
        self.data_generators = {
            "text": self._generate_text_data,
            "numbers": self._generate_numeric_data,
            "files": self._generate_file_data,
            "urls": self._generate_url_data,
            "json": self._generate_json_data,
            "images": self._generate_image_data,
            "code": self._generate_code_data
        }
        
        # Edge case patterns
        self.edge_case_patterns = {
            "empty_input": {},
            "null_values": {"data": None, "value": None},
            "large_input": {"size": "large"},
            "special_characters": {"text": "Special chars: !@#$%^&*()"},
            "unicode": {"text": "Unicode: 你好世界 🌍"},
            "malformed_json": {"json": '{"invalid": json}'},
            "long_text": {"text": "A" * 10000},
            "negative_numbers": {"number": -999999},
            "zero_values": {"number": 0, "count": 0},
            "boundary_values": {"min": 0, "max": 2147483647}
        }
        
        # Performance test patterns
        self.performance_patterns = {
            "stress_test": {"volume": "high", "concurrent": True},
            "load_test": {"requests": 100, "duration": 60},
            "spike_test": {"sudden_load": True, "multiplier": 10},
            "endurance_test": {"duration": 300, "steady_load": True}
        }
    
    def _initialize_scenario_templates(self) -> Dict[str, List[ScenarioTemplate]]:
        """Initialize scenario templates for different categories"""
        templates = {}
        
        # NLP Processing Templates
        templates["nlp_processing"] = [
            ScenarioTemplate(
                name="Text Analysis",
                description="Analyze text content for entities, sentiment, and structure",
                category="nlp_processing",
                difficulty="basic",
                input_patterns=[
                    {"text": "Sample text for analysis"},
                    {"text": "Long article with multiple paragraphs and complex sentences."},
                    {"text": "Short phrase"},
                    {"text": "Text with numbers 123 and dates 2024-01-01"}
                ],
                expected_outputs=["entities", "sentiment", "tokens"],
                edge_cases=[
                    {"text": ""},
                    {"text": "🌍🚀💡"},
                    {"text": "A" * 50000}
                ],
                performance_expectations={"max_time_ms": 5000, "max_memory_mb": 512}
            ),
            ScenarioTemplate(
                name="Language Detection",
                description="Detect language of input text",
                category="nlp_processing", 
                difficulty="basic",
                input_patterns=[
                    {"text": "Hello world"},
                    {"text": "Bonjour le monde"},
                    {"text": "Hola mundo"},
                    {"text": "你好世界"}
                ],
                expected_outputs=["language", "confidence"],
                edge_cases=[
                    {"text": "123456"},
                    {"text": "!@#$%^&*()"},
                    {"text": ""}
                ],
                performance_expectations={"max_time_ms": 2000, "max_memory_mb": 256}
            )
        ]
        
        # Graphics and Media Templates
        templates["graphics_media"] = [
            ScenarioTemplate(
                name="Image Processing",
                description="Process and transform images",
                category="graphics_media",
                difficulty="intermediate",
                input_patterns=[
                    {"image_path": "test_image.jpg", "operation": "resize"},
                    {"image_path": "large_image.png", "operation": "compress"},
                    {"image_path": "photo.jpeg", "operation": "filter"}
                ],
                expected_outputs=["processed_image", "metadata"],
                edge_cases=[
                    {"image_path": "corrupted.jpg"},
                    {"image_path": "tiny_1x1.png"},
                    {"image_path": "huge_10000x10000.jpg"}
                ],
                performance_expectations={"max_time_ms": 15000, "max_memory_mb": 2048}
            ),
            ScenarioTemplate(
                name="Data Visualization",
                description="Create charts and graphs from data",
                category="graphics_media",
                difficulty="intermediate",
                input_patterns=[
                    {"data": [1, 2, 3, 4, 5], "chart_type": "line"},
                    {"data": {"A": 10, "B": 20, "C": 15}, "chart_type": "bar"},
                    {"data": [[1, 2], [3, 4], [5, 6]], "chart_type": "scatter"}
                ],
                expected_outputs=["chart_image", "chart_data"],
                edge_cases=[
                    {"data": [], "chart_type": "line"},
                    {"data": [1], "chart_type": "bar"},
                    {"data": list(range(10000)), "chart_type": "line"}
                ],
                performance_expectations={"max_time_ms": 10000, "max_memory_mb": 1024}
            )
        ]
        
        # Database Operations Templates
        templates["database_ops"] = [
            ScenarioTemplate(
                name="Data Query",
                description="Query and retrieve data from database",
                category="database_ops",
                difficulty="basic",
                input_patterns=[
                    {"query": "SELECT * FROM users WHERE active = true"},
                    {"query": "SELECT COUNT(*) FROM orders WHERE date > '2024-01-01'"},
                    {"query": "SELECT name, email FROM customers ORDER BY name"}
                ],
                expected_outputs=["results", "row_count"],
                edge_cases=[
                    {"query": "SELECT * FROM nonexistent_table"},
                    {"query": "INVALID SQL SYNTAX"},
                    {"query": "SELECT * FROM huge_table"}  # Performance test
                ],
                performance_expectations={"max_time_ms": 8000, "max_memory_mb": 512}
            ),
            ScenarioTemplate(
                name="Data Insertion",
                description="Insert new data into database",
                category="database_ops",
                difficulty="intermediate",
                input_patterns=[
                    {"table": "users", "data": {"name": "John", "email": "john@example.com"}},
                    {"table": "orders", "data": {"user_id": 1, "amount": 99.99, "date": "2024-01-01"}},
                    {"table": "products", "data": {"name": "Widget", "price": 19.99, "category": "tools"}}
                ],
                expected_outputs=["insert_id", "affected_rows"],
                edge_cases=[
                    {"table": "users", "data": {}},  # Empty data
                    {"table": "users", "data": {"email": "invalid-email"}},  # Invalid data
                    {"table": "nonexistent", "data": {"field": "value"}}  # Invalid table
                ],
                performance_expectations={"max_time_ms": 5000, "max_memory_mb": 256}
            )
        ]
        
        # Web UI Creation Templates
        templates["web_ui_creation"] = [
            ScenarioTemplate(
                name="Component Creation",
                description="Create UI components",
                category="web_ui_creation",
                difficulty="advanced",
                input_patterns=[
                    {"component": "button", "props": {"text": "Click me", "color": "blue"}},
                    {"component": "form", "props": {"fields": ["name", "email", "message"]}},
                    {"component": "table", "props": {"data": [{"id": 1, "name": "Item 1"}]}}
                ],
                expected_outputs=["component_code", "component_html"],
                edge_cases=[
                    {"component": "unknown_component", "props": {}},
                    {"component": "button", "props": {"text": "A" * 1000}},  # Very long text
                    {"component": "table", "props": {"data": []}}  # Empty data
                ],
                performance_expectations={"max_time_ms": 12000, "max_memory_mb": 1024}
            )
        ]
        
        # Development Workflow Templates
        templates["development"] = [
            ScenarioTemplate(
                name="Code Analysis",
                description="Analyze code for issues and improvements",
                category="development",
                difficulty="intermediate",
                input_patterns=[
                    {"code": "def hello():\n    print('Hello, world!')", "language": "python"},
                    {"code": "function add(a, b) { return a + b; }", "language": "javascript"},
                    {"code": "public class Test { }", "language": "java"}
                ],
                expected_outputs=["analysis", "suggestions", "metrics"],
                edge_cases=[
                    {"code": "", "language": "python"},  # Empty code
                    {"code": "invalid syntax here", "language": "python"},  # Syntax errors
                    {"code": "# " + "A" * 100000, "language": "python"}  # Very large file
                ],
                performance_expectations={"max_time_ms": 10000, "max_memory_mb": 512}
            )
        ]

        # Coding Templates
        templates["coding"] = [
            ScenarioTemplate(
                name="Code Generation",
                description="Generate code from specifications",
                category="coding",
                difficulty="advanced",
                input_patterns=[
                    {"specification": "Create a function that calculates fibonacci numbers", "language": "python"},
                    {"specification": "Build a REST API endpoint for user management", "language": "javascript"},
                    {"specification": "Implement a binary search algorithm", "language": "java"},
                    {"specification": "Create a data structure for a priority queue", "language": "cpp"}
                ],
                expected_outputs=["generated_code", "documentation", "tests"],
                edge_cases=[
                    {"specification": "", "language": "python"},  # Empty specification
                    {"specification": "Create something impossible", "language": "python"},  # Impossible request
                    {"specification": "A" * 10000, "language": "python"}  # Very long specification
                ],
                performance_expectations={"max_time_ms": 15000, "max_memory_mb": 1024}
            ),
            ScenarioTemplate(
                name="Code Optimization",
                description="Optimize existing code for performance and readability",
                category="coding",
                difficulty="expert",
                input_patterns=[
                    {"code": "def slow_fibonacci(n):\n    if n <= 1: return n\n    return slow_fibonacci(n-1) + slow_fibonacci(n-2)", "language": "python"},
                    {"code": "function inefficientSort(arr) {\n    for(let i = 0; i < arr.length; i++) {\n        for(let j = 0; j < arr.length; j++) {\n            if(arr[i] < arr[j]) {\n                let temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;\n            }\n        }\n    }\n    return arr;\n}", "language": "javascript"},
                    {"code": "public class SlowSearch {\n    public static int linearSearch(int[] arr, int target) {\n        for(int i = 0; i < arr.length; i++) {\n            if(arr[i] == target) return i;\n        }\n        return -1;\n    }\n}", "language": "java"}
                ],
                expected_outputs=["optimized_code", "performance_analysis", "complexity_comparison"],
                edge_cases=[
                    {"code": "def broken_code():\n    return undefined_variable", "language": "python"},  # Broken code
                    {"code": "# Already optimal code\ndef optimal(n): return n * 2", "language": "python"},  # Already optimal
                    {"code": "", "language": "python"}  # Empty code
                ],
                performance_expectations={"max_time_ms": 12000, "max_memory_mb": 768}
            ),
            ScenarioTemplate(
                name="Code Review",
                description="Perform comprehensive code review",
                category="coding",
                difficulty="intermediate",
                input_patterns=[
                    {"code": "def process_data(data):\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result", "language": "python"},
                    {"code": "class UserManager {\n    constructor() {\n        this.users = [];\n    }\n    addUser(user) {\n        this.users.push(user);\n    }\n    getUser(id) {\n        return this.users.find(u => u.id === id);\n    }\n}", "language": "javascript"}
                ],
                expected_outputs=["review_comments", "quality_score", "improvement_suggestions"],
                edge_cases=[
                    {"code": "def security_issue():\n    exec(user_input)", "language": "python"},  # Security issues
                    {"code": "def perfect_code():\n    '''Perfect function'''\n    return 42", "language": "python"},  # Perfect code
                    {"code": "def " + "very_long_function_name" * 100 + "():\n    pass", "language": "python"}  # Very long names
                ],
                performance_expectations={"max_time_ms": 8000, "max_memory_mb": 512}
            )
        ]

        # Academic Research Templates
        templates["academic_research"] = [
            ScenarioTemplate(
                name="Literature Search",
                description="Search and filter academic literature",
                category="academic_research",
                difficulty="intermediate",
                input_patterns=[
                    {"query": "machine learning in healthcare", "field": "computer science", "years": "2020-2024"},
                    {"query": "climate change impact on agriculture", "field": "environmental science", "years": "2015-2024"},
                    {"query": "quantum computing algorithms", "field": "physics", "years": "2018-2024"},
                    {"query": "neural networks for natural language processing", "field": "artificial intelligence", "years": "2019-2024"}
                ],
                expected_outputs=["paper_list", "relevance_scores", "abstracts", "citations"],
                edge_cases=[
                    {"query": "", "field": "computer science", "years": "2020-2024"},  # Empty query
                    {"query": "nonexistent research topic xyz123", "field": "computer science", "years": "2020-2024"},  # No results
                    {"query": "machine learning", "field": "all", "years": "1900-2024"}  # Too broad
                ],
                performance_expectations={"max_time_ms": 20000, "max_memory_mb": 1024}
            ),
            ScenarioTemplate(
                name="Data Analysis",
                description="Analyze research data and generate insights",
                category="academic_research",
                difficulty="advanced",
                input_patterns=[
                    {"dataset": "survey_responses.csv", "analysis_type": "descriptive_statistics", "variables": ["age", "satisfaction", "usage"]},
                    {"dataset": "experiment_results.json", "analysis_type": "hypothesis_testing", "variables": ["treatment", "outcome", "control"]},
                    {"dataset": "longitudinal_study.xlsx", "analysis_type": "time_series", "variables": ["time", "measurement", "group"]},
                    {"dataset": "correlation_data.csv", "analysis_type": "correlation_analysis", "variables": ["var1", "var2", "var3"]}
                ],
                expected_outputs=["statistical_summary", "visualizations", "significance_tests", "interpretation"],
                edge_cases=[
                    {"dataset": "empty_data.csv", "analysis_type": "descriptive_statistics", "variables": []},  # Empty dataset
                    {"dataset": "corrupted_data.csv", "analysis_type": "descriptive_statistics", "variables": ["invalid"]},  # Corrupted data
                    {"dataset": "huge_dataset.csv", "analysis_type": "complex_analysis", "variables": list(range(1000))}  # Very large dataset
                ],
                performance_expectations={"max_time_ms": 25000, "max_memory_mb": 2048}
            ),
            ScenarioTemplate(
                name="Paper Writing",
                description="Assist in academic paper writing and formatting",
                category="academic_research",
                difficulty="expert",
                input_patterns=[
                    {"section": "introduction", "topic": "machine learning applications", "style": "IEEE", "length": "500 words"},
                    {"section": "methodology", "topic": "experimental design", "style": "APA", "length": "800 words"},
                    {"section": "results", "topic": "statistical analysis", "style": "Nature", "length": "600 words"},
                    {"section": "conclusion", "topic": "research implications", "style": "ACM", "length": "400 words"}
                ],
                expected_outputs=["formatted_text", "citations", "references", "structure_check"],
                edge_cases=[
                    {"section": "unknown_section", "topic": "machine learning", "style": "IEEE", "length": "500 words"},  # Unknown section
                    {"section": "introduction", "topic": "", "style": "IEEE", "length": "500 words"},  # Empty topic
                    {"section": "introduction", "topic": "machine learning", "style": "unknown_style", "length": "10000 words"}  # Unknown style, very long
                ],
                performance_expectations={"max_time_ms": 18000, "max_memory_mb": 1024}
            ),
            ScenarioTemplate(
                name="Citation Management",
                description="Manage and format academic citations",
                category="academic_research",
                difficulty="intermediate",
                input_patterns=[
                    {"papers": ["Smith et al. 2023", "Johnson 2022", "Brown & Davis 2024"], "style": "APA", "type": "bibliography"},
                    {"papers": ["Nature Paper 2023", "Science Article 2022"], "style": "IEEE", "type": "in_text"},
                    {"papers": ["Conference Paper 2024", "Journal Article 2023"], "style": "MLA", "type": "works_cited"},
                    {"papers": ["Book Chapter 2022", "Thesis 2023"], "style": "Chicago", "type": "footnotes"}
                ],
                expected_outputs=["formatted_citations", "bibliography", "reference_list"],
                edge_cases=[
                    {"papers": [], "style": "APA", "type": "bibliography"},  # No papers
                    {"papers": ["Invalid Citation Format"], "style": "APA", "type": "bibliography"},  # Invalid format
                    {"papers": ["Paper " + str(i) for i in range(1000)], "style": "APA", "type": "bibliography"}  # Too many papers
                ],
                performance_expectations={"max_time_ms": 10000, "max_memory_mb": 512}
            )
        ]
        
        return templates
    
    async def generate_scenarios_for_recipe(self, 
                                          recipe: RecipeDefinition,
                                          categories: Optional[List[str]] = None,
                                          max_scenarios: int = 10) -> List[Dict[str, Any]]:
        """
        Generate test scenarios for a specific recipe.
        
        Args:
            recipe: Recipe to generate scenarios for
            categories: Specific scenario categories to include
            max_scenarios: Maximum number of scenarios to generate
            
        Returns:
            List of test scenarios
        """
        logger.info(f"Generating scenarios for recipe: {recipe.name}")
        
        scenarios = []
        
        # Get relevant templates
        recipe_category = recipe.category.value
        templates = self.scenario_templates.get(recipe_category, [])
        
        # Add general templates if specific ones not found
        if not templates:
            templates = self._get_general_templates(recipe)
        
        # Generate scenarios from templates
        for template in templates:
            if categories and template.category not in categories:
                continue
            
            template_scenarios = await self._generate_from_template(template, recipe)
            scenarios.extend(template_scenarios)
            
            if len(scenarios) >= max_scenarios:
                break
        
        # Generate edge case scenarios
        edge_scenarios = await self._generate_edge_case_scenarios(recipe)
        scenarios.extend(edge_scenarios[:max_scenarios // 4])  # 25% edge cases
        
        # Generate performance scenarios
        perf_scenarios = await self._generate_performance_scenarios(recipe)
        scenarios.extend(perf_scenarios[:max_scenarios // 4])  # 25% performance tests
        
        # Limit total scenarios
        scenarios = scenarios[:max_scenarios]
        
        logger.info(f"Generated {len(scenarios)} scenarios for recipe {recipe.name}")
        return scenarios
    
    async def _generate_from_template(self, 
                                    template: ScenarioTemplate,
                                    recipe: RecipeDefinition) -> List[Dict[str, Any]]:
        """Generate scenarios from a template"""
        scenarios = []
        
        # Generate normal scenarios
        for i, input_pattern in enumerate(template.input_patterns):
            scenario = {
                "scenario_id": f"{template.name.lower().replace(' ', '_')}_{i}",
                "name": f"{template.name} - Case {i+1}",
                "description": template.description,
                "category": template.category,
                "input_data": await self._expand_input_pattern(input_pattern, recipe),
                "expected_outputs": template.expected_outputs,
                "success_criteria": {
                    "required_outputs": template.expected_outputs,
                    "max_execution_time_ms": template.performance_expectations.get("max_time_ms", 10000),
                    "max_memory_mb": template.performance_expectations.get("max_memory_mb", 512)
                },
                "step_inputs": await self._generate_step_inputs(input_pattern, recipe),
                "metadata": {
                    "template": template.name,
                    "difficulty": template.difficulty,
                    "type": "normal"
                }
            }
            scenarios.append(scenario)
        
        # Generate edge case scenarios from template
        for i, edge_case in enumerate(template.edge_cases):
            scenario = {
                "scenario_id": f"{template.name.lower().replace(' ', '_')}_edge_{i}",
                "name": f"{template.name} - Edge Case {i+1}",
                "description": f"{template.description} (Edge Case)",
                "category": template.category,
                "input_data": await self._expand_input_pattern(edge_case, recipe),
                "expected_outputs": template.expected_outputs,
                "success_criteria": {
                    "required_outputs": [],  # Edge cases may not produce all outputs
                    "max_execution_time_ms": template.performance_expectations.get("max_time_ms", 10000) * 2,  # More lenient
                    "max_memory_mb": template.performance_expectations.get("max_memory_mb", 512) * 2
                },
                "step_inputs": await self._generate_step_inputs(edge_case, recipe),
                "metadata": {
                    "template": template.name,
                    "difficulty": template.difficulty,
                    "type": "edge_case"
                }
            }
            scenarios.append(scenario)
        
        return scenarios
    
    async def _expand_input_pattern(self, 
                                  pattern: Dict[str, Any],
                                  recipe: RecipeDefinition) -> Dict[str, Any]:
        """Expand input pattern with generated data"""
        expanded = {}
        
        for key, value in pattern.items():
            if isinstance(value, str) and value.startswith("generate:"):
                # Generate data based on type
                data_type = value.split(":")[1]
                if data_type in self.data_generators:
                    expanded[key] = await self.data_generators[data_type](recipe)
                else:
                    expanded[key] = value
            else:
                expanded[key] = value
        
        return expanded
    
    async def _generate_step_inputs(self, 
                                  input_pattern: Dict[str, Any],
                                  recipe: RecipeDefinition) -> Dict[str, Dict[str, Any]]:
        """Generate step-specific inputs"""
        step_inputs = {}
        
        # Generate inputs for each step based on pattern
        for step in recipe.steps:
            step_input = {}
            
            # Add step-specific data based on step action
            if step.agent_action in ["search", "query"]:
                step_input.update({"query": input_pattern.get("text", "default query")})
            elif step.agent_action in ["process", "analyze"]:
                step_input.update({"data": input_pattern.get("data", input_pattern)})
            elif step.agent_action in ["store", "save"]:
                step_input.update({"output_format": "json"})
            
            if step_input:
                step_inputs[step.step_id] = step_input
        
        return step_inputs
    
    async def _generate_edge_case_scenarios(self, recipe: RecipeDefinition) -> List[Dict[str, Any]]:
        """Generate edge case scenarios"""
        scenarios = []
        
        for edge_name, edge_pattern in self.edge_case_patterns.items():
            scenario = {
                "scenario_id": f"edge_{edge_name}",
                "name": f"Edge Case: {edge_name.replace('_', ' ').title()}",
                "description": f"Test recipe behavior with {edge_name.replace('_', ' ')}",
                "category": "edge_case",
                "input_data": edge_pattern.copy(),
                "expected_outputs": [],  # Edge cases may not produce normal outputs
                "success_criteria": {
                    "required_outputs": [],
                    "max_execution_time_ms": recipe.validation_criteria.performance_budget_ms * 2,
                    "max_memory_mb": recipe.validation_criteria.memory_budget_mb * 2,
                    "allow_failure": True  # Edge cases are allowed to fail gracefully
                },
                "step_inputs": {},
                "metadata": {
                    "type": "edge_case",
                    "pattern": edge_name,
                    "difficulty": "high"
                }
            }
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_performance_scenarios(self, recipe: RecipeDefinition) -> List[Dict[str, Any]]:
        """Generate performance test scenarios"""
        scenarios = []
        
        for perf_name, perf_pattern in self.performance_patterns.items():
            scenario = {
                "scenario_id": f"perf_{perf_name}",
                "name": f"Performance Test: {perf_name.replace('_', ' ').title()}",
                "description": f"Test recipe performance under {perf_name.replace('_', ' ')} conditions",
                "category": "performance",
                "input_data": await self._generate_performance_data(perf_pattern, recipe),
                "expected_outputs": [],  # Focus on performance, not outputs
                "success_criteria": {
                    "required_outputs": [],
                    "max_execution_time_ms": recipe.validation_criteria.performance_budget_ms,
                    "max_memory_mb": recipe.validation_criteria.memory_budget_mb,
                    "performance_focus": True
                },
                "step_inputs": {},
                "metadata": {
                    "type": "performance",
                    "pattern": perf_name,
                    "difficulty": "high"
                }
            }
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_performance_data(self, 
                                       pattern: Dict[str, Any],
                                       recipe: RecipeDefinition) -> Dict[str, Any]:
        """Generate data for performance testing"""
        data = {}
        
        if pattern.get("volume") == "high":
            data["items"] = list(range(1000))  # Large dataset
            data["text"] = "Large text content " * 1000
        
        if pattern.get("concurrent"):
            data["concurrent_requests"] = 10
        
        if pattern.get("requests"):
            data["request_count"] = pattern["requests"]
        
        if pattern.get("duration"):
            data["test_duration_seconds"] = pattern["duration"]
        
        return data
    
    def _get_general_templates(self, recipe: RecipeDefinition) -> List[ScenarioTemplate]:
        """Get general templates when specific ones aren't available"""
        return [
            ScenarioTemplate(
                name="Basic Execution",
                description="Basic recipe execution test",
                category="general",
                difficulty="basic",
                input_patterns=[
                    {"data": "test input"},
                    {"value": 42},
                    {"text": "sample text"}
                ],
                expected_outputs=["result"],
                edge_cases=[
                    {"data": None},
                    {"data": ""}
                ],
                performance_expectations={"max_time_ms": 10000, "max_memory_mb": 512}
            )
        ]
    
    # Data generators
    async def _generate_text_data(self, recipe: RecipeDefinition) -> str:
        """Generate text data for testing"""
        text_samples = [
            "This is a sample text for testing natural language processing capabilities.",
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Testing with numbers: 123, dates: 2024-01-01, and emails: test@example.com"
        ]
        return random.choice(text_samples)
    
    async def _generate_numeric_data(self, recipe: RecipeDefinition) -> List[float]:
        """Generate numeric data for testing"""
        return [random.uniform(0, 100) for _ in range(random.randint(5, 20))]
    
    async def _generate_file_data(self, recipe: RecipeDefinition) -> Dict[str, str]:
        """Generate file data for testing"""
        return {
            "filename": f"test_file_{random.randint(1, 100)}.txt",
            "content": "Sample file content for testing",
            "size": random.randint(100, 10000)
        }
    
    async def _generate_url_data(self, recipe: RecipeDefinition) -> str:
        """Generate URL data for testing"""
        urls = [
            "https://example.com",
            "https://test.example.org/path",
            "https://api.example.com/v1/data",
            "https://docs.example.com/guide"
        ]
        return random.choice(urls)
    
    async def _generate_json_data(self, recipe: RecipeDefinition) -> Dict[str, Any]:
        """Generate JSON data for testing"""
        return {
            "id": random.randint(1, 1000),
            "name": f"Test Item {random.randint(1, 100)}",
            "value": random.uniform(0, 100),
            "active": random.choice([True, False]),
            "tags": [f"tag{i}" for i in range(random.randint(1, 5))]
        }
    
    async def _generate_image_data(self, recipe: RecipeDefinition) -> Dict[str, Any]:
        """Generate image data for testing"""
        return {
            "width": random.choice([100, 200, 500, 1000]),
            "height": random.choice([100, 200, 500, 1000]),
            "format": random.choice(["jpg", "png", "gif"]),
            "path": f"test_image_{random.randint(1, 100)}.jpg"
        }
    
    async def _generate_code_data(self, recipe: RecipeDefinition) -> Dict[str, str]:
        """Generate code data for testing"""
        code_samples = {
            "python": "def hello():\n    print('Hello, world!')",
            "javascript": "function hello() {\n    console.log('Hello, world!');\n}",
            "java": "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, world!\");\n    }\n}"
        }
        
        language = random.choice(list(code_samples.keys()))
        return {
            "language": language,
            "code": code_samples[language],
            "filename": f"test.{language}"
        }
    
    def get_scenario_stats(self) -> Dict[str, Any]:
        """Get scenario generation statistics"""
        return {
            "template_categories": list(self.scenario_templates.keys()),
            "total_templates": sum(len(templates) for templates in self.scenario_templates.values()),
            "data_generators": list(self.data_generators.keys()),
            "edge_case_patterns": list(self.edge_case_patterns.keys()),
            "performance_patterns": list(self.performance_patterns.keys())
        }
