"""
Smart Test Selection for CI/CD

This module provides intelligent test selection based on code changes,
impact analysis, and risk assessment to optimize build validation time.
"""

import asyncio
import logging
import subprocess
import ast
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib

from ..recipes.schema import RecipeDefinition
from ..recipes.registry import RecipeRegistry
from ..core.framework import RecipeTestResult


logger = logging.getLogger(__name__)


@dataclass
class CodeChange:
    """Represents a code change in the repository"""
    file_path: str
    change_type: str  # added, modified, deleted, renamed
    lines_added: int = 0
    lines_removed: int = 0
    functions_changed: List[str] = field(default_factory=list)
    classes_changed: List[str] = field(default_factory=list)
    imports_changed: List[str] = field(default_factory=list)


@dataclass
class ImpactAnalysis:
    """Analysis of change impact on recipes"""
    affected_recipes: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    confidence: float = 0.0
    impact_reasons: List[str] = field(default_factory=list)
    recommended_test_level: int = 1  # 1-4 (basic to comprehensive)


@dataclass
class TestSelection:
    """Selected tests for execution"""
    recipes_to_test: List[str] = field(default_factory=list)
    test_level: int = 1
    estimated_duration_minutes: float = 0.0
    priority_order: List[str] = field(default_factory=list)
    skip_reasons: Dict[str, str] = field(default_factory=dict)


class SmartTestSelector:
    """
    Intelligent test selection system for CI/CD pipelines.
    
    Analyzes code changes to determine which recipe tests need to be run,
    optimizing build time while maintaining comprehensive coverage.
    """
    
    def __init__(self, 
                 repo_path: str = ".",
                 cache_dir: str = "./data/test_selection_cache"):
        self.repo_path = Path(repo_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Component dependencies mapping
        self.component_dependencies = {
            "src/core/": ["agent", "factory", "capability"],
            "src/agents/": ["agent", "specialized", "factory"],
            "src/mcp/": ["mcp", "server", "tools"],
            "src/rag/": ["rag", "retrieval", "indexing"],
            "src/storage/": ["storage", "vector", "database"],
            "src/communication/": ["communication", "protocol", "messaging"],
            "src/utils/": ["utils", "logging", "performance"],
            "src/testing/": ["testing", "recipes", "framework"],
            # New AI components
            "src/ai/nlp/": ["nlp", "text_processing", "recipe_parsing"],
            "src/ai/multi_agent/": ["multi_agent", "coordination", "workflow"],
            "src/ai/predictive/": ["predictive", "optimization", "recommendation"],
            "src/integration/": ["integration", "orchestration", "workflow"]
        }
        
        # Recipe impact patterns
        self.recipe_impact_patterns = {
            "agent_changes": {
                "patterns": ["src/agents/", "src/core/agent"],
                "affects": "agent_requirements"
            },
            "mcp_changes": {
                "patterns": ["src/mcp/", "src/testing/mcp/"],
                "affects": "mcp_requirements"
            },
            "storage_changes": {
                "patterns": ["src/storage/", "src/database/"],
                "affects": "storage_dependent"
            },
            "communication_changes": {
                "patterns": ["src/communication/", "src/api/"],
                "affects": "communication_dependent"
            },
            "utils_changes": {
                "patterns": ["src/utils/"],
                "affects": "all"  # Utils affect everything
            },
            # New AI component patterns
            "nlp_changes": {
                "patterns": ["src/ai/nlp/", "tests/ai/nlp/"],
                "affects": "nlp_dependent"
            },
            "multiagent_changes": {
                "patterns": ["src/ai/multi_agent/", "tests/ai/multi_agent/"],
                "affects": "multiagent_dependent"
            },
            "predictive_changes": {
                "patterns": ["src/ai/predictive/", "tests/ai/predictive/"],
                "affects": "predictive_dependent"
            },
            "integration_changes": {
                "patterns": ["src/integration/", "tests/integration/"],
                "affects": "integration_dependent"
            }
        }
        
        # Test level definitions
        self.test_levels = {
            1: {
                "name": "Basic Validation",
                "duration_minutes": 0.5,
                "description": "Syntax and basic connectivity tests"
            },
            2: {
                "name": "Core Functionality", 
                "duration_minutes": 2.0,
                "description": "Core functionality smoke tests"
            },
            3: {
                "name": "Integration Testing",
                "duration_minutes": 5.0,
                "description": "Integration and performance validation"
            },
            4: {
                "name": "Comprehensive Testing",
                "duration_minutes": 15.0,
                "description": "Full test suite execution"
            }
        }
        
        # Historical data
        self.test_history: List[RecipeTestResult] = []
        self.change_impact_cache: Dict[str, ImpactAnalysis] = {}
        
        # Registry reference
        self.recipe_registry: Optional[RecipeRegistry] = None
    
    async def initialize(self, recipe_registry: RecipeRegistry) -> None:
        """Initialize the smart test selector"""
        self.recipe_registry = recipe_registry
        
        # Load cached data
        await self._load_cache()
        
        logger.info("Smart Test Selector initialized")
    
    async def select_tests_for_changes(self, 
                                     base_commit: str = "HEAD~1",
                                     target_commit: str = "HEAD",
                                     max_test_duration_minutes: float = 5.0) -> TestSelection:
        """
        Select tests based on code changes between commits.
        
        Args:
            base_commit: Base commit for comparison
            target_commit: Target commit for comparison  
            max_test_duration_minutes: Maximum allowed test duration
            
        Returns:
            TestSelection with recommended tests
        """
        logger.info(f"Selecting tests for changes: {base_commit}..{target_commit}")
        
        try:
            # Analyze code changes
            changes = await self._analyze_code_changes(base_commit, target_commit)
            
            if not changes:
                logger.info("No code changes detected")
                return TestSelection(
                    test_level=1,
                    estimated_duration_minutes=0.0,
                    skip_reasons={"no_changes": "No code changes detected"}
                )
            
            # Analyze impact on recipes
            impact_analysis = await self._analyze_change_impact(changes)
            
            # Select appropriate test level
            test_level = self._determine_test_level(impact_analysis, max_test_duration_minutes)
            
            # Select specific recipes to test
            selected_recipes = await self._select_recipes_for_testing(
                impact_analysis, 
                test_level,
                max_test_duration_minutes
            )
            
            # Prioritize test execution order
            priority_order = await self._prioritize_test_execution(selected_recipes, impact_analysis)
            
            # Calculate estimated duration
            estimated_duration = self._estimate_test_duration(selected_recipes, test_level)
            
            selection = TestSelection(
                recipes_to_test=selected_recipes,
                test_level=test_level,
                estimated_duration_minutes=estimated_duration,
                priority_order=priority_order
            )
            
            logger.info(f"Selected {len(selected_recipes)} recipes for testing (Level {test_level}, ~{estimated_duration:.1f}min)")
            
            # Cache the analysis
            change_hash = self._hash_changes(changes)
            self.change_impact_cache[change_hash] = impact_analysis
            await self._save_cache()
            
            return selection
        
        except Exception as e:
            logger.error(f"Test selection failed: {e}")
            # Fallback to basic testing
            return TestSelection(
                test_level=1,
                estimated_duration_minutes=0.5,
                skip_reasons={"error": f"Selection failed: {str(e)}"}
            )
    
    async def select_tests_for_files(self, 
                                   changed_files: List[str],
                                   max_test_duration_minutes: float = 5.0) -> TestSelection:
        """
        Select tests based on specific file changes.
        
        Args:
            changed_files: List of changed file paths
            max_test_duration_minutes: Maximum allowed test duration
            
        Returns:
            TestSelection with recommended tests
        """
        logger.info(f"Selecting tests for {len(changed_files)} changed files")
        
        # Create synthetic changes from file list
        changes = []
        for file_path in changed_files:
            changes.append(CodeChange(
                file_path=file_path,
                change_type="modified",
                lines_added=10,  # Estimate
                lines_removed=5   # Estimate
            ))
        
        # Analyze impact
        impact_analysis = await self._analyze_change_impact(changes)
        
        # Select tests
        test_level = self._determine_test_level(impact_analysis, max_test_duration_minutes)
        selected_recipes = await self._select_recipes_for_testing(
            impact_analysis, 
            test_level,
            max_test_duration_minutes
        )
        
        priority_order = await self._prioritize_test_execution(selected_recipes, impact_analysis)
        estimated_duration = self._estimate_test_duration(selected_recipes, test_level)
        
        return TestSelection(
            recipes_to_test=selected_recipes,
            test_level=test_level,
            estimated_duration_minutes=estimated_duration,
            priority_order=priority_order
        )
    
    async def get_minimal_test_set(self, max_duration_minutes: float = 2.0) -> TestSelection:
        """
        Get minimal test set for fast validation.
        
        Args:
            max_duration_minutes: Maximum test duration
            
        Returns:
            Minimal test selection
        """
        if not self.recipe_registry:
            return TestSelection(test_level=1, estimated_duration_minutes=0.0)
        
        # Get high-priority recipes (frequently failing or critical)
        all_recipes = await self.recipe_registry.list_recipes()
        
        # Select based on historical reliability and importance
        critical_recipes = []
        for recipe in all_recipes:
            # Prioritize recipes that:
            # 1. Have failed recently
            # 2. Are marked as critical
            # 3. Cover core functionality
            
            if (recipe.success_rate < 0.9 or 
                "critical" in recipe.tags or
                recipe.category.value in ["development", "integration"]):
                critical_recipes.append(recipe.id)
        
        # Limit to fit duration budget
        selected_recipes = critical_recipes[:5]  # Limit to 5 recipes for speed
        
        return TestSelection(
            recipes_to_test=selected_recipes,
            test_level=1,
            estimated_duration_minutes=min(len(selected_recipes) * 0.5, max_duration_minutes),
            priority_order=selected_recipes
        )
    
    async def _analyze_code_changes(self, base_commit: str, target_commit: str) -> List[CodeChange]:
        """Analyze code changes between commits"""
        changes = []
        
        try:
            # Get changed files using git diff
            cmd = ["git", "diff", "--name-status", f"{base_commit}..{target_commit}"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                logger.error(f"Git diff failed: {result.stderr}")
                return changes
            
            # Parse git diff output
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    status = parts[0]
                    file_path = parts[1]
                    
                    # Map git status to change type
                    if status == 'A':
                        change_type = 'added'
                    elif status == 'M':
                        change_type = 'modified'
                    elif status == 'D':
                        change_type = 'deleted'
                    elif status.startswith('R'):
                        change_type = 'renamed'
                    else:
                        change_type = 'modified'
                    
                    # Get detailed change info for Python files
                    if file_path.endswith('.py') and change_type in ['added', 'modified']:
                        detailed_change = await self._analyze_python_file_changes(
                            file_path, base_commit, target_commit
                        )
                        changes.append(detailed_change)
                    else:
                        changes.append(CodeChange(
                            file_path=file_path,
                            change_type=change_type
                        ))
        
        except Exception as e:
            logger.error(f"Failed to analyze code changes: {e}")
        
        return changes
    
    async def _analyze_python_file_changes(self, 
                                         file_path: str, 
                                         base_commit: str, 
                                         target_commit: str) -> CodeChange:
        """Analyze changes in a Python file"""
        change = CodeChange(file_path=file_path, change_type="modified")
        
        try:
            # Get file content at both commits
            base_content = await self._get_file_at_commit(file_path, base_commit)
            target_content = await self._get_file_at_commit(file_path, target_commit)
            
            if base_content is None or target_content is None:
                return change
            
            # Parse AST to find changed functions/classes
            try:
                base_ast = ast.parse(base_content)
                target_ast = ast.parse(target_content)
                
                # Extract function and class names
                base_functions = self._extract_function_names(base_ast)
                base_classes = self._extract_class_names(base_ast)
                base_imports = self._extract_imports(base_ast)
                
                target_functions = self._extract_function_names(target_ast)
                target_classes = self._extract_class_names(target_ast)
                target_imports = self._extract_imports(target_ast)
                
                # Find changes
                change.functions_changed = list(
                    (set(target_functions) - set(base_functions)) |
                    (set(base_functions) - set(target_functions))
                )
                
                change.classes_changed = list(
                    (set(target_classes) - set(base_classes)) |
                    (set(base_classes) - set(target_classes))
                )
                
                change.imports_changed = list(
                    (set(target_imports) - set(base_imports)) |
                    (set(base_imports) - set(target_imports))
                )
            
            except SyntaxError:
                # If AST parsing fails, just note that the file changed
                logger.warning(f"Could not parse AST for {file_path}")
            
            # Count line changes (simple diff)
            base_lines = base_content.split('\n')
            target_lines = target_content.split('\n')
            
            change.lines_added = max(0, len(target_lines) - len(base_lines))
            change.lines_removed = max(0, len(base_lines) - len(target_lines))
        
        except Exception as e:
            logger.warning(f"Failed to analyze Python file {file_path}: {e}")
        
        return change
    
    async def _get_file_at_commit(self, file_path: str, commit: str) -> Optional[str]:
        """Get file content at specific commit"""
        try:
            cmd = ["git", "show", f"{commit}:{file_path}"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                return result.stdout
            else:
                return None
        
        except Exception:
            return None
    
    def _extract_function_names(self, tree: ast.AST) -> List[str]:
        """Extract function names from AST"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions
    
    def _extract_class_names(self, tree: ast.AST) -> List[str]:
        """Extract class names from AST"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    
    async def _analyze_change_impact(self, changes: List[CodeChange]) -> ImpactAnalysis:
        """Analyze impact of changes on recipes"""
        impact = ImpactAnalysis()
        
        if not self.recipe_registry:
            return impact
        
        # Get all recipes
        all_recipes = await self.recipe_registry.list_recipes()
        
        # Analyze each change
        total_risk = 0.0
        impact_count = 0
        
        for change in changes:
            file_path = change.file_path
            
            # Check component impact patterns
            for pattern_name, pattern_info in self.recipe_impact_patterns.items():
                if any(pattern in file_path for pattern in pattern_info["patterns"]):
                    # This change affects recipes with specific requirements
                    affects = pattern_info["affects"]
                    
                    for recipe in all_recipes:
                        if self._recipe_affected_by_change(recipe, affects, change):
                            if recipe.id not in impact.affected_recipes:
                                impact.affected_recipes.append(recipe.id)
                                impact.impact_reasons.append(f"Recipe uses {affects} affected by {pattern_name}")
                    
                    # Calculate risk based on change magnitude
                    change_risk = self._calculate_change_risk(change)
                    total_risk += change_risk
                    impact_count += 1
        
        # Calculate overall risk and confidence
        if impact_count > 0:
            impact.risk_score = min(total_risk / impact_count, 1.0)
            impact.confidence = min(impact_count / 10.0, 1.0)  # More changes = higher confidence
        
        # Determine recommended test level
        if impact.risk_score > 0.8 or len(impact.affected_recipes) > 10:
            impact.recommended_test_level = 4  # Comprehensive
        elif impact.risk_score > 0.6 or len(impact.affected_recipes) > 5:
            impact.recommended_test_level = 3  # Integration
        elif impact.risk_score > 0.3 or len(impact.affected_recipes) > 0:
            impact.recommended_test_level = 2  # Core functionality
        else:
            impact.recommended_test_level = 1  # Basic
        
        return impact
    
    def _recipe_affected_by_change(self, 
                                 recipe: RecipeDefinition, 
                                 affects: str, 
                                 change: CodeChange) -> bool:
        """Check if a recipe is affected by a specific change"""
        if affects == "all":
            return True
        
        if affects == "agent_requirements":
            return len(recipe.agent_requirements) > 0
        
        if affects == "mcp_requirements":
            return len(recipe.mcp_requirements) > 0
        
        if affects == "storage_dependent":
            # Check if recipe uses storage-related tools
            storage_tools = ["database", "vector", "storage", "postgres", "mongo"]
            return any(tool in recipe.get_required_tools() for tool in storage_tools)
        
        if affects == "communication_dependent":
            # Check if recipe uses communication features
            return any(step.agent_action in ["communicate", "send", "receive"] for step in recipe.steps)
        
        return False
    
    def _calculate_change_risk(self, change: CodeChange) -> float:
        """Calculate risk score for a change"""
        risk = 0.0
        
        # File type risk
        if change.file_path.endswith('.py'):
            risk += 0.3
        elif change.file_path.endswith(('.json', '.yaml', '.yml')):
            risk += 0.2
        
        # Change type risk
        if change.change_type == 'added':
            risk += 0.2
        elif change.change_type == 'modified':
            risk += 0.4
        elif change.change_type == 'deleted':
            risk += 0.6
        elif change.change_type == 'renamed':
            risk += 0.3
        
        # Change magnitude risk
        total_lines = change.lines_added + change.lines_removed
        if total_lines > 100:
            risk += 0.4
        elif total_lines > 50:
            risk += 0.3
        elif total_lines > 20:
            risk += 0.2
        elif total_lines > 5:
            risk += 0.1
        
        # Function/class changes
        if change.functions_changed:
            risk += min(len(change.functions_changed) * 0.1, 0.3)
        
        if change.classes_changed:
            risk += min(len(change.classes_changed) * 0.15, 0.4)
        
        if change.imports_changed:
            risk += min(len(change.imports_changed) * 0.05, 0.2)
        
        return min(risk, 1.0)
    
    def _determine_test_level(self, 
                            impact: ImpactAnalysis, 
                            max_duration_minutes: float) -> int:
        """Determine appropriate test level"""
        # Start with recommended level from impact analysis
        recommended_level = impact.recommended_test_level
        
        # Adjust based on time constraints
        for level in range(recommended_level, 0, -1):
            if self.test_levels[level]["duration_minutes"] <= max_duration_minutes:
                return level
        
        return 1  # Fallback to basic level
    
    async def _select_recipes_for_testing(self, 
                                        impact: ImpactAnalysis,
                                        test_level: int,
                                        max_duration_minutes: float) -> List[str]:
        """Select specific recipes for testing"""
        if test_level == 1:
            # Basic level - just a few critical recipes
            return impact.affected_recipes[:3]
        
        elif test_level == 2:
            # Core functionality - affected recipes plus some critical ones
            selected = impact.affected_recipes[:8]
            
            # Add critical recipes if not already included
            if self.recipe_registry:
                all_recipes = await self.recipe_registry.list_recipes()
                critical_recipes = [r.id for r in all_recipes if "critical" in r.tags]
                
                for recipe_id in critical_recipes:
                    if recipe_id not in selected and len(selected) < 10:
                        selected.append(recipe_id)
            
            return selected
        
        elif test_level == 3:
            # Integration testing - most affected recipes
            return impact.affected_recipes[:15]
        
        else:
            # Comprehensive testing - all affected recipes
            return impact.affected_recipes
    
    async def _prioritize_test_execution(self, 
                                       selected_recipes: List[str],
                                       impact: ImpactAnalysis) -> List[str]:
        """Prioritize test execution order"""
        if not self.recipe_registry:
            return selected_recipes
        
        # Get recipe details
        recipe_priorities = []
        
        for recipe_id in selected_recipes:
            recipe = await self.recipe_registry.get_recipe(recipe_id)
            if recipe:
                # Calculate priority score
                priority = 0.0
                
                # Historical reliability (lower success rate = higher priority)
                priority += (1.0 - recipe.success_rate) * 0.4
                
                # Criticality
                if "critical" in recipe.tags:
                    priority += 0.3
                
                # Recent failures
                if recipe.success_rate < 0.8:
                    priority += 0.2
                
                # Complexity (more complex = higher priority to catch early)
                complexity = len(recipe.steps) + len(recipe.mcp_requirements)
                priority += min(complexity / 20.0, 0.1)
                
                recipe_priorities.append((recipe_id, priority))
        
        # Sort by priority (highest first)
        recipe_priorities.sort(key=lambda x: x[1], reverse=True)
        
        return [recipe_id for recipe_id, _ in recipe_priorities]
    
    def _estimate_test_duration(self, selected_recipes: List[str], test_level: int) -> float:
        """Estimate total test duration"""
        base_duration = self.test_levels[test_level]["duration_minutes"]
        
        # Estimate per-recipe overhead
        per_recipe_overhead = 0.1  # 6 seconds per recipe
        
        return base_duration + (len(selected_recipes) * per_recipe_overhead)
    
    def _hash_changes(self, changes: List[CodeChange]) -> str:
        """Create hash of changes for caching"""
        change_data = []
        for change in changes:
            change_data.append(f"{change.file_path}:{change.change_type}:{change.lines_added}:{change.lines_removed}")
        
        change_string = "|".join(sorted(change_data))
        return hashlib.md5(change_string.encode()).hexdigest()
    
    async def _save_cache(self) -> None:
        """Save cache data to disk"""
        try:
            cache_data = {
                "change_impact_cache": {
                    hash_key: {
                        "affected_recipes": analysis.affected_recipes,
                        "risk_score": analysis.risk_score,
                        "confidence": analysis.confidence,
                        "impact_reasons": analysis.impact_reasons,
                        "recommended_test_level": analysis.recommended_test_level
                    }
                    for hash_key, analysis in self.change_impact_cache.items()
                }
            }
            
            cache_file = self.cache_dir / "test_selection_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    async def _load_cache(self) -> None:
        """Load cache data from disk"""
        try:
            cache_file = self.cache_dir / "test_selection_cache.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Reconstruct impact analysis objects
                for hash_key, analysis_data in cache_data.get("change_impact_cache", {}).items():
                    self.change_impact_cache[hash_key] = ImpactAnalysis(
                        affected_recipes=analysis_data["affected_recipes"],
                        risk_score=analysis_data["risk_score"],
                        confidence=analysis_data["confidence"],
                        impact_reasons=analysis_data["impact_reasons"],
                        recommended_test_level=analysis_data["recommended_test_level"]
                    )
                
                logger.info(f"Loaded {len(self.change_impact_cache)} cached impact analyses")
        
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get test selection statistics"""
        return {
            "cached_analyses": len(self.change_impact_cache),
            "test_levels": self.test_levels,
            "component_dependencies": list(self.component_dependencies.keys()),
            "impact_patterns": list(self.recipe_impact_patterns.keys())
        }
