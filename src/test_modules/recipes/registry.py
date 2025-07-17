"""
Recipe Registry

This module provides centralized registry and management for Agent + MCP recipes,
including storage, versioning, search, and metadata management.
"""

import asyncio
import logging
import json
import sqlite3
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import uuid

from .schema import RecipeDefinition, RecipeCategory, RecipeDifficulty, RecipeStatus


logger = logging.getLogger(__name__)


@dataclass
class RecipeSearchFilter:
    """Filter criteria for recipe search"""
    categories: Optional[List[RecipeCategory]] = None
    difficulties: Optional[List[RecipeDifficulty]] = None
    statuses: Optional[List[RecipeStatus]] = None
    tags: Optional[List[str]] = None
    agent_types: Optional[List[str]] = None
    mcp_servers: Optional[List[str]] = None
    min_success_rate: Optional[float] = None
    max_execution_time: Optional[int] = None
    text_search: Optional[str] = None


class RecipeRegistry:
    """
    Centralized registry for Agent + MCP recipes.
    
    Provides storage, versioning, search, and metadata management
    for recipe definitions with SQLite backend.
    """
    
    def __init__(self, data_dir: str = "./data/recipes"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Database setup
        self.db_path = self.data_dir / "recipes.db"
        self.connection: Optional[sqlite3.Connection] = None
        
        # In-memory cache
        self.recipe_cache: Dict[str, RecipeDefinition] = {}
        self.cache_dirty = False
        
        # Registry statistics
        self.total_recipes = 0
        self.recipes_by_category: Dict[str, int] = {}
        self.recipes_by_status: Dict[str, int] = {}
    
    async def initialize(self) -> None:
        """Initialize the recipe registry"""
        # Setup database
        await self._setup_database()
        
        # Load recipes into cache
        await self._load_recipes_cache()
        
        # Update statistics
        await self._update_statistics()
        
        logger.info(f"Recipe Registry initialized with {self.total_recipes} recipes")
    
    async def shutdown(self) -> None:
        """Shutdown the recipe registry"""
        # Save any pending changes
        if self.cache_dirty:
            await self._save_cache_to_db()
        
        # Close database connection
        if self.connection:
            self.connection.close()
        
        logger.info("Recipe Registry shutdown complete")
    
    async def register_recipe(self, recipe: RecipeDefinition) -> bool:
        """
        Register a new recipe or update existing one.
        
        Args:
            recipe: Recipe to register
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Validate recipe
            validation_issues = recipe.validate_structure()
            if validation_issues:
                logger.error(f"Recipe validation failed: {validation_issues}")
                return False
            
            # Check if recipe already exists
            existing_recipe = await self.get_recipe(recipe.id)
            if existing_recipe:
                # Update existing recipe
                recipe.updated_at = datetime.utcnow()
                logger.info(f"Updating existing recipe: {recipe.name}")
            else:
                # New recipe
                recipe.created_at = datetime.utcnow()
                recipe.updated_at = datetime.utcnow()
                logger.info(f"Registering new recipe: {recipe.name}")
            
            # Store in cache
            self.recipe_cache[recipe.id] = recipe
            self.cache_dirty = True
            
            # Store in database
            await self._save_recipe_to_db(recipe)
            
            # Update statistics
            await self._update_statistics()
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to register recipe {recipe.name}: {e}")
            return False
    
    async def get_recipe(self, recipe_id: str) -> Optional[RecipeDefinition]:
        """
        Get recipe by ID.
        
        Args:
            recipe_id: Recipe ID to retrieve
            
        Returns:
            Recipe definition or None if not found
        """
        # Check cache first
        if recipe_id in self.recipe_cache:
            return self.recipe_cache[recipe_id]
        
        # Load from database
        recipe = await self._load_recipe_from_db(recipe_id)
        if recipe:
            self.recipe_cache[recipe_id] = recipe
        
        return recipe
    
    async def list_recipes(self, 
                          limit: Optional[int] = None,
                          offset: int = 0) -> List[RecipeDefinition]:
        """
        List all recipes with optional pagination.
        
        Args:
            limit: Maximum number of recipes to return
            offset: Number of recipes to skip
            
        Returns:
            List of recipe definitions
        """
        recipes = list(self.recipe_cache.values())
        
        # Sort by updated_at descending
        recipes.sort(key=lambda r: r.updated_at, reverse=True)
        
        # Apply pagination
        if limit:
            recipes = recipes[offset:offset + limit]
        else:
            recipes = recipes[offset:]
        
        return recipes
    
    async def find_recipes(self, filters: Optional[Dict[str, Any]] = None) -> List[RecipeDefinition]:
        """
        Find recipes matching filter criteria.
        
        Args:
            filters: Search filter criteria
            
        Returns:
            List of matching recipes
        """
        if not filters:
            return await self.list_recipes()
        
        # Convert dict filters to RecipeSearchFilter
        search_filter = RecipeSearchFilter()
        
        if "categories" in filters:
            search_filter.categories = [RecipeCategory(c) for c in filters["categories"]]
        if "difficulties" in filters:
            search_filter.difficulties = [RecipeDifficulty(d) for d in filters["difficulties"]]
        if "statuses" in filters:
            search_filter.statuses = [RecipeStatus(s) for s in filters["statuses"]]
        if "tags" in filters:
            search_filter.tags = filters["tags"]
        if "agent_types" in filters:
            search_filter.agent_types = filters["agent_types"]
        if "mcp_servers" in filters:
            search_filter.mcp_servers = filters["mcp_servers"]
        if "min_success_rate" in filters:
            search_filter.min_success_rate = filters["min_success_rate"]
        if "max_execution_time" in filters:
            search_filter.max_execution_time = filters["max_execution_time"]
        if "text_search" in filters:
            search_filter.text_search = filters["text_search"]
        
        return await self.search_recipes(search_filter)
    
    async def search_recipes(self, search_filter: RecipeSearchFilter) -> List[RecipeDefinition]:
        """
        Search recipes with advanced filtering.
        
        Args:
            search_filter: Search filter criteria
            
        Returns:
            List of matching recipes
        """
        matching_recipes = []
        
        for recipe in self.recipe_cache.values():
            if self._recipe_matches_filter(recipe, search_filter):
                matching_recipes.append(recipe)
        
        # Sort by relevance (simplified scoring)
        matching_recipes.sort(key=lambda r: self._calculate_relevance_score(r, search_filter), reverse=True)
        
        return matching_recipes
    
    async def delete_recipe(self, recipe_id: str) -> bool:
        """
        Delete a recipe.
        
        Args:
            recipe_id: Recipe ID to delete
            
        Returns:
            bool: True if deletion successful
        """
        try:
            # Remove from cache
            if recipe_id in self.recipe_cache:
                del self.recipe_cache[recipe_id]
                self.cache_dirty = True
            
            # Remove from database
            await self._delete_recipe_from_db(recipe_id)
            
            # Update statistics
            await self._update_statistics()
            
            logger.info(f"Deleted recipe: {recipe_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete recipe {recipe_id}: {e}")
            return False
    
    async def update_recipe_performance(self, 
                                      recipe_id: str,
                                      success_rate: float,
                                      avg_execution_time: int,
                                      usage_count: int) -> bool:
        """
        Update recipe performance metrics.
        
        Args:
            recipe_id: Recipe ID to update
            success_rate: New success rate
            avg_execution_time: Average execution time in ms
            usage_count: Total usage count
            
        Returns:
            bool: True if update successful
        """
        recipe = await self.get_recipe(recipe_id)
        if not recipe:
            return False
        
        try:
            recipe.success_rate = success_rate
            recipe.average_execution_time_ms = avg_execution_time
            recipe.usage_count = usage_count
            recipe.last_tested = datetime.utcnow()
            recipe.updated_at = datetime.utcnow()
            
            # Update cache
            self.recipe_cache[recipe_id] = recipe
            self.cache_dirty = True
            
            # Update database
            await self._save_recipe_to_db(recipe)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to update recipe performance {recipe_id}: {e}")
            return False
    
    async def get_recipes_by_category(self, category: RecipeCategory) -> List[RecipeDefinition]:
        """Get all recipes in a specific category"""
        return [recipe for recipe in self.recipe_cache.values() if recipe.category == category]
    
    async def get_recipes_by_agent_type(self, agent_type: str) -> List[RecipeDefinition]:
        """Get all recipes that use a specific agent type"""
        matching_recipes = []
        for recipe in self.recipe_cache.values():
            if any(req.agent_type == agent_type for req in recipe.agent_requirements):
                matching_recipes.append(recipe)
        return matching_recipes
    
    async def get_recipes_by_mcp_server(self, server_name: str) -> List[RecipeDefinition]:
        """Get all recipes that use a specific MCP server"""
        matching_recipes = []
        for recipe in self.recipe_cache.values():
            if any(req.server_name == server_name for req in recipe.mcp_requirements):
                matching_recipes.append(recipe)
        return matching_recipes
    
    async def get_top_recipes(self, 
                            limit: int = 10,
                            sort_by: str = "success_rate") -> List[RecipeDefinition]:
        """
        Get top recipes by specified criteria.
        
        Args:
            limit: Number of recipes to return
            sort_by: Sort criteria (success_rate, usage_count, avg_execution_time)
            
        Returns:
            List of top recipes
        """
        recipes = list(self.recipe_cache.values())
        
        if sort_by == "success_rate":
            recipes.sort(key=lambda r: r.success_rate, reverse=True)
        elif sort_by == "usage_count":
            recipes.sort(key=lambda r: r.usage_count, reverse=True)
        elif sort_by == "avg_execution_time":
            recipes.sort(key=lambda r: r.average_execution_time_ms)
        else:
            recipes.sort(key=lambda r: r.updated_at, reverse=True)
        
        return recipes[:limit]
    
    def _recipe_matches_filter(self, recipe: RecipeDefinition, search_filter: RecipeSearchFilter) -> bool:
        """Check if recipe matches search filter"""
        # Category filter
        if search_filter.categories and recipe.category not in search_filter.categories:
            return False
        
        # Difficulty filter
        if search_filter.difficulties and recipe.difficulty not in search_filter.difficulties:
            return False
        
        # Status filter
        if search_filter.statuses and recipe.status not in search_filter.statuses:
            return False
        
        # Tags filter
        if search_filter.tags:
            if not any(tag in recipe.tags for tag in search_filter.tags):
                return False
        
        # Agent types filter
        if search_filter.agent_types:
            recipe_agent_types = [req.agent_type for req in recipe.agent_requirements]
            if not any(agent_type in recipe_agent_types for agent_type in search_filter.agent_types):
                return False
        
        # MCP servers filter
        if search_filter.mcp_servers:
            recipe_servers = [req.server_name for req in recipe.mcp_requirements]
            if not any(server in recipe_servers for server in search_filter.mcp_servers):
                return False
        
        # Success rate filter
        if search_filter.min_success_rate and recipe.success_rate < search_filter.min_success_rate:
            return False
        
        # Execution time filter
        if search_filter.max_execution_time and recipe.average_execution_time_ms > search_filter.max_execution_time:
            return False
        
        # Text search filter
        if search_filter.text_search:
            search_text = search_filter.text_search.lower()
            searchable_text = f"{recipe.name} {recipe.description} {' '.join(recipe.tags)}".lower()
            if search_text not in searchable_text:
                return False
        
        return True
    
    def _calculate_relevance_score(self, recipe: RecipeDefinition, search_filter: RecipeSearchFilter) -> float:
        """Calculate relevance score for search results"""
        score = 0.0
        
        # Base score from success rate and usage
        score += recipe.success_rate * 0.3
        score += min(recipe.usage_count / 100.0, 1.0) * 0.2
        
        # Recency bonus
        days_old = (datetime.utcnow() - recipe.updated_at).days
        recency_score = max(0.0, 1.0 - (days_old / 365.0))  # Decay over a year
        score += recency_score * 0.2
        
        # Text search relevance
        if search_filter.text_search:
            search_text = search_filter.text_search.lower()
            if search_text in recipe.name.lower():
                score += 0.3  # Name match bonus
            if search_text in recipe.description.lower():
                score += 0.2  # Description match bonus
        
        # Performance bonus (faster execution)
        if recipe.average_execution_time_ms > 0:
            time_score = max(0.0, 1.0 - (recipe.average_execution_time_ms / 30000.0))  # 30 second baseline
            score += time_score * 0.1
        
        return score
    
    async def _setup_database(self) -> None:
        """Setup SQLite database for recipe storage"""
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        
        # Create recipes table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                version TEXT,
                category TEXT,
                difficulty TEXT,
                status TEXT,
                tags TEXT,  -- JSON array
                agent_requirements TEXT,  -- JSON array
                mcp_requirements TEXT,  -- JSON array
                steps TEXT,  -- JSON array
                execution_order TEXT,  -- JSON array
                parallel_steps TEXT,  -- JSON array
                validation_criteria TEXT,  -- JSON object
                test_scenarios TEXT,  -- JSON array
                author TEXT,
                created_at TEXT,
                updated_at TEXT,
                success_rate REAL DEFAULT 0.0,
                average_execution_time_ms INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                last_tested TEXT,
                documentation_url TEXT,
                example_usage TEXT,
                known_issues TEXT,  -- JSON array
                related_recipes TEXT  -- JSON array
            )
        """)
        
        # Create indexes for better search performance
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_recipes_category ON recipes(category)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_recipes_difficulty ON recipes(difficulty)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_recipes_status ON recipes(status)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_recipes_success_rate ON recipes(success_rate)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_recipes_updated_at ON recipes(updated_at)")
        
        self.connection.commit()
    
    async def _save_recipe_to_db(self, recipe: RecipeDefinition) -> None:
        """Save recipe to database"""
        recipe_dict = recipe.to_dict()
        
        # Convert complex fields to JSON
        json_fields = [
            'tags', 'agent_requirements', 'mcp_requirements', 'steps',
            'execution_order', 'parallel_steps', 'validation_criteria',
            'test_scenarios', 'known_issues', 'related_recipes'
        ]
        
        for field in json_fields:
            if field in recipe_dict:
                recipe_dict[field] = json.dumps(recipe_dict[field])
        
        # Insert or replace recipe
        placeholders = ', '.join(['?' for _ in recipe_dict])
        columns = ', '.join(recipe_dict.keys())
        values = list(recipe_dict.values())
        
        self.connection.execute(
            f"INSERT OR REPLACE INTO recipes ({columns}) VALUES ({placeholders})",
            values
        )
        self.connection.commit()
    
    async def _load_recipe_from_db(self, recipe_id: str) -> Optional[RecipeDefinition]:
        """Load recipe from database"""
        cursor = self.connection.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Convert row to dict
        recipe_dict = dict(row)
        
        # Parse JSON fields
        json_fields = [
            'tags', 'agent_requirements', 'mcp_requirements', 'steps',
            'execution_order', 'parallel_steps', 'validation_criteria',
            'test_scenarios', 'known_issues', 'related_recipes'
        ]
        
        for field in json_fields:
            if recipe_dict[field]:
                try:
                    recipe_dict[field] = json.loads(recipe_dict[field])
                except json.JSONDecodeError:
                    recipe_dict[field] = []
        
        return RecipeDefinition.from_dict(recipe_dict)
    
    async def _load_recipes_cache(self) -> None:
        """Load all recipes into memory cache"""
        cursor = self.connection.execute("SELECT id FROM recipes")
        recipe_ids = [row[0] for row in cursor.fetchall()]
        
        for recipe_id in recipe_ids:
            recipe = await self._load_recipe_from_db(recipe_id)
            if recipe:
                self.recipe_cache[recipe_id] = recipe
    
    async def _save_cache_to_db(self) -> None:
        """Save cache changes to database"""
        for recipe in self.recipe_cache.values():
            await self._save_recipe_to_db(recipe)
        
        self.cache_dirty = False
    
    async def _delete_recipe_from_db(self, recipe_id: str) -> None:
        """Delete recipe from database"""
        self.connection.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
        self.connection.commit()
    
    async def _update_statistics(self) -> None:
        """Update registry statistics"""
        self.total_recipes = len(self.recipe_cache)
        
        # Count by category
        self.recipes_by_category = {}
        for recipe in self.recipe_cache.values():
            category = recipe.category.value
            self.recipes_by_category[category] = self.recipes_by_category.get(category, 0) + 1
        
        # Count by status
        self.recipes_by_status = {}
        for recipe in self.recipe_cache.values():
            status = recipe.status.value
            self.recipes_by_status[status] = self.recipes_by_status.get(status, 0) + 1
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_recipes": self.total_recipes,
            "recipes_by_category": self.recipes_by_category,
            "recipes_by_status": self.recipes_by_status,
            "cache_size": len(self.recipe_cache),
            "cache_dirty": self.cache_dirty
        }
