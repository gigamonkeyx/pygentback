"""
Recipe Ranking System

This module provides intelligent ranking and recommendation of Agent + MCP recipes
based on performance metrics, success rates, and contextual factors.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..core.framework import RecipeTestResult
from ..recipes.schema import RecipeDefinition, RecipeCategory, RecipeDifficulty
from ..core.scoring import RecipeScorer, ScoringMetrics, ScoreBreakdown


logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Available ranking strategies"""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    POPULARITY = "popularity"
    RECENCY = "recency"
    COMPOSITE = "composite"
    CONTEXTUAL = "contextual"


@dataclass
class RankingCriteria:
    """Criteria for ranking recipes"""
    strategy: RankingStrategy = RankingStrategy.COMPOSITE
    category_preference: Optional[RecipeCategory] = None
    difficulty_preference: Optional[RecipeDifficulty] = None
    max_execution_time_ms: Optional[int] = None
    min_success_rate: Optional[float] = None
    agent_type_preference: Optional[str] = None
    mcp_server_preference: Optional[List[str]] = None
    recency_weight: float = 0.1
    popularity_weight: float = 0.1
    custom_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class RankedRecipe:
    """A recipe with its ranking information"""
    recipe: RecipeDefinition
    rank: int
    score: float
    score_breakdown: ScoreBreakdown
    ranking_factors: Dict[str, float] = field(default_factory=dict)
    recommendation_reason: str = ""


@dataclass
class RankingResult:
    """Result of recipe ranking"""
    ranked_recipes: List[RankedRecipe]
    total_recipes: int
    ranking_criteria: RankingCriteria
    ranking_metadata: Dict[str, Any] = field(default_factory=dict)


class RecipeRanker:
    """
    Intelligent ranking system for Agent + MCP recipes.
    
    Ranks recipes based on multiple factors including performance, reliability,
    popularity, recency, and contextual relevance.
    """
    
    def __init__(self, scorer: Optional[RecipeScorer] = None):
        self.scorer = scorer or RecipeScorer()
        
        # Historical data for ranking
        self.recipe_results: Dict[str, List[RecipeTestResult]] = {}
        self.recipe_usage_counts: Dict[str, int] = {}
        self.recipe_last_used: Dict[str, datetime] = {}
        self.recipe_user_ratings: Dict[str, List[float]] = {}
        
        # Ranking weights for different strategies
        self.strategy_weights = {
            RankingStrategy.PERFORMANCE: {
                "performance_score": 0.6,
                "reliability_score": 0.3,
                "recency": 0.1
            },
            RankingStrategy.RELIABILITY: {
                "reliability_score": 0.7,
                "consistency_score": 0.2,
                "robustness_score": 0.1
            },
            RankingStrategy.POPULARITY: {
                "usage_count": 0.5,
                "user_rating": 0.3,
                "success_rate": 0.2
            },
            RankingStrategy.RECENCY: {
                "recency": 0.6,
                "performance_score": 0.2,
                "success_rate": 0.2
            },
            RankingStrategy.COMPOSITE: {
                "performance_score": 0.25,
                "reliability_score": 0.25,
                "popularity": 0.2,
                "recency": 0.15,
                "contextual_fit": 0.15
            }
        }
    
    def add_test_results(self, recipe_id: str, results: List[RecipeTestResult]) -> None:
        """Add test results for a recipe"""
        if recipe_id not in self.recipe_results:
            self.recipe_results[recipe_id] = []
        self.recipe_results[recipe_id].extend(results)
        
        # Update usage count
        self.recipe_usage_counts[recipe_id] = self.recipe_usage_counts.get(recipe_id, 0) + len(results)
        
        # Update last used timestamp
        self.recipe_last_used[recipe_id] = datetime.utcnow()
    
    def add_user_rating(self, recipe_id: str, rating: float) -> None:
        """Add user rating for a recipe (0.0 to 5.0)"""
        if recipe_id not in self.recipe_user_ratings:
            self.recipe_user_ratings[recipe_id] = []
        self.recipe_user_ratings[recipe_id].append(max(0.0, min(5.0, rating)))
    
    def rank_recipes(self, 
                    recipes: List[RecipeDefinition],
                    criteria: Optional[RankingCriteria] = None,
                    limit: Optional[int] = None) -> RankingResult:
        """
        Rank recipes based on specified criteria.
        
        Args:
            recipes: List of recipes to rank
            criteria: Ranking criteria
            limit: Maximum number of recipes to return
            
        Returns:
            Ranking result with ranked recipes
        """
        criteria = criteria or RankingCriteria()
        
        logger.info(f"Ranking {len(recipes)} recipes using {criteria.strategy.value} strategy")
        
        # Filter recipes based on criteria
        filtered_recipes = self._filter_recipes(recipes, criteria)
        
        if not filtered_recipes:
            return RankingResult(
                ranked_recipes=[],
                total_recipes=len(recipes),
                ranking_criteria=criteria,
                ranking_metadata={"filtered_out": len(recipes)}
            )
        
        # Calculate scores for each recipe
        scored_recipes = []
        for recipe in filtered_recipes:
            score, score_breakdown, ranking_factors = self._calculate_recipe_score(recipe, criteria)
            
            scored_recipes.append({
                "recipe": recipe,
                "score": score,
                "score_breakdown": score_breakdown,
                "ranking_factors": ranking_factors
            })
        
        # Sort by score (descending)
        scored_recipes.sort(key=lambda x: x["score"], reverse=True)
        
        # Create ranked recipes
        ranked_recipes = []
        for i, item in enumerate(scored_recipes):
            if limit and i >= limit:
                break
            
            reason = self._generate_recommendation_reason(item, criteria)
            
            ranked_recipe = RankedRecipe(
                recipe=item["recipe"],
                rank=i + 1,
                score=item["score"],
                score_breakdown=item["score_breakdown"],
                ranking_factors=item["ranking_factors"],
                recommendation_reason=reason
            )
            ranked_recipes.append(ranked_recipe)
        
        # Create ranking metadata
        metadata = {
            "total_evaluated": len(filtered_recipes),
            "filtered_out": len(recipes) - len(filtered_recipes),
            "strategy_used": criteria.strategy.value,
            "average_score": sum(item["score"] for item in scored_recipes) / len(scored_recipes) if scored_recipes else 0.0
        }
        
        return RankingResult(
            ranked_recipes=ranked_recipes,
            total_recipes=len(recipes),
            ranking_criteria=criteria,
            ranking_metadata=metadata
        )
    
    def get_recommendations(self, 
                           recipes: List[RecipeDefinition],
                           context: Dict[str, Any],
                           limit: int = 5) -> List[RankedRecipe]:
        """
        Get personalized recipe recommendations based on context.
        
        Args:
            recipes: Available recipes
            context: Context information (user preferences, current task, etc.)
            limit: Number of recommendations
            
        Returns:
            List of recommended recipes
        """
        # Create contextual ranking criteria
        criteria = self._create_contextual_criteria(context)
        
        # Rank recipes
        ranking_result = self.rank_recipes(recipes, criteria, limit)
        
        return ranking_result.ranked_recipes
    
    def _filter_recipes(self, 
                       recipes: List[RecipeDefinition],
                       criteria: RankingCriteria) -> List[RecipeDefinition]:
        """Filter recipes based on criteria"""
        filtered = []
        
        for recipe in recipes:
            # Category filter
            if criteria.category_preference and recipe.category != criteria.category_preference:
                continue
            
            # Difficulty filter
            if criteria.difficulty_preference and recipe.difficulty != criteria.difficulty_preference:
                continue
            
            # Agent type filter
            if criteria.agent_type_preference:
                agent_types = [req.agent_type for req in recipe.agent_requirements]
                if criteria.agent_type_preference not in agent_types:
                    continue
            
            # MCP server filter
            if criteria.mcp_server_preference:
                recipe_servers = [req.server_name for req in recipe.mcp_requirements]
                if not any(server in recipe_servers for server in criteria.mcp_server_preference):
                    continue
            
            # Performance filters
            if criteria.max_execution_time_ms:
                if recipe.average_execution_time_ms > criteria.max_execution_time_ms:
                    continue
            
            if criteria.min_success_rate:
                if recipe.success_rate < criteria.min_success_rate:
                    continue
            
            filtered.append(recipe)
        
        return filtered
    
    def _calculate_recipe_score(self, 
                               recipe: RecipeDefinition,
                               criteria: RankingCriteria) -> Tuple[float, ScoreBreakdown, Dict[str, float]]:
        """Calculate score for a recipe based on criteria"""
        ranking_factors = {}
        
        # Get test results for this recipe
        results = self.recipe_results.get(recipe.id, [])
        
        if criteria.strategy == RankingStrategy.PERFORMANCE:
            score_breakdown = self._score_performance_strategy(recipe, results)
        elif criteria.strategy == RankingStrategy.RELIABILITY:
            score_breakdown = self._score_reliability_strategy(recipe, results)
        elif criteria.strategy == RankingStrategy.POPULARITY:
            score_breakdown = self._score_popularity_strategy(recipe, results)
        elif criteria.strategy == RankingStrategy.RECENCY:
            score_breakdown = self._score_recency_strategy(recipe, results)
        elif criteria.strategy == RankingStrategy.CONTEXTUAL:
            score_breakdown = self._score_contextual_strategy(recipe, results, criteria)
        else:  # COMPOSITE
            score_breakdown = self._score_composite_strategy(recipe, results, criteria)
        
        # Apply custom weights if provided
        if criteria.custom_weights:
            score_breakdown = self._apply_custom_weights(score_breakdown, criteria.custom_weights)
        
        return score_breakdown.total_score, score_breakdown, ranking_factors
    
    def _score_performance_strategy(self, 
                                   recipe: RecipeDefinition,
                                   results: List[RecipeTestResult]) -> ScoreBreakdown:
        """Score recipe using performance strategy"""
        if results:
            return self.scorer.score_multiple_results(results, metric=ScoringMetrics.PERFORMANCE)
        else:
            # Use recipe metadata for scoring
            base_score = 0.5  # Default score
            
            # Adjust based on recipe characteristics
            if recipe.average_execution_time_ms > 0:
                time_score = max(0.0, min(1.0, 5000 / recipe.average_execution_time_ms))
                base_score = 0.7 * base_score + 0.3 * time_score
            
            return ScoreBreakdown(
                total_score=base_score,
                explanation="Performance score based on recipe metadata (no test results)"
            )
    
    def _score_reliability_strategy(self, 
                                   recipe: RecipeDefinition,
                                   results: List[RecipeTestResult]) -> ScoreBreakdown:
        """Score recipe using reliability strategy"""
        if results:
            return self.scorer.score_multiple_results(results, metric=ScoringMetrics.RELIABILITY)
        else:
            # Use recipe success rate
            base_score = recipe.success_rate
            
            return ScoreBreakdown(
                total_score=base_score,
                explanation=f"Reliability score based on historical success rate: {base_score:.1%}"
            )
    
    def _score_popularity_strategy(self, 
                                  recipe: RecipeDefinition,
                                  results: List[RecipeTestResult]) -> ScoreBreakdown:
        """Score recipe using popularity strategy"""
        usage_count = self.recipe_usage_counts.get(recipe.id, 0)
        user_ratings = self.recipe_user_ratings.get(recipe.id, [])
        
        # Usage score (normalized)
        max_usage = max(self.recipe_usage_counts.values()) if self.recipe_usage_counts else 1
        usage_score = usage_count / max_usage if max_usage > 0 else 0.0
        
        # User rating score
        avg_rating = sum(user_ratings) / len(user_ratings) if user_ratings else 2.5
        rating_score = avg_rating / 5.0  # Normalize to 0-1
        
        # Success rate
        success_score = recipe.success_rate
        
        # Combine scores
        weights = self.strategy_weights[RankingStrategy.POPULARITY]
        total_score = (
            usage_score * weights["usage_count"] +
            rating_score * weights["user_rating"] +
            success_score * weights["success_rate"]
        )
        
        return ScoreBreakdown(
            total_score=total_score,
            component_scores={
                "usage_count": usage_score,
                "user_rating": rating_score,
                "success_rate": success_score
            },
            weights=weights,
            explanation=f"Popularity: {usage_count} uses, {avg_rating:.1f}/5.0 rating, {success_score:.1%} success"
        )
    
    def _score_recency_strategy(self, 
                               recipe: RecipeDefinition,
                               results: List[RecipeTestResult]) -> ScoreBreakdown:
        """Score recipe using recency strategy"""
        # Recency score
        last_used = self.recipe_last_used.get(recipe.id, recipe.created_at)
        days_since_used = (datetime.utcnow() - last_used).days
        recency_score = max(0.0, 1.0 - days_since_used / 365.0)  # Decay over a year
        
        # Performance and success scores
        if results:
            perf_breakdown = self.scorer.score_multiple_results(results, metric=ScoringMetrics.PERFORMANCE)
            performance_score = perf_breakdown.total_score
        else:
            performance_score = 0.5
        
        success_score = recipe.success_rate
        
        # Combine scores
        weights = self.strategy_weights[RankingStrategy.RECENCY]
        total_score = (
            recency_score * weights["recency"] +
            performance_score * weights["performance_score"] +
            success_score * weights["success_rate"]
        )
        
        return ScoreBreakdown(
            total_score=total_score,
            component_scores={
                "recency": recency_score,
                "performance": performance_score,
                "success_rate": success_score
            },
            weights=weights,
            explanation=f"Recency: {days_since_used} days ago, performance {performance_score:.2f}"
        )
    
    def _score_contextual_strategy(self, 
                                  recipe: RecipeDefinition,
                                  results: List[RecipeTestResult],
                                  criteria: RankingCriteria) -> ScoreBreakdown:
        """Score recipe using contextual strategy"""
        # Base performance score
        if results:
            base_breakdown = self.scorer.score_multiple_results(results, metric=ScoringMetrics.COMPOSITE)
            base_score = base_breakdown.total_score
        else:
            base_score = recipe.success_rate
        
        # Contextual fit score
        contextual_score = 1.0
        
        # Category match bonus
        if criteria.category_preference and recipe.category == criteria.category_preference:
            contextual_score += 0.2
        
        # Difficulty match bonus
        if criteria.difficulty_preference and recipe.difficulty == criteria.difficulty_preference:
            contextual_score += 0.1
        
        # Agent type match bonus
        if criteria.agent_type_preference:
            agent_types = [req.agent_type for req in recipe.agent_requirements]
            if criteria.agent_type_preference in agent_types:
                contextual_score += 0.15
        
        # MCP server match bonus
        if criteria.mcp_server_preference:
            recipe_servers = [req.server_name for req in recipe.mcp_requirements]
            matches = sum(1 for server in criteria.mcp_server_preference if server in recipe_servers)
            contextual_score += 0.1 * matches / len(criteria.mcp_server_preference)
        
        # Normalize contextual score
        contextual_score = min(1.0, contextual_score)
        
        # Combine scores
        total_score = 0.7 * base_score + 0.3 * contextual_score
        
        return ScoreBreakdown(
            total_score=total_score,
            component_scores={
                "base_score": base_score,
                "contextual_fit": contextual_score
            },
            weights={"base": 0.7, "contextual": 0.3},
            explanation=f"Contextual: base {base_score:.2f}, fit {contextual_score:.2f}"
        )
    
    def _score_composite_strategy(self, 
                                 recipe: RecipeDefinition,
                                 results: List[RecipeTestResult],
                                 criteria: RankingCriteria) -> ScoreBreakdown:
        """Score recipe using composite strategy"""
        # Get component scores
        if results:
            perf_breakdown = self.scorer.score_multiple_results(results, metric=ScoringMetrics.PERFORMANCE)
            rel_breakdown = self.scorer.score_multiple_results(results, metric=ScoringMetrics.RELIABILITY)
            performance_score = perf_breakdown.total_score
            reliability_score = rel_breakdown.total_score
        else:
            performance_score = 0.5
            reliability_score = recipe.success_rate
        
        # Popularity score
        usage_count = self.recipe_usage_counts.get(recipe.id, 0)
        max_usage = max(self.recipe_usage_counts.values()) if self.recipe_usage_counts else 1
        popularity_score = usage_count / max_usage if max_usage > 0 else 0.0
        
        # Recency score
        last_used = self.recipe_last_used.get(recipe.id, recipe.created_at)
        days_since_used = (datetime.utcnow() - last_used).days
        recency_score = max(0.0, 1.0 - days_since_used / 365.0)
        
        # Contextual fit score (simplified)
        contextual_score = 1.0
        if criteria.category_preference and recipe.category == criteria.category_preference:
            contextual_score = 1.2
        
        # Combine scores
        weights = self.strategy_weights[RankingStrategy.COMPOSITE]
        total_score = (
            performance_score * weights["performance_score"] +
            reliability_score * weights["reliability_score"] +
            popularity_score * weights["popularity"] +
            recency_score * weights["recency"] +
            contextual_score * weights["contextual_fit"]
        )
        
        return ScoreBreakdown(
            total_score=min(1.0, total_score),  # Cap at 1.0
            component_scores={
                "performance": performance_score,
                "reliability": reliability_score,
                "popularity": popularity_score,
                "recency": recency_score,
                "contextual_fit": contextual_score
            },
            weights=weights,
            explanation=f"Composite: P={performance_score:.2f}, R={reliability_score:.2f}, Pop={popularity_score:.2f}"
        )
    
    def _apply_custom_weights(self, 
                             score_breakdown: ScoreBreakdown,
                             custom_weights: Dict[str, float]) -> ScoreBreakdown:
        """Apply custom weights to score breakdown"""
        # Recalculate total score with custom weights
        total_weight = sum(custom_weights.values())
        if total_weight == 0:
            return score_breakdown
        
        # Normalize weights
        normalized_weights = {k: v / total_weight for k, v in custom_weights.items()}
        
        # Recalculate score
        new_score = 0.0
        for component, weight in normalized_weights.items():
            if component in score_breakdown.component_scores:
                new_score += score_breakdown.component_scores[component] * weight
        
        # Update breakdown
        score_breakdown.total_score = new_score
        score_breakdown.weights.update(normalized_weights)
        score_breakdown.explanation += f" (Custom weights applied)"
        
        return score_breakdown
    
    def _create_contextual_criteria(self, context: Dict[str, Any]) -> RankingCriteria:
        """Create ranking criteria from context"""
        criteria = RankingCriteria(strategy=RankingStrategy.CONTEXTUAL)
        
        # Extract preferences from context
        if "category" in context:
            try:
                criteria.category_preference = RecipeCategory(context["category"])
            except ValueError:
                pass
        
        if "difficulty" in context:
            try:
                criteria.difficulty_preference = RecipeDifficulty(context["difficulty"])
            except ValueError:
                pass
        
        if "agent_type" in context:
            criteria.agent_type_preference = context["agent_type"]
        
        if "mcp_servers" in context:
            criteria.mcp_server_preference = context["mcp_servers"]
        
        if "max_time" in context:
            criteria.max_execution_time_ms = context["max_time"]
        
        if "min_success_rate" in context:
            criteria.min_success_rate = context["min_success_rate"]
        
        return criteria
    
    def _generate_recommendation_reason(self, 
                                       scored_recipe: Dict[str, Any],
                                       criteria: RankingCriteria) -> str:
        """Generate explanation for why recipe is recommended"""
        recipe = scored_recipe["recipe"]
        score = scored_recipe["score"]
        factors = scored_recipe["ranking_factors"]
        
        reasons = []
        
        # High score
        if score > 0.8:
            reasons.append("excellent overall performance")
        elif score > 0.6:
            reasons.append("good performance metrics")
        
        # Success rate
        if recipe.success_rate > 0.9:
            reasons.append("very reliable")
        elif recipe.success_rate > 0.8:
            reasons.append("reliable")
        
        # Usage popularity
        usage_count = self.recipe_usage_counts.get(recipe.id, 0)
        if usage_count > 50:
            reasons.append("widely used")
        elif usage_count > 10:
            reasons.append("popular choice")
        
        # Recency
        last_used = self.recipe_last_used.get(recipe.id, recipe.created_at)
        days_since_used = (datetime.utcnow() - last_used).days
        if days_since_used < 7:
            reasons.append("recently tested")
        
        # Contextual fit
        if criteria.category_preference and recipe.category == criteria.category_preference:
            reasons.append(f"matches {criteria.category_preference.value} category")
        
        if criteria.agent_type_preference:
            agent_types = [req.agent_type for req in recipe.agent_requirements]
            if criteria.agent_type_preference in agent_types:
                reasons.append(f"uses {criteria.agent_type_preference} agent")
        
        # Combine reasons
        if not reasons:
            return "meets basic requirements"
        elif len(reasons) == 1:
            return reasons[0]
        elif len(reasons) == 2:
            return f"{reasons[0]} and {reasons[1]}"
        else:
            return f"{', '.join(reasons[:-1])}, and {reasons[-1]}"
    
    def get_ranking_statistics(self) -> Dict[str, Any]:
        """Get ranking system statistics"""
        total_recipes_tracked = len(self.recipe_results)
        total_results = sum(len(results) for results in self.recipe_results.values())
        total_ratings = sum(len(ratings) for ratings in self.recipe_user_ratings.values())
        
        return {
            "recipes_tracked": total_recipes_tracked,
            "total_test_results": total_results,
            "total_user_ratings": total_ratings,
            "average_results_per_recipe": total_results / total_recipes_tracked if total_recipes_tracked > 0 else 0,
            "recipes_with_ratings": len(self.recipe_user_ratings),
            "most_used_recipe": max(self.recipe_usage_counts.items(), key=lambda x: x[1]) if self.recipe_usage_counts else None
        }
