"""
Recipe Neural Architecture Search

Main orchestrator for neural architecture search optimization of PyGent Factory recipes.
Combines architecture encoding, search space, performance prediction, and search strategies.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .architecture_encoder import ArchitectureEncoder, RecipeArchitecture
from .search_space import SearchSpace, SearchConstraints
from .performance_predictor import PerformancePredictor
from .search_strategies import SearchStrategyFactory, SearchConfig, SearchResult

try:
    from ...testing.recipes.schema import RecipeDefinition
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    
    @dataclass
    class RecipeDefinition:
        name: str = ""
        description: str = ""

logger = logging.getLogger(__name__)


@dataclass
class NASConfig:
    """Configuration for Recipe NAS"""
    # Search configuration
    search_strategy: str = "evolutionary"
    max_evaluations: int = 1000
    max_time_seconds: float = 3600
    target_performance: float = 0.95
    
    # Search space configuration
    max_nodes: int = 30
    max_edges: int = 60
    complexity_budget: float = 15.0
    
    # Population settings (for evolutionary search)
    population_size: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    
    # Performance prediction
    use_predictor: bool = True
    train_predictor: bool = True
    predictor_confidence_threshold: float = 0.7
    
    # Advanced settings
    multi_objective: bool = True
    diversity_preservation: bool = True
    early_stopping_patience: int = 100
    
    # Logging and monitoring
    log_interval: int = 50
    checkpoint_interval: int = 200


@dataclass
class NASResult:
    """Result of Recipe NAS optimization"""
    best_recipe: RecipeDefinition
    best_architecture: RecipeArchitecture
    best_performance: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time_seconds: float
    search_strategy_used: str
    termination_reason: str
    convergence_data: List[float]
    architecture_diversity: List[float]
    predictor_accuracy: Optional[float] = None


class RecipeNAS:
    """
    Main Neural Architecture Search system for recipe optimization.
    
    Orchestrates the complete NAS pipeline:
    1. Architecture encoding and search space definition
    2. Performance prediction for efficient evaluation
    3. Search strategy execution for optimization
    4. Result analysis and recipe generation
    """
    
    def __init__(self, config: Optional[NASConfig] = None):
        self.config = config or NASConfig()
        
        # Initialize components
        self.search_space = SearchSpace(
            SearchConstraints(
                max_nodes=self.config.max_nodes,
                max_edges=self.config.max_edges,
                complexity_budget=self.config.complexity_budget
            )
        )
        
        self.architecture_encoder = ArchitectureEncoder()
        self.performance_predictor = PerformancePredictor(self.search_space)
        
        # NAS state
        self.is_running = False
        self.start_time = None
        self.optimization_history = []
        self.best_result = None
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
    
    async def optimize_recipes(self, 
                              seed_recipes: Optional[List[RecipeDefinition]] = None,
                              progress_callback: Optional[Callable] = None) -> NASResult:
        """
        Optimize recipes using neural architecture search.
        
        Args:
            seed_recipes: Optional initial recipes to start optimization
            progress_callback: Optional callback for progress updates
            
        Returns:
            NAS optimization result
        """
        self.start_time = time.time()
        self.is_running = True
        
        try:
            logger.info(f"Starting Recipe NAS optimization with {self.config.search_strategy} strategy")
            
            # Prepare initial architectures
            initial_architectures = []
            if seed_recipes:
                for recipe in seed_recipes:
                    architecture = self.architecture_encoder.encode_recipe(recipe)
                    initial_architectures.append(architecture)
                logger.info(f"Encoded {len(initial_architectures)} seed recipes")
            
            # Train performance predictor if enabled
            if self.config.use_predictor and self.config.train_predictor:
                await self._train_performance_predictor(initial_architectures)
            
            # Create search strategy
            search_config = SearchConfig(
                max_evaluations=self.config.max_evaluations,
                max_time_seconds=self.config.max_time_seconds,
                target_performance=self.config.target_performance,
                population_size=self.config.population_size,
                mutation_rate=self.config.mutation_rate,
                crossover_rate=self.config.crossover_rate,
                early_stopping_patience=self.config.early_stopping_patience
            )
            
            search_strategy = SearchStrategyFactory.create_strategy(
                self.config.search_strategy,
                self.search_space,
                self.performance_predictor,
                search_config
            )
            
            # Add progress callback
            if progress_callback:
                self.progress_callbacks.append(progress_callback)
            
            # Perform search
            search_result = await self._execute_search_with_monitoring(
                search_strategy, initial_architectures
            )
            
            # Create final result
            nas_result = await self._create_nas_result(search_result)
            
            logger.info(f"Recipe NAS completed: {nas_result.total_evaluations} evaluations, "
                       f"best performance: {nas_result.best_performance}")
            
            return nas_result
            
        except Exception as e:
            logger.error(f"Recipe NAS optimization failed: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _train_performance_predictor(self, initial_architectures: List[RecipeArchitecture]):
        """Train the performance predictor with available data"""
        if not initial_architectures:
            logger.info("No initial architectures for predictor training")
            return
        
        logger.info("Training performance predictor...")
        
        # Generate training data by evaluating some architectures
        training_data = []
        
        # Evaluate initial architectures
        for arch in initial_architectures:
            # For now, use heuristic evaluation
            # In production, this would use actual recipe execution results
            performance_data = self._heuristic_evaluation(arch)
            training_data.append((arch, performance_data))
        
        # Generate additional random architectures for training
        num_additional = min(50, self.config.max_evaluations // 10)
        for _ in range(num_additional):
            arch = self.search_space.generate_random_architecture()
            performance_data = self._heuristic_evaluation(arch)
            training_data.append((arch, performance_data))
        
        # Train the predictor
        self.performance_predictor.train_models(training_data)
        
        logger.info(f"Trained performance predictor with {len(training_data)} samples")
    
    def _heuristic_evaluation(self, architecture: RecipeArchitecture) -> Dict[str, float]:
        """Heuristic evaluation for training data generation"""
        # Calculate complexity-based performance estimates
        complexity = self.architecture_encoder.calculate_architecture_complexity(architecture)
        
        # Simple heuristics
        success_probability = max(0.1, min(0.95, 0.8 - complexity * 0.3))
        execution_time = 1000 + complexity * 2000 + len(architecture.nodes) * 100
        memory_usage = 256 + complexity * 512 + len(architecture.nodes) * 32
        
        return {
            'success_probability': success_probability,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'complexity_score': complexity
        }
    
    async def _execute_search_with_monitoring(self, search_strategy, initial_architectures):
        """Execute search with progress monitoring"""
        
        # Wrap the search strategy's evaluate method to add monitoring
        original_evaluate = search_strategy.evaluate_architecture
        
        async def monitored_evaluate(architecture):
            result = await original_evaluate(architecture)
            
            # Record in optimization history
            self.optimization_history.append({
                'evaluation': search_strategy.evaluations,
                'performance': result.success_probability,
                'architecture_complexity': self.architecture_encoder.calculate_architecture_complexity(architecture),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Call progress callbacks
            if search_strategy.evaluations % self.config.log_interval == 0:
                await self._call_progress_callbacks(search_strategy)
            
            return result
        
        # Replace the evaluate method
        search_strategy.evaluate_architecture = monitored_evaluate
        
        # Execute search
        return await search_strategy.search(initial_architectures)
    
    async def _call_progress_callbacks(self, search_strategy):
        """Call progress callbacks with current status"""
        try:
            progress_data = {
                'evaluations': search_strategy.evaluations,
                'best_performance': search_strategy.best_performance.success_probability if search_strategy.best_performance else 0.0,
                'elapsed_time': time.time() - self.start_time,
                'convergence_history': search_strategy.convergence_history[-10:],  # Last 10 values
                'search_strategy': self.config.search_strategy
            }
            
            for callback in self.progress_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress_data)
                else:
                    callback(progress_data)
                    
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")
    
    async def _create_nas_result(self, search_result: SearchResult) -> NASResult:
        """Create NAS result from search result"""
        # Decode best architecture to recipe
        best_recipe = self.architecture_encoder.decode_architecture(search_result.best_architecture)
        
        # Extract performance metrics
        best_performance = {
            'success_probability': search_result.best_performance.success_probability,
            'execution_time_ms': search_result.best_performance.execution_time_ms,
            'memory_usage_mb': search_result.best_performance.memory_usage_mb,
            'complexity_score': search_result.best_performance.complexity_score,
            'confidence': search_result.best_performance.confidence
        }
        
        # Calculate architecture diversity over time
        architecture_diversity = self._calculate_architecture_diversity_history(search_result)
        
        # Calculate predictor accuracy if possible
        predictor_accuracy = self._calculate_predictor_accuracy(search_result)
        
        nas_result = NASResult(
            best_recipe=best_recipe,
            best_architecture=search_result.best_architecture,
            best_performance=best_performance,
            optimization_history=self.optimization_history.copy(),
            total_evaluations=search_result.total_evaluations,
            optimization_time_seconds=search_result.search_time_seconds,
            search_strategy_used=search_result.search_strategy,
            termination_reason=search_result.termination_reason,
            convergence_data=search_result.convergence_history,
            architecture_diversity=architecture_diversity,
            predictor_accuracy=predictor_accuracy
        )
        
        self.best_result = nas_result
        return nas_result
    
    def _calculate_architecture_diversity_history(self, search_result: SearchResult) -> List[float]:
        """Calculate diversity of architectures over search history"""
        diversity_history = []
        
        # Calculate diversity in sliding windows
        window_size = 20
        for i in range(window_size, len(search_result.search_history), 10):
            window_architectures = [arch for arch, _ in search_result.search_history[i-window_size:i]]
            
            # Calculate pairwise diversity
            total_diversity = 0.0
            comparisons = 0
            
            for j in range(len(window_architectures)):
                for k in range(j + 1, len(window_architectures)):
                    arch1, arch2 = window_architectures[j], window_architectures[k]
                    
                    # Simple diversity metric based on node and edge differences
                    node_diff = abs(len(arch1.nodes) - len(arch2.nodes))
                    edge_diff = abs(len(arch1.edges) - len(arch2.edges))
                    
                    diversity = (node_diff + edge_diff) / (len(arch1.nodes) + len(arch1.edges) + 1)
                    total_diversity += diversity
                    comparisons += 1
            
            avg_diversity = total_diversity / max(comparisons, 1)
            diversity_history.append(avg_diversity)
        
        return diversity_history
    
    def _calculate_predictor_accuracy(self, search_result: SearchResult) -> Optional[float]:
        """Calculate performance predictor accuracy"""
        # This would require actual vs predicted performance comparison
        # For now, return None as we don't have ground truth
        return None
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        if not self.is_running:
            return {"status": "not_running"}
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        status = {
            "status": "running",
            "elapsed_time": elapsed_time,
            "evaluations_completed": len(self.optimization_history),
            "best_performance": self.optimization_history[-1]['performance'] if self.optimization_history else 0.0,
            "search_strategy": self.config.search_strategy,
            "predictor_status": self.performance_predictor.get_predictor_status()
        }
        
        return status
    
    def stop_optimization(self):
        """Stop the optimization process"""
        self.is_running = False
        logger.info("Recipe NAS optimization stop requested")
    
    def add_progress_callback(self, callback: Callable):
        """Add a progress callback function"""
        self.progress_callbacks.append(callback)
    
    def export_optimization_data(self, filepath: str):
        """Export optimization data for analysis"""
        if not self.best_result:
            logger.warning("No optimization result to export")
            return
        
        export_data = {
            "config": {
                "search_strategy": self.config.search_strategy,
                "max_evaluations": self.config.max_evaluations,
                "max_nodes": self.config.max_nodes,
                "complexity_budget": self.config.complexity_budget
            },
            "result": {
                "best_performance": self.best_result.best_performance,
                "total_evaluations": self.best_result.total_evaluations,
                "optimization_time": self.best_result.optimization_time_seconds,
                "termination_reason": self.best_result.termination_reason
            },
            "optimization_history": self.optimization_history,
            "convergence_data": self.best_result.convergence_data,
            "architecture_diversity": self.best_result.architecture_diversity
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Optimization data exported to {filepath}")
    
    async def validate_recipe(self, recipe: RecipeDefinition) -> Dict[str, Any]:
        """Validate a recipe using the NAS system"""
        # Encode recipe to architecture
        architecture = self.architecture_encoder.encode_recipe(recipe)
        
        # Validate architecture constraints
        is_valid, violations = self.search_space.validate_architecture_constraints(architecture)
        
        # Predict performance
        performance_prediction = await self.performance_predictor.predict_performance(architecture)
        
        # Calculate complexity
        complexity = self.architecture_encoder.calculate_architecture_complexity(architecture)
        
        validation_result = {
            "is_valid": is_valid,
            "constraint_violations": violations,
            "predicted_performance": {
                "success_probability": performance_prediction.success_probability,
                "execution_time_ms": performance_prediction.execution_time_ms,
                "memory_usage_mb": performance_prediction.memory_usage_mb,
                "confidence": performance_prediction.confidence
            },
            "complexity_score": complexity,
            "risk_factors": performance_prediction.risk_factors,
            "architecture_stats": {
                "node_count": len(architecture.nodes),
                "edge_count": len(architecture.edges),
                "depth": self.architecture_encoder._calculate_depth(architecture),
                "width": self.architecture_encoder._calculate_width(architecture)
            }
        }
        
        return validation_result
    
    def get_search_space_summary(self) -> Dict[str, Any]:
        """Get summary of the search space"""
        return self.search_space.get_search_space_summary()

    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status for health monitoring"""
        return {
            'is_running': self.is_running,
            'status': 'running' if self.is_running else 'idle',
            'evaluations_completed': len(self.optimization_history),
            'best_performance': self.optimization_history[-1]['performance'] if self.optimization_history else 0.0,
            'search_strategy': self.config.search_strategy,
            'max_evaluations': self.config.max_evaluations,
            'has_result': self.best_result is not None
        }


async def test_recipe_nas():
    """Test function for Recipe NAS"""
    # Create test configuration
    config = NASConfig(
        search_strategy="evolutionary",
        max_evaluations=100,
        max_time_seconds=60,
        population_size=20
    )
    
    # Create Recipe NAS
    nas = RecipeNAS(config)
    
    # Progress callback
    async def progress_callback(data):
        print(f"Evaluation {data['evaluations']}: "
              f"Best performance: {data['best_performance']:.3f}, "
              f"Time: {data['elapsed_time']:.1f}s")
    
    # Run optimization
    result = await nas.optimize_recipes(progress_callback=progress_callback)
    
    print(f"\nRecipe NAS completed:")
    print(f"Best performance: {result.best_performance}")
    print(f"Evaluations: {result.total_evaluations}")
    print(f"Time: {result.optimization_time_seconds:.1f}s")
    print(f"Strategy: {result.search_strategy_used}")
    print(f"Termination: {result.termination_reason}")
    print(f"Best recipe: {result.best_recipe.name}")


if __name__ == "__main__":
    asyncio.run(test_recipe_nas())
