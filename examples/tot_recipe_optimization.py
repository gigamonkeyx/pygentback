"""
Tree of Thought Recipe Optimization Example

Demonstrates how to use the ToT framework for optimizing PyGent Factory recipes
with multi-path reasoning and systematic evaluation.
"""

import asyncio
import logging
from typing import Dict, Any

from src.ai.reasoning.tot.models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from src.ai.reasoning.tot.tasks.recipe_optimization import RecipeOptimizationTask, RecipeOptimizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def optimize_simple_recipe():
    """Example: Optimize a simple recipe using ToT"""
    
    print("🌳 Tree of Thought Recipe Optimization Example")
    print("=" * 50)
    
    # Define the recipe to optimize
    recipe_description = """
    Recipe: Data Processing Pipeline
    Steps:
    1. Load data from CSV file
    2. Clean missing values
    3. Apply transformations
    4. Save to database
    
    Current implementation uses pandas and takes 5 minutes for 1M rows.
    """
    
    # Define current issues
    current_issues = [
        "Slow performance with large datasets",
        "High memory usage",
        "No error handling for corrupted data"
    ]
    
    # Define target improvements
    target_improvements = [
        "Reduce processing time by 50%",
        "Lower memory footprint",
        "Add robust error handling"
    ]
    
    # Configure ToT for recipe optimization
    tot_config = ToTConfig(
        generation_strategy=GenerationStrategy.PROPOSE,
        evaluation_method=EvaluationMethod.VALUE,
        search_method=SearchMethod.BFS,
        n_generate_sample=3,
        n_evaluate_sample=2,
        n_select_sample=3,
        max_depth=5,
        temperature=0.7,
        model_name="phi4-fast"  # Use your optimized model
    )
    
    # Configure optimization parameters
    optimization_config = RecipeOptimizationConfig(
        optimization_goals=["performance", "reliability", "maintainability"],
        constraints=["maintain data accuracy", "use existing infrastructure"],
        evaluation_criteria=["feasibility", "impact", "implementation_complexity"]
    )
    
    # Create optimization task
    optimizer = RecipeOptimizationTask(tot_config, optimization_config)
    
    print("🚀 Starting recipe optimization...")
    print(f"Recipe: {recipe_description.strip()}")
    print(f"Issues: {', '.join(current_issues)}")
    print(f"Goals: {', '.join(target_improvements)}")
    print()
    
    try:
        # Run optimization
        result = await optimizer.optimize_recipe(
            recipe_description=recipe_description,
            current_issues=current_issues,
            target_improvements=target_improvements
        )
        
        # Display results
        print("📊 Optimization Results:")
        print(f"Success: {result['success']}")
        print(f"Time taken: {result['total_time']:.2f} seconds")
        print(f"Nodes explored: {result['reasoning_stats']['nodes_explored']}")
        print(f"Max depth: {result['reasoning_stats']['max_depth_reached']}")
        print()
        
        if result['optimization_found']:
            best_opt = result['best_optimization']
            print("🎯 Best Optimization Found:")
            print(f"Confidence: {best_opt['confidence_score']:.3f}")
            print(f"Description: {best_opt['description']}")
            print()
            
            print("🛤️ Reasoning Path:")
            print(best_opt['reasoning_path'])
            print()
            
            # Show alternatives if available
            if 'alternative_optimizations' in result:
                print("🔄 Alternative Optimizations:")
                for i, alt in enumerate(result['alternative_optimizations'][:3]):
                    print(f"{i+1}. (Score: {alt['confidence_score']:.3f}) {alt['description'][:100]}...")
                print()
        else:
            print("❌ No optimization found")
            if 'error' in result:
                print(f"Error: {result['error']}")
            elif 'partial_optimization' in result:
                partial = result['partial_optimization']
                print(f"Partial result: {partial['description']}")
                print(f"Note: {partial['note']}")
    
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print(f"❌ Optimization failed: {e}")


async def compare_optimization_strategies():
    """Example: Compare different optimization strategies"""
    
    print("\n🔍 Comparing Optimization Strategies")
    print("=" * 40)
    
    recipe_description = """
    Recipe: Machine Learning Training Pipeline
    Steps:
    1. Load training data
    2. Preprocess features
    3. Train model
    4. Validate performance
    5. Save model
    
    Current issues: Training takes 2 hours, uses 16GB RAM
    """
    
    # Create optimizer
    tot_config = ToTConfig(
        max_depth=4,
        n_generate_sample=2,
        model_name="phi4-fast"
    )
    
    optimizer = RecipeOptimizationTask(tot_config)
    
    # Compare different strategies
    strategies = ["performance", "memory_efficiency", "scalability"]
    
    print(f"🔬 Testing strategies: {', '.join(strategies)}")
    
    try:
        comparison_result = await optimizer.compare_optimization_strategies(
            recipe_description, strategies
        )
        
        print("\n📈 Strategy Comparison Results:")
        
        for strategy in strategies:
            if strategy in comparison_result:
                result = comparison_result[strategy]
                print(f"\n{strategy.upper()}:")
                print(f"  Success: {result.get('success', False)}")
                if result.get('optimization_found'):
                    opt = result['best_optimization']
                    print(f"  Confidence: {opt['confidence_score']:.3f}")
                    print(f"  Solution: {opt['description'][:100]}...")
        
        # Show recommendation
        if 'comparison_summary' in comparison_result:
            summary = comparison_result['comparison_summary']
            print(f"\n🏆 Recommendation: {summary.get('recommended_strategy', 'None')}")
            print(f"Reason: {summary.get('recommendation_reason', 'N/A')}")
    
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        print(f"❌ Strategy comparison failed: {e}")


async def demonstrate_tot_features():
    """Demonstrate various ToT features"""
    
    print("\n🧠 Tree of Thought Features Demo")
    print("=" * 35)
    
    # Test different search strategies
    recipe = "Simple data validation recipe with performance issues"
    
    tot_configs = [
        ("BFS", ToTConfig(search_method=SearchMethod.BFS, max_depth=3)),
        ("DFS", ToTConfig(search_method=SearchMethod.DFS, max_depth=3)),
    ]
    
    for strategy_name, config in tot_configs:
        print(f"\n🔍 Testing {strategy_name} Search Strategy:")
        
        optimizer = RecipeOptimizationTask(config)
        
        try:
            result = await optimizer.optimize_recipe(
                recipe_description=recipe,
                current_issues=["slow execution"],
                target_improvements=["faster processing"]
            )
            
            print(f"  Success: {result['success']}")
            print(f"  Nodes explored: {result['reasoning_stats']['nodes_explored']}")
            print(f"  Time: {result['total_time']:.2f}s")
            
            if result.get('optimization_found'):
                score = result['best_optimization']['confidence_score']
                print(f"  Best score: {score:.3f}")
        
        except Exception as e:
            print(f"  Error: {e}")


async def main():
    """Run all examples"""
    print("🌳 PyGent Factory Tree of Thought Examples")
    print("=" * 50)
    
    # Run examples
    await optimize_simple_recipe()
    await compare_optimization_strategies()
    await demonstrate_tot_features()
    
    print("\n✅ All examples completed!")
    print("\nNext steps:")
    print("1. Try with your own recipes")
    print("2. Experiment with different ToT configurations")
    print("3. Integrate with your existing PyGent Factory workflows")


if __name__ == "__main__":
    # Note: This example requires Ollama to be running with phi4-fast model
    print("📋 Prerequisites:")
    print("- Ollama running on localhost:11434")
    print("- phi4-fast model available")
    print("- PyGent Factory environment set up")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        logger.exception("Example execution failed")
