"""
Tree of Thought with GPU Vector Search Example

Demonstrates the integration of ToT reasoning with GPU-accelerated
vector search for enhanced semantic similarity and solution retrieval.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List

from src.ai.reasoning.tot.models import ToTConfig, GenerationStrategy, EvaluationMethod
from src.ai.reasoning.tot.tot_engine import ToTEngine
from src.ai.reasoning.tot.tasks.recipe_optimization import RecipeOptimizationTask
from src.ai.reasoning.tot.integrations.vector_search_integration import VectorSearchIntegration
from src.search.gpu_search import FAISS_AVAILABLE, FAISS_GPU_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_semantic_thought_search():
    """Demonstrate semantic search over thought states"""
    print("🧠 Semantic Thought Search Demo")
    print("=" * 35)
    
    # Create vector search integration
    vector_search = VectorSearchIntegration(
        embedding_dim=384,  # Smaller for demo
        use_gpu=FAISS_GPU_AVAILABLE
    )
    
    # Create some sample thought states
    from src.ai.reasoning.tot.models import ThoughtState
    
    sample_thoughts = [
        ThoughtState(content="Optimize database queries for better performance", depth=1),
        ThoughtState(content="Implement caching layer to reduce latency", depth=1),
        ThoughtState(content="Use connection pooling for database efficiency", depth=1),
        ThoughtState(content="Add monitoring and alerting for system health", depth=2),
        ThoughtState(content="Scale horizontally with load balancers", depth=2),
        ThoughtState(content="Implement data compression to save bandwidth", depth=1),
        ThoughtState(content="Use CDN for static content delivery", depth=2),
        ThoughtState(content="Optimize memory usage in application code", depth=1),
    ]
    
    # Mark some as solutions
    sample_thoughts[1].is_solution = True
    sample_thoughts[1].value_score = 0.9
    sample_thoughts[4].is_solution = True
    sample_thoughts[4].value_score = 0.85
    
    print(f"Created {len(sample_thoughts)} sample thoughts")
    
    # Add thoughts to vector search
    for thought in sample_thoughts:
        vector_search.add_thought(thought)
    
    # Test semantic search
    query_thought = ThoughtState(
        content="Improve application response time and reduce delays",
        depth=0
    )
    
    print(f"\nQuery: {query_thought.content}")
    print("\nSimilar thoughts:")
    
    similar_thoughts = vector_search.find_similar_thoughts(query_thought, k=3)
    
    for i, (thought_emb, similarity) in enumerate(similar_thoughts):
        print(f"{i+1}. (Similarity: {similarity:.3f}) {thought_emb.metadata}")
    
    # Test solution search
    print("\nSimilar solutions:")
    similar_solutions = vector_search.find_similar_solutions(query_thought, k=2)
    
    for i, (thought_id, similarity) in enumerate(similar_solutions):
        print(f"{i+1}. (Similarity: {similarity:.3f}) ID: {thought_id}")
    
    # Test clustering
    thought_ids = [t.id for t in sample_thoughts]
    clusters = vector_search.cluster_thoughts(thought_ids, n_clusters=3)
    
    print(f"\nClustered {len(thought_ids)} thoughts into {len(clusters)} clusters:")
    for cluster_id, cluster_thoughts in clusters.items():
        print(f"Cluster {cluster_id}: {len(cluster_thoughts)} thoughts")
    
    # Get statistics
    stats = vector_search.get_search_stats()
    print(f"\nVector Search Stats:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    vector_search.cleanup()


async def demonstrate_tot_with_vector_search():
    """Demonstrate ToT reasoning enhanced with vector search"""
    print("\n🌳 ToT with Vector Search Demo")
    print("=" * 35)
    
    # Configure ToT with vector search enabled
    tot_config = ToTConfig(
        generation_strategy=GenerationStrategy.PROPOSE,
        evaluation_method=EvaluationMethod.VALUE,
        n_generate_sample=2,
        n_evaluate_sample=1,
        n_select_sample=2,
        max_depth=4,
        temperature=0.7
    )
    
    # Create ToT engine with vector search
    tot_engine = ToTEngine(tot_config, enable_vector_search=True)
    
    # Test problem
    problem = """
    Recipe Performance Issue:
    
    Current recipe processes 1000 records per minute but needs to handle 10,000.
    The bottleneck appears to be in the data transformation step.
    Memory usage is also high at 8GB for the current load.
    
    Requirements:
    - 10x performance improvement
    - Reduce memory usage
    - Maintain data accuracy
    """
    
    task_context = {
        "propose_prompt": """
Problem: {task_description}

Current step: {current_thought}

Propose the next optimization step considering:
- Performance bottlenecks
- Memory efficiency
- Scalability
- Data accuracy

Next step:""",
        
        "value_prompt": """
Optimization step: {thought_content}

Rate this step (0.0-1.0) based on:
- Impact on performance
- Memory efficiency
- Implementation feasibility
- Risk level

Score:""",
        
        "solution_prompt": """
Proposed solution: {thought_content}

Is this a complete solution that addresses the 10x performance requirement?

Answer:"""
    }
    
    print("🚀 Starting ToT reasoning with vector search...")
    print(f"Problem: {problem.strip()}")
    print()
    
    # Solve with ToT
    result = await tot_engine.solve(problem, task_context)
    
    print("📊 ToT Results:")
    print(f"Success: {result.success}")
    print(f"Time: {result.total_time:.2f}s")
    print(f"Nodes explored: {result.nodes_explored}")
    print(f"Solutions found: {len(result.all_solutions)}")
    
    if result.best_solution:
        print(f"\n🎯 Best Solution (Score: {result.best_solution.value_score:.3f}):")
        print(result.best_solution.content)
        
        print(f"\n🛤️ Reasoning Path:")
        path = tot_engine.get_solution_path(result.best_solution, result.tree)
        for i, state in enumerate(path):
            prefix = "Problem:" if i == 0 else f"Step {i}:"
            print(f"{prefix} {state.content}")
    
    # Demonstrate vector search integration
    if tot_engine.vector_search:
        print(f"\n🔍 Vector Search Integration:")
        
        # Add the reasoning tree to vector search
        added_count = tot_engine.vector_search.add_thought_tree(result.tree)
        print(f"Added {added_count} thoughts to vector index")
        
        # Find similar thoughts to the best solution
        if result.best_solution:
            similar = tot_engine.vector_search.find_similar_thoughts(
                result.best_solution, k=3
            )
            print(f"Found {len(similar)} similar thoughts to the best solution")
        
        # Get vector search stats
        vs_stats = tot_engine.vector_search.get_search_stats()
        print(f"Vector search stats: {vs_stats['total_thoughts']} thoughts indexed")
    
    return result


async def demonstrate_recipe_optimization_with_search():
    """Demonstrate recipe optimization with semantic search"""
    print("\n🍳 Recipe Optimization with Search")
    print("=" * 40)
    
    # Create recipe optimization task
    tot_config = ToTConfig(
        generation_strategy=GenerationStrategy.PROPOSE,
        evaluation_method=EvaluationMethod.VALUE,
        max_depth=3,
        n_generate_sample=2
    )
    
    from src.ai.reasoning.tot.tasks.recipe_optimization import (
        RecipeOptimizationTask, RecipeOptimizationConfig
    )
    
    optimization_config = RecipeOptimizationConfig(
        optimization_goals=["performance", "scalability"],
        constraints=["maintain accuracy", "use existing infrastructure"],
        evaluation_criteria=["feasibility", "impact", "complexity"]
    )
    
    optimizer = RecipeOptimizationTask(tot_config, optimization_config)
    
    # Recipe to optimize
    recipe = """
    Data Processing Recipe:
    1. Read CSV files from disk
    2. Parse and validate each row
    3. Apply business rules transformation
    4. Write to database
    
    Current performance: 100 rows/second
    Target: 1000 rows/second
    """
    
    issues = ["slow file I/O", "inefficient parsing", "database bottleneck"]
    improvements = ["batch processing", "parallel execution", "memory optimization"]
    
    print(f"Optimizing recipe: {recipe.strip()}")
    print(f"Issues: {', '.join(issues)}")
    print(f"Target improvements: {', '.join(improvements)}")
    print()
    
    # Run optimization
    result = await optimizer.optimize_recipe(
        recipe_description=recipe,
        current_issues=issues,
        target_improvements=improvements
    )
    
    print("📈 Optimization Results:")
    print(f"Success: {result['success']}")
    print(f"Time: {result['total_time']:.2f}s")
    
    if result.get('optimization_found'):
        opt = result['best_optimization']
        print(f"\n✨ Best Optimization (Confidence: {opt['confidence_score']:.3f}):")
        print(opt['description'])
        
        print(f"\n📋 Reasoning Steps:")
        steps = opt['reasoning_path'].split('\n')
        for step in steps[:5]:  # Show first 5 steps
            if step.strip():
                print(f"  • {step.strip()}")
    
    return result


async def main():
    """Run all examples"""
    print("🚀 PyGent Factory: ToT + Vector Search Examples")
    print("=" * 55)
    
    # Check prerequisites
    if not FAISS_AVAILABLE:
        print("❌ FAISS not available. Install with:")
        print("   pip install faiss-cpu  # or faiss-gpu")
        return
    
    print(f"✅ FAISS available")
    print(f"🎮 GPU support: {FAISS_GPU_AVAILABLE}")
    print()
    
    try:
        # Run demonstrations
        await demonstrate_semantic_thought_search()
        await demonstrate_tot_with_vector_search()
        await demonstrate_recipe_optimization_with_search()
        
        print("\n🎉 All demonstrations completed!")
        print("\n💡 Key Benefits Demonstrated:")
        print("1. Semantic similarity search over reasoning states")
        print("2. Solution clustering and deduplication")
        print("3. Enhanced ToT with vector-based retrieval")
        print("4. GPU acceleration for large-scale reasoning")
        print("\n🔧 Next Steps:")
        print("1. Integrate with real embedding models")
        print("2. Scale to larger reasoning trees")
        print("3. Add persistent storage for thought libraries")
        print("4. Implement cross-session solution retrieval")
        
    except Exception as e:
        logger.exception("Example failed")
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    # Note: Requires FAISS and preferably GPU support
    print("📋 Prerequisites:")
    print("- FAISS library (pip install faiss-cpu or faiss-gpu)")
    print("- NVIDIA GPU with CUDA for optimal performance")
    print("- PyGent Factory environment")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        logger.exception("Example execution failed")
