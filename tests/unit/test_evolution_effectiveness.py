#!/usr/bin/env python3
"""
Test Evolution System Effectiveness - Are we doing real evolution or just "voodoo"?

This script will:
1. Start a small evolution run
2. Monitor what actually happens 
3. Check if we're generating meaningful improvements
4. Validate against DGM principles
"""

import asyncio
import time
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_recipe_evolution():
    """Test the recipe evolution system to see if it does real work"""
    print("=" * 60)
    print("TESTING RECIPE EVOLUTION SYSTEM EFFECTIVENESS")
    print("=" * 60)
    
    try:
        from src.evolution.advanced_recipe_evolution import RecipeEvolutionSystem, EvolutionConfig, EvolutionStrategy
        
        # Simple config for testing
        config = EvolutionConfig(
            population_size=10,  # Small for testing
            max_generations=3,   # Quick test
            evolution_strategy=EvolutionStrategy.HYBRID,
            use_reasoning=False,  # Disable reasoning to test core GA
            use_rag=False,       # Disable RAG to test core GA
            use_tot_reasoning=False
        )
        
        print(f"Creating evolution system with config: {config}")
        evolution = RecipeEvolutionSystem(config)
        
        # Create some initial test recipes
        initial_recipes = [
            {
                'name': 'Basic Web Scraper',
                'description': 'Simple web scraping script',
                'steps': ['Import requests', 'Send GET request', 'Parse HTML', 'Extract data'],
                'parameters': {'timeout': 30, 'retries': 3}
            },
            {
                'name': 'API Client',
                'description': 'REST API client implementation',
                'steps': ['Setup authentication', 'Send request', 'Handle response', 'Error handling'],
                'parameters': {'timeout': 60, 'max_retries': 5}
            }
        ]
        
        print(f"Starting evolution with {len(initial_recipes)} initial recipes...")
        
        # Run evolution
        start_time = time.time()
        results = await evolution.evolve(
            initial_recipes=initial_recipes,
            target_objectives=['performance', 'reliability', 'maintainability']
        )
        end_time = time.time()
        
        print(f"\nEvolution completed in {end_time - start_time:.2f} seconds")
        print(f"Results: {results}")
        
        # Analyze results
        if results.get('success'):
            print("\n" + "=" * 40)
            print("EVOLUTION ANALYSIS")
            print("=" * 40)
            
            best_recipes = results.get('best_recipes', [])
            print(f"Generated {len(best_recipes)} best recipes")
            
            for i, recipe in enumerate(best_recipes[:3]):  # Show top 3
                print(f"\nRecipe {i+1}: {recipe.get('name', 'Unknown')}")
                print(f"  Description: {recipe.get('description', 'N/A')}")
                print(f"  Generation: {recipe.get('generation', 0)}")
                print(f"  Fitness scores: {recipe.get('fitness_scores', {})}")
                print(f"  Steps: {len(recipe.get('steps', []))}")
                
            print(f"\nEvolution Stats:")
            print(f"  Generations: {results.get('generations_completed', 0)}")
            print(f"  Evaluations: {results.get('evaluations_performed', 0)}")
            print(f"  Average generation time: {results.get('average_generation_time', 0):.2f}s")
            print(f"  Convergence achieved: {results.get('convergence_achieved', False)}")
            
            # Check if we did real work
            evaluation_count = results.get('evaluations_performed', 0)
            if evaluation_count > 0:
                print(f"\n✅ REAL WORK DETECTED: {evaluation_count} fitness evaluations performed")
            else:
                print(f"\n❌ NO REAL WORK: 0 fitness evaluations performed")
                
            fitness_history = results.get('fitness_history', [])
            if len(fitness_history) > 1:
                improvement = fitness_history[-1] - fitness_history[0]
                print(f"✅ FITNESS IMPROVEMENT: {improvement:.4f} ({fitness_history[0]:.4f} -> {fitness_history[-1]:.4f})")
            else:
                print(f"❌ NO FITNESS TRACKING: {len(fitness_history)} fitness points recorded")
                
        else:
            print("❌ EVOLUTION FAILED")
            
    except Exception as e:
        print(f"❌ ERROR testing recipe evolution: {e}")
        import traceback
        traceback.print_exc()

async def test_distributed_genetic_algorithm():
    """Test the distributed genetic algorithm"""
    print("\n" + "=" * 60)
    print("TESTING DISTRIBUTED GENETIC ALGORITHM")
    print("=" * 60)
    
    try:
        from src.orchestration.distributed_genetic_algorithm import DistributedGeneticAlgorithm, EvolutionaryParameters
        
        # Simple fitness function for testing
        def test_fitness_function(genome):
            """Simple test fitness: sum of genome values"""
            return sum(genome)
        
        params = EvolutionaryParameters(
            population_size=20,
            genome_length=10,
            max_generations=5
        )
        
        print(f"Creating DGA with params: population_size={params.population_size}, generations={params.max_generations}")
        dga = DistributedGeneticAlgorithm("test_agent", test_fitness_function, params)
        
        # Initialize without A2A for testing
        await dga.initialize()
        
        print(f"Initial population size: {len(dga.local_population)}")
        
        # Run a few generations
        for gen in range(3):
            print(f"\nRunning generation {gen + 1}...")
            start_time = time.time()
            
            await dga.run_generation()
            
            gen_time = time.time() - start_time
            
            # Get statistics
            stats = await dga.get_population_statistics()
            print(f"  Generation {gen + 1} completed in {gen_time:.2f}s")
            print(f"  Population size: {stats.get('population_size', 0)}")
            print(f"  Best fitness: {stats.get('best_fitness', 0):.4f}")
            print(f"  Average fitness: {stats.get('average_fitness', 0):.4f}")
            print(f"  Diversity: {stats.get('diversity_score', 0):.4f}")
            
        # Final analysis
        final_stats = await dga.get_population_statistics()
        print(f"\n" + "=" * 40)
        print("DISTRIBUTED GA ANALYSIS")
        print("=" * 40)
        
        if final_stats.get('population_size', 0) > 0:
            print(f"✅ POPULATION MAINTAINED: {final_stats['population_size']} individuals")
        else:
            print(f"❌ POPULATION LOST: {final_stats.get('population_size', 0)} individuals")
            
        if final_stats.get('best_fitness', 0) > 0:
            print(f"✅ FITNESS EVOLUTION: best={final_stats['best_fitness']:.4f}, avg={final_stats['average_fitness']:.4f}")
        else:
            print(f"❌ NO FITNESS: best={final_stats.get('best_fitness', 0):.4f}")
            
        if final_stats.get('diversity_score', 0) > 0:
            print(f"✅ DIVERSITY MAINTAINED: {final_stats['diversity_score']:.4f}")
        else:
            print(f"❌ NO DIVERSITY: {final_stats.get('diversity_score', 0):.4f}")
            
    except Exception as e:
        print(f"❌ ERROR testing distributed GA: {e}")
        import traceback
        traceback.print_exc()

async def test_collaborative_self_improvement():
    """Test collaborative self-improvement system"""
    print("\n" + "=" * 60)
    print("TESTING COLLABORATIVE SELF-IMPROVEMENT")
    print("=" * 60)
    
    try:
        from src.orchestration.collaborative_self_improvement import CollaborativeSelfImprovement, ImprovementType
        
        # Mock orchestrator for testing
        class MockOrchestrator:
            def __init__(self):
                self.agent_id = "test_agent"
                
            async def get_peer_agents(self):
                return [
                    {'id': 'peer1', 'capabilities': ['problem_solving']},
                    {'id': 'peer2', 'capabilities': ['problem_solving', 'code_generation']}
                ]
        
        orchestrator = MockOrchestrator()
        csi = CollaborativeSelfImprovement(orchestrator)
        
        # Test improvement session
        session_id = await csi.start_improvement_session()
        print(f"Started improvement session: {session_id}")
        
        # Test problem decomposition
        problem = "Optimize agent task allocation algorithm for better performance"
        print(f"Testing problem decomposition for: {problem}")
        
        task = await csi.collaborative_problem_decomposition(problem)
        print(f"Created collaborative task: {task.id}")
        print(f"  Participants: {task.participants}")
        print(f"  Decomposition steps: {len(task.decomposition)}")
        print(f"  Task assignments: {task.task_assignments}")
        
        # Check if we did real work
        if task.decomposition and len(task.decomposition) > 0:
            print(f"✅ PROBLEM DECOMPOSITION: {len(task.decomposition)} subtasks created")
        else:
            print(f"❌ NO DECOMPOSITION: {len(task.decomposition)} subtasks")
            
        if task.participants and len(task.participants) > 0:
            print(f"✅ COLLABORATION: {len(task.participants)} agents involved")
        else:
            print(f"❌ NO COLLABORATION: {len(task.participants)} agents")
            
    except Exception as e:
        print(f"❌ ERROR testing collaborative self-improvement: {e}")
        import traceback
        traceback.print_exc()

async def analyze_evolution_vs_dgm_principles():
    """Analyze how our evolution aligns with DGM principles"""
    print("\n" + "=" * 60)
    print("ANALYSIS: PYGENT VS SAKANA AI DARWIN GÖDEL MACHINE")
    print("=" * 60)
    
    dgm_principles = {
        "Self-modification": "DGM rewrites its own code to improve performance",
        "Empirical validation": "Improvements are tested on real benchmarks (SWE-bench, Polyglot)",
        "Open-ended exploration": "Archive of diverse agents enables parallel evolutionary paths",
        "Performance improvement": "DGM improved from 20% to 50% on SWE-bench",
        "Transferable improvements": "Improvements work across different models and languages",
        "Traceable lineage": "Every change is tracked and can be audited"
    }
    
    pygent_current = {
        "Self-modification": "❓ Recipes modify steps/parameters, but not core code",
        "Empirical validation": "❓ Fitness functions exist but may not run real benchmarks",
        "Open-ended exploration": "✅ Population-based evolution with diversity",
        "Performance improvement": "❓ Improvements tracked but effectiveness unclear",
        "Transferable improvements": "❓ No clear transfer mechanism between contexts",
        "Traceable lineage": "✅ Parent tracking and generation history"
    }
    
    print("COMPARISON:")
    for principle, dgm_desc in dgm_principles.items():
        pygent_status = pygent_current.get(principle, "❌ Not implemented")
        print(f"\n{principle}:")
        print(f"  DGM: {dgm_desc}")
        print(f"  PyGent: {pygent_status}")
    
    print("\n" + "=" * 40)
    print("RECOMMENDATIONS FOR REAL DGM-STYLE EVOLUTION:")
    print("=" * 40)
    print("1. Implement real code modification (not just parameter tuning)")
    print("2. Add benchmark-based fitness evaluation (SWE-bench style)")
    print("3. Create agent performance archive with lineage tracking")
    print("4. Implement cross-context improvement transfer")
    print("5. Add safety mechanisms for self-modification")
    print("6. Create empirical validation pipeline")

async def main():
    """Main test function"""
    print("PYGENT FACTORY EVOLUTION EFFECTIVENESS TEST")
    print("Comparing our system to Sakana AI Darwin Gödel Machine principles")
    print("=" * 80)
    
    await test_recipe_evolution()
    await test_distributed_genetic_algorithm()
    await test_collaborative_self_improvement()
    await analyze_evolution_vs_dgm_principles()
    
    print("\n" + "=" * 80)
    print("EVOLUTION EFFECTIVENESS TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
