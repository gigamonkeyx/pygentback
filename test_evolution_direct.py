#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct test of evolution loop without import chain issues
"""

import sys
import os
import asyncio

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_evolution_loop_direct():
    """Test evolution loop by importing directly."""
    try:
        print("Testing Observer Evolution Loop (direct import)...")
        
        # Import directly from the fixed module
        from ai.evolution.evo_loop_fixed import ObserverEvolutionLoop
        print("✅ Evolution Loop: IMPORTED")
        
        # Create evolution loop
        config = {
            'max_generations': 3,
            'max_runtime_seconds': 30,
            'bloat_penalty_enabled': True
        }
        
        evolution_loop = ObserverEvolutionLoop(config)
        print("✅ Evolution Loop: INITIALIZED")
        
        # Create simple test population
        population = ['agent1', 'agent2', 'agent3', 'agent4', 'agent5']
        
        # Define simple test functions
        async def fitness_function(individual):
            return 0.5 + (len(str(individual)) * 0.01)
        
        async def mutation_function(individual):
            return individual + '_mutated'
        
        async def crossover_function(parent1, parent2):
            return f'{parent1}_{parent2}'
        
        # Run evolution
        print("Running evolution loop...")
        result = await evolution_loop.run_evolution(
            population, fitness_function, mutation_function, crossover_function
        )
        
        if result['success']:
            print(f"✅ Evolution completed successfully!")
            print(f"Generations: {result['generations_completed']}")
            print(f"Best fitness: {result['best_fitness']:.3f}")
            print(f"Runtime: {result['runtime_seconds']:.1f}s")
            print(f"Termination reason: {result['termination_reason']}")
            return True
        else:
            print(f"❌ Evolution failed: {result.get('error', 'unknown')}")
            return False
            
    except Exception as e:
        print(f"❌ Evolution loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_world_simulation_direct():
    """Test world simulation directly."""
    try:
        print("\nTesting Observer World Simulation (direct)...")
        
        from sim.world_sim import WorldSimulation
        print("✅ World Simulation: IMPORTED")
        
        # Create simulation
        sim = WorldSimulation()
        
        # Initialize with 20 agents (medium scale)
        init_success = await sim.initialize(num_agents=20)
        if not init_success:
            print("❌ Failed to initialize simulation")
            return False
        
        print(f"✅ Initialized with {len(sim.agents)} agents")
        
        # Run short simulation
        result = await sim.sim_loop(generations=3)
        
        if result['simulation_success']:
            print(f"✅ Simulation completed!")
            print(f"Generations: {result['generations_completed']}")
            print(f"Final fitness: {result['final_average_fitness']:.3f}")
            print(f"Emergent behaviors: {result['emergent_behaviors_detected']}")
            print(f"Cooperation events: {result['cooperation_events']}")
            return True
        else:
            print(f"❌ Simulation failed: {result.get('error', 'unknown')}")
            return False
            
    except Exception as e:
        print(f"❌ World simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run direct tests to bypass import issues."""
    print("=" * 60)
    print("DIRECT OBSERVER SYSTEM TESTING")
    print("Unicode Encoding: FIXED")
    print("Import Method: DIRECT")
    print("=" * 60)
    
    # Test evolution loop directly
    evolution_success = await test_evolution_loop_direct()
    
    # Test world simulation directly
    simulation_success = await test_world_simulation_direct()
    
    # Summary
    print("\n" + "=" * 60)
    print("DIRECT TEST RESULTS:")
    print(f"Evolution Loop: {'SUCCESS' if evolution_success else 'FAILED'}")
    print(f"World Simulation: {'SUCCESS' if simulation_success else 'FAILED'}")
    
    overall_success = evolution_success and simulation_success
    print(f"Overall: {'SUCCESS' if overall_success else 'PARTIAL'}")
    
    if overall_success:
        print("\n✅ Unicode issue FIXED")
        print("✅ Observer systems FUNCTIONAL")
        print("✅ Ready for large-scale testing")
    
    return overall_success

if __name__ == "__main__":
    asyncio.run(main())
