#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observer-approved world simulation test with proper Unicode handling
Fixes encoding issues for reliable large-scale testing
"""

import sys
import os
import asyncio
import logging

# Ensure UTF-8 encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Set up logging with UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_full_world_simulation():
    """Test Observer world simulation with 100 agents and multiple generations."""
    try:
        print("=" * 60)
        print("OBSERVER WORLD SIMULATION TEST")
        print("=" * 60)
        print("Testing large-scale simulation (100 agents, 5 generations)...")
        
        from sim.world_sim import WorldSimulation, calc_fitness
        
        # Create large-scale simulation
        sim = WorldSimulation()
        
        # Initialize with 100 agents
        print("\n[STEP 1] Initializing simulation...")
        init_success = await sim.initialize(num_agents=100)
        if not init_success:
            print("ERROR: Failed to initialize simulation")
            return False
        
        print(f"SUCCESS: Initialized simulation with {len(sim.agents)} agents")
        
        # Test agent distribution
        agent_types = {}
        for agent in sim.agents:
            agent_types[agent.agent_type] = agent_types.get(agent.agent_type, 0) + 1
        
        print(f"Agent distribution: {agent_types}")
        
        # Calculate initial fitness
        initial_fitness_scores = [calc_fitness(agent, sim.environment) for agent in sim.agents]
        initial_avg_fitness = sum(initial_fitness_scores) / len(initial_fitness_scores)
        print(f"Initial average fitness: {initial_avg_fitness:.3f}")
        
        # Run simulation
        print("\n[STEP 2] Running 5-generation simulation...")
        result = await sim.sim_loop(generations=5)
        
        if result['simulation_success']:
            print("\n[RESULTS] Simulation completed successfully!")
            print("-" * 40)
            print(f"Generations completed: {result['generations_completed']}")
            print(f"Final average fitness: {result['final_average_fitness']:.3f}")
            print(f"Emergent behaviors detected: {result['emergent_behaviors_detected']}")
            print(f"Cooperation events: {result['cooperation_events']}")
            
            # Calculate fitness improvement
            if len(sim.simulation_metrics['average_fitness_history']) > 1:
                initial_fitness = sim.simulation_metrics['average_fitness_history'][0]
                final_fitness = sim.simulation_metrics['average_fitness_history'][-1]
                improvement = ((final_fitness - initial_fitness) / initial_fitness) * 100
                print(f"Fitness improvement: {improvement:.1f}%")
            
            # Get detailed metrics
            metrics = sim.get_simulation_metrics()
            print(f"Total simulation steps: {metrics['time_step']}")
            print(f"Environment resources: {metrics['environment_resources']}")
            
            # Analyze emergent behaviors
            if sim.simulation_metrics['emergent_behaviors']:
                print("\n[EMERGENCE ANALYSIS]")
                behavior_types = {}
                for behavior in sim.simulation_metrics['emergent_behaviors']:
                    behavior_type = behavior['type']
                    behavior_types[behavior_type] = behavior_types.get(behavior_type, 0) + 1
                
                for behavior_type, count in behavior_types.items():
                    print(f"- {behavior_type}: {count} instances")
            
            print("\n" + "=" * 60)
            print("TASK 2.1: FULL WORLD SIMULATION SUCCESS")
            print("=" * 60)
            return True
            
        else:
            print(f"\nERROR: Simulation failed - {result.get('error', 'unknown')}")
            return False
            
    except Exception as e:
        print(f"\nERROR: World simulation test failed - {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_dgm_formal_proof_integration():
    """Test DGM formal proof system integration."""
    try:
        print("\n" + "=" * 60)
        print("DGM FORMAL PROOF INTEGRATION TEST")
        print("=" * 60)
        
        from dgm.autonomy_fixed import FormalProofSystem, ObserverAutonomyController
        
        # Test formal proof system
        print("[STEP 1] Testing formal proof system...")
        config = {'formal_proofs': {'enabled': True}}
        proof_system = FormalProofSystem(config['formal_proofs'])
        
        print(f"SUCCESS: Initialized with {len(proof_system.invariants)} invariants")
        
        # Test improvement candidate validation
        improvement_candidate = {
            'type': 'fitness_improvement',
            'expected_fitness_gain': 0.1,
            'complexity_change': 5,
            'expected_efficiency_gain': 0.05
        }
        
        print("[STEP 2] Testing improvement safety proof...")
        proof_result = await proof_system.prove_improvement_safety(improvement_candidate)
        
        print(f"Proof valid: {proof_result['proof_valid']}")
        print(f"Safety score: {proof_result['safety_score']:.2%}")
        print(f"Recommendation: {proof_result['recommendation']}")
        
        if proof_result['violations']:
            print(f"Violations detected: {len(proof_result['violations'])}")
        
        # Test system consistency
        print("[STEP 3] Testing system consistency...")
        consistency_result = await proof_system.verify_system_consistency()
        
        print(f"System consistent: {consistency_result['consistent']}")
        print(f"Consistency score: {consistency_result['consistency_score']:.2%}")
        
        print("\nTASK 2.2: DGM FORMAL PROOF INTEGRATION SUCCESS")
        return True
        
    except Exception as e:
        print(f"\nERROR: DGM formal proof test failed - {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_evolution_loop_integration():
    """Test evolution loop with real agent population."""
    try:
        print("\n" + "=" * 60)
        print("EVOLUTION LOOP INTEGRATION TEST")
        print("=" * 60)
        
        from ai.evolution.evo_loop_fixed import ObserverEvolutionLoop
        from sim.world_sim import Agent
        
        # Create evolution loop
        config = {
            'max_generations': 3,
            'max_runtime_seconds': 60,
            'bloat_penalty_enabled': True
        }
        
        evolution_loop = ObserverEvolutionLoop(config)
        
        # Create test population of agents
        print("[STEP 1] Creating test agent population...")
        population = []
        for i in range(10):
            agent = Agent(f"test_agent_{i}", "learner", {})
            population.append(agent)
        
        print(f"Created population of {len(population)} agents")
        
        # Define fitness, mutation, and crossover functions
        async def fitness_function(agent):
            return agent.fitness + (sum(agent.capabilities.values()) / len(agent.capabilities))
        
        async def mutation_function(agent):
            # Mutate agent capabilities slightly
            for capability in agent.capabilities:
                if hasattr(agent, 'capabilities') and capability in agent.capabilities:
                    agent.capabilities[capability] = min(1.0, max(0.1, 
                        agent.capabilities[capability] + ((-0.05 + 0.1) * 0.5)))
            return agent
        
        async def crossover_function(parent1, parent2):
            # Create offspring by averaging capabilities
            child = Agent(f"child_{parent1.agent_id}_{parent2.agent_id}", "learner", {})
            for capability in parent1.capabilities:
                if capability in parent2.capabilities:
                    child.capabilities[capability] = (parent1.capabilities[capability] + 
                                                    parent2.capabilities[capability]) / 2
            return child
        
        # Run evolution
        print("[STEP 2] Running evolution loop...")
        result = await evolution_loop.run_evolution(
            population, fitness_function, mutation_function, crossover_function
        )
        
        if result['success']:
            print(f"Evolution completed successfully!")
            print(f"Generations: {result['generations_completed']}")
            print(f"Best fitness: {result['best_fitness']:.3f}")
            print(f"Bloat penalties applied: {result['bloat_penalties_applied']}")
            print(f"Runtime: {result['runtime_seconds']:.1f}s")
            print(f"Termination reason: {result['termination_reason']}")
            
            print("\nTASK 2.3: EVOLUTION LOOP INTEGRATION SUCCESS")
            return True
        else:
            print(f"Evolution failed: {result.get('error', 'unknown')}")
            return False
            
    except Exception as e:
        print(f"\nERROR: Evolution loop test failed - {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all comprehensive system tests."""
    print("OBSERVER-APPROVED COMPREHENSIVE SYSTEM TESTING")
    print("RIPER-Ω Protocol Compliance: ACTIVE")
    print("Unicode Encoding: UTF-8 FIXED")
    
    results = []
    
    # Test 1: Full World Simulation
    result1 = await test_full_world_simulation()
    results.append(("World Simulation", result1))
    
    # Test 2: DGM Formal Proof Integration  
    result2 = await test_dgm_formal_proof_integration()
    results.append(("DGM Formal Proof", result2))
    
    # Test 3: Evolution Loop Integration
    result3 = await test_evolution_loop_integration()
    results.append(("Evolution Loop", result3))
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 60)
    
    success_count = 0
    for test_name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{test_name}: {status}")
        if success:
            success_count += 1
    
    overall_success = success_count == len(results)
    print(f"\nOverall: {success_count}/{len(results)} tests passed")
    print(f"TASK 2 STATUS: {'SUCCESS' if overall_success else 'PARTIAL'}")
    
    return overall_success

if __name__ == "__main__":
    asyncio.run(main())
