#!/usr/bin/env python3
"""Test Phase 3 World Simulation - Observer-approved validation"""

import asyncio
import sys
sys.path.append('.')

async def test_world_simulation():
    print("🌍 PHASE 3 WORLD SIMULATION TEST")
    
    try:
        # Test the standalone sim_loop function
        from src.sim.world_sim import sim_loop
        
        print("Running 10-generation simulation...")
        results = sim_loop(generations=10)
        
        print(f"\n🎯 SIMULATION RESULTS:")
        print(f"Generations completed: {results.get('generations', 0)}")
        print(f"Agents count: {results.get('agents_count', 0)}")
        print(f"Emergence detected: {results.get('emergence_detected', False)}")
        
        if results.get('emergence_detected'):
            print(f"Emergence generation: {results.get('emergence_generation', 'Unknown')}")
        
        final_fitness = results.get('final_agent_fitness', [])
        if final_fitness:
            total_fitness = sum(final_fitness)
            avg_fitness = total_fitness / len(final_fitness)
            print(f"Final fitness sum: {total_fitness:.2f}")
            print(f"Average fitness: {avg_fitness:.2f}")
            print(f"Fitness > 2.0 threshold: {avg_fitness > 2.0}")
        
        # Check visualization
        viz_status = results.get('visualization', {}).get('status', 'unknown')
        print(f"Visualization: {viz_status}")
        
        # Test success criteria
        success_criteria = {
            'generations_completed': results.get('generations', 0) >= 10,
            'agents_present': results.get('agents_count', 0) >= 10,
            'fitness_calculated': len(final_fitness) > 0,
            'emergence_system': 'emergence_detected' in results,
            'visualization_attempted': 'visualization' in results
        }
        
        print(f"\n✅ SUCCESS CRITERIA:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {criterion}: {status}")
        
        overall_success = all(success_criteria.values())
        print(f"\n🎯 OVERALL TEST: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_world_simulation())
    print(f"\nPhase 3 test result: {'SUCCESS' if result else 'FAILED'}")
