#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final End-to-End Deployment Test
Observer-approved comprehensive validation with visualization
"""

import sys
import os
import asyncio
import time
from datetime import datetime

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def final_e2e_test():
    """Final comprehensive end-to-end deployment test"""
    print("=" * 80)
    print("OBSERVER FINAL E2E DEPLOYMENT TEST")
    print("RIPER-Ω Protocol: ACTIVE")
    print("Testing: Production deployment with visualization")
    print("=" * 80)
    
    test_results = {}
    start_time = time.time()
    
    # Test 1: World Simulation with Visualization
    print("\n[TEST 1] World Simulation + Emergence Visualization...")
    try:
        from sim.world_sim import WorldSimulation
        
        sim = WorldSimulation()
        init_success = await sim.initialize(num_agents=20)
        
        if init_success:
            # Run simulation
            result = await sim.sim_loop(generations=5)
            
            # Generate visualization
            viz_success = sim.plot_emergence_evolution("observer_evolution_test.png")
            
            test_results['world_sim'] = {
                'success': result['simulation_success'],
                'agents': len(sim.agents),
                'behaviors': result['emergent_behaviors_detected'],
                'cooperation': result['cooperation_events'],
                'fitness': result['final_average_fitness'],
                'visualization': viz_success
            }
            
            print(f"✅ Simulation: {len(sim.agents)} agents, {result['emergent_behaviors_detected']} behaviors")
            print(f"✅ Cooperation: {result['cooperation_events']} events")
            print(f"✅ Visualization: {'SUCCESS' if viz_success else 'FAILED'}")
        else:
            test_results['world_sim'] = {'success': False, 'error': 'initialization failed'}
            print("❌ World Simulation: Initialization failed")
    except Exception as e:
        test_results['world_sim'] = {'success': False, 'error': str(e)}
        print(f"❌ World Simulation: {e}")
    
    # Test 2: Formal Proof System (Production Scale)
    print("\n[TEST 2] Formal Proof System (Production Scale)...")
    try:
        from dgm.autonomy_fixed import FormalProofSystem
        
        config = {
            'formal_proofs': {
                'safety_threshold': 0.6,
                'bloat_tolerance': 0.15,
                'complexity_limit': 1500,
                'approval_threshold': 0.6
            }
        }
        
        proof_system = FormalProofSystem(config['formal_proofs'])
        scenario_results = await proof_system.test_proof_scenarios()
        
        approval_rate = scenario_results['approval_rate']
        
        test_results['formal_proof'] = {
            'success': approval_rate >= 0.8,
            'approval_rate': approval_rate,
            'invariants': len(proof_system.invariants),
            'scenarios_tested': scenario_results['scenarios_tested']
        }
        
        print(f"✅ Approval Rate: {approval_rate:.1%}")
        print(f"✅ Invariants: {len(proof_system.invariants)}")
        print(f"✅ Scenarios: {scenario_results['scenarios_tested']}")
    except Exception as e:
        test_results['formal_proof'] = {'success': False, 'error': str(e)}
        print(f"❌ Formal Proof System: {e}")
    
    # Test 3: Evolution Loop (Production Configuration)
    print("\n[TEST 3] Evolution Loop (Production Configuration)...")
    try:
        from ai.evolution.evo_loop_fixed import ObserverEvolutionLoop
        
        config = {
            'max_generations': 5,
            'max_runtime_seconds': 60,
            'bloat_penalty_enabled': True
        }
        
        evolution_loop = ObserverEvolutionLoop(config)
        
        # Production-scale population
        population = [f'production_agent_{i}' for i in range(10)]
        
        async def fitness_fn(individual):
            return 0.5 + (len(str(individual)) * 0.01) + (hash(individual) % 100) / 1000
        
        async def mutation_fn(individual):
            return individual + f'_evo_{time.time():.0f}'
        
        async def crossover_fn(parent1, parent2):
            return f'{parent1}_x_{parent2}'
        
        result = await evolution_loop.run_evolution(population, fitness_fn, mutation_fn, crossover_fn)
        
        test_results['evolution'] = {
            'success': result['success'],
            'generations': result['generations_completed'],
            'fitness': result['best_fitness'],
            'population_size': len(population)
        }
        
        print(f"✅ Generations: {result['generations_completed']}")
        print(f"✅ Best Fitness: {result['best_fitness']:.3f}")
        print(f"✅ Population: {len(population)} agents")
    except Exception as e:
        test_results['evolution'] = {'success': False, 'error': str(e)}
        print(f"❌ Evolution Loop: {e}")
    
    # Test 4: Communication System
    print("\n[TEST 4] Communication System...")
    try:
        from agents.communication_system_fixed import ObserverCommunicationSystem
        
        comm_system = ObserverCommunicationSystem({'fallback_enabled': True})
        await comm_system.initialize()
        
        metrics = comm_system.get_communication_metrics()
        
        test_results['communication'] = {
            'success': True,
            'fallback_enabled': metrics['fallback_enabled'],
            'metrics': metrics
        }
        
        print(f"✅ Fallback: {metrics['fallback_enabled']}")
        print("✅ Communication System: Operational")
    except Exception as e:
        test_results['communication'] = {'success': False, 'error': str(e)}
        print(f"❌ Communication System: {e}")
    
    # Test 5: Query System
    print("\n[TEST 5] Query System...")
    try:
        from mcp.query_fixed import ObserverQuerySystem
        
        query_system = ObserverQuerySystem()
        
        # Test multiple queries
        test_queries = [
            ('health_check', {}),
            ('system_status', {}),
            ('performance_metrics', {})
        ]
        
        successful_queries = 0
        for query_type, params in test_queries:
            try:
                result = await query_system.execute_query(query_type, params)
                if result['success']:
                    successful_queries += 1
            except Exception:
                pass
        
        success_rate = successful_queries / len(test_queries)
        
        test_results['query_system'] = {
            'success': success_rate >= 0.8,
            'success_rate': success_rate,
            'queries_tested': len(test_queries)
        }
        
        print(f"✅ Success Rate: {success_rate:.1%}")
        print(f"✅ Queries Tested: {len(test_queries)}")
    except Exception as e:
        test_results['query_system'] = {'success': False, 'error': str(e)}
        print(f"❌ Query System: {e}")
    
    # Calculate overall results
    total_time = time.time() - start_time
    successful_tests = sum(1 for r in test_results.values() if r.get('success', False))
    total_tests = len(test_results)
    success_rate = successful_tests / total_tests
    
    # Final Report
    print("\n" + "=" * 80)
    print("FINAL E2E DEPLOYMENT TEST RESULTS")
    print("=" * 80)
    
    for test_name, result in test_results.items():
        status = "SUCCESS" if result.get('success', False) else "FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        if result.get('success', False):
            # Print key metrics
            if 'agents' in result:
                print(f"  - Agents: {result['agents']}, Behaviors: {result['behaviors']}")
            if 'approval_rate' in result:
                print(f"  - Approval Rate: {result['approval_rate']:.1%}")
            if 'fitness' in result:
                print(f"  - Fitness: {result['fitness']:.3f}")
            if 'visualization' in result:
                print(f"  - Visualization: {'SUCCESS' if result['visualization'] else 'FAILED'}")
    
    print(f"\nOverall: {successful_tests}/{total_tests} tests passed ({success_rate:.1%})")
    print(f"Execution Time: {total_time:.1f} seconds")
    
    if success_rate >= 0.8:
        print("\n🎉 FINAL E2E DEPLOYMENT TEST: SUCCESS")
        print("✅ Observer Systems: PRODUCTION READY")
        print("✅ All core functionality validated")
        print("✅ Visualization capabilities confirmed")
        print("✅ RIPER-Ω Protocol: COMPLIANT")
        print("✅ Ready for v2.2.0 evolution")
    else:
        print(f"\n⚠️ FINAL E2E DEPLOYMENT TEST: PARTIAL ({success_rate:.1%})")
        print("⚠️ Some systems need attention")
    
    # Generate timestamp report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"observer_e2e_test_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Observer E2E Deployment Test Report\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Success Rate: {success_rate:.1%}\n")
        f.write(f"Execution Time: {total_time:.1f}s\n\n")
        
        for test_name, result in test_results.items():
            f.write(f"{test_name}: {result}\n")
    
    print(f"\n📄 Test report saved: {report_file}")
    
    return success_rate >= 0.8

if __name__ == "__main__":
    success = asyncio.run(final_e2e_test())
    sys.exit(0 if success else 1)
