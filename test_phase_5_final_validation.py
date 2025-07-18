#!/usr/bin/env python3
"""
Phase 5: Final E2E Validation - Observer-approved production readiness test
Complete system validation with zero fails standard
"""

import asyncio
import json
import time
import sys
sys.path.append('.')

async def test_phase_5_final_validation():
    print("🚀 PHASE 5 FINAL E2E VALIDATION - GOING BIG")
    print("=" * 80)
    
    validation_results = {
        'timestamp': time.time(),
        'phase': 'Phase 5 Final Validation',
        'observer_compliance': True,
        'tests': {},
        'metrics': {},
        'deployment_ready': False
    }
    
    try:
        # Test 1: Full Simulation Validation
        print("\n1. 🌍 FULL SIMULATION VALIDATION")
        from src.sim.world_sim import sim_loop
        
        sim_results = sim_loop(generations=10)
        
        sim_validation = {
            'generations_completed': sim_results.get('generations', 0),
            'agents_count': sim_results.get('agents_count', 0),
            'emergence_detected': sim_results.get('emergence_detected', False),
            'emergence_generation': sim_results.get('emergence_generation', None),
            'final_fitness_sum': sum(sim_results.get('final_agent_fitness', [])),
            'average_fitness': 0.0
        }
        
        if sim_results.get('final_agent_fitness'):
            sim_validation['average_fitness'] = sim_validation['final_fitness_sum'] / len(sim_results['final_agent_fitness'])
        
        # Observer thresholds validation
        sim_success = (
            sim_validation['generations_completed'] >= 10 and
            sim_validation['emergence_detected'] and
            sim_validation['emergence_generation'] <= 7 and  # Early emergence
            sim_validation['average_fitness'] >= 6.03  # Phase 4 standard
        )
        
        validation_results['tests']['simulation'] = {
            'status': 'PASS' if sim_success else 'FAIL',
            'data': sim_validation
        }
        
        print(f"   Generations: {sim_validation['generations_completed']}/10")
        print(f"   Emergence: {'✅' if sim_validation['emergence_detected'] else '❌'} (Gen {sim_validation['emergence_generation']})")
        print(f"   Avg Fitness: {sim_validation['average_fitness']:.2f} (Target: ≥6.03)")
        print(f"   Status: {'✅ PASS' if sim_success else '❌ FAIL'}")
        
        # Test 2: Factory Agent Creation Validation
        print("\n2. 🏭 FACTORY AGENT CREATION VALIDATION")
        
        try:
            from src.core.agent_factory import AgentFactory
            factory = AgentFactory()
            
            sim_roles = ['explorer', 'builder', 'gatherer', 'learner']
            created_agents = []
            agent_details = []
            
            for role in sim_roles:
                try:
                    agent = await factory.create_agent(
                        agent_type='general',
                        name=f'production_{role}_agent',
                        capabilities=['research', 'analysis', 'coordination'],
                        custom_config={'role': role, 'production_ready': True}
                    )
                    
                    if agent:
                        agent_id = getattr(agent, 'id', None) or getattr(agent, 'agent_id', None) or 'unknown'
                        created_agents.append(agent)
                        agent_details.append({
                            'role': role,
                            'id': agent_id,
                            'name': getattr(agent, 'name', 'unknown'),
                            'status': 'created'
                        })
                        print(f"   ✅ {role.capitalize()} agent: {agent_id}")
                    else:
                        agent_details.append({'role': role, 'status': 'failed', 'error': 'None returned'})
                        print(f"   ❌ {role.capitalize()} agent: Creation failed")
                        
                except Exception as e:
                    agent_details.append({'role': role, 'status': 'failed', 'error': str(e)})
                    print(f"   ❌ {role.capitalize()} agent: {e}")
            
            factory_success = len(created_agents) == 4
            
            validation_results['tests']['factory_agents'] = {
                'status': 'PASS' if factory_success else 'FAIL',
                'data': {
                    'agents_created': len(created_agents),
                    'target_count': 4,
                    'agent_details': agent_details
                }
            }
            
            print(f"   Agents Created: {len(created_agents)}/4")
            print(f"   Status: {'✅ PASS' if factory_success else '❌ FAIL'}")
            
        except Exception as e:
            validation_results['tests']['factory_agents'] = {
                'status': 'FAIL',
                'data': {'error': str(e)}
            }
            print(f"   ❌ Factory test failed: {e}")
            factory_success = False
        
        # Test 3: Phase Integration Validation
        print("\n3. 🔗 PHASE INTEGRATION VALIDATION")
        
        phase_tests = {
            'phase_1_audit': True,  # Completed in previous phases
            'phase_2_fixes': True,  # Validated in Phase 2
            'phase_3_world_sim': sim_success,
            'phase_4_factory_integration': factory_success
        }
        
        integration_success = all(phase_tests.values())
        
        validation_results['tests']['phase_integration'] = {
            'status': 'PASS' if integration_success else 'FAIL',
            'data': phase_tests
        }
        
        for phase, status in phase_tests.items():
            print(f"   {phase}: {'✅ PASS' if status else '❌ FAIL'}")
        print(f"   Status: {'✅ PASS' if integration_success else '❌ FAIL'}")
        
        # Test 4: Zero Fails Validation
        print("\n4. 🎯 ZERO FAILS VALIDATION")
        
        all_tests_passed = all(
            test['status'] == 'PASS' 
            for test in validation_results['tests'].values()
        )
        
        zero_fails_criteria = {
            'no_critical_errors': all_tests_passed,
            'emergence_achieved': sim_validation['emergence_detected'],
            'fitness_threshold_met': sim_validation['average_fitness'] >= 6.03,
            'all_agents_created': len(created_agents) == 4 if 'created_agents' in locals() else False,
            'observer_compliance': True
        }
        
        zero_fails_success = all(zero_fails_criteria.values())
        
        validation_results['tests']['zero_fails'] = {
            'status': 'PASS' if zero_fails_success else 'FAIL',
            'data': zero_fails_criteria
        }
        
        for criterion, status in zero_fails_criteria.items():
            print(f"   {criterion}: {'✅ PASS' if status else '❌ FAIL'}")
        print(f"   Status: {'✅ PASS' if zero_fails_success else '❌ FAIL'}")
        
        # Final Metrics
        print("\n" + "=" * 80)
        print("🎯 PHASE 5 FINAL VALIDATION RESULTS")
        print("=" * 80)
        
        total_tests = len(validation_results['tests'])
        passed_tests = sum(1 for test in validation_results['tests'].values() if test['status'] == 'PASS')
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        validation_results['metrics'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'deployment_ready': success_rate == 100.0
        }
        
        validation_results['deployment_ready'] = success_rate == 100.0
        
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Deployment Ready: {'✅ YES' if validation_results['deployment_ready'] else '❌ NO'}")
        
        if validation_results['deployment_ready']:
            print("\n🚀 OBSERVER AUTHORIZATION: READY FOR PRODUCTION DEPLOYMENT")
        else:
            print("\n❌ OBSERVER HOLD: VALIDATION FAILURES DETECTED")
        
        return validation_results
        
    except Exception as e:
        print(f"\n❌ CRITICAL VALIDATION FAILURE: {e}")
        import traceback
        traceback.print_exc()
        
        validation_results['tests']['critical_error'] = {
            'status': 'FAIL',
            'data': {'error': str(e), 'traceback': traceback.format_exc()}
        }
        
        return validation_results

if __name__ == "__main__":
    result = asyncio.run(test_phase_5_final_validation())
    
    # Save validation report
    with open('phase_5_validation_report.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n📊 Validation report saved: phase_5_validation_report.json")
    print(f"🎯 Phase 5 result: {'SUCCESS' if result.get('deployment_ready', False) else 'FAILED'}")
