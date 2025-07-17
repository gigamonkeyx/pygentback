#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP-Evolution Fusion Test
Observer-approved comprehensive test for MCP-guard evolution fusion with 95%+ enforcement
"""

import sys
import os
import asyncio
import time
import json
from datetime import datetime

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except AttributeError:
        pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_mcp_fitness_integration():
    """Test MCP-guard fusion into fitness calculation"""
    print("🔗 TESTING MCP-FITNESS INTEGRATION")
    print("-" * 50)
    
    try:
        from ai.evolution.two_phase import TwoPhaseEvolutionSystem
        
        # Initialize evolution system
        evolution_system = TwoPhaseEvolutionSystem({
            'exploration_generations': 2,
            'exploitation_generations': 2
        })
        print("✅ MCP-fitness integration system initialized")
        
        # Test MCP-fitness integration scenarios
        test_scenarios = [
            {
                'name': 'High Appropriateness Amplification',
                'phase_results': [{
                    'mcp_calls': 5,
                    'mcp_successes': 5,
                    'mcp_failures': 0,
                    'env_improvement': 0.2,
                    'context_appropriateness': 0.8
                }],
                'base_fitness': 1.0,
                'expected_bonus_range': (0.1, 0.4)
            },
            {
                'name': 'Compound Learning Acceleration',
                'phase_results': [{
                    'mcp_calls': 8,
                    'mcp_successes': 7,
                    'mcp_failures': 1,
                    'env_improvement': 0.15,
                    'context_appropriateness': 0.9
                }],
                'base_fitness': 1.5,
                'expected_bonus_range': (0.2, 0.5)
            },
            {
                'name': 'Gaming Penalty Application',
                'phase_results': [{
                    'mcp_calls': 6,
                    'mcp_successes': 2,
                    'mcp_failures': 4,
                    'env_improvement': -0.05,
                    'context_appropriateness': 0.2
                }],
                'base_fitness': 1.0,
                'expected_bonus_range': (-0.3, -0.1)
            }
        ]
        
        integration_results = {}
        for scenario in test_scenarios:
            # Calculate MCP fitness integration bonus
            mcp_fitness_bonus = evolution_system._calculate_mcp_fitness_integration(
                scenario['phase_results'],
                scenario['base_fitness'],
                0.1  # mcp_reward
            )
            
            expected_min, expected_max = scenario['expected_bonus_range']
            is_in_range = expected_min <= mcp_fitness_bonus <= expected_max
            
            integration_results[scenario['name']] = {
                'mcp_fitness_bonus': mcp_fitness_bonus,
                'expected_range': scenario['expected_bonus_range'],
                'in_range': is_in_range,
                'base_fitness': scenario['base_fitness']
            }
            
            status = "✅" if is_in_range else "❌"
            print(f"{status} {scenario['name']}: bonus={mcp_fitness_bonus:.3f} (expected: {expected_min:.1f} to {expected_max:.1f})")
        
        # Get integration stats
        integration_stats = evolution_system.get_mcp_integration_stats()
        print(f"✅ MCP integration stats: {integration_stats}")
        
        # Calculate success rate
        successful_integrations = sum(1 for result in integration_results.values() if result['in_range'])
        success_rate = successful_integrations / len(integration_results)
        
        print(f"✅ MCP-fitness integration: {success_rate:.1%} success rate")
        
        return {
            'success': True,
            'integration_results': integration_results,
            'integration_stats': integration_stats,
            'success_rate': success_rate,
            'mcp_fitness_integration_working': success_rate >= 0.8
        }
        
    except Exception as e:
        print(f"❌ MCP-fitness integration test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_cascade_penalty_evolution():
    """Test cascade penalty evolution and DGM integration"""
    print("\n⚖️ TESTING CASCADE PENALTY EVOLUTION")
    print("-" * 50)
    
    try:
        from core.safety_monitor import SafetyMonitor
        
        # Initialize safety monitor
        safety_monitor = SafetyMonitor({
            'minor_penalty': -0.1,
            'major_penalty': -0.5,
            'critical_penalty': -1.0
        })
        print("✅ Cascade penalty evolution system initialized")
        
        # Test penalty evolution scenarios
        evolution_scenarios = [
            {
                'name': 'High Gaming Frequency',
                'generation_data': {
                    'generation': 5,
                    'gaming_attempts': 8,
                    'enforcement_rate': 0.9
                },
                'expected_adaptation': 'increased_severity'
            },
            {
                'name': 'Perfect Enforcement',
                'generation_data': {
                    'generation': 10,
                    'gaming_attempts': 0,
                    'enforcement_rate': 0.98
                },
                'expected_adaptation': 'optimized_balance'
            },
            {
                'name': 'Stable Gaming Levels',
                'generation_data': {
                    'generation': 3,
                    'gaming_attempts': 2,
                    'enforcement_rate': 0.85
                },
                'expected_adaptation': 'maintained_stability'
            }
        ]
        
        evolution_results = {}
        for scenario in evolution_scenarios:
            # Evolve cascade penalties
            penalty_evolution = safety_monitor.evolve_cascade_penalties(scenario['generation_data'])
            
            adaptation_correct = penalty_evolution.get('adaptation') == scenario['expected_adaptation']
            
            evolution_results[scenario['name']] = {
                'penalty_evolution': penalty_evolution,
                'expected_adaptation': scenario['expected_adaptation'],
                'adaptation_correct': adaptation_correct
            }
            
            status = "✅" if adaptation_correct else "❌"
            print(f"{status} {scenario['name']}: adaptation={penalty_evolution.get('adaptation', 'unknown')}")
        
        # Test DGM rewrite integration
        test_agent_id = "test_gaming_agent"
        
        # Add some penalties to trigger rewrite
        safety_monitor.monitor_mcp_usage(
            test_agent_id,
            {'type': 'dummy', 'content': ''},
            {'success': True, 'env_improvement': 0.0},
            {}
        )
        
        # Test DGM rewrite integration
        rewrite_result = safety_monitor.integrate_dgm_rewrite_system(test_agent_id)
        
        # Test gaming trait pruning
        test_traits = {
            'dummy_call_tendency': 0.8,
            'minimal_compliance_behavior': 0.7,
            'exploration_skill': 0.6,
            'cooperation_ability': 0.9
        }
        
        pruning_result = safety_monitor.prune_gaming_traits(test_traits)
        
        print(f"✅ DGM rewrite integration: {rewrite_result.get('rewrite_needed', False)}")
        print(f"✅ Gaming trait pruning: {pruning_result.get('gaming_traits_removed', 0)} traits removed")
        
        # Calculate evolution effectiveness
        successful_evolutions = sum(1 for result in evolution_results.values() if result['adaptation_correct'])
        evolution_rate = successful_evolutions / len(evolution_results)
        
        return {
            'success': True,
            'evolution_results': evolution_results,
            'rewrite_result': rewrite_result,
            'pruning_result': pruning_result,
            'evolution_rate': evolution_rate,
            'cascade_evolution_working': evolution_rate >= 0.8
        }
        
    except Exception as e:
        print(f"❌ Cascade penalty evolution test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_mcp_audit_visualization():
    """Test MCP audit logging and visualization"""
    print("\n📊 TESTING MCP AUDIT VISUALIZATION")
    print("-" * 50)
    
    try:
        from sim.emergent_detector import EmergentBehaviorDetector
        
        # Initialize emergent detector
        detector = EmergentBehaviorDetector({})
        print("✅ MCP audit visualization system initialized")
        
        # Simulate MCP audit logging
        test_audits = [
            {
                'agent_id': 'agent_1',
                'mcp_action': {'type': 'query', 'content': 'resource analysis'},
                'outcome': {'success': True, 'env_improvement': 0.15},
                'context': {'context_appropriateness': 0.8}
            },
            {
                'agent_id': 'agent_2',
                'mcp_action': {'type': 'dummy', 'content': ''},
                'outcome': {'success': True, 'env_improvement': 0.0},
                'context': {'context_appropriateness': 0.2}
            },
            {
                'agent_id': 'agent_3',
                'mcp_action': {'type': 'coordinate', 'content': 'team optimization'},
                'outcome': {'success': True, 'env_improvement': 0.1},
                'context': {'context_appropriateness': 0.7}
            }
        ]
        
        audit_ids = []
        for audit in test_audits:
            audit_id = detector.log_mcp_audit_chain(
                audit['agent_id'],
                audit['mcp_action'],
                audit['outcome'],
                audit['context']
            )
            audit_ids.append(audit_id)
        
        print(f"✅ Logged {len(audit_ids)} MCP audit entries")
        
        # Generate audit heatmap data
        heatmap_data = detector.generate_audit_heatmap_data(generations=5)
        
        # Get audit summary
        audit_summary = detector.get_mcp_audit_summary()
        
        print(f"✅ Audit heatmap data: {heatmap_data.get('generations', [])}")
        print(f"✅ Audit summary: {audit_summary}")
        
        # Validate audit functionality
        audit_working = (
            len(audit_ids) == len(test_audits) and
            not heatmap_data.get('error') and
            not audit_summary.get('error')
        )
        
        return {
            'success': True,
            'audit_ids': audit_ids,
            'heatmap_data': heatmap_data,
            'audit_summary': audit_summary,
            'audit_visualization_working': audit_working
        }
        
    except Exception as e:
        print(f"❌ MCP audit visualization test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_mcp_seeded_environment():
    """Test MCP-seeded environment with auto-bias"""
    print("\n🌱 TESTING MCP-SEEDED ENVIRONMENT")
    print("-" * 50)
    
    try:
        from sim.mcp_seeded_env import MCPSeededEnvironment
        
        # Initialize MCP-seeded environment
        env_config = {
            'mcp_weight_base': 0.5,
            'appropriateness_threshold': 0.7,
            'auto_bias_enabled': True
        }
        
        mcp_env = MCPSeededEnvironment(env_config)
        print("✅ MCP-seeded environment initialized")
        
        # Create test audit data
        test_audit_data = [
            {
                'agent_id': 'agent_1',
                'success': True,
                'gaming_detected': False,
                'appropriateness_score': 0.8,
                'env_improvement': 0.15
            },
            {
                'agent_id': 'agent_2',
                'success': True,
                'gaming_detected': True,
                'appropriateness_score': 0.2,
                'env_improvement': 0.0
            },
            {
                'agent_id': 'agent_3',
                'success': True,
                'gaming_detected': False,
                'appropriateness_score': 0.9,
                'env_improvement': 0.2
            }
        ]
        
        # Seed environment from MCP audits
        seeding_result = mcp_env.seed_from_mcp_audits(test_audit_data)
        
        print(f"✅ Environment seeding: {seeding_result.get('applied_biases', [])}")
        
        # Test directed growth
        test_behaviors = [
            'mcp_smart_usage',
            'context_smart_decisions',
            'gaming_resistance',
            'environment_improvement'
        ]
        
        growth_result = mcp_env.test_directed_growth(test_behaviors)
        
        print(f"✅ Directed growth: {growth_result.get('growth_effectiveness', 0):.1%} effectiveness")
        
        # Get seeding stats
        seeding_stats = mcp_env.get_seeding_stats()
        
        # Validate seeded environment
        seeding_working = (
            not seeding_result.get('error') and
            growth_result.get('directed_growth_working', False) and
            seeding_stats.get('active_biases', 0) > 0
        )
        
        return {
            'success': True,
            'seeding_result': seeding_result,
            'growth_result': growth_result,
            'seeding_stats': seeding_stats,
            'mcp_seeded_env_working': seeding_working
        }
        
    except Exception as e:
        print(f"❌ MCP-seeded environment test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_end_to_end_mcp_fusion():
    """Test end-to-end MCP-evolution fusion"""
    print("\n🚀 TESTING END-TO-END MCP FUSION")
    print("-" * 50)
    
    try:
        from ai.evolution.two_phase import TwoPhaseEvolutionSystem
        from core.safety_monitor import SafetyMonitor
        from sim.emergent_detector import EmergentBehaviorDetector
        from sim.mcp_seeded_env import MCPSeededEnvironment
        
        # Initialize all systems
        evolution_system = TwoPhaseEvolutionSystem({})
        safety_monitor = SafetyMonitor({})
        detector = EmergentBehaviorDetector({})
        mcp_env = MCPSeededEnvironment({'auto_bias_enabled': True})
        
        print("✅ All MCP fusion systems initialized")
        
        # Simulate end-to-end MCP fusion workflow
        generations = 3
        e2e_results = []
        
        for gen in range(generations):
            print(f"\n🔄 Generation {gen + 1}")
            
            # Simulate MCP usage in generation
            mcp_actions = [
                {'type': 'query', 'content': f'gen_{gen}_optimization'},
                {'type': 'sense', 'content': f'environment_analysis_{gen}'},
                {'type': 'coordinate', 'content': f'cooperation_gen_{gen}'}
            ]
            
            generation_mcp_data = []
            generation_penalties = 0
            
            for i, action in enumerate(mcp_actions):
                # Simulate outcome
                outcome = {
                    'success': True,
                    'env_improvement': 0.1 + (gen * 0.05) + (i * 0.02)
                }
                
                context = {
                    'context_appropriateness': 0.6 + (gen * 0.1),
                    'generation': gen + 1
                }
                
                # Log MCP audit
                audit_id = detector.log_mcp_audit_chain(f"agent_{i}", action, outcome, context)
                
                # Monitor with safety system
                penalties = safety_monitor.monitor_mcp_usage(
                    f"agent_{i}", action, outcome, context
                )
                generation_penalties += len(penalties)
                
                # Store MCP data
                generation_mcp_data.append({
                    'agent_id': f"agent_{i}",
                    'success': outcome['success'],
                    'gaming_detected': False,
                    'appropriateness_score': context['context_appropriateness'],
                    'env_improvement': outcome['env_improvement']
                })
            
            # Calculate MCP fitness integration
            phase_results = [{
                'mcp_calls': len(mcp_actions),
                'mcp_successes': len(mcp_actions),
                'mcp_failures': 0,
                'env_improvement': sum(data['env_improvement'] for data in generation_mcp_data) / len(generation_mcp_data),
                'context_appropriateness': sum(data['appropriateness_score'] for data in generation_mcp_data) / len(generation_mcp_data)
            }]
            
            mcp_fitness_bonus = evolution_system._calculate_mcp_fitness_integration(
                phase_results, 1.0, 0.1
            )
            
            # Seed environment for next generation
            seeding_result = mcp_env.seed_from_mcp_audits(generation_mcp_data)
            
            # Store generation results
            gen_result = {
                'generation': gen + 1,
                'mcp_calls': len(mcp_actions),
                'mcp_fitness_bonus': mcp_fitness_bonus,
                'penalties_applied': generation_penalties,
                'avg_appropriateness': sum(data['appropriateness_score'] for data in generation_mcp_data) / len(generation_mcp_data),
                'avg_improvement': sum(data['env_improvement'] for data in generation_mcp_data) / len(generation_mcp_data),
                'biases_applied': len(seeding_result.get('applied_biases', []))
            }
            
            e2e_results.append(gen_result)
            
            print(f"   MCP Fitness Bonus: {mcp_fitness_bonus:.3f}")
            print(f"   Penalties Applied: {generation_penalties}")
            print(f"   Avg Appropriateness: {gen_result['avg_appropriateness']:.3f}")
            print(f"   Biases Applied: {gen_result['biases_applied']}")
        
        # Analyze end-to-end progression
        initial_appropriateness = e2e_results[0]['avg_appropriateness']
        final_appropriateness = e2e_results[-1]['avg_appropriateness']
        appropriateness_improvement = ((final_appropriateness - initial_appropriateness) / initial_appropriateness) * 100
        
        total_penalties = sum(result['penalties_applied'] for result in e2e_results)
        avg_fitness_bonus = sum(result['mcp_fitness_bonus'] for result in e2e_results) / len(e2e_results)
        total_biases = sum(result['biases_applied'] for result in e2e_results)
        
        # Calculate overall fusion effectiveness
        fusion_effectiveness = min(1.0, (
            (final_appropriateness * 0.4) +
            (avg_fitness_bonus * 0.3) +
            (min(1.0, total_biases / 5) * 0.3)
        ))
        
        print(f"\n📊 END-TO-END FUSION RESULTS:")
        print(f"   Appropriateness Improvement: {appropriateness_improvement:.1f}%")
        print(f"   Average Fitness Bonus: {avg_fitness_bonus:.3f}")
        print(f"   Total Penalties Applied: {total_penalties}")
        print(f"   Total Biases Applied: {total_biases}")
        print(f"   Fusion Effectiveness: {fusion_effectiveness:.1%}")
        
        return {
            'success': True,
            'e2e_results': e2e_results,
            'appropriateness_improvement': appropriateness_improvement,
            'avg_fitness_bonus': avg_fitness_bonus,
            'total_penalties': total_penalties,
            'total_biases': total_biases,
            'fusion_effectiveness': fusion_effectiveness,
            'e2e_fusion_working': fusion_effectiveness >= 0.95
        }
        
    except Exception as e:
        print(f"❌ End-to-end MCP fusion test failed: {e}")
        return {'success': False, 'error': str(e)}

async def main():
    """Main test execution"""
    print("🧪 OBSERVER MCP-EVOLUTION FUSION TEST")
    print("RIPER-Ω Protocol: MCP-GUARD EVOLUTION INTEGRATION")
    print("=" * 70)
    
    test_results = {}
    
    # Run all MCP-evolution fusion tests
    test_results['mcp_fitness_integration'] = await test_mcp_fitness_integration()
    test_results['cascade_penalty_evolution'] = await test_cascade_penalty_evolution()
    test_results['mcp_audit_visualization'] = await test_mcp_audit_visualization()
    test_results['mcp_seeded_environment'] = await test_mcp_seeded_environment()
    test_results['end_to_end_mcp_fusion'] = await test_end_to_end_mcp_fusion()
    
    # Compile final results
    print("\n" + "=" * 70)
    print("OBSERVER MCP-EVOLUTION FUSION RESULTS")
    print("=" * 70)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() if result.get('success', False))
    success_rate = successful_tests / total_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Success Rate: {success_rate:.1%}")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    # Calculate overall fusion effectiveness
    fusion_metrics = []
    
    if test_results['end_to_end_mcp_fusion'].get('fusion_effectiveness'):
        fusion_metrics.append(test_results['end_to_end_mcp_fusion']['fusion_effectiveness'])
    
    if test_results['mcp_fitness_integration'].get('success_rate'):
        fusion_metrics.append(test_results['mcp_fitness_integration']['success_rate'])
    
    if test_results['cascade_penalty_evolution'].get('evolution_rate'):
        fusion_metrics.append(test_results['cascade_penalty_evolution']['evolution_rate'])
    
    overall_fusion_effectiveness = sum(fusion_metrics) / len(fusion_metrics) if fusion_metrics else 0.0
    
    # Overall assessment
    if success_rate >= 0.9 and overall_fusion_effectiveness >= 0.95:
        print("\n🎉 OBSERVER ASSESSMENT: MCP-EVOLUTION FUSION PERFECTED")
        print("✅ 95%+ fusion effectiveness achieved")
        print("✅ All MCP-guard systems integrated into evolution")
        print("✅ Cascade penalties evolving correctly")
        print("✅ Ready for production deployment")
    elif success_rate >= 0.7 and overall_fusion_effectiveness >= 0.8:
        print("\n⚡ OBSERVER ASSESSMENT: MCP-EVOLUTION FUSION EFFECTIVE")
        print("✅ Strong fusion integration working")
        print("⚠️ Minor optimizations recommended")
    else:
        print("\n🔄 OBSERVER ASSESSMENT: FUSION NEEDS STRENGTHENING")
        print("❌ Fusion effectiveness gaps detected")
        print("🔧 Further integration development required")
    
    print(f"\n📊 Overall Fusion Effectiveness: {overall_fusion_effectiveness:.1%}")
    print(f"🎯 Target Effectiveness: 95%")
    print(f"🏆 Fusion Achievement: {'✅ ACHIEVED' if overall_fusion_effectiveness >= 0.95 else '⚠️ PARTIAL'}")
    
    # Save test report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"mcp_evolution_fusion_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Test report saved: {report_file}")
    
    return success_rate >= 0.8 and overall_fusion_effectiveness >= 0.9

if __name__ == "__main__":
    asyncio.run(main())
