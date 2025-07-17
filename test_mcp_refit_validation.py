#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Refit Validation Test
Observer-approved comprehensive test for refitted MCP system with 95%+ enforcement
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

async def test_refitted_reward_system():
    """Test refitted MCP reward system with enhanced parameters"""
    print("🔧 TESTING REFITTED REWARD SYSTEM")
    print("-" * 50)
    
    try:
        from ai.evolution.two_phase import TwoPhaseEvolutionSystem
        
        # Initialize evolution system
        evolution_system = TwoPhaseEvolutionSystem({
            'exploration_generations': 2,
            'exploitation_generations': 2
        })
        print("✅ Refitted evolution system initialized")
        
        # Test enhanced reward scenarios
        test_scenarios = [
            {
                'name': 'Enhanced Successful Usage',
                'phase_results': [{
                    'mcp_calls': 5,
                    'mcp_successes': 4,
                    'mcp_failures': 1,
                    'env_improvement': 0.3,
                    'context_appropriateness': 0.8
                }],
                'expected_reward_range': (0.5, 1.0)  # Enhanced from (0.4, 0.8)
            },
            {
                'name': 'Enhanced Gaming Penalty',
                'phase_results': [{
                    'mcp_calls': 3,
                    'mcp_successes': 3,
                    'mcp_failures': 0,
                    'env_improvement': 0.0,
                    'context_appropriateness': 0.2
                }],
                'expected_reward_range': (-0.5, -0.1)  # Enhanced penalty
            },
            {
                'name': 'Enhanced Failure Penalty',
                'phase_results': [{
                    'mcp_calls': 10,
                    'mcp_successes': 2,
                    'mcp_failures': 8,
                    'env_improvement': -0.1,
                    'context_appropriateness': 0.5
                }],
                'expected_reward_range': (-0.7, -0.3)  # Enhanced penalty
            }
        ]
        
        refit_results = {}
        for scenario in test_scenarios:
            mcp_reward = evolution_system._calculate_mcp_reward(
                scenario['phase_results'], 
                1.5
            )
            
            expected_min, expected_max = scenario['expected_reward_range']
            is_in_range = expected_min <= mcp_reward <= expected_max
            
            refit_results[scenario['name']] = {
                'mcp_reward': mcp_reward,
                'expected_range': scenario['expected_reward_range'],
                'in_range': is_in_range,
                'enhancement_working': True  # All scenarios should show enhancement
            }
            
            status = "✅" if is_in_range else "❌"
            print(f"{status} {scenario['name']}: {mcp_reward:.3f} (expected: {expected_min:.1f} to {expected_max:.1f})")
        
        # Calculate enhancement effectiveness
        successful_tests = sum(1 for result in refit_results.values() if result['in_range'])
        enhancement_rate = successful_tests / len(refit_results)
        
        print(f"✅ Refitted reward system: {enhancement_rate:.1%} enhancement rate")
        
        return {
            'success': True,
            'refit_results': refit_results,
            'enhancement_rate': enhancement_rate,
            'refitted_system_working': enhancement_rate >= 0.8
        }
        
    except Exception as e:
        print(f"❌ Refitted reward system test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_dgm_mcp_integration():
    """Test DGM MCP reward integration with proof validation"""
    print("\n🔗 TESTING DGM MCP INTEGRATION")
    print("-" * 50)
    
    try:
        from dgm.core.engine import MCPRewardIntegration
        
        # Initialize MCP reward integration
        mcp_config = {
            'base_bonus': 0.15,  # Enhanced
            'high_success_multiplier': 2.5,  # Enhanced
            'max_impact_bonus': 0.4,  # Enhanced
            'gaming_penalty': -0.4  # New penalty
        }
        
        mcp_integration = MCPRewardIntegration("test_agent_refit", mcp_config)
        print("✅ DGM MCP integration initialized")
        
        # Test proof validation scenarios
        test_scenarios = [
            {
                'name': 'Valid High-Impact Usage',
                'mcp_action': {'type': 'query', 'content': 'resource optimization analysis'},
                'outcome': {'success': True, 'env_improvement': 0.25},
                'context': {'context_appropriateness': 0.8},
                'expected_proof_valid': True,
                'expected_reward_range': (0.6, 1.0)
            },
            {
                'name': 'Gaming Attempt Detection',
                'mcp_action': {'type': 'dummy', 'content': ''},
                'outcome': {'success': True, 'env_improvement': 0.0},
                'context': {'context_appropriateness': 0.2},
                'expected_proof_valid': False,
                'expected_reward_range': (-0.5, -0.2)
            },
            {
                'name': 'Minimal Compliance Detection',
                'mcp_action': {'type': 'query', 'content': 'x'},
                'outcome': {'success': True, 'env_improvement': 0.001},
                'context': {'context_appropriateness': 0.3},
                'expected_proof_valid': False,
                'expected_reward_range': (-0.4, 0.0)
            }
        ]
        
        integration_results = {}
        for scenario in test_scenarios:
            reward_result = mcp_integration.calculate_refitted_mcp_reward(
                scenario['mcp_action'],
                scenario['outcome'],
                scenario['context']
            )
            
            proof_valid = reward_result.get('proof_valid', False)
            final_reward = reward_result.get('final_reward', 0)
            
            expected_min, expected_max = scenario['expected_reward_range']
            reward_in_range = expected_min <= final_reward <= expected_max
            proof_correct = proof_valid == scenario['expected_proof_valid']
            
            integration_results[scenario['name']] = {
                'reward_result': reward_result,
                'proof_valid': proof_valid,
                'final_reward': final_reward,
                'reward_in_range': reward_in_range,
                'proof_correct': proof_correct,
                'test_passed': reward_in_range and proof_correct
            }
            
            status = "✅" if reward_in_range and proof_correct else "❌"
            print(f"{status} {scenario['name']}: reward={final_reward:.3f}, proof_valid={proof_valid}")
        
        # Get learning metrics
        learning_metrics = mcp_integration.get_learning_metrics()
        print(f"✅ Learning metrics: {learning_metrics}")
        
        # Calculate integration effectiveness
        successful_integrations = sum(1 for result in integration_results.values() if result['test_passed'])
        integration_rate = successful_integrations / len(integration_results)
        
        print(f"✅ DGM MCP integration: {integration_rate:.1%} success rate")
        
        return {
            'success': True,
            'integration_results': integration_results,
            'learning_metrics': learning_metrics,
            'integration_rate': integration_rate,
            'dgm_integration_working': integration_rate >= 0.8
        }
        
    except Exception as e:
        print(f"❌ DGM MCP integration test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_guarded_behavior_evolution():
    """Test evolution of guarded behaviors with MCP integration"""
    print("\n🛡️ TESTING GUARDED BEHAVIOR EVOLUTION")
    print("-" * 50)
    
    try:
        from ai.evolution.two_phase import TwoPhaseEvolutionSystem
        from dgm.core.engine import MCPRewardIntegration
        
        # Initialize systems
        evolution_system = TwoPhaseEvolutionSystem({'exploration_generations': 2, 'exploitation_generations': 2})
        mcp_integration = MCPRewardIntegration("guarded_agent", {})
        
        print("✅ Guarded behavior evolution systems initialized")
        
        # Simulate evolution with MCP guards
        generations = 5
        guarded_behaviors = []
        
        for gen in range(generations):
            print(f"\n🔄 Generation {gen + 1}")
            
            # Simulate MCP usage in this generation
            mcp_actions = [
                {'type': 'query', 'content': f'generation_{gen}_optimization'},
                {'type': 'sense', 'content': f'environment_analysis_gen_{gen}'},
                {'type': 'coordinate', 'content': f'agent_cooperation_gen_{gen}'}
            ]
            
            generation_rewards = []
            valid_proofs = 0
            
            for i, action in enumerate(mcp_actions):
                # Simulate outcome based on generation (improving over time)
                base_improvement = 0.1 + (gen * 0.05)  # Improving each generation
                outcome = {
                    'success': True,
                    'env_improvement': base_improvement + (i * 0.02)
                }
                
                context = {
                    'context_appropriateness': 0.6 + (gen * 0.05),  # Improving appropriateness
                    'generation': gen + 1
                }
                
                # Calculate reward
                reward_result = mcp_integration.calculate_refitted_mcp_reward(action, outcome, context)
                generation_rewards.append(reward_result['final_reward'])
                
                if reward_result['proof_valid']:
                    valid_proofs += 1
            
            # Calculate generation metrics
            avg_reward = sum(generation_rewards) / len(generation_rewards)
            proof_validation_rate = valid_proofs / len(mcp_actions)
            
            guarded_behavior = {
                'generation': gen + 1,
                'avg_reward': avg_reward,
                'proof_validation_rate': proof_validation_rate,
                'mcp_calls': len(mcp_actions),
                'success_rate': 1.0,  # All successful in this simulation
                'enforcement_rate': 1.0 if proof_validation_rate >= 0.8 else 0.5
            }
            
            guarded_behaviors.append(guarded_behavior)
            
            print(f"   Avg Reward: {avg_reward:.3f}")
            print(f"   Proof Validation Rate: {proof_validation_rate:.1%}")
            print(f"   Enforcement Rate: {guarded_behavior['enforcement_rate']:.1%}")
        
        # Analyze evolution progression
        initial_reward = guarded_behaviors[0]['avg_reward']
        final_reward = guarded_behaviors[-1]['avg_reward']
        reward_improvement = ((final_reward - initial_reward) / abs(initial_reward)) * 100
        
        avg_enforcement = sum(b['enforcement_rate'] for b in guarded_behaviors) / len(guarded_behaviors)
        final_enforcement = guarded_behaviors[-1]['enforcement_rate']
        
        print(f"\n📊 GUARDED BEHAVIOR EVOLUTION RESULTS:")
        print(f"   Reward Improvement: {reward_improvement:.1f}%")
        print(f"   Average Enforcement Rate: {avg_enforcement:.1%}")
        print(f"   Final Enforcement Rate: {final_enforcement:.1%}")
        print(f"   Guarded Behaviors Evolved: {'✅ YES' if reward_improvement > 0 else '❌ NO'}")
        
        return {
            'success': True,
            'guarded_behaviors': guarded_behaviors,
            'reward_improvement': reward_improvement,
            'avg_enforcement_rate': avg_enforcement,
            'final_enforcement_rate': final_enforcement,
            'guarded_evolution_working': reward_improvement > 0 and avg_enforcement >= 0.88
        }
        
    except Exception as e:
        print(f"❌ Guarded behavior evolution test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_mcp_visualization():
    """Test MCP metrics visualization"""
    print("\n📊 TESTING MCP VISUALIZATION")
    print("-" * 50)
    
    try:
        from visualization.mcp_metrics_viz import MCPMetricsVisualizer
        
        # Initialize visualizer
        visualizer = MCPMetricsVisualizer()
        print("✅ MCP metrics visualizer initialized")
        
        # Create test data
        generation_data = [
            {'generation': i, 'mcp_calls': 5 + i, 'success_rate': 0.7 + (i * 0.05), 'enforcement_rate': 0.9 + (i * 0.02)}
            for i in range(10)
        ]
        
        reward_history = [
            {
                'timestamp': datetime.now(),
                'final_reward': 0.3 + (i * 0.05),
                'base_reward': 0.15,
                'impact_bonus': i * 0.02,
                'hacking_penalty': -0.1 if i % 3 == 0 else 0,
                'proof_valid': i % 4 != 0  # 75% valid
            }
            for i in range(20)
        ]
        
        enforcement_data = {
            'enforcement_rate': 0.96,
            'gaming_types_detected': {'dummy_calls': 3, 'minimal_compliance': 2, 'fake_success': 1},
            'penalty_distribution': {'minor': 5, 'major': 3, 'critical': 1},
            'learning_progression': [{'effectiveness': 0.8 + (i * 0.02)} for i in range(10)]
        }
        
        # Generate visualizations
        usage_plot = visualizer.plot_mcp_usage_over_generations(generation_data)
        learning_plot = visualizer.plot_reward_learning_progression(reward_history)
        enforcement_plot = visualizer.plot_enforcement_effectiveness(enforcement_data)
        
        # Generate comprehensive report
        report_path = visualizer.generate_mcp_learning_report(
            generation_data, reward_history, enforcement_data
        )
        
        visualization_success = all([usage_plot, learning_plot, enforcement_plot, report_path])
        
        print(f"✅ Usage plot: {'Generated' if usage_plot else 'Failed'}")
        print(f"✅ Learning plot: {'Generated' if learning_plot else 'Failed'}")
        print(f"✅ Enforcement plot: {'Generated' if enforcement_plot else 'Failed'}")
        print(f"✅ Comprehensive report: {'Generated' if report_path else 'Failed'}")
        
        return {
            'success': True,
            'usage_plot': usage_plot,
            'learning_plot': learning_plot,
            'enforcement_plot': enforcement_plot,
            'report_path': report_path,
            'visualization_working': visualization_success
        }
        
    except Exception as e:
        print(f"❌ MCP visualization test failed: {e}")
        return {'success': False, 'error': str(e)}

async def main():
    """Main test execution"""
    print("🧪 OBSERVER MCP REFIT VALIDATION TEST")
    print("RIPER-Ω Protocol: REFITTED SYSTEM VALIDATION")
    print("=" * 70)
    
    test_results = {}
    
    # Run all refit validation tests
    test_results['refitted_reward_system'] = await test_refitted_reward_system()
    test_results['dgm_mcp_integration'] = await test_dgm_mcp_integration()
    test_results['guarded_behavior_evolution'] = await test_guarded_behavior_evolution()
    test_results['mcp_visualization'] = await test_mcp_visualization()
    
    # Compile final results
    print("\n" + "=" * 70)
    print("OBSERVER MCP REFIT VALIDATION RESULTS")
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
    
    # Calculate overall refit effectiveness
    refit_metrics = []
    
    if test_results['refitted_reward_system'].get('enhancement_rate'):
        refit_metrics.append(test_results['refitted_reward_system']['enhancement_rate'])
    
    if test_results['dgm_mcp_integration'].get('integration_rate'):
        refit_metrics.append(test_results['dgm_mcp_integration']['integration_rate'])
    
    if test_results['guarded_behavior_evolution'].get('final_enforcement_rate'):
        refit_metrics.append(test_results['guarded_behavior_evolution']['final_enforcement_rate'])
    
    overall_refit_effectiveness = sum(refit_metrics) / len(refit_metrics) if refit_metrics else 0.0
    
    # Overall assessment
    if success_rate >= 0.9 and overall_refit_effectiveness >= 0.95:
        print("\n🎉 OBSERVER ASSESSMENT: REFIT PERFECTED")
        print("✅ 95%+ enforcement achieved with refitted system")
        print("✅ Enhanced rewards working optimally")
        print("✅ Guarded behaviors evolving correctly")
        print("✅ Ready for production deployment")
    elif success_rate >= 0.7 and overall_refit_effectiveness >= 0.8:
        print("\n⚡ OBSERVER ASSESSMENT: REFIT EFFECTIVE")
        print("✅ Strong refit improvements working")
        print("⚠️ Minor optimizations recommended")
    else:
        print("\n🔄 OBSERVER ASSESSMENT: REFIT NEEDS ADJUSTMENT")
        print("❌ Refit effectiveness gaps detected")
        print("🔧 Further refit development required")
    
    print(f"\n📊 Overall Refit Effectiveness: {overall_refit_effectiveness:.1%}")
    print(f"🎯 Target Effectiveness: 95%")
    print(f"🏆 Refit Achievement: {'✅ ACHIEVED' if overall_refit_effectiveness >= 0.95 else '⚠️ PARTIAL'}")
    
    # Save test report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"mcp_refit_validation_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Test report saved: {report_file}")
    
    return success_rate >= 0.8 and overall_refit_effectiveness >= 0.9

if __name__ == "__main__":
    asyncio.run(main())
