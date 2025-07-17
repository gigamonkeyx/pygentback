#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Anti-Hacking System Test
Observer-approved comprehensive test for MCP reward system with 95%+ enforcement
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

async def test_tiered_mcp_rewards():
    """Test tiered MCP reward system in two-phase evolution"""
    print("🎯 TESTING TIERED MCP REWARDS")
    print("-" * 50)
    
    try:
        from ai.evolution.two_phase import TwoPhaseEvolutionSystem
        
        # Initialize evolution system
        evolution_config = {
            'exploration_generations': 2,
            'exploitation_generations': 2,
            'rl_enabled': True
        }
        
        evolution_system = TwoPhaseEvolutionSystem(evolution_config)
        print("✅ Two-phase evolution system initialized")
        
        # Test MCP reward calculation with different scenarios
        test_scenarios = [
            {
                'name': 'Successful MCP Usage',
                'phase_results': [{
                    'mcp_calls': 5,
                    'mcp_successes': 4,
                    'mcp_failures': 1,
                    'env_improvement': 0.3,
                    'context_appropriateness': 0.8
                }],
                'expected_reward_range': (0.4, 0.8)
            },
            {
                'name': 'Dummy Call Gaming',
                'phase_results': [{
                    'mcp_calls': 3,
                    'mcp_successes': 3,
                    'mcp_failures': 0,
                    'env_improvement': 0.0,  # No improvement despite "success"
                    'context_appropriateness': 0.2
                }],
                'expected_reward_range': (-0.3, 0.1)
            },
            {
                'name': 'High Failure Rate',
                'phase_results': [{
                    'mcp_calls': 10,
                    'mcp_successes': 2,
                    'mcp_failures': 8,
                    'env_improvement': -0.1,
                    'context_appropriateness': 0.5
                }],
                'expected_reward_range': (-0.5, -0.1)
            },
            {
                'name': 'No MCP Usage',
                'phase_results': [{
                    'mcp_calls': 0,
                    'mcp_successes': 0,
                    'mcp_failures': 0,
                    'env_improvement': 0.0,
                    'context_appropriateness': 0.5
                }],
                'expected_reward_range': (0.0, 0.0)
            }
        ]
        
        test_results = {}
        for scenario in test_scenarios:
            mcp_reward = evolution_system._calculate_mcp_reward(
                scenario['phase_results'], 
                1.5  # best_fitness
            )
            
            expected_min, expected_max = scenario['expected_reward_range']
            is_in_range = expected_min <= mcp_reward <= expected_max
            
            test_results[scenario['name']] = {
                'mcp_reward': mcp_reward,
                'expected_range': scenario['expected_reward_range'],
                'in_range': is_in_range
            }
            
            status = "✅" if is_in_range else "❌"
            print(f"{status} {scenario['name']}: {mcp_reward:.3f} (expected: {expected_min:.1f} to {expected_max:.1f})")
        
        # Calculate success rate
        successful_tests = sum(1 for result in test_results.values() if result['in_range'])
        success_rate = successful_tests / len(test_results)
        
        print(f"✅ Tiered reward system test: {success_rate:.1%} success rate")
        
        return {
            'success': True,
            'test_results': test_results,
            'success_rate': success_rate,
            'tiered_rewards_working': success_rate >= 0.75
        }
        
    except Exception as e:
        print(f"❌ Tiered MCP rewards test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_context_aware_rl_guard():
    """Test context-aware RL guard system"""
    print("\n🛡️ TESTING CONTEXT-AWARE RL GUARD")
    print("-" * 50)
    
    try:
        from ai.rl.reward_functions import ContextAwareRLGuard
        
        # Initialize RL guard
        guard_config = {
            'ambiguity_threshold': 0.3,
            'appropriateness_threshold': 0.5
        }
        
        rl_guard = ContextAwareRLGuard(guard_config)
        print("✅ Context-aware RL guard initialized")
        
        # Test scenarios with different appropriateness levels
        test_scenarios = [
            {
                'name': 'Appropriate High Ambiguity',
                'environment_state': {
                    'resource_availability': 0.5,  # Moderate = ambiguous
                    'agent_count': 50,
                    'fitness_variance': 0.4
                },
                'mcp_action': {
                    'type': 'query',
                    'content': 'sense resource distribution'
                },
                'outcome': {
                    'success': True,
                    'env_improvement': 0.2
                },
                'expected_appropriateness': 'high'
            },
            {
                'name': 'Inappropriate Low Ambiguity',
                'environment_state': {
                    'resource_availability': 0.9,  # High = clear
                    'agent_count': 50,
                    'fitness_variance': 0.1
                },
                'mcp_action': {
                    'type': 'query',
                    'content': 'unnecessary query'
                },
                'outcome': {
                    'success': True,
                    'env_improvement': 0.0
                },
                'expected_appropriateness': 'low'
            },
            {
                'name': 'Gaming Attempt',
                'environment_state': {
                    'resource_availability': 0.6,
                    'agent_count': 50,
                    'fitness_variance': 0.3
                },
                'mcp_action': {
                    'type': 'dummy',
                    'content': ''
                },
                'outcome': {
                    'success': True,
                    'env_improvement': 0.0
                },
                'expected_appropriateness': 'gaming'
            }
        ]
        
        evaluation_results = {}
        for scenario in test_scenarios:
            contextual_reward = rl_guard.evaluate_mcp_appropriateness(
                scenario['environment_state'],
                scenario['mcp_action'],
                scenario['outcome']
            )
            
            # Evaluate appropriateness
            if scenario['expected_appropriateness'] == 'high':
                is_correct = contextual_reward.total_reward > 0.2
            elif scenario['expected_appropriateness'] == 'low':
                is_correct = contextual_reward.total_reward < 0.1
            elif scenario['expected_appropriateness'] == 'gaming':
                is_correct = contextual_reward.anti_hacking_penalty < -0.2
            else:
                is_correct = False
            
            evaluation_results[scenario['name']] = {
                'contextual_reward': contextual_reward,
                'is_correct': is_correct
            }
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {scenario['name']}: reward={contextual_reward.total_reward:.3f}, "
                  f"appropriateness={contextual_reward.context_appropriateness:.3f}")
        
        # Get enforcement stats
        enforcement_stats = rl_guard.get_enforcement_stats()
        print(f"✅ Enforcement stats: {enforcement_stats}")
        
        # Calculate success rate
        successful_evaluations = sum(1 for result in evaluation_results.values() if result['is_correct'])
        success_rate = successful_evaluations / len(evaluation_results)
        
        print(f"✅ Context-aware RL guard test: {success_rate:.1%} success rate")
        
        return {
            'success': True,
            'evaluation_results': evaluation_results,
            'enforcement_stats': enforcement_stats,
            'success_rate': success_rate,
            'context_guard_working': success_rate >= 0.8
        }
        
    except Exception as e:
        print(f"❌ Context-aware RL guard test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_safety_monitor_penalties():
    """Test safety monitor cascade penalties"""
    print("\n⚖️ TESTING SAFETY MONITOR PENALTIES")
    print("-" * 50)
    
    try:
        from core.safety_monitor import SafetyMonitor
        
        # Initialize safety monitor
        monitor_config = {
            'minor_penalty': -0.1,
            'major_penalty': -0.5,
            'critical_penalty': -1.0,
            'minor_threshold': 3,
            'major_threshold': 2,
            'rewrite_threshold': 3
        }
        
        safety_monitor = SafetyMonitor(monitor_config)
        print("✅ Safety monitor initialized")
        
        # Test penalty scenarios
        test_agent_id = "test_agent_penalties"
        
        # Scenario 1: Minor penalty for unused call
        unused_call_penalties = safety_monitor.monitor_mcp_usage(
            test_agent_id,
            {'type': 'query', 'content': 'test query'},
            {'success': True, 'env_improvement': 0.0, 'benefit': 0.0},
            {'resource_availability': 0.5}
        )
        
        print(f"✅ Unused call penalties: {len(unused_call_penalties)} applied")
        
        # Scenario 2: Major penalty for failed outcome
        failed_outcome_penalties = safety_monitor.monitor_mcp_usage(
            test_agent_id,
            {'type': 'action', 'content': 'risky action'},
            {'success': False, 'error': 'execution failed'},
            {'resource_availability': 0.5}
        )
        
        print(f"✅ Failed outcome penalties: {len(failed_outcome_penalties)} applied")
        
        # Scenario 3: Gaming attempt detection
        gaming_penalties = safety_monitor.monitor_mcp_usage(
            test_agent_id,
            {'type': 'dummy', 'content': 'fake call'},
            {'success': True, 'env_improvement': 0.0},
            {'resource_availability': 0.5}
        )
        
        print(f"✅ Gaming attempt penalties: {len(gaming_penalties)} applied")
        
        # Check if agent is marked for rewrite
        agent_summary = safety_monitor.get_agent_penalty_summary(test_agent_id)
        print(f"✅ Agent penalty summary: {agent_summary}")
        
        # Get overall monitoring stats
        monitoring_stats = safety_monitor.get_monitoring_stats()
        print(f"✅ Monitoring stats: {monitoring_stats}")
        
        # Verify penalty escalation
        total_penalties = len(unused_call_penalties) + len(failed_outcome_penalties) + len(gaming_penalties)
        rewrite_candidate = agent_summary.get('rewrite_candidate', False)
        
        return {
            'success': True,
            'total_penalties_applied': total_penalties,
            'agent_marked_for_rewrite': rewrite_candidate,
            'agent_summary': agent_summary,
            'monitoring_stats': monitoring_stats,
            'penalty_system_working': total_penalties >= 3
        }
        
    except Exception as e:
        print(f"❌ Safety monitor penalties test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_dgm_audit_trails():
    """Test DGM audit trail system"""
    print("\n📋 TESTING DGM AUDIT TRAILS")
    print("-" * 50)
    
    try:
        from dgm.core.engine import DGMAuditTrail
        
        # Initialize audit trail
        audit_trail = DGMAuditTrail("test_agent_audit")
        print("✅ DGM audit trail initialized")
        
        # Test different intent scenarios
        test_scenarios = [
            {
                'name': 'Valid Environment Sensing',
                'mcp_action': {'type': 'sense', 'content': 'check resource levels'},
                'intent': 'env_sense',
                'outcome': {'success': True, 'env_improvement': 0.1},
                'context': {'ambiguity_score': 0.6, 'resource_scarcity': True}
            },
            {
                'name': 'Invalid Gaming Attempt',
                'mcp_action': {'type': 'dummy', 'content': 'test'},
                'intent': 'env_sense',
                'outcome': {'success': True, 'env_improvement': 0.0},
                'context': {'ambiguity_score': 0.1}
            },
            {
                'name': 'Valid Resource Query',
                'mcp_action': {'type': 'query', 'content': 'resource availability check'},
                'intent': 'resource_query',
                'outcome': {'success': True, 'env_improvement': 0.2},
                'context': {'ambiguity_score': 0.5, 'resource_scarcity': True}
            }
        ]
        
        audit_results = {}
        for scenario in test_scenarios:
            # Log call chain
            entry_id = audit_trail.log_mcp_call_chain(
                scenario['mcp_action'],
                scenario['intent'],
                scenario['outcome'],
                scenario['context']
            )
            
            # Validate call chain
            is_valid = audit_trail.validate_call_chain(entry_id)
            
            audit_results[scenario['name']] = {
                'entry_id': entry_id,
                'is_valid': is_valid
            }
            
            status = "✅" if is_valid else "❌"
            print(f"{status} {scenario['name']}: entry_id={entry_id}, valid={is_valid}")
        
        # Discard gaming proofs
        discarded_count = audit_trail.discard_gaming_proofs()
        print(f"✅ Discarded {discarded_count} gaming proofs")
        
        # Get audit stats
        audit_stats = audit_trail.get_audit_stats()
        print(f"✅ Audit stats: {audit_stats}")
        
        # Calculate validation rate
        valid_entries = sum(1 for result in audit_results.values() if result['is_valid'])
        validation_rate = valid_entries / len(audit_results)
        
        return {
            'success': True,
            'audit_results': audit_results,
            'discarded_gaming_proofs': discarded_count,
            'audit_stats': audit_stats,
            'validation_rate': validation_rate,
            'audit_system_working': validation_rate >= 0.6
        }
        
    except Exception as e:
        print(f"❌ DGM audit trails test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_gaming_attempt_simulation():
    """Test system response to deliberate gaming attempts"""
    print("\n🎮 TESTING GAMING ATTEMPT SIMULATION")
    print("-" * 50)
    
    try:
        from ai.rl.reward_functions import ContextAwareRLGuard
        from core.safety_monitor import SafetyMonitor
        
        # Initialize systems
        rl_guard = ContextAwareRLGuard({'ambiguity_threshold': 0.3})
        safety_monitor = SafetyMonitor({})
        
        print("✅ Anti-hacking systems initialized")
        
        # Simulate gaming attempts
        gaming_attempts = [
            {
                'name': 'Dummy Call Spam',
                'mcp_action': {'type': 'dummy', 'content': ''},
                'outcome': {'success': True, 'env_improvement': 0.0}
            },
            {
                'name': 'Minimal Compliance',
                'mcp_action': {'type': 'query', 'content': 'x'},
                'outcome': {'success': True, 'env_improvement': 0.001}
            },
            {
                'name': 'Fake Success',
                'mcp_action': {'type': 'test', 'content': 'fake action'},
                'outcome': {'success': True, 'env_improvement': 0.0}
            }
        ]
        
        gaming_agent_id = "gaming_agent_test"
        detection_results = {}
        
        for attempt in gaming_attempts:
            # Test with RL guard
            environment_state = {'resource_availability': 0.8, 'agent_count': 50}
            contextual_reward = rl_guard.evaluate_mcp_appropriateness(
                environment_state,
                attempt['mcp_action'],
                attempt['outcome']
            )
            
            # Test with safety monitor
            penalties = safety_monitor.monitor_mcp_usage(
                gaming_agent_id,
                attempt['mcp_action'],
                attempt['outcome'],
                environment_state
            )
            
            # Check if gaming was detected
            gaming_detected = (
                contextual_reward.anti_hacking_penalty < -0.1 or
                len(penalties) > 0
            )
            
            detection_results[attempt['name']] = {
                'gaming_detected': gaming_detected,
                'contextual_reward': contextual_reward.total_reward,
                'penalties_applied': len(penalties)
            }
            
            status = "✅" if gaming_detected else "❌"
            print(f"{status} {attempt['name']}: detected={gaming_detected}, "
                  f"reward={contextual_reward.total_reward:.3f}, penalties={len(penalties)}")
        
        # Calculate detection rate
        detected_attempts = sum(1 for result in detection_results.values() if result['gaming_detected'])
        detection_rate = detected_attempts / len(gaming_attempts)
        
        # Get enforcement stats
        rl_enforcement = rl_guard.get_enforcement_stats()
        safety_enforcement = safety_monitor.get_monitoring_stats()
        
        print(f"✅ Gaming detection rate: {detection_rate:.1%}")
        print(f"✅ RL enforcement: {rl_enforcement}")
        print(f"✅ Safety enforcement: {safety_enforcement}")
        
        return {
            'success': True,
            'detection_results': detection_results,
            'detection_rate': detection_rate,
            'rl_enforcement': rl_enforcement,
            'safety_enforcement': safety_enforcement,
            'enforcement_effective': detection_rate >= 0.95  # 95%+ target
        }
        
    except Exception as e:
        print(f"❌ Gaming attempt simulation test failed: {e}")
        return {'success': False, 'error': str(e)}

async def main():
    """Main test execution"""
    print("🧪 OBSERVER MCP ANTI-HACKING SYSTEM TEST")
    print("RIPER-Ω Protocol: 95%+ ENFORCEMENT VALIDATION")
    print("=" * 70)
    
    test_results = {}
    
    # Run all anti-hacking tests
    test_results['tiered_mcp_rewards'] = await test_tiered_mcp_rewards()
    test_results['context_aware_rl_guard'] = await test_context_aware_rl_guard()
    test_results['safety_monitor_penalties'] = await test_safety_monitor_penalties()
    test_results['dgm_audit_trails'] = await test_dgm_audit_trails()
    test_results['gaming_attempt_simulation'] = await test_gaming_attempt_simulation()
    
    # Compile final results
    print("\n" + "=" * 70)
    print("OBSERVER MCP ANTI-HACKING TEST RESULTS")
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
    
    # Calculate overall enforcement effectiveness
    enforcement_metrics = []
    if test_results['gaming_attempt_simulation'].get('detection_rate'):
        enforcement_metrics.append(test_results['gaming_attempt_simulation']['detection_rate'])
    if test_results['context_aware_rl_guard'].get('success_rate'):
        enforcement_metrics.append(test_results['context_aware_rl_guard']['success_rate'])
    if test_results['dgm_audit_trails'].get('validation_rate'):
        enforcement_metrics.append(test_results['dgm_audit_trails']['validation_rate'])
    
    overall_enforcement = sum(enforcement_metrics) / len(enforcement_metrics) if enforcement_metrics else 0.0
    
    # Overall assessment
    if success_rate >= 0.9 and overall_enforcement >= 0.95:
        print("\n🎉 OBSERVER ASSESSMENT: ANTI-HACKING PERFECTED")
        print("✅ 95%+ enforcement rate achieved")
        print("✅ All anti-hacking systems operational")
        print("✅ Gaming attempts successfully blocked")
        print("✅ Ready for production deployment")
    elif success_rate >= 0.7 and overall_enforcement >= 0.8:
        print("\n⚡ OBSERVER ASSESSMENT: ANTI-HACKING EFFECTIVE")
        print("✅ Strong enforcement systems working")
        print("⚠️ Minor optimizations recommended")
    else:
        print("\n🔄 OBSERVER ASSESSMENT: NEEDS STRENGTHENING")
        print("❌ Enforcement gaps detected")
        print("🔧 Further anti-hacking development required")
    
    print(f"\n📊 Overall Enforcement Rate: {overall_enforcement:.1%}")
    print(f"🎯 Target Enforcement Rate: 95%")
    print(f"🏆 Enforcement Achievement: {'✅ ACHIEVED' if overall_enforcement >= 0.95 else '⚠️ PARTIAL'}")
    
    # Save test report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"mcp_anti_hacking_test_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Test report saved: {report_file}")
    
    return success_rate >= 0.8 and overall_enforcement >= 0.9

if __name__ == "__main__":
    asyncio.run(main())
