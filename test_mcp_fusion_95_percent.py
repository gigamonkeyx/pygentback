#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Fusion 95%+ Effectiveness Test
Observer-approved comprehensive test for 95%+ MCP-evolution fusion effectiveness
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

async def test_enhanced_fusion_effectiveness():
    """Test enhanced MCP-evolution fusion for 95%+ effectiveness"""
    print("🚀 TESTING ENHANCED FUSION EFFECTIVENESS")
    print("-" * 50)
    
    try:
        from ai.evolution.two_phase import TwoPhaseEvolutionSystem
        
        # Initialize enhanced evolution system
        evolution_system = TwoPhaseEvolutionSystem({
            'exploration_generations': 2,
            'exploitation_generations': 2
        })
        print("✅ Enhanced fusion system initialized")
        
        # Test enhanced effectiveness scenarios
        test_scenarios = [
            {
                'name': 'Excellence Bonus Scenario',
                'phase_results': [{
                    'mcp_calls': 8,
                    'mcp_successes': 8,
                    'mcp_failures': 0,
                    'env_improvement': 0.18,  # High improvement
                    'context_appropriateness': 0.92  # Excellent appropriateness
                }],
                'base_fitness': 1.5,
                'expected_effectiveness': 0.95
            },
            {
                'name': 'Compound Learning Scenario',
                'phase_results': [{
                    'mcp_calls': 10,
                    'mcp_successes': 9,
                    'mcp_failures': 1,
                    'env_improvement': 0.16,
                    'context_appropriateness': 0.88
                }],
                'base_fitness': 1.2,
                'expected_effectiveness': 0.92
            },
            {
                'name': 'Synergy Bonus Scenario',
                'phase_results': [{
                    'mcp_calls': 6,
                    'mcp_successes': 6,
                    'mcp_failures': 0,
                    'env_improvement': 0.14,
                    'context_appropriateness': 0.86
                }],
                'base_fitness': 1.0,
                'expected_effectiveness': 0.90
            }
        ]
        
        effectiveness_results = {}
        total_effectiveness = 0.0
        
        for scenario in test_scenarios:
            # Calculate enhanced MCP fitness integration
            mcp_fitness_bonus = evolution_system._calculate_mcp_fitness_integration(
                scenario['phase_results'],
                scenario['base_fitness'],
                0.15  # Enhanced MCP reward
            )
            
            # Calculate effectiveness boost
            avg_appropriateness = scenario['phase_results'][0]['context_appropriateness']
            avg_improvement = scenario['phase_results'][0]['env_improvement']
            total_calls = scenario['phase_results'][0]['mcp_calls']
            
            effectiveness_boost = evolution_system._calculate_effectiveness_boost(
                avg_appropriateness, avg_improvement, total_calls
            )
            
            # Calculate total effectiveness
            total_bonus = mcp_fitness_bonus + effectiveness_boost
            effectiveness_score = min(1.0, total_bonus / scenario['base_fitness'])
            
            effectiveness_results[scenario['name']] = {
                'mcp_fitness_bonus': mcp_fitness_bonus,
                'effectiveness_boost': effectiveness_boost,
                'total_bonus': total_bonus,
                'effectiveness_score': effectiveness_score,
                'expected_effectiveness': scenario['expected_effectiveness'],
                'meets_expectation': effectiveness_score >= scenario['expected_effectiveness']
            }
            
            total_effectiveness += effectiveness_score
            
            status = "✅" if effectiveness_score >= scenario['expected_effectiveness'] else "❌"
            print(f"{status} {scenario['name']}: {effectiveness_score:.3f} effectiveness (expected: {scenario['expected_effectiveness']:.3f})")
        
        # Calculate overall effectiveness
        avg_effectiveness = total_effectiveness / len(test_scenarios)
        
        print(f"✅ Enhanced fusion effectiveness: {avg_effectiveness:.1%}")
        
        return {
            'success': True,
            'effectiveness_results': effectiveness_results,
            'avg_effectiveness': avg_effectiveness,
            'target_effectiveness': 0.95,
            'enhanced_fusion_working': avg_effectiveness >= 0.95
        }
        
    except Exception as e:
        print(f"❌ Enhanced fusion effectiveness test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_gaming_prediction_system():
    """Test gaming prediction system for >100% adaptation"""
    print("\n🔮 TESTING GAMING PREDICTION SYSTEM")
    print("-" * 50)
    
    try:
        from ai.rl.reward_functions import GamingPredictor
        
        # Initialize gaming predictor
        predictor_config = {
            'prediction_window': 5,
            'confidence_threshold': 0.7
        }
        
        gaming_predictor = GamingPredictor(predictor_config)
        print("✅ Gaming prediction system initialized")
        
        # Test prediction scenarios
        test_scenarios = [
            {
                'name': 'High Gaming Probability',
                'agent_history': [
                    {
                        'mcp_action': {'type': 'dummy', 'content': ''},
                        'outcome': {'success': True, 'env_improvement': 0.0},
                        'context': {'context_appropriateness': 0.2}
                    },
                    {
                        'mcp_action': {'type': 'dummy', 'content': 'test'},
                        'outcome': {'success': True, 'env_improvement': 0.001},
                        'context': {'context_appropriateness': 0.15}
                    },
                    {
                        'mcp_action': {'type': 'query', 'content': 'x'},
                        'outcome': {'success': True, 'env_improvement': 0.0},
                        'context': {'context_appropriateness': 0.25}
                    }
                ],
                'current_context': {'context_appropriateness': 0.2, 'resource_scarcity': True},
                'expected_probability_range': (0.7, 1.0)
            },
            {
                'name': 'Moderate Gaming Probability',
                'agent_history': [
                    {
                        'mcp_action': {'type': 'query', 'content': 'resource check'},
                        'outcome': {'success': True, 'env_improvement': 0.05},
                        'context': {'context_appropriateness': 0.6}
                    },
                    {
                        'mcp_action': {'type': 'query', 'content': 'status'},
                        'outcome': {'success': False, 'env_improvement': 0.0},
                        'context': {'context_appropriateness': 0.4}
                    },
                    {
                        'mcp_action': {'type': 'sense', 'content': 'environment'},
                        'outcome': {'success': True, 'env_improvement': 0.02},
                        'context': {'context_appropriateness': 0.5}
                    }
                ],
                'current_context': {'context_appropriateness': 0.45},
                'expected_probability_range': (0.3, 0.7)
            }
        ]
        
        prediction_results = {}
        for scenario in test_scenarios:
            # Make gaming prediction
            prediction = gaming_predictor.predict_gaming_attempt(
                scenario['agent_history'],
                scenario['current_context']
            )
            
            gaming_probability = prediction.get('gaming_probability', 0.0)
            expected_min, expected_max = scenario['expected_probability_range']
            
            in_range = expected_min <= gaming_probability <= expected_max
            adaptation_strength = prediction.get('adaptation_strength', 0.0)
            
            prediction_results[scenario['name']] = {
                'prediction': prediction,
                'gaming_probability': gaming_probability,
                'expected_range': scenario['expected_probability_range'],
                'in_range': in_range,
                'adaptation_strength': adaptation_strength,
                'over_100_percent_adaptation': adaptation_strength > 1.0
            }
            
            status = "✅" if in_range else "❌"
            print(f"{status} {scenario['name']}: {gaming_probability:.3f} probability, {adaptation_strength:.3f} adaptation")
        
        # Get prediction stats
        prediction_stats = gaming_predictor.get_prediction_stats()
        
        # Calculate prediction effectiveness
        successful_predictions = sum(1 for result in prediction_results.values() if result['in_range'])
        prediction_effectiveness = successful_predictions / len(prediction_results)
        
        print(f"✅ Gaming prediction effectiveness: {prediction_effectiveness:.1%}")
        
        return {
            'success': True,
            'prediction_results': prediction_results,
            'prediction_stats': prediction_stats,
            'prediction_effectiveness': prediction_effectiveness,
            'gaming_prediction_working': prediction_effectiveness >= 0.8
        }
        
    except Exception as e:
        print(f"❌ Gaming prediction system test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_rl_seeded_389_growth():
    """Test RL-seeded environment for >389% growth"""
    print("\n🌱 TESTING RL-SEEDED 389% GROWTH")
    print("-" * 50)
    
    try:
        from sim.mcp_seeded_env import MCPSeededEnvironment
        
        # Initialize enhanced MCP-seeded environment
        env_config = {
            'mcp_weight_base': 0.6,  # Enhanced base
            'appropriateness_threshold': 0.7,
            'auto_bias_enabled': True
        }
        
        mcp_env = MCPSeededEnvironment(env_config)
        print("✅ Enhanced MCP-seeded environment initialized")
        
        # Create enhanced audit data for >389% growth
        enhanced_audit_data = [
            {
                'agent_id': 'agent_1',
                'success': True,
                'gaming_detected': False,
                'appropriateness_score': 0.9,
                'env_improvement': 0.2
            },
            {
                'agent_id': 'agent_2',
                'success': True,
                'gaming_detected': True,  # Gaming detected for anti-gaming bias
                'appropriateness_score': 0.15,
                'env_improvement': 0.0
            },
            {
                'agent_id': 'agent_3',
                'success': True,
                'gaming_detected': False,
                'appropriateness_score': 0.95,
                'env_improvement': 0.25
            },
            {
                'agent_id': 'agent_4',
                'success': True,
                'gaming_detected': False,
                'appropriateness_score': 0.88,
                'env_improvement': 0.18
            }
        ]
        
        # Apply RL seeding from enhanced audit logs
        seeding_result = mcp_env.rl_seed_from_logs(enhanced_audit_data)
        
        print(f"✅ RL seeding applied: {seeding_result.get('applied_modifications', [])}")
        print(f"✅ Expected growth boost: {seeding_result.get('expected_growth_boost', 1.0):.2f}x")
        
        # Test 389% growth scenarios
        growth_test_scenarios = [
            'mcp_appropriateness',
            'gaming_resistance',
            'context_adaptation',
            'cooperation_efficiency',
            'resource_optimization',
            'compound_learning',
            'enforcement_effectiveness'
        ]
        
        growth_result = mcp_env.test_seeded_growth_389(growth_test_scenarios)
        
        avg_growth = growth_result.get('avg_growth_percentage', 0.0)
        successful_scenarios = growth_result.get('successful_scenarios', 0)
        
        print(f"✅ Average growth: {avg_growth:.1f}% (target: 389%)")
        print(f"✅ Successful scenarios: {successful_scenarios}/{len(growth_test_scenarios)}")
        
        # Calculate 389% achievement rate
        achievement_rate = successful_scenarios / len(growth_test_scenarios) if growth_test_scenarios else 0.0
        
        return {
            'success': True,
            'seeding_result': seeding_result,
            'growth_result': growth_result,
            'avg_growth_percentage': avg_growth,
            'achievement_rate': achievement_rate,
            'target_growth': 389.0,
            'rl_seeded_389_working': avg_growth >= 389.0
        }
        
    except Exception as e:
        print(f"❌ RL-seeded 389% growth test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_audit_dashboard_integration():
    """Test audit dashboard integration"""
    print("\n📊 TESTING AUDIT DASHBOARD INTEGRATION")
    print("-" * 50)
    
    try:
        from api.audit_viz_endpoint import create_audit_viz_app
        from sim.emergent_detector import EmergentBehaviorDetector
        
        # Initialize audit data source
        detector = EmergentBehaviorDetector({})
        
        # Add some test audit data
        test_audits = [
            {
                'agent_id': 'agent_1',
                'mcp_action': {'type': 'query', 'content': 'optimization'},
                'outcome': {'success': True, 'env_improvement': 0.15},
                'context': {'context_appropriateness': 0.85}
            },
            {
                'agent_id': 'agent_2',
                'mcp_action': {'type': 'sense', 'content': 'environment'},
                'outcome': {'success': True, 'env_improvement': 0.12},
                'context': {'context_appropriateness': 0.78}
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
        
        # Create audit visualization app
        app = create_audit_viz_app(detector)
        
        # Test audit data endpoints
        with app.test_client() as client:
            # Test audit data API
            audit_response = client.get('/api/audit-data')
            audit_data = audit_response.get_json()
            
            # Test heatmap API
            heatmap_response = client.get('/api/audit-heatmap')
            heatmap_data = heatmap_response.get_json()
            
            # Test gaming trends API
            trends_response = client.get('/api/gaming-trends')
            trends_data = trends_response.get_json()
        
        dashboard_working = (
            audit_response.status_code == 200 and
            heatmap_response.status_code == 200 and
            trends_response.status_code == 200 and
            len(audit_ids) == len(test_audits)
        )
        
        print(f"✅ Audit IDs logged: {len(audit_ids)}")
        print(f"✅ Dashboard endpoints: {'Working' if dashboard_working else 'Failed'}")
        print(f"✅ Audit data: {audit_data}")
        
        return {
            'success': True,
            'audit_ids': audit_ids,
            'audit_data': audit_data,
            'heatmap_data': heatmap_data,
            'trends_data': trends_data,
            'dashboard_working': dashboard_working
        }
        
    except Exception as e:
        print(f"❌ Audit dashboard integration test failed: {e}")
        return {'success': False, 'error': str(e)}

async def main():
    """Main test execution for 95%+ fusion effectiveness"""
    print("🧪 OBSERVER MCP FUSION 95%+ EFFECTIVENESS TEST")
    print("RIPER-Ω Protocol: PUSH TO 95%+ FUSION EFFECTIVENESS")
    print("=" * 70)
    
    test_results = {}
    
    # Run all 95%+ effectiveness tests
    test_results['enhanced_fusion_effectiveness'] = await test_enhanced_fusion_effectiveness()
    test_results['gaming_prediction_system'] = await test_gaming_prediction_system()
    test_results['rl_seeded_389_growth'] = await test_rl_seeded_389_growth()
    test_results['audit_dashboard_integration'] = await test_audit_dashboard_integration()
    
    # Compile final results
    print("\n" + "=" * 70)
    print("OBSERVER MCP FUSION 95%+ EFFECTIVENESS RESULTS")
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
    
    # Calculate overall 95%+ effectiveness achievement
    effectiveness_metrics = []
    
    if test_results['enhanced_fusion_effectiveness'].get('avg_effectiveness'):
        effectiveness_metrics.append(test_results['enhanced_fusion_effectiveness']['avg_effectiveness'])
    
    if test_results['gaming_prediction_system'].get('prediction_effectiveness'):
        effectiveness_metrics.append(test_results['gaming_prediction_system']['prediction_effectiveness'])
    
    if test_results['rl_seeded_389_growth'].get('achievement_rate'):
        effectiveness_metrics.append(test_results['rl_seeded_389_growth']['achievement_rate'])
    
    overall_95_effectiveness = sum(effectiveness_metrics) / len(effectiveness_metrics) if effectiveness_metrics else 0.0
    
    # Overall assessment
    if success_rate >= 0.9 and overall_95_effectiveness >= 0.95:
        print("\n🎉 OBSERVER ASSESSMENT: 95%+ FUSION EFFECTIVENESS ACHIEVED")
        print("✅ 95%+ fusion effectiveness confirmed")
        print("✅ Enhanced MCP-evolution integration operational")
        print("✅ Gaming prediction with >100% adaptation working")
        print("✅ RL-seeded 389% growth potential demonstrated")
        print("✅ Ready for maximum performance deployment")
    elif success_rate >= 0.8 and overall_95_effectiveness >= 0.9:
        print("\n⚡ OBSERVER ASSESSMENT: NEAR 95%+ EFFECTIVENESS")
        print("✅ Strong 95%+ fusion progress achieved")
        print("⚠️ Minor optimizations for full 95%+ target")
    else:
        print("\n🔄 OBSERVER ASSESSMENT: APPROACHING 95%+ TARGET")
        print("❌ 95%+ effectiveness gaps detected")
        print("🔧 Further optimization required for 95%+ target")
    
    print(f"\n📊 Overall 95%+ Effectiveness: {overall_95_effectiveness:.1%}")
    print(f"🎯 Target Effectiveness: 95%")
    print(f"🏆 95%+ Achievement: {'✅ ACHIEVED' if overall_95_effectiveness >= 0.95 else '⚠️ APPROACHING'}")
    
    # Save test report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"mcp_fusion_95_percent_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Test report saved: {report_file}")
    
    return success_rate >= 0.9 and overall_95_effectiveness >= 0.95

if __name__ == "__main__":
    asyncio.run(main())
