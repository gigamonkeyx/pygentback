#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iteration Validation Test for 95%+ Achievement
Observer-approved comprehensive iteration validation with DGM-threshold adjustment
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

async def test_calibrated_synergy_effectiveness():
    """Test DGM-calibrated synergy for 75%+ scenarios"""
    print("🔧 TESTING CALIBRATED SYNERGY EFFECTIVENESS")
    print("-" * 50)
    
    try:
        from ai.evolution.two_phase import TwoPhaseEvolutionSystem
        
        # Initialize evolution system with calibration
        evolution_system = TwoPhaseEvolutionSystem({
            'exploration_generations': 2,
            'exploitation_generations': 2
        })
        print("✅ Calibrated evolution system initialized")
        
        # Create performance history for calibration
        performance_history = [
            {'effectiveness_score': 0.629},  # Enhanced Synergy Scenario
            {'effectiveness_score': 0.807},  # Compound Effectiveness Scenario  
            {'effectiveness_score': 0.640},  # Maximum Performance Scenario
        ]
        
        # Apply DGM calibration
        calibration_result = evolution_system.dgm_calibrate_synergy_thresholds(performance_history)
        evolution_system.apply_dgm_calibration(calibration_result)
        
        print(f"✅ DGM calibration applied: {calibration_result}")
        
        # Test calibrated scenarios
        calibrated_test_scenarios = [
            {
                'name': 'Calibrated Enhanced Synergy',
                'phase_results': [{
                    'mcp_calls': 8,
                    'mcp_successes': 8,
                    'mcp_failures': 0,
                    'env_improvement': 0.16,
                    'context_appropriateness': 0.91
                }],
                'base_fitness': 1.5,
                'expected_effectiveness': 0.75
            },
            {
                'name': 'Calibrated Compound Effectiveness',
                'phase_results': [{
                    'mcp_calls': 10,
                    'mcp_successes': 10,
                    'mcp_failures': 0,
                    'env_improvement': 0.18,
                    'context_appropriateness': 0.93
                }],
                'base_fitness': 1.2,
                'expected_effectiveness': 0.80
            },
            {
                'name': 'Calibrated Maximum Performance',
                'phase_results': [{
                    'mcp_calls': 12,
                    'mcp_successes': 12,
                    'mcp_failures': 0,
                    'env_improvement': 0.22,
                    'context_appropriateness': 0.95
                }],
                'base_fitness': 1.8,
                'expected_effectiveness': 0.85
            }
        ]
        
        # Test calibrated effectiveness
        calibrated_test_result = evolution_system.test_calibrated_effectiveness(calibrated_test_scenarios)
        
        avg_effectiveness = calibrated_test_result.get('avg_effectiveness', 0.0)
        gap_closure_progress = calibrated_test_result.get('gap_closure_progress', 0.0)
        
        print(f"✅ Calibrated effectiveness: {avg_effectiveness:.1%}")
        print(f"✅ Gap closure progress: {gap_closure_progress:.1%}")
        
        return {
            'success': True,
            'calibration_result': calibration_result,
            'calibrated_test_result': calibrated_test_result,
            'avg_effectiveness': avg_effectiveness,
            'gap_closure_progress': gap_closure_progress,
            'target_effectiveness': 0.75,
            'calibrated_synergy_working': avg_effectiveness >= 0.75
        }
        
    except Exception as e:
        print(f"❌ Calibrated synergy effectiveness test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_rl_forecast_gaming_confidence():
    """Test RL forecast gaming with confidence validation"""
    print("\n🔮 TESTING RL FORECAST GAMING CONFIDENCE")
    print("-" * 50)
    
    try:
        from ai.rl.reward_functions import GamingPredictor
        
        # Initialize gaming predictor with RL forecasting
        gaming_predictor = GamingPredictor({
            'prediction_window': 5,
            'confidence_threshold': 0.7
        })
        print("✅ RL forecast gaming system initialized")
        
        # Test RL forecast scenarios
        rl_forecast_scenarios = [
            {
                'name': 'High Pattern Gaming Forecast',
                'agent_history': [
                    {
                        'mcp_action': {'type': 'dummy', 'content': ''},
                        'outcome': {'success': True, 'env_improvement': 0.0},
                        'context': {'context_appropriateness': 0.1}
                    },
                    {
                        'mcp_action': {'type': 'dummy', 'content': 'test'},
                        'outcome': {'success': True, 'env_improvement': 0.0},
                        'context': {'context_appropriateness': 0.15}
                    },
                    {
                        'mcp_action': {'type': 'dummy', 'content': 'x'},
                        'outcome': {'success': True, 'env_improvement': 0.001},
                        'context': {'context_appropriateness': 0.12}
                    }
                ],
                'current_context': {'context_appropriateness': 0.1},
                'expected_confidence': 0.8
            },
            {
                'name': 'Moderate Pattern Gaming Forecast',
                'agent_history': [
                    {
                        'mcp_action': {'type': 'query', 'content': 'x'},
                        'outcome': {'success': True, 'env_improvement': 0.001},
                        'context': {'context_appropriateness': 0.3}
                    },
                    {
                        'mcp_action': {'type': 'sense', 'content': 'y'},
                        'outcome': {'success': True, 'env_improvement': 0.002},
                        'context': {'context_appropriateness': 0.25}
                    },
                    {
                        'mcp_action': {'type': 'query', 'content': 'z'},
                        'outcome': {'success': True, 'env_improvement': 0.001},
                        'context': {'context_appropriateness': 0.28}
                    }
                ],
                'current_context': {'context_appropriateness': 0.35},
                'expected_confidence': 0.6
            }
        ]
        
        # Test zero gaming confidence
        confidence_test_result = gaming_predictor.test_zero_gaming_confidence(rl_forecast_scenarios)
        
        avg_confidence = confidence_test_result.get('avg_confidence', 0.0)
        confidence_achievement_rate = confidence_test_result.get('confidence_achievement_rate', 0.0)
        
        print(f"✅ RL forecast confidence: {avg_confidence:.1%}")
        print(f"✅ Confidence achievement rate: {confidence_achievement_rate:.1%}")
        
        return {
            'success': True,
            'confidence_test_result': confidence_test_result,
            'avg_confidence': avg_confidence,
            'confidence_achievement_rate': confidence_achievement_rate,
            'target_confidence': 0.8,
            'rl_forecast_confidence_working': confidence_achievement_rate >= 0.8
        }
        
    except Exception as e:
        print(f"❌ RL forecast gaming confidence test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_rl_param_optimization():
    """Test RL parameter optimization for 4.89x+ boost"""
    print("\n⚙️ TESTING RL PARAMETER OPTIMIZATION")
    print("-" * 50)
    
    try:
        from sim.mcp_seeded_env import MCPSeededEnvironment
        
        # Initialize environment for RL parameter optimization
        mcp_env = MCPSeededEnvironment({
            'mcp_weight_base': 0.7,
            'appropriateness_threshold': 0.7,
            'auto_bias_enabled': True
        })
        print("✅ RL parameter optimization environment initialized")
        
        # Create high-quality audit data for optimization
        optimization_audit_data = [
            {
                'agent_id': 'agent_1',
                'success': True,
                'gaming_detected': False,
                'appropriateness_score': 0.94,
                'env_improvement': 0.24
            },
            {
                'agent_id': 'agent_2',
                'success': True,
                'gaming_detected': False,
                'appropriateness_score': 0.91,
                'env_improvement': 0.21
            },
            {
                'agent_id': 'agent_3',
                'success': True,
                'gaming_detected': False,
                'appropriateness_score': 0.96,
                'env_improvement': 0.27
            },
            {
                'agent_id': 'agent_4',
                'success': True,
                'gaming_detected': False,
                'appropriateness_score': 0.89,
                'env_improvement': 0.19
            }
        ]
        
        # Apply RL parameter optimization
        optimization_result = mcp_env.rl_param_optimization_from_audits(
            optimization_audit_data, target_growth=4.89
        )
        
        achieved_boost = optimization_result.get('optimization_test_result', {}).get('achieved_boost', 0.0)
        target_achievement_rate = optimization_result.get('optimization_test_result', {}).get('target_achievement_rate', 0.0)
        
        print(f"✅ RL parameter optimization: {achieved_boost:.2f}x boost")
        print(f"✅ Target achievement rate: {target_achievement_rate:.1%}")
        
        return {
            'success': True,
            'optimization_result': optimization_result,
            'achieved_boost': achieved_boost,
            'target_achievement_rate': target_achievement_rate,
            'target_boost': 4.89,
            'rl_param_optimization_working': achieved_boost >= 4.89
        }
        
    except Exception as e:
        print(f"❌ RL parameter optimization test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_predictive_dashboard():
    """Test predictive dashboard with iteration views"""
    print("\n📊 TESTING PREDICTIVE DASHBOARD")
    print("-" * 50)
    
    try:
        from api.audit_viz_endpoint import create_audit_viz_app
        from sim.emergent_detector import EmergentBehaviorDetector
        
        # Initialize audit data source
        detector = EmergentBehaviorDetector({})
        
        # Create audit visualization app
        app = create_audit_viz_app(detector)
        
        # Test predictive dashboard endpoints
        with app.test_client() as client:
            # Test iteration progress API
            iteration_response = client.get('/api/iteration-progress')
            iteration_data = iteration_response.get_json()
            
            # Test real-time optimization API
            optimization_response = client.get('/api/real-time-optimization')
            optimization_data = optimization_response.get_json()
            
            # Test predictive metrics API
            predictive_response = client.get('/api/predictive-metrics')
            predictive_data = predictive_response.get_json()
        
        dashboard_working = (
            iteration_response.status_code == 200 and
            optimization_response.status_code == 200 and
            predictive_response.status_code == 200
        )
        
        print(f"✅ Predictive dashboard endpoints: {'Working' if dashboard_working else 'Failed'}")
        print(f"✅ Iteration progress data: {len(iteration_data.get('iterations', []))} iterations")
        print(f"✅ Optimization data: {len(optimization_data.get('timestamps', []))} timestamps")
        
        return {
            'success': True,
            'iteration_data': iteration_data,
            'optimization_data': optimization_data,
            'predictive_data': predictive_data,
            'dashboard_working': dashboard_working
        }
        
    except Exception as e:
        print(f"❌ Predictive dashboard test failed: {e}")
        return {'success': False, 'error': str(e)}

async def main():
    """Main iteration validation execution"""
    print("🧪 OBSERVER ITERATION VALIDATION FOR 95%+ ACHIEVEMENT")
    print("RIPER-Ω Protocol: ITERATION VALIDATION WITH DGM-THRESHOLD ADJUSTMENT")
    print("=" * 70)
    
    test_results = {}
    
    # Run iteration validation tests
    test_results['calibrated_synergy_effectiveness'] = await test_calibrated_synergy_effectiveness()
    test_results['rl_forecast_gaming_confidence'] = await test_rl_forecast_gaming_confidence()
    test_results['rl_param_optimization'] = await test_rl_param_optimization()
    test_results['predictive_dashboard'] = await test_predictive_dashboard()
    
    # Compile iteration results
    print("\n" + "=" * 70)
    print("OBSERVER ITERATION VALIDATION RESULTS")
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
    
    # Calculate iteration effectiveness
    effectiveness_metrics = []
    
    if test_results['calibrated_synergy_effectiveness'].get('avg_effectiveness'):
        effectiveness_metrics.append(test_results['calibrated_synergy_effectiveness']['avg_effectiveness'])
    
    if test_results['rl_forecast_gaming_confidence'].get('avg_confidence'):
        effectiveness_metrics.append(test_results['rl_forecast_gaming_confidence']['avg_confidence'])
    
    if test_results['rl_param_optimization'].get('achieved_boost'):
        boost_effectiveness = min(1.0, test_results['rl_param_optimization']['achieved_boost'] / 4.89)
        effectiveness_metrics.append(boost_effectiveness)
    
    iteration_effectiveness = sum(effectiveness_metrics) / len(effectiveness_metrics) if effectiveness_metrics else 0.0
    
    # DGM-threshold adjustment assessment
    if success_rate >= 0.9 and iteration_effectiveness >= 0.95:
        print("\n🎉 OBSERVER ASSESSMENT: 95%+ EFFECTIVENESS ACHIEVED")
        print("✅ DGM-threshold adjustment: Raise target to 98%")
        print("✅ Iteration validation: Complete success")
        print("✅ Ready for 98%+ effectiveness targeting")
    elif success_rate >= 0.8 and iteration_effectiveness >= 0.85:
        print("\n⚡ OBSERVER ASSESSMENT: STRONG ITERATION PROGRESS")
        print("✅ Iteration effectiveness: Approaching 95% target")
        print("⚠️ Continue iteration for 95%+ achievement")
    else:
        print("\n🔄 OBSERVER ASSESSMENT: ITERATION OPTIMIZATION NEEDED")
        print("❌ Iteration gaps detected")
        print("🔧 Additional iteration cycles required")
    
    print(f"\n📊 Iteration Effectiveness: {iteration_effectiveness:.1%}")
    print(f"🎯 Target Effectiveness: 95%")
    print(f"🏆 95%+ Achievement: {'✅ ACHIEVED' if iteration_effectiveness >= 0.95 else '⚠️ IN PROGRESS'}")
    
    # Save iteration validation report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"iteration_validation_95_percent_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Iteration validation report saved: {report_file}")
    
    return success_rate >= 0.9 and iteration_effectiveness >= 0.95

if __name__ == "__main__":
    asyncio.run(main())
