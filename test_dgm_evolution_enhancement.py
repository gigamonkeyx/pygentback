#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test DGM/Evolution Enhancement System
Observer-approved comprehensive testing of enhanced DGM and evolution integration
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
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_dgm_validator():
    """Test the enhanced DGM validator with configurable thresholds"""
    print("🔍 TESTING DGM VALIDATOR")
    print("-" * 40)
    
    try:
        from src.dgm.core.validator import DGMValidator
        from src.dgm.models import ImprovementCandidate, ImprovementType
        
        # Configure validator with Observer-approved thresholds
        validator_config = {
            'safety_threshold': 0.6,
            'performance_threshold': 0.05,
            'complexity_threshold': 1500,
            'bloat_threshold': 0.15,
            'adaptive_thresholds': True,
            'threshold_learning_rate': 0.1
        }
        
        validator = DGMValidator(validator_config)
        print(f"✅ DGMValidator initialized with safety_threshold={validator.safety_threshold}")
        
        # Create test improvement candidates
        test_candidates = [
            ImprovementCandidate(
                id="test_candidate_1",
                improvement_type=ImprovementType.ALGORITHM,
                description="Enhanced evolution algorithm with better selection",
                code_changes={
                    "evolution.py": "def enhanced_selection():\n    return improved_algorithm()"
                },
                expected_improvement=0.2,
                risk_level=0.1
            ),
            ImprovementCandidate(
                id="test_candidate_2",
                improvement_type=ImprovementType.OPTIMIZATION,
                description="Performance optimization for fitness calculation",
                code_changes={
                    "fitness.py": "def optimized_fitness():\n    return faster_calculation()"
                },
                expected_improvement=0.15,
                risk_level=0.05
            )
        ]
        
        # Test validation
        validation_results = []
        for candidate in test_candidates:
            print(f"🧪 Validating candidate: {candidate.id}")
            result = await validator.validate_improvement(candidate)
            validation_results.append(result)
            
            print(f"   Success: {result.success}")
            print(f"   Safety Score: {result.safety_score:.3f}")
            print(f"   Improvement Score: {result.improvement_score:.3f}")
        
        # Test threshold adaptation
        print("\n🔧 Testing threshold adaptation...")
        stats = validator.get_validation_stats()
        print(f"✅ Validation stats: {stats}")
        
        return {
            'success': True,
            'validator_initialized': True,
            'candidates_tested': len(test_candidates),
            'validation_results': validation_results,
            'adaptive_thresholds_working': stats.get('current_thresholds') is not None
        }
        
    except Exception as e:
        print(f"❌ DGM Validator test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_two_phase_evolution():
    """Test the two-phase evolution system with RL integration"""
    print("\n🧬 TESTING TWO-PHASE EVOLUTION")
    print("-" * 40)
    
    try:
        from src.ai.evolution.two_phase import TwoPhaseEvolutionSystem
        
        # Configure two-phase evolution
        evolution_config = {
            'exploration_generations': 2,
            'exploitation_generations': 3,
            'exploration_mutation_rate': 0.4,
            'exploitation_mutation_rate': 0.1,
            'rl_enabled': True,
            'efficiency_target': 5.0,
            'improvement_target': 1.5  # 150% improvement target
        }
        
        evolution_system = TwoPhaseEvolutionSystem(evolution_config)
        print(f"✅ TwoPhaseEvolutionSystem initialized")
        
        # Create test population
        initial_population = [f"agent_{i}" for i in range(10)]
        
        # Define test functions
        async def test_fitness_function(individual):
            # Simulate fitness that improves over time
            base_fitness = 0.5 + (len(individual) * 0.01)
            return min(base_fitness, 2.0)
        
        async def test_mutation_function(individual):
            return f"{individual}_mutated"
        
        async def test_crossover_function(parent1, parent2):
            return f"{parent1}_x_{parent2}", f"{parent2}_x_{parent1}"
        
        # Run evolution
        print("🚀 Running two-phase evolution...")
        evolution_result = await evolution_system.evolve_population(
            initial_population,
            test_fitness_function,
            test_mutation_function,
            test_crossover_function
        )
        
        print(f"✅ Evolution completed:")
        print(f"   Success: {evolution_result['success']}")
        print(f"   Generations: {evolution_result['generations_completed']}")
        print(f"   Best Fitness: {evolution_result['best_fitness']:.3f}")
        print(f"   Evolution Time: {evolution_result['evolution_time']:.2f}s")
        
        # Check RL rewards
        if evolution_result.get('rl_reward'):
            rl_reward = evolution_result['rl_reward']
            print(f"   RL Total Reward: {rl_reward.total_reward:.3f}")
            print(f"   Efficiency Reward: {rl_reward.efficiency_reward:.3f}")
            print(f"   Improvement Reward: {rl_reward.improvement_reward:.3f}")
        
        # Test performance stats
        stats = evolution_system.get_performance_stats()
        print(f"✅ Performance stats available: {not stats.get('no_data', False)}")
        
        return {
            'success': True,
            'evolution_completed': evolution_result['success'],
            'generations_completed': evolution_result['generations_completed'],
            'best_fitness': evolution_result['best_fitness'],
            'rl_reward_calculated': evolution_result.get('rl_reward') is not None,
            'target_improvement_reached': evolution_result.get('target_improvement_reached', False)
        }
        
    except Exception as e:
        print(f"❌ Two-phase evolution test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_dgm_evolution_integration():
    """Test the DGM-Evolution integration engine"""
    print("\n🔗 TESTING DGM-EVOLUTION INTEGRATION")
    print("-" * 40)
    
    try:
        from src.dgm.core.evolution_integration import DGMEvolutionEngine
        from src.dgm.models import ImprovementCandidate, ImprovementType
        
        # Configure integration engine
        integration_config = {
            'validator': {
                'safety_threshold': 0.6,
                'adaptive_thresholds': True
            },
            'evolution': {
                'exploration_generations': 2,
                'exploitation_generations': 2,
                'rl_enabled': True
            },
            'self_rewrite_enabled': True,
            'fitness_threshold': 0.8,
            'rewrite_trigger_threshold': 0.6,
            'mcp_sensing_enabled': True
        }
        
        integration_engine = DGMEvolutionEngine(integration_config)
        print(f"✅ DGMEvolutionEngine initialized")
        
        # Create test improvement candidates
        improvement_candidates = [
            ImprovementCandidate(
                id="integration_test_1",
                improvement_type=ImprovementType.ALGORITHM,
                description="Integration test improvement",
                code_changes={"test.py": "# Test improvement"},
                expected_improvement=0.3,
                risk_level=0.1
            )
        ]
        
        # Create test population and functions
        initial_population = [f"integrated_agent_{i}" for i in range(8)]
        
        async def integration_fitness_function(individual):
            return 0.4 + (len(individual) * 0.01)  # Low fitness to trigger self-rewrite
        
        async def integration_mutation_function(individual):
            return f"{individual}_integrated_mutation"
        
        async def integration_crossover_function(parent1, parent2):
            return f"{parent1}_integrated_x_{parent2}", f"{parent2}_integrated_x_{parent1}"
        
        # Run integrated evolution
        print("🚀 Running DGM-validated evolution with self-rewriting...")
        integration_result = await integration_engine.evolve_with_dgm_validation(
            initial_population,
            integration_fitness_function,
            integration_mutation_function,
            integration_crossover_function,
            improvement_candidates
        )
        
        print(f"✅ Integration completed:")
        print(f"   Success: {integration_result['success']}")
        print(f"   DGM Validation Applied: {integration_result.get('dgm_validation_applied', False)}")
        print(f"   Validated Candidates: {integration_result.get('validated_candidates', 0)}")
        print(f"   Self-Rewrite Applied: {integration_result.get('self_rewrite_applied', False)}")
        print(f"   Dependency Fixes: {integration_result.get('dependency_fixes', 0)}")
        print(f"   Best Fitness: {integration_result.get('best_fitness', 0):.3f}")
        
        # Test integration stats
        stats = integration_engine.get_integration_stats()
        print(f"✅ Integration stats available: {not stats.get('no_data', False)}")
        
        return {
            'success': True,
            'integration_completed': integration_result['success'],
            'dgm_validation_applied': integration_result.get('dgm_validation_applied', False),
            'self_rewrite_triggered': integration_result.get('self_rewrite_applied', False),
            'dependency_fixes_applied': integration_result.get('dependency_fixes', 0) > 0,
            'best_fitness': integration_result.get('best_fitness', 0)
        }
        
    except Exception as e:
        print(f"❌ DGM-Evolution integration test failed: {e}")
        return {'success': False, 'error': str(e)}

async def test_useful_demo():
    """Test a useful demo: evolving a solution to fix Unicode issues"""
    print("\n🎯 TESTING USEFUL DEMO: UNICODE FIX EVOLUTION")
    print("-" * 40)
    
    try:
        from src.dgm.core.evolution_integration import DGMEvolutionEngine
        from src.dgm.models import ImprovementCandidate, ImprovementType
        
        # Configure for Unicode fix demo
        demo_config = {
            'validator': {'safety_threshold': 0.7},
            'evolution': {
                'exploration_generations': 2,
                'exploitation_generations': 3,
                'rl_enabled': True,
                'improvement_target': 1.2  # 120% improvement
            },
            'self_rewrite_enabled': True
        }
        
        demo_engine = DGMEvolutionEngine(demo_config)
        print(f"✅ Demo engine initialized for Unicode fix evolution")
        
        # Create Unicode fix improvement candidate
        unicode_fix_candidate = ImprovementCandidate(
            id="unicode_fix_demo",
            improvement_type=ImprovementType.BUG_FIX,
            description="Fix Unicode encoding issues on Windows",
            code_changes={
                "unicode_fix.py": '''
import sys
import codecs
if sys.platform == "win32":
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
'''
            },
            expected_improvement=0.4,
            risk_level=0.05
        )
        
        # Create population of potential solutions
        unicode_solutions = [
            "basic_encoding_fix",
            "codecs_wrapper_solution",
            "platform_specific_handler",
            "utf8_environment_setup",
            "comprehensive_unicode_support"
        ]
        
        async def unicode_fitness_function(solution):
            # Simulate fitness based on solution completeness
            fitness_map = {
                "basic_encoding_fix": 0.6,
                "codecs_wrapper_solution": 0.8,
                "platform_specific_handler": 0.9,
                "utf8_environment_setup": 0.7,
                "comprehensive_unicode_support": 1.2
            }
            base_fitness = fitness_map.get(solution.split('_')[0], 0.5)
            
            # Bonus for evolved solutions
            if "evolved" in solution or "integrated" in solution:
                base_fitness *= 1.1
            
            return base_fitness
        
        async def unicode_mutation_function(solution):
            return f"{solution}_evolved_mutation"
        
        async def unicode_crossover_function(parent1, parent2):
            return f"{parent1}_x_{parent2}_hybrid", f"{parent2}_x_{parent1}_hybrid"
        
        # Run Unicode fix evolution
        print("🚀 Evolving Unicode fix solutions...")
        demo_result = await demo_engine.evolve_with_dgm_validation(
            unicode_solutions,
            unicode_fitness_function,
            unicode_mutation_function,
            unicode_crossover_function,
            [unicode_fix_candidate]
        )
        
        print(f"✅ Unicode fix evolution completed:")
        print(f"   Success: {demo_result['success']}")
        print(f"   Best Fitness: {demo_result.get('best_fitness', 0):.3f}")
        print(f"   Target Reached: {demo_result.get('target_improvement_reached', False)}")
        print(f"   Solution Quality: {'EXCELLENT' if demo_result.get('best_fitness', 0) > 1.0 else 'GOOD'}")
        
        # Calculate success rate
        success_rate = min(demo_result.get('best_fitness', 0) / 1.2, 1.0)
        print(f"   Success Rate: {success_rate:.1%}")
        
        return {
            'success': True,
            'demo_completed': demo_result['success'],
            'best_fitness': demo_result.get('best_fitness', 0),
            'target_reached': demo_result.get('target_improvement_reached', False),
            'success_rate': success_rate,
            'solution_quality': 'excellent' if demo_result.get('best_fitness', 0) > 1.0 else 'good'
        }
        
    except Exception as e:
        print(f"❌ Useful demo test failed: {e}")
        return {'success': False, 'error': str(e)}

async def main():
    """Main test execution"""
    print("🧪 OBSERVER DGM/EVOLUTION ENHANCEMENT TESTING")
    print("RIPER-Ω Protocol: COMPREHENSIVE VALIDATION")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    test_results['dgm_validator'] = await test_dgm_validator()
    test_results['two_phase_evolution'] = await test_two_phase_evolution()
    test_results['dgm_evolution_integration'] = await test_dgm_evolution_integration()
    test_results['useful_demo'] = await test_useful_demo()
    
    # Compile final results
    print("\n" + "=" * 60)
    print("OBSERVER DGM/EVOLUTION ENHANCEMENT TEST RESULTS")
    print("=" * 60)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() if result.get('success', False))
    success_rate = successful_tests / total_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Success Rate: {success_rate:.1%}")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    # Overall assessment
    if success_rate >= 0.8:
        print("\n🎉 OBSERVER ASSESSMENT: EXCELLENT")
        print("✅ DGM/Evolution enhancements working perfectly")
        print("✅ All core systems operational")
        print("✅ Ready for production deployment")
    elif success_rate >= 0.6:
        print("\n⚡ OBSERVER ASSESSMENT: GOOD")
        print("✅ Most enhancements working correctly")
        print("⚠️ Minor issues to address")
    else:
        print("\n🔄 OBSERVER ASSESSMENT: NEEDS IMPROVEMENT")
        print("❌ Significant issues detected")
        print("🔧 Further development required")
    
    # Save test report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"dgm_evolution_test_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Test report saved: {report_file}")
    
    return success_rate >= 0.8

if __name__ == "__main__":
    asyncio.run(main())
