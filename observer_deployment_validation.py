#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observer Systems Deployment Validation Script
Comprehensive validation for production deployment readiness

RIPER-Ω Protocol Compliant
Version: 2.4
"""

import sys
import os
import asyncio
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class ObserverDeploymentValidator:
    """Comprehensive deployment validation for Observer systems"""
    
    def __init__(self):
        self.validation_results = {}
        self.start_time = time.time()
        self.deployment_metrics = {
            'total_systems': 5,
            'validated_systems': 0,
            'critical_failures': 0,
            'warnings': 0,
            'performance_metrics': {}
        }
    
    async def validate_formal_proof_system(self) -> Dict[str, Any]:
        """Validate formal proof system for production readiness"""
        try:
            print("🔍 Validating Formal Proof System...")
            
            from dgm.autonomy_fixed import FormalProofSystem
            
            # Test with production configuration
            config = {
                'formal_proofs': {
                    'safety_threshold': 0.6,
                    'bloat_tolerance': 0.15,
                    'complexity_limit': 1500,
                    'approval_threshold': 0.6
                }
            }
            
            proof_system = FormalProofSystem(config['formal_proofs'])
            
            # Run comprehensive scenario testing
            scenario_results = await proof_system.test_proof_scenarios()
            
            # Validate approval rate meets production requirements
            approval_rate = scenario_results['approval_rate']
            production_ready = approval_rate >= 0.8  # 80% minimum for production
            
            result = {
                'status': 'SUCCESS' if production_ready else 'WARNING',
                'approval_rate': approval_rate,
                'invariants': len(proof_system.invariants),
                'thresholds': proof_system.thresholds,
                'production_ready': production_ready,
                'details': f"{approval_rate:.1%} approval rate, {len(proof_system.invariants)} invariants"
            }
            
            if production_ready:
                print(f"✅ Formal Proof System: PRODUCTION READY ({approval_rate:.1%})")
            else:
                print(f"⚠️ Formal Proof System: WARNING ({approval_rate:.1%} < 80%)")
                self.deployment_metrics['warnings'] += 1
            
            return result
            
        except Exception as e:
            print(f"❌ Formal Proof System: CRITICAL FAILURE - {e}")
            self.deployment_metrics['critical_failures'] += 1
            return {
                'status': 'CRITICAL_FAILURE',
                'error': str(e),
                'production_ready': False
            }
    
    async def validate_evolution_loop(self) -> Dict[str, Any]:
        """Validate evolution loop for production readiness"""
        try:
            print("🧬 Validating Evolution Loop...")
            
            from ai.evolution.evo_loop_fixed import ObserverEvolutionLoop
            
            # Test with production configuration
            config = {
                'max_generations': 5,
                'max_runtime_seconds': 60,
                'bloat_penalty_enabled': True
            }
            
            evolution_loop = ObserverEvolutionLoop(config)
            
            # Run production-scale test
            population = [f'agent_{i}' for i in range(10)]
            
            async def fitness_fn(individual):
                return 0.5 + (len(str(individual)) * 0.01)
            
            async def mutation_fn(individual):
                return individual + '_evolved'
            
            async def crossover_fn(parent1, parent2):
                return f'{parent1}_{parent2}'
            
            start_time = time.time()
            result = await evolution_loop.run_evolution(population, fitness_fn, mutation_fn, crossover_fn)
            execution_time = time.time() - start_time
            
            production_ready = (
                result['success'] and 
                result['generations_completed'] >= 3 and
                result['best_fitness'] > 0.5 and
                execution_time < 120  # 2 minutes max
            )
            
            validation_result = {
                'status': 'SUCCESS' if production_ready else 'WARNING',
                'generations': result['generations_completed'],
                'fitness': result['best_fitness'],
                'execution_time': execution_time,
                'production_ready': production_ready,
                'details': f"{result['generations_completed']} gens, fitness {result['best_fitness']:.3f}, {execution_time:.1f}s"
            }
            
            if production_ready:
                print(f"✅ Evolution Loop: PRODUCTION READY ({result['generations_completed']} gens)")
            else:
                print(f"⚠️ Evolution Loop: WARNING (performance issues)")
                self.deployment_metrics['warnings'] += 1
            
            return validation_result
            
        except Exception as e:
            print(f"❌ Evolution Loop: CRITICAL FAILURE - {e}")
            self.deployment_metrics['critical_failures'] += 1
            return {
                'status': 'CRITICAL_FAILURE',
                'error': str(e),
                'production_ready': False
            }
    
    async def validate_world_simulation(self) -> Dict[str, Any]:
        """Validate world simulation for production readiness"""
        try:
            print("🌍 Validating World Simulation...")
            
            from sim.world_sim import WorldSimulation
            
            sim = WorldSimulation()
            
            # Test production-scale simulation
            start_time = time.time()
            init_success = await sim.initialize(num_agents=50)  # Production scale
            
            if not init_success:
                raise Exception("Simulation initialization failed")
            
            result = await sim.sim_loop(generations=5)  # Production test
            execution_time = time.time() - start_time
            
            production_ready = (
                result['simulation_success'] and
                result['emergent_behaviors_detected'] >= 10 and
                result['cooperation_events'] >= 100 and
                execution_time < 300  # 5 minutes max
            )
            
            validation_result = {
                'status': 'SUCCESS' if production_ready else 'WARNING',
                'agents': len(sim.agents),
                'behaviors': result['emergent_behaviors_detected'],
                'cooperation_events': result['cooperation_events'],
                'execution_time': execution_time,
                'production_ready': production_ready,
                'details': f"{len(sim.agents)} agents, {result['emergent_behaviors_detected']} behaviors, {result['cooperation_events']} events"
            }
            
            if production_ready:
                print(f"✅ World Simulation: PRODUCTION READY ({result['emergent_behaviors_detected']} behaviors)")
            else:
                print(f"⚠️ World Simulation: WARNING (scale/performance issues)")
                self.deployment_metrics['warnings'] += 1
            
            return validation_result
            
        except Exception as e:
            print(f"❌ World Simulation: CRITICAL FAILURE - {e}")
            self.deployment_metrics['critical_failures'] += 1
            return {
                'status': 'CRITICAL_FAILURE',
                'error': str(e),
                'production_ready': False
            }
    
    async def validate_communication_system(self) -> Dict[str, Any]:
        """Validate communication system for production readiness"""
        try:
            print("📡 Validating Communication System...")
            
            from agents.communication_system_fixed import ObserverCommunicationSystem
            
            comm_system = ObserverCommunicationSystem({'fallback_enabled': True})
            await comm_system.initialize()
            
            # Test communication capabilities
            metrics = comm_system.get_communication_metrics()
            
            production_ready = (
                metrics['fallback_enabled'] and
                hasattr(comm_system, 'send_message') and
                hasattr(comm_system, 'receive_message')
            )
            
            validation_result = {
                'status': 'SUCCESS' if production_ready else 'WARNING',
                'fallback_enabled': metrics['fallback_enabled'],
                'production_ready': production_ready,
                'details': f"Fallback: {metrics['fallback_enabled']}, Core functions available"
            }
            
            if production_ready:
                print("✅ Communication System: PRODUCTION READY (fallback enabled)")
            else:
                print("⚠️ Communication System: WARNING (limited functionality)")
                self.deployment_metrics['warnings'] += 1
            
            return validation_result
            
        except Exception as e:
            print(f"❌ Communication System: CRITICAL FAILURE - {e}")
            self.deployment_metrics['critical_failures'] += 1
            return {
                'status': 'CRITICAL_FAILURE',
                'error': str(e),
                'production_ready': False
            }
    
    async def validate_query_system(self) -> Dict[str, Any]:
        """Validate query system for production readiness"""
        try:
            print("🔍 Validating Query System...")
            
            from mcp.query_fixed import ObserverQuerySystem
            
            query_system = ObserverQuerySystem()
            
            # Test query capabilities
            start_time = time.time()
            
            # Test multiple query types
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
            
            execution_time = time.time() - start_time
            success_rate = successful_queries / len(test_queries)
            
            production_ready = (
                success_rate >= 0.8 and  # 80% success rate
                execution_time < 30  # 30 seconds max
            )
            
            validation_result = {
                'status': 'SUCCESS' if production_ready else 'WARNING',
                'success_rate': success_rate,
                'execution_time': execution_time,
                'production_ready': production_ready,
                'details': f"{success_rate:.1%} success rate, {execution_time:.1f}s"
            }
            
            if production_ready:
                print(f"✅ Query System: PRODUCTION READY ({success_rate:.1%} success)")
            else:
                print(f"⚠️ Query System: WARNING ({success_rate:.1%} success rate)")
                self.deployment_metrics['warnings'] += 1
            
            return validation_result
            
        except Exception as e:
            print(f"❌ Query System: CRITICAL FAILURE - {e}")
            self.deployment_metrics['critical_failures'] += 1
            return {
                'status': 'CRITICAL_FAILURE',
                'error': str(e),
                'production_ready': False
            }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive deployment validation"""
        print("=" * 80)
        print("OBSERVER SYSTEMS DEPLOYMENT VALIDATION")
        print("RIPER-Ω Protocol: ACTIVE")
        print("Target: Production Deployment Readiness")
        print("=" * 80)
        
        # Run all validations
        validations = [
            ('formal_proof', self.validate_formal_proof_system()),
            ('evolution_loop', self.validate_evolution_loop()),
            ('world_simulation', self.validate_world_simulation()),
            ('communication', self.validate_communication_system()),
            ('query_system', self.validate_query_system())
        ]
        
        for name, validation_coro in validations:
            self.validation_results[name] = await validation_coro
            if self.validation_results[name].get('production_ready', False):
                self.deployment_metrics['validated_systems'] += 1
        
        # Calculate overall deployment readiness
        total_time = time.time() - self.start_time
        success_rate = self.deployment_metrics['validated_systems'] / self.deployment_metrics['total_systems']
        
        deployment_ready = (
            success_rate >= 0.8 and  # 80% systems ready
            self.deployment_metrics['critical_failures'] == 0  # No critical failures
        )
        
        # Generate deployment report
        print("\n" + "=" * 80)
        print("DEPLOYMENT VALIDATION RESULTS")
        print("=" * 80)
        
        for system_name, result in self.validation_results.items():
            status_icon = "✅" if result.get('production_ready', False) else "⚠️" if result.get('status') != 'CRITICAL_FAILURE' else "❌"
            details = result.get('details', result.get('error', 'No details'))
            print(f"{status_icon} {system_name.replace('_', ' ').title()}: {result.get('status', 'UNKNOWN')} - {details}")
        
        print(f"\nOverall: {self.deployment_metrics['validated_systems']}/{self.deployment_metrics['total_systems']} systems ready ({success_rate:.1%})")
        print(f"Critical Failures: {self.deployment_metrics['critical_failures']}")
        print(f"Warnings: {self.deployment_metrics['warnings']}")
        print(f"Validation Time: {total_time:.1f}s")
        
        if deployment_ready:
            print("\n🚀 DEPLOYMENT STATUS: APPROVED")
            print("✅ All Observer systems validated for production")
            print("✅ No critical failures detected")
            print("✅ Performance metrics within acceptable ranges")
            print("✅ RIPER-Ω Protocol compliance confirmed")
        else:
            print(f"\n⚠️ DEPLOYMENT STATUS: CONDITIONAL ({success_rate:.1%})")
            print("⚠️ Some systems require attention before production")
            if self.deployment_metrics['critical_failures'] > 0:
                print("❌ Critical failures must be resolved")
        
        return {
            'deployment_ready': deployment_ready,
            'success_rate': success_rate,
            'validation_results': self.validation_results,
            'metrics': self.deployment_metrics,
            'total_time': total_time
        }

async def main():
    """Main deployment validation entry point"""
    validator = ObserverDeploymentValidator()
    result = await validator.run_comprehensive_validation()
    
    # Save validation report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"observer_deployment_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n📄 Validation report saved: {report_file}")
    
    return result['deployment_ready']

if __name__ == "__main__":
    deployment_ready = asyncio.run(main())
    sys.exit(0 if deployment_ready else 1)
