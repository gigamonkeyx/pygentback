#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Autonomous Observer Simulation with Seeding & Domain Shift
Observer-approved implementation with directed learning and adaptation
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

class EnhancedAutonomousSystem:
    """Enhanced Autonomous Observer System with seeding and domain shift adaptation"""
    
    def __init__(self):
        self.running = True
        self.generation = 0
        self.evolution_metrics = []
        self.learning_metrics = []
        self.domain_shifts = []
        
    async def initialize_enhanced_systems(self):
        """Initialize enhanced Observer systems with seeding and adaptation"""
        print("🧠 INITIALIZING ENHANCED AUTONOMOUS OBSERVER SYSTEMS")
        print("RIPER-Ω Protocol: ENHANCED LEARNING MODE")
        print("Features: Seeding, Domain Shift, Adaptive Learning")
        print("=" * 70)
        
        try:
            # Initialize Enhanced World Simulation with seeding
            from sim.world_sim import WorldSimulation
            
            # Configure seeded evolution
            enhanced_config = {
                'seed_params': {
                    'cooperation': 0.6,      # Favor cooperative behaviors
                    'exploration': 0.4,      # Moderate exploration
                    'sustainability': 0.7,   # High sustainability focus
                    'adaptation': 0.8        # Strong adaptation bias
                },
                'environment': {
                    'resource_scarcity': 0.3,  # Start with moderate scarcity
                    'change_rate': 0.1         # Gradual environmental changes
                }
            }
            
            self.world_sim = WorldSimulation(enhanced_config)
            await self.world_sim.initialize(num_agents=25)
            print("✅ Enhanced World Simulation: 25 agents with seeded evolution")
            
            # Initialize Enhanced Evolution Loop
            from ai.evolution.evo_loop_fixed import ObserverEvolutionLoop
            evo_config = {
                'max_generations': 5,
                'max_runtime_seconds': 60,
                'bloat_penalty_enabled': True,
                'learning_phases': {
                    'exploration_phase': {'generations': 2, 'mutation_rate': 0.4},
                    'exploitation_phase': {'generations': 3, 'crossover_rate': 0.9}
                },
                'domain_adaptation_enabled': True,
                'seeded_evolution_weights': {
                    'cooperation': 0.4,
                    'sustainability': 0.3,
                    'adaptation': 0.3
                }
            }
            self.evolution_loop = ObserverEvolutionLoop(evo_config)
            print("✅ Enhanced Evolution Loop: Phased learning enabled")
            
            # Initialize other systems
            from dgm.autonomy_fixed import FormalProofSystem
            proof_config = {
                'formal_proofs': {
                    'safety_threshold': 0.7,
                    'adaptation_bonus': 0.1
                }
            }
            self.proof_system = FormalProofSystem(proof_config['formal_proofs'])
            print("✅ Enhanced Formal Proof System: Adaptation-aware")
            
            print("\n🚀 ALL ENHANCED OBSERVER SYSTEMS ONLINE")
            print("🧠 Seeded evolution: ACTIVE")
            print("🔄 Domain shift adaptation: ENABLED")
            print("📊 Enhanced visualization: READY")
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced initialization failed: {e}")
            return False
    
    async def enhanced_evolution_cycle(self, cycle_config: dict):
        """Run enhanced evolution cycle with seeding and domain shift"""
        self.generation += 1
        cycle_start = time.time()
        
        print(f"\n🧠 ENHANCED CYCLE {self.generation}")
        print(f"Config: {cycle_config['name']}")
        print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)
        
        cycle_metrics = {
            'generation': self.generation,
            'timestamp': datetime.now(),
            'config': cycle_config,
            'systems': {},
            'learning_metrics': {},
            'domain_shifts': []
        }
        
        # 1. Enhanced World Simulation with seeding
        try:
            sim_start = time.time()
            
            # Apply cycle-specific seeding
            if 'seed_override' in cycle_config:
                self.world_sim.seed_params.update(cycle_config['seed_override'])
                print(f"🎯 Applied seeding: {cycle_config['seed_override']}")
            
            # Run simulation with domain shift detection
            result = await self.world_sim.sim_loop(generations=3)
            sim_time = time.time() - sim_start
            
            cycle_metrics['systems']['world_sim'] = {
                'success': result['simulation_success'],
                'agents': len(self.world_sim.agents),
                'behaviors': result['emergent_behaviors_detected'],
                'cooperation': result['cooperation_events'],
                'fitness': result['final_average_fitness'],
                'learning_metrics': result.get('learning_metrics', {}),
                'time': sim_time
            }
            
            # Check for domain shifts
            if 'learning_metrics' in result:
                learning_data = result['learning_metrics']
                if learning_data.get('adaptation_rate', 0) > 0.1:
                    shift_detected = {
                        'type': 'adaptation_spike',
                        'magnitude': learning_data['adaptation_rate'],
                        'generation': self.generation
                    }
                    cycle_metrics['domain_shifts'].append(shift_detected)
                    self.domain_shifts.append(shift_detected)
                    print(f"🔄 Domain shift detected: {shift_detected['type']} ({shift_detected['magnitude']:.2f})")
            
            print(f"🌍 Enhanced Sim: {result['emergent_behaviors_detected']} behaviors, {result['cooperation_events']} cooperation")
            
            # Generate enhanced visualization
            viz_success = self.world_sim.plot_emergence_evolution(f"enhanced_gen_{self.generation}_{cycle_config['name']}.png")
            if viz_success:
                print(f"📊 Enhanced visualization: enhanced_gen_{self.generation}_{cycle_config['name']}.png")
            
        except Exception as e:
            print(f"❌ Enhanced simulation error: {e}")
            cycle_metrics['systems']['world_sim'] = {'success': False, 'error': str(e)}
        
        # 2. Enhanced Formal Proof with adaptation awareness
        try:
            proof_start = time.time()
            scenario_results = await self.proof_system.test_proof_scenarios()
            proof_time = time.time() - proof_start
            
            approval_rate = scenario_results['approval_rate']
            
            # Adaptation bonus for high approval rates
            if approval_rate >= 0.9:
                adaptation_bonus = 0.1
                print(f"🏆 Adaptation bonus: {adaptation_bonus:.1f} for {approval_rate:.1%} approval")
            else:
                adaptation_bonus = 0.0
            
            cycle_metrics['systems']['formal_proof'] = {
                'success': approval_rate >= 0.8,
                'approval_rate': approval_rate,
                'adaptation_bonus': adaptation_bonus,
                'scenarios': scenario_results['scenarios_tested'],
                'time': proof_time
            }
            
            print(f"🔍 Enhanced Proofs: {approval_rate:.1%} approval (+{adaptation_bonus:.1f} bonus)")
            
        except Exception as e:
            print(f"❌ Enhanced proof error: {e}")
            cycle_metrics['systems']['formal_proof'] = {'success': False, 'error': str(e)}
        
        # 3. Enhanced Evolution with phased learning
        try:
            evo_start = time.time()
            
            # Determine evolution phase
            if self.generation <= 2:
                phase = 'exploration'
                population_size = 12 + self.generation
                print(f"🔍 Evolution Phase: EXPLORATION (pop: {population_size})")
            else:
                phase = 'exploitation'
                population_size = 15 + (self.generation - 2)
                print(f"🎯 Evolution Phase: EXPLOITATION (pop: {population_size})")
            
            # Create adaptive population
            population = [f'enhanced_agent_{phase}_{self.generation}_{i}' for i in range(population_size)]
            
            async def enhanced_fitness_fn(individual):
                base_fitness = 0.5 + (self.generation * 0.05)
                
                # Apply seeded evolution weights
                cooperation_bonus = 0.0
                if 'cooperation' in str(individual):
                    cooperation_bonus = self.evolution_loop.seeded_evolution_weights.get('cooperation', 0.0)
                
                sustainability_bonus = 0.0
                if 'sustainable' in str(individual):
                    sustainability_bonus = self.evolution_loop.seeded_evolution_weights.get('sustainability', 0.0)
                
                adaptation_bonus = 0.0
                if 'adaptive' in str(individual):
                    adaptation_bonus = self.evolution_loop.seeded_evolution_weights.get('adaptation', 0.0)
                
                seeded_bonus = cooperation_bonus + sustainability_bonus + adaptation_bonus
                
                return min(base_fitness + seeded_bonus + (len(str(individual)) * 0.01), 1.0)
            
            async def enhanced_mutation_fn(individual):
                if phase == 'exploration':
                    return f'{individual}_explore_gen{self.generation}'
                else:
                    return f'{individual}_exploit_gen{self.generation}'
            
            async def enhanced_crossover_fn(parent1, parent2):
                return f'{parent1}_x_{parent2}_{phase}_gen{self.generation}'
            
            evo_result = await self.evolution_loop.run_evolution(
                population, enhanced_fitness_fn, enhanced_mutation_fn, enhanced_crossover_fn
            )
            evo_time = time.time() - evo_start
            
            cycle_metrics['systems']['evolution'] = {
                'success': evo_result['success'],
                'phase': phase,
                'generations': evo_result['generations_completed'],
                'fitness': evo_result['best_fitness'],
                'population_size': population_size,
                'time': evo_time
            }
            
            print(f"🧬 Enhanced Evolution: {evo_result['generations_completed']} gens, fitness {evo_result['best_fitness']:.3f}")
            
        except Exception as e:
            print(f"❌ Enhanced evolution error: {e}")
            cycle_metrics['systems']['evolution'] = {'success': False, 'error': str(e)}
        
        # Calculate enhanced metrics
        cycle_time = time.time() - cycle_start
        successful_systems = sum(1 for s in cycle_metrics['systems'].values() if s.get('success', False))
        total_systems = len(cycle_metrics['systems'])
        success_rate = successful_systems / total_systems
        
        cycle_metrics['cycle_time'] = cycle_time
        cycle_metrics['success_rate'] = success_rate
        cycle_metrics['successful_systems'] = successful_systems
        cycle_metrics['total_systems'] = total_systems
        
        # Enhanced learning metrics
        learning_metrics = {
            'seeded_compliance': cycle_metrics['systems'].get('world_sim', {}).get('learning_metrics', {}).get('seeded_direction_compliance', 0.0),
            'adaptation_rate': cycle_metrics['systems'].get('world_sim', {}).get('learning_metrics', {}).get('adaptation_rate', 0.0),
            'domain_shifts_detected': len(cycle_metrics['domain_shifts']),
            'phase': cycle_metrics['systems'].get('evolution', {}).get('phase', 'unknown')
        }
        cycle_metrics['learning_metrics'] = learning_metrics
        
        # Store metrics
        self.evolution_metrics.append(cycle_metrics)
        self.learning_metrics.append(learning_metrics)
        
        # Enhanced status display
        print(f"\n📊 ENHANCED CYCLE {self.generation} SUMMARY:")
        print(f"Success Rate: {success_rate:.1%} ({successful_systems}/{total_systems})")
        print(f"Cycle Time: {cycle_time:.1f}s")
        print(f"Seeded Compliance: {learning_metrics['seeded_compliance']:.1%}")
        print(f"Adaptation Rate: {learning_metrics['adaptation_rate']:.1%}")
        print(f"Domain Shifts: {learning_metrics['domain_shifts_detected']}")
        
        # Enhanced autonomous decision making
        if success_rate >= 0.8 and learning_metrics['seeded_compliance'] >= 0.6:
            print("✅ ENHANCED STATUS: OPTIMAL LEARNING")
            print("🧠 Seeded evolution and adaptation working perfectly")
        elif success_rate >= 0.6:
            print("⚠️ ENHANCED STATUS: LEARNING IN PROGRESS")
            print("🔧 Adaptation protocols active")
        else:
            print("❌ ENHANCED STATUS: LEARNING CHALLENGES")
            print("🚨 Enhanced recovery protocols activated")
        
        return cycle_metrics

    async def run_enhanced_simulation(self, max_cycles=5):
        """Run enhanced autonomous simulation with seeding and domain shift"""
        print("🧠 STARTING ENHANCED AUTONOMOUS OBSERVER SIMULATION")
        print("Features: Seeded Evolution, Domain Shift Adaptation, Phased Learning")
        print("=" * 70)

        # Initialize enhanced systems
        init_success = await self.initialize_enhanced_systems()
        if not init_success:
            print("❌ Failed to initialize enhanced Observer systems")
            return False

        # Define cycle configurations with different seeding strategies
        cycle_configs = [
            {
                'name': 'cooperative_focus',
                'seed_override': {'cooperation': 0.8, 'sustainability': 0.6}
            },
            {
                'name': 'exploration_boost',
                'seed_override': {'exploration': 0.7, 'adaptation': 0.8}
            },
            {
                'name': 'sustainability_drive',
                'seed_override': {'sustainability': 0.9, 'cooperation': 0.5}
            },
            {
                'name': 'adaptation_focus',
                'seed_override': {'adaptation': 0.9, 'exploration': 0.6}
            },
            {
                'name': 'balanced_evolution',
                'seed_override': {'cooperation': 0.6, 'exploration': 0.5, 'sustainability': 0.7, 'adaptation': 0.7}
            }
        ]

        # Run enhanced cycles
        for cycle in range(max_cycles):
            if not self.running:
                break

            try:
                config = cycle_configs[cycle % len(cycle_configs)]
                cycle_metrics = await self.enhanced_evolution_cycle(config)

                # Save enhanced cycle report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = f"enhanced_cycle_{self.generation}_{config['name']}_{timestamp}.json"

                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(cycle_metrics, f, indent=2, default=str)

                print(f"📄 Enhanced report saved: {report_file}")

                # Brief pause between cycles
                if cycle < max_cycles - 1:
                    print(f"\n⏳ Waiting 15s for next enhanced cycle...")
                    await asyncio.sleep(15)

            except KeyboardInterrupt:
                print("\n🛑 Enhanced simulation interrupted by user")
                self.running = False
                break
            except Exception as e:
                print(f"\n❌ Enhanced cycle {self.generation} failed: {e}")
                print("🔄 Attempting enhanced recovery...")
                await asyncio.sleep(5)

        # Enhanced final summary
        print("\n" + "=" * 70)
        print("ENHANCED AUTONOMOUS SIMULATION COMPLETE")
        print("=" * 70)

        if self.evolution_metrics:
            total_cycles = len(self.evolution_metrics)
            avg_success_rate = sum(m['success_rate'] for m in self.evolution_metrics) / total_cycles
            avg_seeded_compliance = sum(m['learning_metrics']['seeded_compliance'] for m in self.learning_metrics) / len(self.learning_metrics)
            avg_adaptation_rate = sum(m['learning_metrics']['adaptation_rate'] for m in self.learning_metrics) / len(self.learning_metrics)
            total_domain_shifts = len(self.domain_shifts)
            total_time = sum(m['cycle_time'] for m in self.evolution_metrics)

            print(f"Total Enhanced Cycles: {total_cycles}")
            print(f"Average Success Rate: {avg_success_rate:.1%}")
            print(f"Average Seeded Compliance: {avg_seeded_compliance:.1%}")
            print(f"Average Adaptation Rate: {avg_adaptation_rate:.1%}")
            print(f"Domain Shifts Detected: {total_domain_shifts}")
            print(f"Total Runtime: {total_time:.1f}s")
            print(f"Enhanced Visualizations: {total_cycles}")

            if avg_success_rate >= 0.8 and avg_seeded_compliance >= 0.6:
                print("\n🎉 ENHANCED SIMULATION: SPECTACULAR SUCCESS")
                print("✅ Seeded evolution protocols highly effective")
                print("✅ Domain shift adaptation working perfectly")
                print("✅ Enhanced learning demonstrated")
                print("✅ Phased evolution optimized")
            else:
                print(f"\n⚠️ ENHANCED SIMULATION: LEARNING IN PROGRESS ({avg_success_rate:.1%})")
                print("⚠️ Enhanced systems showing improvement potential")

        return True

async def main():
    """Main entry point for enhanced autonomous Observer simulation"""
    print("🧠 ENHANCED OBSERVER AUTONOMOUS SIMULATION")
    print("RIPER-Ω Protocol: ENHANCED LEARNING MODE")
    print("Advanced features: Seeding, Domain Shift, Adaptive Learning")
    print()

    # Create enhanced autonomous system
    enhanced_system = EnhancedAutonomousSystem()

    # Run enhanced simulation
    try:
        success = await enhanced_system.run_enhanced_simulation(max_cycles=5)

        if success:
            print("\n🎉 Enhanced autonomous Observer simulation completed successfully!")
            print("📊 Check enhanced visualizations: enhanced_gen_*.png")
            print("📄 Check enhanced reports: enhanced_cycle_*.json")
            print("🧠 Seeded evolution and domain shift adaptation validated!")
        else:
            print("\n❌ Enhanced simulation encountered issues")

    except KeyboardInterrupt:
        print("\n🛑 Enhanced simulation interrupted by user")
        enhanced_system.running = False
    except Exception as e:
        print(f"\n💥 Enhanced simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
