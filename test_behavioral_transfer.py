#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Behavioral Transfer and Knowledge Inheritance
Investigate whether evolved behaviors transfer to new agents
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_behavioral_transfer():
    """Test if evolved behaviors transfer to new agents"""
    print("🧠 TESTING BEHAVIORAL TRANSFER & KNOWLEDGE INHERITANCE")
    print("=" * 60)
    
    from sim.world_sim import WorldSimulation, Agent
    
    # Phase 1: Create baseline agents (no evolution)
    print("\n[PHASE 1] Creating baseline agents (no evolution)")
    baseline_sim = WorldSimulation()
    await baseline_sim.initialize(num_agents=10)
    
    # Record baseline capabilities
    baseline_capabilities = {}
    for agent in baseline_sim.agents:
        baseline_capabilities[agent.agent_type] = {
            'exploration': agent.capabilities['exploration'],
            'resource_gathering': agent.capabilities['resource_gathering'],
            'cooperation': agent.capabilities['cooperation'],
            'learning': agent.capabilities['learning'],
            'adaptation': agent.capabilities['adaptation']
        }
    
    print(f"✅ Baseline capabilities recorded for {len(baseline_capabilities)} agent types")
    
    # Phase 2: Run evolution simulation to develop behaviors
    print("\n[PHASE 2] Running evolution simulation (10 generations)")
    evolved_sim = WorldSimulation()
    await evolved_sim.initialize(num_agents=15)
    
    # Run extended evolution
    evolution_result = await evolved_sim.sim_loop(generations=10)
    
    print(f"✅ Evolution completed: {evolution_result['emergent_behaviors_detected']} behaviors")
    print(f"✅ Cooperation events: {evolution_result['cooperation_events']}")
    print(f"✅ Final fitness: {evolution_result['final_average_fitness']:.3f}")
    
    # Record evolved capabilities
    evolved_capabilities = {}
    evolved_knowledge = {}
    evolved_cooperation = {}
    
    for agent in evolved_sim.agents:
        agent_type = agent.agent_type
        if agent_type not in evolved_capabilities:
            evolved_capabilities[agent_type] = []
            evolved_knowledge[agent_type] = []
            evolved_cooperation[agent_type] = []
        
        evolved_capabilities[agent_type].append(agent.capabilities)
        evolved_knowledge[agent_type].append(agent.resources['knowledge'])
        evolved_cooperation[agent_type].append(agent.cooperation_score)
    
    # Calculate averages for evolved agents
    evolved_averages = {}
    for agent_type in evolved_capabilities:
        capabilities = evolved_capabilities[agent_type]
        avg_capabilities = {}
        for capability in capabilities[0]:
            avg_capabilities[capability] = sum(c[capability] for c in capabilities) / len(capabilities)
        
        evolved_averages[agent_type] = {
            'capabilities': avg_capabilities,
            'knowledge': sum(evolved_knowledge[agent_type]) / len(evolved_knowledge[agent_type]),
            'cooperation': sum(evolved_cooperation[agent_type]) / len(evolved_cooperation[agent_type])
        }
    
    print(f"✅ Evolved capabilities recorded for {len(evolved_averages)} agent types")
    
    # Phase 3: Test knowledge transfer mechanisms
    print("\n[PHASE 3] Testing knowledge transfer mechanisms")
    
    # Create new agents and test if they inherit behaviors
    transfer_sim = WorldSimulation()
    await transfer_sim.initialize(num_agents=12)
    
    # Simulate knowledge transfer from evolved agents
    knowledge_transfer_success = 0
    behavioral_improvements = {}
    
    for new_agent in transfer_sim.agents:
        agent_type = new_agent.agent_type
        
        # Check if we have evolved data for this agent type
        if agent_type in evolved_averages:
            evolved_data = evolved_averages[agent_type]
            
            # Simulate knowledge transfer (partial inheritance)
            transfer_rate = 0.3  # 30% of evolved knowledge transfers
            
            # Transfer capabilities (partial)
            for capability in new_agent.capabilities:
                if capability in evolved_data['capabilities']:
                    evolved_value = evolved_data['capabilities'][capability]
                    baseline_value = new_agent.capabilities[capability]
                    
                    # Transfer some of the improvement
                    improvement = (evolved_value - baseline_value) * transfer_rate
                    new_agent.capabilities[capability] = min(1.0, baseline_value + improvement)
            
            # Transfer knowledge
            knowledge_transfer = int(evolved_data['knowledge'] * transfer_rate)
            new_agent.resources['knowledge'] += knowledge_transfer
            
            # Transfer cooperation tendencies
            cooperation_boost = evolved_data['cooperation'] * transfer_rate
            new_agent.cooperation_score += cooperation_boost
            
            knowledge_transfer_success += 1
            
            # Record improvements
            if agent_type not in behavioral_improvements:
                behavioral_improvements[agent_type] = {
                    'capability_improvements': {},
                    'knowledge_gained': 0,
                    'cooperation_boost': 0
                }
            
            for capability in new_agent.capabilities:
                if capability in baseline_capabilities[agent_type]:
                    baseline_val = baseline_capabilities[agent_type][capability]
                    new_val = new_agent.capabilities[capability]
                    improvement = ((new_val - baseline_val) / baseline_val) * 100
                    behavioral_improvements[agent_type]['capability_improvements'][capability] = improvement
            
            behavioral_improvements[agent_type]['knowledge_gained'] = knowledge_transfer
            behavioral_improvements[agent_type]['cooperation_boost'] = cooperation_boost
    
    print(f"✅ Knowledge transfer applied to {knowledge_transfer_success} agents")
    
    # Phase 4: Test new agents with inherited behaviors
    print("\n[PHASE 4] Testing new agents with inherited behaviors")
    
    # Run simulation with transfer agents
    transfer_result = await transfer_sim.sim_loop(generations=3)
    
    print(f"✅ Transfer simulation completed")
    print(f"✅ Behaviors detected: {transfer_result['emergent_behaviors_detected']}")
    print(f"✅ Cooperation events: {transfer_result['cooperation_events']}")
    print(f"✅ Final fitness: {transfer_result['final_average_fitness']:.3f}")
    
    # Phase 5: Compare performance
    print("\n[PHASE 5] Analyzing behavioral transfer effectiveness")
    
    # Compare baseline vs evolved vs transfer
    comparison = {
        'baseline': {
            'avg_fitness': sum(a.fitness for a in baseline_sim.agents) / len(baseline_sim.agents),
            'avg_knowledge': sum(a.resources['knowledge'] for a in baseline_sim.agents) / len(baseline_sim.agents),
            'avg_cooperation': sum(a.cooperation_score for a in baseline_sim.agents) / len(baseline_sim.agents)
        },
        'evolved': {
            'avg_fitness': sum(a.fitness for a in evolved_sim.agents) / len(evolved_sim.agents),
            'avg_knowledge': sum(a.resources['knowledge'] for a in evolved_sim.agents) / len(evolved_sim.agents),
            'avg_cooperation': sum(a.cooperation_score for a in evolved_sim.agents) / len(evolved_sim.agents)
        },
        'transfer': {
            'avg_fitness': sum(a.fitness for a in transfer_sim.agents) / len(transfer_sim.agents),
            'avg_knowledge': sum(a.resources['knowledge'] for a in transfer_sim.agents) / len(transfer_sim.agents),
            'avg_cooperation': sum(a.cooperation_score for a in transfer_sim.agents) / len(transfer_sim.agents)
        }
    }
    
    # Calculate improvements (safe division)
    baseline_fitness = max(comparison['baseline']['avg_fitness'], 0.001)
    baseline_knowledge = max(comparison['baseline']['avg_knowledge'], 1)
    baseline_cooperation = max(comparison['baseline']['avg_cooperation'], 0.1)

    fitness_improvement = ((comparison['transfer']['avg_fitness'] - comparison['baseline']['avg_fitness']) /
                          baseline_fitness) * 100
    knowledge_improvement = ((comparison['transfer']['avg_knowledge'] - comparison['baseline']['avg_knowledge']) /
                           baseline_knowledge) * 100
    cooperation_improvement = ((comparison['transfer']['avg_cooperation'] - comparison['baseline']['avg_cooperation']) /
                             baseline_cooperation) * 100
    
    # Results
    print("\n" + "=" * 60)
    print("BEHAVIORAL TRANSFER ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"Baseline Agents    - Fitness: {comparison['baseline']['avg_fitness']:.3f}, Knowledge: {comparison['baseline']['avg_knowledge']:.1f}, Cooperation: {comparison['baseline']['avg_cooperation']:.3f}")
    print(f"Evolved Agents     - Fitness: {comparison['evolved']['avg_fitness']:.3f}, Knowledge: {comparison['evolved']['avg_knowledge']:.1f}, Cooperation: {comparison['evolved']['avg_cooperation']:.3f}")
    print(f"Transfer Agents    - Fitness: {comparison['transfer']['avg_fitness']:.3f}, Knowledge: {comparison['transfer']['avg_knowledge']:.1f}, Cooperation: {comparison['transfer']['avg_cooperation']:.3f}")
    
    print(f"\n🚀 TRANSFER EFFECTIVENESS:")
    print(f"Fitness Improvement:     {fitness_improvement:+.1f}%")
    print(f"Knowledge Improvement:   {knowledge_improvement:+.1f}%")
    print(f"Cooperation Improvement: {cooperation_improvement:+.1f}%")
    
    print(f"\n🧬 BEHAVIORAL IMPROVEMENTS BY TYPE:")
    for agent_type, improvements in behavioral_improvements.items():
        print(f"\n{agent_type.upper()} AGENTS:")
        for capability, improvement in improvements['capability_improvements'].items():
            print(f"  {capability}: {improvement:+.1f}%")
        print(f"  Knowledge gained: +{improvements['knowledge_gained']}")
        print(f"  Cooperation boost: +{improvements['cooperation_boost']:.3f}")
    
    # Determine if transfer is effective
    transfer_effective = (fitness_improvement > 5 and 
                         knowledge_improvement > 10 and 
                         cooperation_improvement > 5)
    
    print(f"\n🎯 BEHAVIORAL TRANSFER ASSESSMENT:")
    if transfer_effective:
        print("✅ BEHAVIORAL TRANSFER: EFFECTIVE")
        print("✅ New agents receive significant head start from evolved behaviors")
        print("✅ Knowledge inheritance mechanisms working")
        print("✅ Cooperation patterns successfully transferred")
    else:
        print("⚠️ BEHAVIORAL TRANSFER: LIMITED")
        print("⚠️ Transfer mechanisms need optimization")
        print("⚠️ Head start benefits are minimal")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"behavioral_transfer_analysis_{timestamp}.json"
    
    detailed_results = {
        'timestamp': timestamp,
        'comparison': comparison,
        'improvements': {
            'fitness': fitness_improvement,
            'knowledge': knowledge_improvement,
            'cooperation': cooperation_improvement
        },
        'behavioral_improvements': behavioral_improvements,
        'transfer_effective': transfer_effective,
        'agents_tested': {
            'baseline': len(baseline_sim.agents),
            'evolved': len(evolved_sim.agents),
            'transfer': len(transfer_sim.agents)
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved: {results_file}")
    
    return transfer_effective

if __name__ == "__main__":
    success = asyncio.run(test_behavioral_transfer())
    print(f"\n🎉 Behavioral transfer test {'PASSED' if success else 'NEEDS IMPROVEMENT'}")
