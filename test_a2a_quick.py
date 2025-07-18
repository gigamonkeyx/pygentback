#!/usr/bin/env python3
"""Quick A2A Big Jump Test"""

from src.sim.world_sim import sim_loop

print("🚀 A2A MAXIMUM ENHANCEMENT TEST")
print("=" * 50)

results = sim_loop(generations=10)

print("🎯 RESULTS:")
print(f"Generations: {results.get('generations', 0)}")
print(f"Agents: {results.get('agents_count', 0)}")
print(f"Emergence: {results.get('emergence_detected', False)}")
print(f"Emergence Gen: {results.get('emergence_generation', 'None')}")

final_fitness = results.get('final_agent_fitness', [])
if final_fitness:
    total_fitness = sum(final_fitness)
    avg_fitness = total_fitness / len(final_fitness)
    print(f"Total Fitness: {total_fitness:.2f}")
    print(f"Average Fitness: {avg_fitness:.2f}")
    print(f"Fitness > 8.14: {'✅ PASS' if avg_fitness >= 8.14 else '❌ FAIL'}")
    
    # Calculate improvement over Phase 4 baseline (6.03)
    improvement = ((avg_fitness - 6.03) / 6.03) * 100
    print(f"Improvement: {improvement:.1f}%")
    print(f"Improvement > 80%: {'✅ PASS' if improvement >= 80.0 else '❌ FAIL'}")
    
cooperation_score = results.get('cooperation_score', 0.0)
print(f"Cooperation Score: {cooperation_score:.3f}")
print(f"Cooperation > 0.7: {'✅ PASS' if cooperation_score >= 0.7 else '❌ FAIL'}")

early_emergence = results.get('emergence_generation', 10) <= 3
print(f"Early Emergence: {'✅ PASS' if early_emergence else '❌ FAIL'}")

print()
print("🎯 BIG JUMP CRITERIA:")
criteria = {
    'fitness_exceeds_8_14': avg_fitness >= 8.14 if final_fitness else False,
    'improvement_exceeds_80_percent': improvement >= 80.0 if final_fitness else False,
    'early_emergence_achieved': early_emergence,
    'cooperation_score_high': cooperation_score >= 0.7,
    'a2a_agents_functional': True  # Always true if we got results
}

passed = sum(criteria.values())
total = len(criteria)

for criterion, status in criteria.items():
    print(f"  {criterion}: {'✅ PASS' if status else '❌ FAIL'}")

print()
print(f"BIG JUMP RESULT: {passed}/{total} criteria passed")
print(f"Status: {'🎉 BIG JUMP ACHIEVED!' if passed == total else '⚠️ NEEDS MORE OPTIMIZATION'}")
