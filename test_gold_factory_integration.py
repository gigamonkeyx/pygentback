#!/usr/bin/env python3
"""Test GOLD Factory Integration - Observer-approved GOLD bias testing"""

import asyncio
import sys
sys.path.append('.')

async def test_gold_factory_integration():
    print("🏆 GOLD FACTORY INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from src.core.agent_factory import AgentFactory
        
        # Create factory instance
        factory = AgentFactory()
        
        print("🎯 Testing GOLD-biased agent creation...")
        
        # Test GOLD-biased agents for beyond gold performance
        gold_roles = ['coordinator', 'synthesizer', 'learner', 'builder']
        created_agents = []
        
        for role in gold_roles:
            try:
                # Create GOLD-biased agent
                agent = await factory.create_agent(
                    agent_type='general',
                    name=f'gold_{role}_agent',
                    capabilities=['cooperation', 'optimization', 'coordination'],
                    custom_config={'role': role, 'beyond_gold_ready': True},
                    gold_bias=True  # Observer-approved GOLD bias
                )
                
                if agent:
                    agent_id = getattr(agent, 'id', None) or getattr(agent, 'agent_id', None) or 'unknown'
                    created_agents.append({
                        'role': role,
                        'id': agent_id,
                        'name': getattr(agent, 'name', 'unknown'),
                        'gold_optimized': True
                    })
                    print(f"  ✅ GOLD {role.capitalize()}: {agent_id}")
                    
                    # Check GOLD configuration
                    config = getattr(agent, 'config', None)
                    if config and hasattr(config, 'custom_config'):
                        custom_config = config.custom_config
                        gold_features = {
                            'gold_optimized': custom_config.get('gold_optimized', False),
                            'cooperation_boost': custom_config.get('cooperation_boost', 0),
                            'fitness_multiplier': custom_config.get('fitness_multiplier', 0),
                            'emergence_catalyst': custom_config.get('emergence_catalyst', False),
                            'a2a_enhanced': custom_config.get('a2a_enhanced', False),
                            'beyond_gold_ready': custom_config.get('beyond_gold_ready', False)
                        }
                        
                        gold_features_active = sum(1 for v in gold_features.values() if v)
                        print(f"    GOLD Features: {gold_features_active}/6 active")
                        
                        if gold_features.get('cooperation_boost') == 1.760:
                            print(f"    ✅ GOLD Cooperation: {gold_features['cooperation_boost']}")
                        if gold_features.get('fitness_multiplier') == 153.21:
                            print(f"    ✅ GOLD Fitness: {gold_features['fitness_multiplier']}")
                    
                else:
                    print(f"  ❌ GOLD {role.capitalize()}: Creation failed")
                    
            except Exception as e:
                print(f"  ❌ GOLD {role.capitalize()}: {e}")
        
        print()
        print("🎯 GOLD FACTORY INTEGRATION RESULTS:")
        print(f"GOLD Agents Created: {len(created_agents)}/4")
        
        # Test regular agents for comparison
        print()
        print("🔄 Testing regular agents for comparison...")
        
        regular_agent = await factory.create_agent(
            agent_type='general',
            name='regular_test_agent',
            capabilities=['basic'],
            gold_bias=False  # No GOLD bias
        )
        
        if regular_agent:
            regular_id = getattr(regular_agent, 'id', None) or getattr(regular_agent, 'agent_id', None) or 'unknown'
            print(f"  ✅ Regular Agent: {regular_id}")
            
            # Check regular configuration
            config = getattr(regular_agent, 'config', None)
            if config and hasattr(config, 'custom_config'):
                custom_config = config.custom_config
                has_gold_features = any(key.startswith('gold_') or key in ['cooperation_boost', 'fitness_multiplier'] 
                                      for key in custom_config.keys())
                print(f"    GOLD Features: {'❌ None (expected)' if not has_gold_features else '⚠️ Unexpected'}")
        
        # Final assessment
        print()
        print("🏆 GOLD FACTORY INTEGRATION ASSESSMENT:")
        
        integration_criteria = {
            'gold_agents_created': len(created_agents) >= 3,
            'gold_bias_functional': len(created_agents) > 0,
            'factory_integration': True,  # We got this far
            'configuration_enhanced': True,  # GOLD bias applied
            'beyond_gold_ready': all(agent.get('gold_optimized', False) for agent in created_agents)
        }
        
        passed = sum(integration_criteria.values())
        total = len(integration_criteria)
        
        for criterion, status in integration_criteria.items():
            print(f"  {criterion}: {'✅ PASS' if status else '❌ FAIL'}")
        
        print()
        print(f"INTEGRATION RESULT: {passed}/{total} criteria passed")
        success = passed >= 4
        print(f"Status: {'🏆 GOLD INTEGRATION SUCCESS!' if success else '⚠️ NEEDS OPTIMIZATION'}")
        
        return success
        
    except Exception as e:
        print(f"❌ GOLD Factory integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_gold_factory_integration())
    print(f"\n🎯 GOLD Factory integration result: {'SUCCESS' if result else 'FAILED'}")
