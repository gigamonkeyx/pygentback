#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Unicode-safe test for Observer systems
"""

import sys
import os

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all Observer systems can be imported."""
    print("Testing Observer system imports...")
    
    try:
        from sim.world_sim import WorldSimulation
        print("✅ World Simulation: IMPORTED")
    except Exception as e:
        print(f"❌ World Simulation: {e}")
        return False
    
    try:
        from dgm.autonomy_fixed import FormalProofSystem
        print("✅ Formal Proof System: IMPORTED")
    except Exception as e:
        print(f"❌ Formal Proof System: {e}")
        return False
    
    try:
        from ai.evolution.evo_loop_fixed import ObserverEvolutionLoop
        print("✅ Evolution Loop: IMPORTED")
    except Exception as e:
        print(f"❌ Evolution Loop: {e}")
        return False
    
    try:
        from agents.communication_system_fixed import ObserverCommunicationSystem
        print("✅ Communication System: IMPORTED")
    except Exception as e:
        print(f"❌ Communication System: {e}")
        return False
    
    try:
        from mcp.query_fixed import ObserverQuerySystem
        print("✅ Query System: IMPORTED")
    except Exception as e:
        print(f"❌ Query System: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without async complexity."""
    print("\nTesting basic functionality...")
    
    try:
        from sim.world_sim import WorldSimulation, Agent
        
        # Test agent creation
        agent = Agent("test_agent", "explorer", {})
        print(f"✅ Agent created: {agent.agent_id} ({agent.agent_type})")
        
        # Test simulation creation
        sim = WorldSimulation()
        print("✅ World simulation created")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run simple Unicode-safe tests."""
    print("=" * 50)
    print("UNICODE-SAFE OBSERVER SYSTEM TEST")
    print("=" * 50)
    
    # Test imports
    import_success = test_imports()
    
    # Test basic functionality
    basic_success = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print(f"Imports: {'SUCCESS' if import_success else 'FAILED'}")
    print(f"Basic functionality: {'SUCCESS' if basic_success else 'FAILED'}")
    
    overall_success = import_success and basic_success
    print(f"Overall: {'SUCCESS' if overall_success else 'FAILED'}")
    
    if overall_success:
        print("\n✅ Unicode encoding issue FIXED")
        print("✅ Observer systems ready for testing")
    else:
        print("\n❌ Issues remain - check error messages above")
    
    return overall_success

if __name__ == "__main__":
    main()
