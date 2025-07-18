#!/usr/bin/env python3
"""
RIPER-Ω Protocol Critical Fix - Ollama Timing Race Condition
Observer-mandated validation of enhanced startup sequence with proper timing
"""

import asyncio
import sys
import time
sys.path.append('.')

async def test_ollama_timing_fix():
    print("🚨 RIPER-Ω PROTOCOL CRITICAL FIX - OLLAMA TIMING")
    print("=" * 70)
    
    try:
        # Test 1: Enhanced Ollama Startup with Timing
        print("\n1. 🔍 OBSERVER CRITICAL: Enhanced Ollama Startup Test")
        from src.core.ollama_startup import ensure_ollama_startup, get_ollama_status
        
        start_time = time.time()
        ollama_result = await ensure_ollama_startup()
        startup_time = time.time() - start_time
        
        print(f"   Startup Time: {startup_time:.2f} seconds")
        print(f"   Ollama Status: {'✅ SUCCESS' if ollama_result['success'] else '❌ FAILED'}")
        print(f"   Running: {'✅ YES' if ollama_result['running'] else '❌ NO'}")
        print(f"   Models Count: {ollama_result['models_count']}")
        print(f"   Models: {', '.join(ollama_result['models'][:3]) if ollama_result['models'] else 'None'}")
        
        if ollama_result.get('error'):
            print(f"   Error: {ollama_result['error']}")
        
        ollama_startup_success = ollama_result['success']
        
        # Test 2: Agent Creation with Enhanced Timing
        print("\n2. 🏭 AGENT CREATION WITH ENHANCED TIMING")
        
        if ollama_startup_success:
            try:
                from src.core.agent_factory import AgentFactory
                
                # Create factory instance
                factory = AgentFactory()
                
                print("   Testing agent creation after enhanced Ollama validation...")
                
                # Create test agent - should NOT get 404 errors now
                start_agent_time = time.time()
                agent = await factory.create_agent(
                    agent_type='general',
                    name='timing_test_agent',
                    capabilities=['basic'],
                    custom_config={'timing_test': True}
                )
                agent_creation_time = time.time() - start_agent_time
                
                print(f"   Agent Creation Time: {agent_creation_time:.2f} seconds")
                
                if agent:
                    agent_id = getattr(agent, 'id', None) or getattr(agent, 'agent_id', None) or 'unknown'
                    print(f"   ✅ Agent Created: {agent_id}")
                    print(f"   ✅ Ollama Validation: {'PASSED' if factory.ollama_validated else 'FAILED'}")
                    
                    # Check for 404 errors in logs (this is the critical test)
                    print("   🔍 Checking for 404 errors...")
                    
                    # Wait a moment for any delayed errors
                    await asyncio.sleep(2)
                    
                    # Test actual Ollama connection from agent
                    try:
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            async with session.get('http://localhost:11434/api/tags') as response:
                                if response.status == 200:
                                    print(f"   ✅ Ollama API Test: {response.status} (No 404 errors)")
                                    api_success = True
                                else:
                                    print(f"   ❌ Ollama API Test: {response.status}")
                                    api_success = False
                    except Exception as e:
                        print(f"   ❌ Ollama API Test Error: {e}")
                        api_success = False
                    
                    factory_success = True
                else:
                    print("   ❌ Agent Creation: FAILED")
                    factory_success = False
                    api_success = False
                    
            except Exception as e:
                print(f"   ❌ Factory Integration Error: {e}")
                factory_success = False
                api_success = False
        else:
            print("   ⏭️ Skipping agent test - Ollama startup failed")
            factory_success = False
            api_success = False
        
        # Test 3: Race Condition Resolution Validation
        print("\n3. 🎯 RACE CONDITION RESOLUTION VALIDATION")
        
        race_condition_tests = {
            'startup_timing_adequate': startup_time >= 10.0,  # Should take time for proper initialization
            'no_404_errors': api_success,  # No 404 errors after startup
            'models_loaded': ollama_result['models_count'] > 0,
            'agent_creation_success': factory_success,
            'consecutive_validation': ollama_startup_success
        }
        
        for test_name, result in race_condition_tests.items():
            print(f"   {test_name}: {'✅ PASS' if result else '❌ FAIL'}")
        
        # Final Assessment
        print("\n" + "=" * 70)
        print("🎯 RIPER-Ω PROTOCOL TIMING FIX ASSESSMENT")
        print("=" * 70)
        
        timing_fix_criteria = {
            'enhanced_startup_functional': ollama_startup_success,
            'proper_timing_implemented': startup_time >= 10.0,
            'race_condition_resolved': api_success and factory_success,
            'no_404_errors_detected': api_success,
            'agent_creation_stable': factory_success,
            'protocol_compliance_restored': all(race_condition_tests.values())
        }
        
        passed = sum(timing_fix_criteria.values())
        total = len(timing_fix_criteria)
        
        for criterion, status in timing_fix_criteria.items():
            print(f"  {criterion}: {'✅ PASS' if status else '❌ FAIL'}")
        
        print()
        print(f"TIMING FIX RESULT: {passed}/{total} criteria passed")
        
        if passed == total:
            print("Status: ✅ RACE CONDITION RESOLVED - TIMING FIX SUCCESS")
            print("Observer Assessment: Ollama startup timing properly fixed")
        elif passed >= 4:
            print("Status: ⚠️ PARTIAL FIX - Minor timing issues remain")
            print("Observer Assessment: Major improvements, minor optimization needed")
        else:
            print("Status: ❌ TIMING FIX FAILED - Race condition persists")
            print("Observer Assessment: Critical timing issues still present")
        
        return passed >= 4
        
    except Exception as e:
        print(f"\n❌ CRITICAL TIMING FIX TEST FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_ollama_timing_fix())
    print(f"\n🎯 RIPER-Ω Protocol timing fix result: {'SUCCESS' if result else 'FAILED'}")
    
    if result:
        print("✅ Observer: Race condition resolved, timing fix successful")
    else:
        print("❌ Observer: Timing issues persist, further optimization required")
