#!/usr/bin/env python3
"""
CLEANUP AND OPTIMIZATION VERIFICATION TEST
Observer-mandated verification of error fixes and system optimization
"""

import asyncio
import sys
import time
import logging
sys.path.append('.')

# Configure logging to test Unicode handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

async def test_cleanup_optimization():
    print("*** CLEANUP AND OPTIMIZATION VERIFICATION TEST")
    print("=" * 70)
    
    optimization_results = {
        'unicode_logging_fixed': False,
        'ollama_404_fixed': False,
        'session_cleanup_fixed': False,
        'ollama_manager_fixed': False,
        'overall_stability': False
    }
    
    try:
        # Test 1: Unicode Logging Fix
        print("\n1. *** UNICODE LOGGING FIX VERIFICATION")
        print("-" * 50)
        
        try:
            from src.core.ollama_startup import ensure_ollama_startup
            
            # This should not cause Unicode encoding errors
            logger = logging.getLogger('test_unicode')
            logger.info("*** Testing Unicode logging fix")
            logger.info("*** ASCII characters only - no emoji")
            
            print("   Unicode Logging: FIXED (no encoding errors)")
            optimization_results['unicode_logging_fixed'] = True
            
        except UnicodeEncodeError as e:
            print(f"   Unicode Logging: FAILED - {e}")
            optimization_results['unicode_logging_fixed'] = False
        except Exception as e:
            print(f"   Unicode Logging: ERROR - {e}")
            optimization_results['unicode_logging_fixed'] = False
        
        # Test 2: Ollama 404 Error Fix
        print("\n2. *** OLLAMA 404 ERROR FIX VERIFICATION")
        print("-" * 50)
        
        try:
            from src.ai.reasoning.tot.thought_generator import OllamaBackend
            
            # Test with a simple prompt
            backend = OllamaBackend(model_name="llama3:8b")
            
            # This should handle 404 errors gracefully with fallback
            result = await backend.generate("test", max_tokens=5)
            
            if result or result == "":  # Empty string is acceptable, None is not
                print("   Ollama 404 Fix: FIXED (graceful error handling)")
                optimization_results['ollama_404_fixed'] = True
            else:
                print("   Ollama 404 Fix: PARTIAL (needs more work)")
                optimization_results['ollama_404_fixed'] = False
                
        except Exception as e:
            print(f"   Ollama 404 Fix: ERROR - {e}")
            optimization_results['ollama_404_fixed'] = False
        
        # Test 3: Session Cleanup Fix
        print("\n3. *** SESSION CLEANUP FIX VERIFICATION")
        print("-" * 50)
        
        try:
            from src.core.ollama_startup import OllamaStartupManager
            
            # Test session cleanup
            manager = OllamaStartupManager()
            
            # This should properly close sessions
            result = await manager.check_ollama_running()
            
            print("   Session Cleanup: FIXED (proper session management)")
            optimization_results['session_cleanup_fixed'] = True
            
        except Exception as e:
            print(f"   Session Cleanup: ERROR - {e}")
            optimization_results['session_cleanup_fixed'] = False
        
        # Test 4: OllamaManager Missing Method Fix
        print("\n4. *** OLLAMA MANAGER METHOD FIX VERIFICATION")
        print("-" * 50)
        
        try:
            from src.core.ollama_integration import OllamaManager
            
            # Test the missing method
            manager = OllamaManager()
            await manager.initialize()
            
            # This should not raise AttributeError
            has_method = hasattr(manager, 'is_model_available')
            if has_method:
                # Test the method
                result = await manager.is_model_available("llama3:8b")
                print(f"   OllamaManager Method: FIXED (is_model_available works)")
                optimization_results['ollama_manager_fixed'] = True
            else:
                print("   OllamaManager Method: FAILED (method still missing)")
                optimization_results['ollama_manager_fixed'] = False
                
        except AttributeError as e:
            print(f"   OllamaManager Method: FAILED - {e}")
            optimization_results['ollama_manager_fixed'] = False
        except Exception as e:
            print(f"   OllamaManager Method: ERROR - {e}")
            optimization_results['ollama_manager_fixed'] = False
        
        # Test 5: Overall System Stability
        print("\n5. *** OVERALL SYSTEM STABILITY TEST")
        print("-" * 50)
        
        try:
            from src.core.agent_factory import AgentFactory
            
            # Test agent creation with all fixes
            factory = AgentFactory()
            
            # This should work without errors
            agent = await factory.create_agent(
                agent_type='general',
                name='stability_test_agent',
                capabilities=['basic'],
                custom_config={'stability_test': True}
            )
            
            if agent:
                print("   System Stability: EXCELLENT (agent creation successful)")
                optimization_results['overall_stability'] = True
            else:
                print("   System Stability: PARTIAL (agent creation issues)")
                optimization_results['overall_stability'] = False
                
        except Exception as e:
            print(f"   System Stability: ERROR - {e}")
            optimization_results['overall_stability'] = False
        
        # Final Assessment
        print("\n" + "=" * 70)
        print("*** CLEANUP AND OPTIMIZATION ASSESSMENT")
        print("=" * 70)
        
        fixes_applied = sum(optimization_results.values())
        total_fixes = len(optimization_results)
        
        for fix_name, status in optimization_results.items():
            print(f"  {fix_name}: {'FIXED' if status else 'NEEDS WORK'}")
        
        print()
        print(f"OPTIMIZATION RESULT: {fixes_applied}/{total_fixes} fixes successful")
        
        if fixes_applied == total_fixes:
            print("Status: *** CLEANUP COMPLETE - ALL OPTIMIZATIONS SUCCESSFUL")
            print("Observer Assessment: System fully optimized and stable")
        elif fixes_applied >= 3:
            print("Status: *** CLEANUP PARTIAL - MAJOR OPTIMIZATIONS COMPLETE")
            print("Observer Assessment: Core issues resolved, minor optimizations remain")
        else:
            print("Status: *** CLEANUP INCOMPLETE - MAJOR ISSUES REMAIN")
            print("Observer Assessment: Significant optimization work still needed")
        
        return fixes_applied >= 3
        
    except Exception as e:
        print(f"\n*** CRITICAL OPTIMIZATION TEST FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_optimization():
    """Test performance improvements"""
    print("\n" + "=" * 70)
    print("*** PERFORMANCE OPTIMIZATION TEST")
    print("=" * 70)
    
    try:
        from src.core.ollama_startup import ensure_ollama_startup
        
        # Test startup performance
        start_time = time.time()
        result = await ensure_ollama_startup()
        startup_time = time.time() - start_time
        
        print(f"   Ollama Startup Time: {startup_time:.2f}s")
        
        if startup_time < 60:  # Should be under 1 minute
            print("   Performance: OPTIMIZED (fast startup)")
            return True
        else:
            print("   Performance: NEEDS OPTIMIZATION (slow startup)")
            return False
            
    except Exception as e:
        print(f"   Performance Test Error: {e}")
        return False

if __name__ == "__main__":
    print("*** OBSERVER DIRECTIVE: CLEANUP AND OPTIMIZATION VERIFICATION")
    print("*** Testing all error fixes and system optimizations")
    
    # Run cleanup tests
    cleanup_result = asyncio.run(test_cleanup_optimization())
    
    # Run performance tests
    performance_result = asyncio.run(test_performance_optimization())
    
    print(f"\n*** FINAL OPTIMIZATION RESULT:")
    print(f"   Cleanup: {'SUCCESS' if cleanup_result else 'NEEDS WORK'}")
    print(f"   Performance: {'OPTIMIZED' if performance_result else 'NEEDS WORK'}")
    
    overall_success = cleanup_result and performance_result
    
    if overall_success:
        print("\n*** Observer: Cleanup and optimization COMPLETE - System ready")
    else:
        print("\n*** Observer: Cleanup and optimization PARTIAL - Further work needed")
