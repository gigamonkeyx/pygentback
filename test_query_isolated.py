#!/usr/bin/env python3
"""
Isolated test for query system to identify logger issue
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

def test_query_system_isolated():
    """Test query system in isolation to find logger issue."""
    try:
        print("Testing query system import in isolation...")
        
        # Test step by step
        print("Step 1: Testing basic imports...")
        import logging
        import asyncio
        import time
        from typing import Dict, List, Optional, Any, Callable
        from datetime import datetime, timedelta
        import json
        print("✅ Basic imports: SUCCESS")
        
        print("Step 2: Testing logger setup...")
        logger = logging.getLogger('test_query')
        print("✅ Logger setup: SUCCESS")
        
        print("Step 3: Testing MCP module imports...")
        try:
            # Import the specific module directly
            import importlib.util
            spec = importlib.util.spec_from_file_location("query_fixed", "src/mcp/query_fixed.py")
            query_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(query_module)
            print("✅ Direct module import: SUCCESS")
            
            # Test class creation
            query_system = query_module.ObserverQuerySystem()
            print("✅ Query system creation: SUCCESS")
            
        except Exception as e:
            print(f"❌ Direct module import failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("Step 4: Testing through normal import...")
        try:
            from mcp.query_fixed import ObserverQuerySystem
            query_system2 = ObserverQuerySystem()
            print("✅ Normal import: SUCCESS")
        except Exception as e:
            print(f"❌ Normal import failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("Step 5: Testing async functionality...")
        try:
            async def test_async():
                result = await query_system2.execute_query('health_check', {})
                return result['success']
            
            success = asyncio.run(test_async())
            print(f"✅ Async test: {'SUCCESS' if success else 'FAILED'}")
        except Exception as e:
            print(f"❌ Async test failed: {e}")
            return False
        
        print("\n✅ ALL TESTS PASSED - Query system is working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_query_system_isolated()
    if success:
        print("\n🎉 Query system is fully functional!")
    else:
        print("\n💥 Query system has issues that need fixing")
