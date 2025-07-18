#!/usr/bin/env python3
import asyncio
from src.mcp.query_fixed import ObserverQuerySystem

async def test_query_system():
    print("🔍 DEBUGGING QUERY SYSTEM ISSUES")
    
    try:
        # Create query system
        query_system = ObserverQuerySystem()
        print("✅ Query system created successfully")
        
        # Test with mock server that works
        class WorkingMockServer:
            async def query(self):
                await asyncio.sleep(0.01)
                return {'status': 'ok', 'timestamp': 'test'}
        
        # Test with mock server that fails
        class FailingMockServer:
            def __init__(self, fail_count=15):
                self.fail_count = fail_count
                self.call_count = 0
            
            async def query(self):
                self.call_count += 1
                if self.call_count <= self.fail_count:
                    return None  # Simulate failure
                return {'status': 'success', 'call_count': self.call_count}
        
        # Test 1: Working server
        print("\n1. Testing with working server...")
        working_server = WorkingMockServer()
        result1 = await query_system.query_env(working_server, max_attempts=3)
        print(f"   Result: {result1.get('status', 'unknown')}")
        
        # Test 2: Failing server
        print("\n2. Testing with failing server...")
        failing_server = FailingMockServer(fail_count=10)
        result2 = await query_system.query_env(failing_server, max_attempts=5)
        print(f"   Result: {result2.get('status', 'unknown')}")
        print(f"   Calls made: {failing_server.call_count}")
        print(f"   Has failure_details: {'failure_details' in result2}")
        
        # Test 3: Load test simulation
        print("\n3. Testing load simulation (10 queries)...")
        fast_server = WorkingMockServer()
        results = []
        for i in range(10):
            result = await query_system.query_env(fast_server, max_attempts=3)
            success = result.get('status') == 'ok'
            results.append(success)
            if i < 3:  # Show first few results
                print(f"   Query {i+1}: {'✅' if success else '❌'}")
        
        success_rate = sum(results) / len(results)
        print(f"   Overall success rate: {success_rate:.1%}")
        print(f"   Passes 80% threshold: {success_rate >= 0.8}")
        
        return success_rate >= 0.8
        
    except Exception as e:
        print(f"❌ Query system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_query_system())
    print(f"\n🎯 QUERY SYSTEM TEST: {'PASS' if result else 'FAIL'}")
