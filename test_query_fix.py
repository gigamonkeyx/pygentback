#!/usr/bin/env python3
import asyncio
from src.mcp.query_fixed import ObserverQuerySystem

async def test_failure_details():
    print("🔧 TESTING QUERY FAILURE_DETAILS FIX")
    
    query_system = ObserverQuerySystem()
    
    class FailingServer:
        def __init__(self):
            self.call_count = 0
        
        async def query(self):
            self.call_count += 1
            return None  # Always fail
    
    failing_server = FailingServer()
    result = await query_system.query_env(failing_server, max_attempts=3)
    
    print(f"Status: {result.get('status')}")
    print(f"Has failure_details: {'failure_details' in result}")
    
    if 'failure_details' in result:
        details = result['failure_details']
        print(f"Attempts made: {details.get('attempts_made')}")
        print(f"Max attempts: {details.get('max_attempts')}")
        print(f"Failure reason: {details.get('failure_reason')}")
        print("✅ FAILURE_DETAILS FIX WORKING")
        return True
    else:
        print("❌ FAILURE_DETAILS STILL MISSING")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_failure_details())
    print(f"Fix status: {'SUCCESS' if result else 'FAILED'}")
