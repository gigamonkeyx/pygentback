#!/usr/bin/env python3
"""
Test 127.0.0.1 Fix

Test performance using 127.0.0.1 instead of localhost.
"""

import time
import requests

def test_performance_with_127():
    """Test performance using 127.0.0.1"""
    print("🚀 Testing Performance with 127.0.0.1...")
    
    servers = {
        'document-processing': 8003,
        'vector-search': 8004,
        'agent-orchestration': 8005
    }
    
    for name, port in servers.items():
        print(f"\n   Testing {name} (port {port}):")
        
        response_times = []
        for i in range(5):
            try:
                start_time = time.time()
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
                req_time = time.time() - start_time
                
                if response.status_code == 200:
                    response_times.append(req_time)
                    print(f"     Request {i+1}: {req_time*1000:.1f}ms")
                else:
                    print(f"     Request {i+1}: HTTP {response.status_code}")
            except Exception as e:
                print(f"     Request {i+1}: ERROR ({str(e)})")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"     Average: {avg_time*1000:.1f}ms")
            print(f"     Range: {min_time*1000:.1f}ms - {max_time*1000:.1f}ms")
            
            if avg_time < 0.1:  # Under 100ms
                print(f"     ✅ EXCELLENT: Under 100ms")
            elif avg_time < 0.5:  # Under 500ms
                print(f"     ✅ GOOD: Under 500ms")
            else:
                print(f"     ⚠️ SLOW: Over 500ms")

if __name__ == "__main__":
    test_performance_with_127()
