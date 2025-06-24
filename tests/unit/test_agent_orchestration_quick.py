#!/usr/bin/env python3
"""
Quick test for agent orchestration server fixes
"""

import time
import requests

def test_quick():
    base_url = "http://localhost:8005"
    
    print("🔧 Quick Agent Orchestration Test")
    print("=" * 40)
    
    # Test 1: Submit a task
    print("1. Submitting test task...")
    task_data = {
        "task_type": "general",
        "description": "Quick test task",
        "input_data": {"test": "data"},
        "priority": "normal",
        "required_capabilities": [],
        "timeout_seconds": 30
    }
    
    response = requests.post(f"{base_url}/v1/tasks", json=task_data, timeout=30)
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            task_id = result.get('data', {}).get('task_id')
            print(f"✅ Task submitted: {task_id}")
            
            # Test 2: Wait and check task status
            print("2. Waiting for task execution...")
            time.sleep(3)
            
            status_response = requests.get(f"{base_url}/v1/tasks/{task_id}", timeout=10)
            if status_response.status_code == 200:
                task_info = status_response.json()
                print(f"✅ Task status: {task_info.get('status')}")
                print(f"   Priority: {task_info.get('priority')}")
                print(f"   Assigned agent: {task_info.get('assigned_agent')}")
            else:
                print(f"❌ Failed to get task status: {status_response.status_code}")
            
            # Test 3: List all tasks
            print("3. Listing all tasks...")
            list_response = requests.get(f"{base_url}/v1/tasks", timeout=10)
            if list_response.status_code == 200:
                tasks = list_response.json()
                print(f"✅ Found {len(tasks)} tasks")
                if tasks:
                    print(f"   First task status: {tasks[0].get('status')}")
            else:
                print(f"❌ Failed to list tasks: {list_response.status_code}")
        else:
            print(f"❌ Task submission failed: {result.get('message')}")
    else:
        print(f"❌ HTTP error: {response.status_code}")
    
    print("\n🎯 Quick test completed!")

if __name__ == "__main__":
    test_quick()
