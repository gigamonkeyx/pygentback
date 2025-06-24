#!/usr/bin/env python3
"""
Test the Research-to-Analysis workflow API endpoints
"""

import requests
import json
import time

def test_workflow_api():
    """Test the complete workflow API"""
    
    print("🧪 Testing Research-to-Analysis Workflow API")
    print("=" * 50)
    
    base_url = "http://localhost:8080"
    
    # Test 1: Check active workflows (should be empty)
    print("\n1️⃣ Testing active workflows endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/workflows/research-analysis/active")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            active_workflows = response.json()
            print(f"   ✅ Active workflows: {len(active_workflows)}")
        else:
            print(f"   ❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Start a workflow
    print("\n2️⃣ Starting a research workflow...")
    workflow_request = {
        "query": "quantum computing feasibility using larger qubits on silicon",
        "analysis_model": "deepseek-r1:8b",
        "max_papers": 5,
        "analysis_depth": 2,
        "export_format": "markdown"
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/workflows/research-analysis",
            json=workflow_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            workflow_data = response.json()
            workflow_id = workflow_data.get("workflow_id")
            print(f"   ✅ Workflow started: {workflow_id}")
            print(f"   Message: {workflow_data.get('message')}")
            
            # Test 3: Monitor workflow progress
            print("\n3️⃣ Monitoring workflow progress...")
            max_attempts = 30  # 30 seconds max
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    status_response = requests.get(
                        f"{base_url}/api/v1/workflows/research-analysis/{workflow_id}/status"
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get("status")
                        progress = status_data.get("progress_percentage", 0)
                        step = status_data.get("current_step", "Unknown")
                        
                        print(f"   Progress: {progress:.1f}% - {status} - {step}")
                        
                        if status == "completed":
                            print("   ✅ Workflow completed successfully!")
                            
                            # Test 4: Get results
                            print("\n4️⃣ Getting workflow results...")
                            result_response = requests.get(
                                f"{base_url}/api/v1/workflows/research-analysis/{workflow_id}/result"
                            )
                            
                            if result_response.status_code == 200:
                                result_data = result_response.json()
                                print(f"   ✅ Results retrieved successfully")
                                print(f"   Query: {result_data.get('query')}")
                                print(f"   Success: {result_data.get('success')}")
                                print(f"   Execution time: {result_data.get('execution_time', 0):.2f}s")
                                print(f"   Citations: {len(result_data.get('citations', []))}")
                                
                                # Show preview of results
                                research_summary = result_data.get('research_summary', '')
                                analysis_summary = result_data.get('analysis_summary', '')
                                
                                if research_summary:
                                    print(f"\n   📚 Research Summary (preview):")
                                    print(f"   {research_summary[:150]}...")
                                
                                if analysis_summary:
                                    print(f"\n   🧠 Analysis Summary (preview):")
                                    print(f"   {analysis_summary[:150]}...")
                                
                                # Test 5: Export results
                                print("\n5️⃣ Testing export functionality...")
                                export_response = requests.get(
                                    f"{base_url}/api/v1/workflows/research-analysis/{workflow_id}/export/markdown"
                                )
                                
                                if export_response.status_code == 200:
                                    print("   ✅ Export successful")
                                    print(f"   Content length: {len(export_response.content)} bytes")
                                else:
                                    print(f"   ⚠️ Export failed: {export_response.status_code}")
                                
                                return True
                            else:
                                print(f"   ❌ Failed to get results: {result_response.status_code}")
                                return False
                        
                        elif status == "failed":
                            error_msg = status_data.get("error_message", "Unknown error")
                            print(f"   ❌ Workflow failed: {error_msg}")
                            return False
                        
                        # Wait before next check
                        time.sleep(2)
                        attempt += 1
                    
                    else:
                        print(f"   ❌ Status check failed: {status_response.status_code}")
                        return False
                        
                except Exception as e:
                    print(f"   ❌ Error checking status: {e}")
                    return False
            
            print(f"   ⏰ Workflow timeout after {max_attempts * 2} seconds")
            return False
            
        else:
            print(f"   ❌ Failed to start workflow: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error starting workflow: {e}")
        return False


def test_api_health():
    """Test basic API health"""
    print("\n🏥 Testing API Health...")
    
    try:
        response = requests.get("http://localhost:8080/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ API is healthy")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            return True
        else:
            print(f"   ❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Research-to-Analysis Workflow API Test")
    print("=" * 60)
    print("Testing the complete automated workflow through the API")
    print("=" * 60)
    
    # Test API health first (optional)
    print("\n🏥 Testing API Health...")
    try:
        response = requests.get("http://localhost:8080/api/v1/health")
        if response.status_code == 200:
            print("   ✅ API is healthy")
        else:
            print(f"   ⚠️ Health endpoint returned {response.status_code} (continuing anyway)")
    except Exception as e:
        print(f"   ⚠️ Health check failed: {e} (continuing anyway)")
    
    # Test the workflow
    success = test_workflow_api()
    
    if success:
        print("\n🎉 All tests passed! The Research-to-Analysis workflow is working!")
        print("\nNext steps:")
        print("1. Open the UI at http://localhost:8080")
        print("2. Navigate to 'Research & Analysis' page")
        print("3. Test the workflow through the web interface")
        print("4. Try different research queries and models")
    else:
        print("\n❌ Tests failed - check the server logs for details")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
