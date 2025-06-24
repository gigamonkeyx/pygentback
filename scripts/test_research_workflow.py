#!/usr/bin/env python3
"""
Test Research and Analytics Workflow

This script tests the fixed research and analytics workflow to ensure
it works properly with the corrected agent communication patterns.
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import os
os.chdir(project_root)


async def test_workflow_api_endpoint():
    """Test the workflow API endpoint directly"""
    print("🔍 Testing Research Workflow API Endpoint...")
    
    try:
        # Test data
        test_request = {
            "query": "quantum computing applications in machine learning",
            "max_papers": 5,
            "analysis_model": "deepseek2:latest",
            "analysis_depth": 3
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8000/api/v1/workflows/research-analysis',
                json=test_request,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    workflow_id = data.get('workflow_id')
                    
                    print(f"✅ Workflow started successfully!")
                    print(f"   Workflow ID: {workflow_id}")
                    print(f"   Status: {data.get('status', 'unknown')}")
                    
                    return workflow_id
                else:
                    error_text = await response.text()
                    print(f"❌ Workflow start failed: HTTP {response.status}")
                    print(f"   Error: {error_text}")
                    return None
                    
    except Exception as e:
        print(f"❌ Workflow API test failed: {e}")
        return None


async def test_workflow_status_tracking(workflow_id: str):
    """Test workflow status tracking"""
    print(f"\n🔍 Testing Workflow Status Tracking for {workflow_id}...")
    
    try:
        max_attempts = 20
        attempt = 0
        
        async with aiohttp.ClientSession() as session:
            while attempt < max_attempts:
                attempt += 1
                
                async with session.get(
                    f'http://localhost:8000/api/v1/workflows/research-analysis/{workflow_id}/status',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get('status', 'unknown')
                        progress = data.get('progress', {})
                        
                        print(f"   Attempt {attempt}: Status = {status}")
                        print(f"   Progress: {progress.get('progress_percentage', 0):.1f}%")
                        print(f"   Step: {progress.get('current_step', 'unknown')}")
                        
                        if progress.get('research_papers_found', 0) > 0:
                            print(f"   Papers found: {progress['research_papers_found']}")
                        
                        if progress.get('analysis_confidence', 0) > 0:
                            print(f"   Analysis confidence: {progress['analysis_confidence']:.2f}")
                        
                        # Check if completed
                        if status in ['completed', 'failed']:
                            print(f"✅ Workflow {status}!")
                            return status == 'completed'
                        
                        # Wait before next check
                        await asyncio.sleep(3)
                    else:
                        print(f"❌ Status check failed: HTTP {response.status}")
                        return False
        
        print(f"⚠️ Workflow did not complete within {max_attempts * 3} seconds")
        return False
        
    except Exception as e:
        print(f"❌ Status tracking failed: {e}")
        return False


async def test_workflow_results(workflow_id: str):
    """Test workflow results retrieval"""
    print(f"\n🔍 Testing Workflow Results for {workflow_id}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'http://localhost:8000/api/v1/workflows/research-analysis/{workflow_id}/results',
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    print("✅ Results retrieved successfully!")
                    
                    # Check result structure
                    if 'formatted_output' in data:
                        output_length = len(data['formatted_output'])
                        print(f"   Formatted output: {output_length} characters")
                    
                    if 'metadata' in data:
                        metadata = data['metadata']
                        print(f"   Papers analyzed: {metadata.get('papers_analyzed', 0)}")
                        print(f"   Analysis confidence: {metadata.get('analysis_confidence', 0):.2f}")
                        print(f"   Execution time: {metadata.get('execution_time', 0):.2f}s")
                    
                    if 'citations' in data:
                        citations_count = len(data['citations'])
                        print(f"   Citations: {citations_count}")
                    
                    # Show a preview of the output
                    if 'formatted_output' in data and data['formatted_output']:
                        preview = data['formatted_output'][:500] + "..." if len(data['formatted_output']) > 500 else data['formatted_output']
                        print(f"\n📄 Output Preview:")
                        print(preview)
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Results retrieval failed: HTTP {response.status}")
                    print(f"   Error: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Results test failed: {e}")
        return False


async def test_workflow_components():
    """Test individual workflow components"""
    print("\n🔍 Testing Workflow Components...")
    
    try:
        # Test agent factory and research agent creation
        from core.agent_factory import AgentFactory
        from config.settings import get_settings
        from core.ollama_manager import get_ollama_manager
        from memory.memory_manager import MemoryManager
        from mcp.mcp_manager import MCPManager
        
        settings = get_settings()
        ollama_manager = get_ollama_manager()
        memory_manager = MemoryManager()
        mcp_manager = MCPManager()
        
        agent_factory = AgentFactory(mcp_manager, memory_manager, settings, ollama_manager)
        
        print("✅ Agent factory created successfully")
        
        # Test research agent creation
        research_agent = await agent_factory.create_agent(
            agent_type="research",
            name="test_research_agent",
            custom_config={
                "enabled_capabilities": ["academic_research"],
                "max_papers": 5
            }
        )
        
        if research_agent:
            print(f"✅ Research agent created: {research_agent.agent_id}")
        else:
            print("❌ Failed to create research agent")
            return False
        
        # Test reasoning agent creation
        reasoning_agent = await agent_factory.create_agent(
            agent_type="reasoning",
            name="test_reasoning_agent",
            custom_config={
                "model_name": "deepseek2:latest",
                "enabled_capabilities": ["tree_of_thought"]
            }
        )
        
        if reasoning_agent:
            print(f"✅ Reasoning agent created: {reasoning_agent.agent_id}")
        else:
            print("❌ Failed to create reasoning agent")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    print("🏭 PyGent Factory Research Workflow Integration Test")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Component validation
    results['components'] = await test_workflow_components()
    
    # Test 2: API endpoint
    workflow_id = await test_workflow_api_endpoint()
    results['api_endpoint'] = workflow_id is not None
    
    if workflow_id:
        # Test 3: Status tracking
        results['status_tracking'] = await test_workflow_status_tracking(workflow_id)
        
        # Test 4: Results retrieval
        results['results_retrieval'] = await test_workflow_results(workflow_id)
    else:
        results['status_tracking'] = False
        results['results_retrieval'] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Research Workflow Test Results:")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 Research workflow is working perfectly!")
        print("   The research and analytics workflow has been fixed!")
    elif passed_tests > 0:
        print(f"\n⚠️ Partial success: {passed_tests}/{total_tests} components working")
        print("   Some components need attention")
    else:
        print("\n💥 Research workflow is still broken!")
        print("   Major issues need to be addressed")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(2)
