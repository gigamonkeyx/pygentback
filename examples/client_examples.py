#!/usr/bin/env python3
"""
PyGent A2A Client Examples

Comprehensive examples demonstrating A2A client usage patterns.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

# Import the A2A client SDK
try:
    from pygent_a2a_client import (
        A2AClient, A2AConfig, TaskResult, TaskState,
        quick_search, quick_analysis
    )
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))
    from pygent_a2a_client import (
        A2AClient, A2AConfig, TaskResult, TaskState,
        quick_search, quick_analysis
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_1_basic_usage():
    """Example 1: Basic A2A client usage"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic A2A Client Usage")
    print("="*60)
    
    # Configure client
    config = A2AConfig(
        base_url="http://localhost:8080",
        timeout=30,
        log_requests=True
    )
    
    # Use client with context manager
    async with A2AClient(config) as client:
        # Check server health
        health = await client.health_check()
        print(f"✅ Server Status: {health.get('status')}")
        print(f"✅ Agents Available: {health.get('agents_registered')}")
        
        # Discover agent capabilities
        agent_info = await client.discover_agents()
        print(f"✅ Agent Name: {agent_info.get('name')}")
        print(f"✅ Skills Available: {len(agent_info.get('skills', []))}")
        
        # Send a simple task
        task = await client.send_task({
            "role": "user",
            "parts": [{"type": "text", "text": "Search for documents about machine learning"}]
        })
        
        print(f"✅ Task Created: {task.task_id}")
        print(f"✅ Initial State: {task.state.value}")
        
        # Wait for completion with progress callback
        def progress_callback(result: TaskResult):
            print(f"   📊 Task Progress: {result.state.value}")
        
        result = await client.wait_for_completion(
            task.task_id,
            timeout=60,
            poll_interval=2.0,
            callback=progress_callback
        )
        
        print(f"✅ Task Completed: {result.state.value}")
        print(f"✅ Artifacts: {len(result.artifacts)}")
        
        # Display results
        if result.artifacts:
            for i, artifact in enumerate(result.artifacts):
                print(f"   📄 Artifact {i+1}: {artifact.get('name', 'Unknown')}")


async def example_2_convenience_methods():
    """Example 2: Using convenience methods"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Convenience Methods")
    print("="*60)
    
    # Quick search without managing client lifecycle
    print("🔍 Quick Document Search:")
    result = await quick_search("quantum computing applications")
    print(f"✅ Search completed: {len(result.artifacts)} artifacts")
    
    # Quick analysis
    print("\n📊 Quick Data Analysis:")
    result = await quick_analysis("statistical trends in AI research publications")
    print(f"✅ Analysis completed: {len(result.artifacts)} artifacts")


async def example_3_batch_processing():
    """Example 3: Batch processing multiple tasks"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    config = A2AConfig(base_url="http://localhost:8080")
    
    async with A2AClient(config) as client:
        # Define multiple search queries
        queries = [
            "artificial intelligence ethics",
            "machine learning bias detection",
            "neural network interpretability",
            "automated decision making fairness",
            "AI transparency requirements"
        ]
        
        print(f"🚀 Starting batch processing of {len(queries)} queries...")
        
        # Send all tasks concurrently
        tasks = []
        for i, query in enumerate(queries):
            task = await client.send_task({
                "role": "user",
                "parts": [{"type": "text", "text": f"Search for documents about {query}"}]
            })
            tasks.append((i+1, query, task.task_id))
            print(f"   📤 Task {i+1} sent: {task.task_id[:8]}...")
        
        # Wait for all completions
        results = []
        for task_num, query, task_id in tasks:
            try:
                result = await client.wait_for_completion(task_id, timeout=30)
                results.append((task_num, query, result))
                print(f"   ✅ Task {task_num} completed: {len(result.artifacts)} artifacts")
            except Exception as e:
                print(f"   ❌ Task {task_num} failed: {e}")
        
        print(f"\n📊 Batch Results: {len(results)}/{len(queries)} tasks completed successfully")


async def example_4_error_handling():
    """Example 4: Comprehensive error handling"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Error Handling")
    print("="*60)
    
    config = A2AConfig(
        base_url="http://localhost:8080",
        timeout=10,
        max_retries=2
    )
    
    async with A2AClient(config) as client:
        # Test 1: Valid request
        print("🧪 Test 1: Valid request")
        try:
            task = await client.send_task({
                "role": "user",
                "parts": [{"type": "text", "text": "Test query"}]
            })
            print(f"   ✅ Success: {task.task_id[:8]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 2: Invalid task ID
        print("\n🧪 Test 2: Invalid task ID")
        try:
            result = await client.get_task("invalid-task-id")
            print(f"   ✅ Unexpected success: {result.task_id}")
        except Exception as e:
            print(f"   ✅ Expected error: {type(e).__name__}: {e}")
        
        # Test 3: Malformed request
        print("\n🧪 Test 3: Malformed request")
        try:
            task = await client.send_task({
                "invalid_role": "user",
                "invalid_parts": "not a list"
            })
            print(f"   ❌ Unexpected success: {task.task_id}")
        except Exception as e:
            print(f"   ✅ Expected error: {type(e).__name__}: {e}")


async def example_5_advanced_features():
    """Example 5: Advanced client features"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Advanced Features")
    print("="*60)
    
    # Configure with authentication and advanced options
    config = A2AConfig(
        base_url="http://localhost:8080",
        api_key="demo-api-key",  # Would be real in production
        timeout=60,
        max_retries=3,
        retry_delay=2.0,
        log_requests=True,
        log_responses=True,
        max_connections=50,
        keepalive_timeout=60
    )
    
    async with A2AClient(config) as client:
        # List all available agents
        print("👥 Available Agents:")
        agents = await client.list_agents()
        for agent in agents:
            print(f"   • {agent.get('name')} ({agent.get('type')}) - {agent.get('status')}")
        
        # Send task with session ID for grouping
        print("\n📝 Sending task with session grouping:")
        session_id = f"demo_session_{int(datetime.now().timestamp())}"
        
        task = await client.send_task({
            "role": "user",
            "parts": [{"type": "text", "text": "Comprehensive analysis of renewable energy trends"}]
        }, session_id=session_id)
        
        print(f"   ✅ Task sent with session ID: {session_id}")
        print(f"   ✅ Task ID: {task.task_id}")
        
        # Monitor task progress
        print("\n📊 Monitoring task progress:")
        start_time = datetime.now()
        
        def detailed_progress(result: TaskResult):
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"   ⏱️  {elapsed:6.1f}s - State: {result.state.value}")
            if result.artifacts:
                print(f"      📄 Artifacts: {len(result.artifacts)}")
        
        try:
            result = await client.wait_for_completion(
                task.task_id,
                timeout=120,
                poll_interval=3.0,
                callback=detailed_progress
            )
            
            print(f"\n✅ Task completed successfully!")
            print(f"   📊 Final state: {result.state.value}")
            print(f"   📄 Total artifacts: {len(result.artifacts)}")
            print(f"   🕒 Session ID: {result.session_id}")
            
            # Display artifact details
            for i, artifact in enumerate(result.artifacts):
                print(f"   📄 Artifact {i+1}:")
                print(f"      Name: {artifact.get('name', 'Unknown')}")
                print(f"      Description: {artifact.get('description', 'No description')}")
                if artifact.get('parts'):
                    content_preview = str(artifact['parts'][0].get('text', ''))[:100]
                    print(f"      Content: {content_preview}...")
        
        except Exception as e:
            print(f"❌ Task failed: {e}")


async def example_6_real_world_workflow():
    """Example 6: Real-world research workflow"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Real-World Research Workflow")
    print("="*60)
    
    config = A2AConfig(base_url="http://localhost:8080")
    
    async with A2AClient(config) as client:
        # Step 1: Research topic discovery
        print("🔍 Step 1: Topic Discovery")
        discovery_task = await client.send_task({
            "role": "user",
            "parts": [{"type": "text", "text": "Search for emerging trends in artificial intelligence research"}]
        })
        
        discovery_result = await client.wait_for_completion(discovery_task.task_id)
        print(f"   ✅ Found {len(discovery_result.artifacts)} research areas")
        
        # Step 2: Deep dive analysis
        print("\n📊 Step 2: Deep Analysis")
        analysis_task = await client.send_task({
            "role": "user",
            "parts": [{"type": "text", "text": "Analyze the impact and potential of large language models in scientific research"}]
        })
        
        analysis_result = await client.wait_for_completion(analysis_task.task_id)
        print(f"   ✅ Analysis completed with {len(analysis_result.artifacts)} insights")
        
        # Step 3: Synthesis and recommendations
        print("\n🔬 Step 3: Synthesis")
        synthesis_task = await client.send_task({
            "role": "user",
            "parts": [{"type": "text", "text": "Synthesize findings and provide recommendations for future AI research directions"}]
        })
        
        synthesis_result = await client.wait_for_completion(synthesis_task.task_id)
        print(f"   ✅ Synthesis completed with {len(synthesis_result.artifacts)} recommendations")
        
        # Summary
        total_artifacts = (
            len(discovery_result.artifacts) + 
            len(analysis_result.artifacts) + 
            len(synthesis_result.artifacts)
        )
        
        print(f"\n🎯 Workflow Summary:")
        print(f"   📊 Total artifacts generated: {total_artifacts}")
        print(f"   ⏱️  Research workflow completed successfully")


async def main():
    """Run all examples"""
    print("🚀 PyGent A2A Client SDK Examples")
    print("=" * 80)
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Convenience Methods", example_2_convenience_methods),
        ("Batch Processing", example_3_batch_processing),
        ("Error Handling", example_4_error_handling),
        ("Advanced Features", example_5_advanced_features),
        ("Real-World Workflow", example_6_real_world_workflow),
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n🎯 Running: {name}")
            await example_func()
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            logger.exception(f"Example {name} failed")
    
    print("\n" + "="*80)
    print("🎉 All examples completed!")
    print("="*80)


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
