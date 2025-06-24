#!/usr/bin/env python3
"""
Test agent orchestrator with Ollama initialized
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.ollama_integration import ollama_manager
from src.core.agent_orchestrator import agent_orchestrator, AgentTask, AgentType

async def test_with_ollama():
    """Test agent orchestrator with Ollama."""
    print("🧪 Testing Agent Orchestrator with Ollama")
    print("=" * 50)
    
    # Initialize Ollama first
    print("Initializing Ollama...")
    if await ollama_manager.initialize():
        print("✅ Ollama initialized")
    else:
        print("❌ Ollama failed to initialize")
        return
    
    # Start orchestrator (should detect Ollama is already initialized)
    print("Starting agent orchestrator...")
    await agent_orchestrator.start()
    print("✅ Agent orchestrator started")
    
    # Create a simple task
    print("\nSubmitting test task...")
    task = AgentTask(
        id="test_task_002",
        agent_type=AgentType.COORDINATOR,
        task_type="research_planning",
        input_data={
            "topic": "Simple Test Topic",
            "scope": "basic"
        }
    )
    
    task_id = await agent_orchestrator.submit_task(task)
    print(f"✅ Task submitted, ID: {task_id}")
    
    # Wait for result
    print("\nWaiting for task result...")
    try:
        result = await agent_orchestrator.get_task_result(task_id, timeout=10.0)
        if result:
            print("✅ Task completed successfully!")
            print(f"Result: {str(result)[:200]}...")
        else:
            print("❌ No result received")
    except Exception as e:
        print(f"❌ Error getting result: {e}")
    
    # Check system status
    status = agent_orchestrator.get_system_status()
    print(f"\nSystem status:")
    print(f"  Queue size: {status.get('queue_size', 0)}")
    print(f"  Running tasks: {status.get('running_tasks', 0)}")
    print(f"  Completed tasks: {status.get('completed_tasks', 0)}")
    
    # Stop
    await agent_orchestrator.stop()
    print("✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_with_ollama())
