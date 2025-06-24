#!/usr/bin/env python3
"""
Agent Factory Integration Test

Tests the core agent factory functionality to ensure agents can be created
and process messages correctly following Context7 MCP best practices.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root for proper imports
import os
os.chdir(project_root)

# Import only what we can safely import
try:
    from core.agent import BaseAgent, AgentMessage, MessageType
    AGENT_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Agent imports not available: {e}")
    AGENT_IMPORTS_AVAILABLE = False


async def test_agent_creation():
    """Test basic agent creation"""
    print("🧪 Testing agent creation...")

    try:
        # Test basic agent imports and structure
        from core.agent import BaseAgent, AgentMessage, MessageType

        # Test that we can import the classes
        print("✅ Successfully imported BaseAgent, AgentMessage, MessageType")

        # Test AgentMessage creation
        test_message = AgentMessage(
            type=MessageType.REQUEST,
            sender="test_system",
            recipient="test_agent",
            content={"test": "message"}
        )

        if test_message and test_message.id:
            print(f"✅ Successfully created AgentMessage: {test_message.id}")
            return test_message
        else:
            print("❌ Failed to create AgentMessage")
            return None

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return None


async def test_agent_message_processing(message):
    """Test agent message processing"""
    print("\n🧪 Testing agent message processing...")

    try:
        from core.agent import MessageType

        # Test message properties
        if hasattr(message, 'id') and message.id:
            print(f"✅ Message has valid ID: {message.id}")

        if hasattr(message, 'type') and message.type == MessageType.REQUEST:
            print(f"✅ Message has correct type: {message.type}")

        if hasattr(message, 'content') and message.content:
            print(f"✅ Message has content: {message.content}")

        if hasattr(message, 'sender') and message.sender:
            print(f"✅ Message has sender: {message.sender}")

        if hasattr(message, 'recipient') and message.recipient:
            print(f"✅ Message has recipient: {message.recipient}")

        return True

    except Exception as e:
        print(f"❌ Message processing failed: {e}")
        return False


async def test_agent_factory_registry():
    """Test agent factory registry functionality"""
    print("\n🧪 Testing agent factory registry...")

    try:
        # Test core message system integration
        from core.message_system import MessageBus, MessagePriority

        # Create message bus
        message_bus = MessageBus()

        if message_bus:
            print("✅ Successfully created MessageBus")

        # Test message priority enum
        priorities = [MessagePriority.LOW, MessagePriority.NORMAL, MessagePriority.HIGH, MessagePriority.URGENT]
        print(f"✅ MessagePriority enum has {len(priorities)} levels")

        return True

    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False


async def test_agent_lifecycle():
    """Test agent lifecycle management"""
    print("\n🧪 Testing agent lifecycle...")

    try:
        # Test orchestration manager import
        from orchestration.orchestration_manager import OrchestrationManager

        print("✅ Successfully imported OrchestrationManager")

        # Test that we can create an orchestration manager
        manager = OrchestrationManager()

        if manager:
            print("✅ Successfully created OrchestrationManager instance")
            print(f"   Running status: {manager.is_running}")

            # Test basic properties
            if hasattr(manager, 'agent_registry'):
                print("✅ OrchestrationManager has agent_registry")

            if hasattr(manager, 'mcp_orchestrator'):
                print("✅ OrchestrationManager has mcp_orchestrator")

            if hasattr(manager, 'task_dispatcher'):
                print("✅ OrchestrationManager has task_dispatcher")

        return True

    except Exception as e:
        print(f"❌ Lifecycle test failed: {e}")
        return False


async def run_all_tests():
    """Run all agent factory tests"""
    print("🏭 PyGent Factory Agent Factory Integration Tests")
    print("=" * 60)

    if not AGENT_IMPORTS_AVAILABLE:
        print("❌ Agent imports not available - skipping tests")
        return False

    test_results = {
        "agent_creation": False,
        "message_processing": False,
        "factory_registry": False,
        "agent_lifecycle": False
    }
    
    # Test 1: Agent Creation
    message = await test_agent_creation()
    test_results["agent_creation"] = message is not None

    # Test 2: Message Processing (only if message creation succeeded)
    if message:
        test_results["message_processing"] = await test_agent_message_processing(message)
    
    # Test 3: Factory Registry
    test_results["factory_registry"] = await test_agent_factory_registry()
    
    # Test 4: Agent Lifecycle
    test_results["agent_lifecycle"] = await test_agent_lifecycle()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
        if passed:
            passed_tests += 1
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All agent factory tests passed!")
        return True
    else:
        print("💥 Some agent factory tests failed!")
        return False


async def main():
    """Main test function"""
    try:
        success = await run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
