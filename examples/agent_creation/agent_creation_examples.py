#!/usr/bin/env python3
"""
Agent Creation Examples

This script demonstrates how to create agents with different providers
and configurations. It shows the proper way to set up dependencies
and handle errors gracefully.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.agent_factory import AgentFactory
from src.core.ollama_manager import get_ollama_manager
from src.ai.providers.openrouter_provider import get_openrouter_manager
from src.config.settings import get_settings


async def create_reasoning_agent_ollama():
    """Example: Create a reasoning agent using Ollama"""
    print("🔧 Creating Reasoning Agent with Ollama...")
    
    # Initialize services
    settings = get_settings()
    ollama_manager = get_ollama_manager(settings)
    openrouter_manager = get_openrouter_manager(settings)
    
    agent_factory = AgentFactory(
        settings=settings,
        ollama_manager=ollama_manager,
        openrouter_manager=openrouter_manager
    )
    
    # Start Ollama
    if not await ollama_manager.start():
        print("❌ Ollama is not available")
        return None
    
    # Check if we have models
    models = await ollama_manager.get_available_models()
    if not models:
        print("❌ No Ollama models available. Please install one:")
        print("   ollama pull phi4-fast")
        return None
    
    # Configuration for Ollama-based reasoning agent
    custom_config = {
        "provider": "ollama",
        "model_name": models[0],  # Use first available model
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # Validate configuration before creation
    validation = await agent_factory.validate_agent_config_before_creation(
        "reasoning", 
        custom_config
    )
    
    if not validation["valid"]:
        print("❌ Configuration validation failed:")
        for error in validation["errors"]:
            print(f"   • {error}")
        return None
    
    try:
        # Create the agent
        agent = await agent_factory.create_agent(
            agent_type="reasoning",
            name="ReasoningAgent-Ollama",
            custom_config=custom_config
        )
        
        print(f"✅ Created reasoning agent: {agent.agent_id}")
        print(f"   Model: {custom_config['model_name']}")
        print(f"   Provider: Ollama")
        
        return agent
    
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return None


async def create_reasoning_agent_openrouter():
    """Example: Create a reasoning agent using OpenRouter"""
    print("🔧 Creating Reasoning Agent with OpenRouter...")
    
    # Initialize services
    settings = get_settings()
    ollama_manager = get_ollama_manager(settings)
    openrouter_manager = get_openrouter_manager(settings)
    
    agent_factory = AgentFactory(
        settings=settings,
        ollama_manager=ollama_manager,
        openrouter_manager=openrouter_manager
    )
    
    # Start OpenRouter
    if not await openrouter_manager.start():
        print("❌ OpenRouter is not available. Please set OPENROUTER_API_KEY environment variable")
        return None
    
    # Configuration for OpenRouter-based reasoning agent
    custom_config = {
        "provider": "openrouter",
        "model_name": "anthropic/claude-3.5-sonnet",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # Validate configuration before creation
    validation = await agent_factory.validate_agent_config_before_creation(
        "reasoning", 
        custom_config
    )
    
    if not validation["valid"]:
        print("❌ Configuration validation failed:")
        for error in validation["errors"]:
            print(f"   • {error}")
        return None
    
    try:
        # Create the agent
        agent = await agent_factory.create_agent(
            agent_type="reasoning",
            name="ReasoningAgent-OpenRouter",
            custom_config=custom_config
        )
        
        print(f"✅ Created reasoning agent: {agent.agent_id}")
        print(f"   Model: {custom_config['model_name']}")
        print(f"   Provider: OpenRouter")
        
        return agent
    
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return None


async def create_research_agent():
    """Example: Create a research agent (requires multiple services)"""
    print("🔧 Creating Research Agent...")
    
    # Initialize services
    settings = get_settings()
    ollama_manager = get_ollama_manager(settings)
    openrouter_manager = get_openrouter_manager(settings)
    
    agent_factory = AgentFactory(
        settings=settings,
        ollama_manager=ollama_manager,
        openrouter_manager=openrouter_manager
    )
    
    # Start services
    await ollama_manager.start()
    await openrouter_manager.start()
    
    # Check what's available
    has_ollama = ollama_manager.is_ready
    has_openrouter = openrouter_manager.is_ready
    
    if not (has_ollama or has_openrouter):
        print("❌ No LLM providers available")
        return None
    
    # Choose provider based on availability
    if has_openrouter:
        provider = "openrouter"
        model_name = "anthropic/claude-3.5-sonnet"
    else:
        provider = "ollama"
        models = await ollama_manager.get_available_models()
        model_name = models[0] if models else "phi4-fast"
    
    # Configuration for research agent
    custom_config = {
        "provider": provider,
        "model_name": model_name,
        "temperature": 0.3,  # Lower temperature for research
        "max_tokens": 2000
    }
    
    # Get agent creation guide
    guide = await agent_factory.get_agent_creation_guide("research")
    
    if not guide["system_ready"]:
        print("❌ System not ready for research agents:")
        for missing in guide["missing_requirements"]:
            print(f"   • Missing: {missing}")
        return None
    
    try:
        # Create the agent
        agent = await agent_factory.create_agent(
            agent_type="research",
            name="ResearchAgent",
            custom_config=custom_config
        )
        
        print(f"✅ Created research agent: {agent.agent_id}")
        print(f"   Model: {custom_config['model_name']}")
        print(f"   Provider: {provider}")
        
        return agent
    
    except Exception as e:
        print(f"❌ Failed to create research agent: {e}")
        return None


async def demonstrate_system_check():
    """Demonstrate system checking capabilities"""
    print("🔍 System Check Demonstration")
    print("=" * 50)
    
    # Initialize services
    settings = get_settings()
    ollama_manager = get_ollama_manager(settings)
    openrouter_manager = get_openrouter_manager(settings)
    
    agent_factory = AgentFactory(
        settings=settings,
        ollama_manager=ollama_manager,
        openrouter_manager=openrouter_manager
    )
    
    # Start services
    await ollama_manager.start()
    await openrouter_manager.start()
    
    # Get system readiness
    readiness = await agent_factory.get_system_readiness()
    
    print("📊 System Status:")
    print(f"   Ollama: {'✅ Ready' if readiness['providers']['ollama']['available'] else '❌ Not Ready'}")
    print(f"   OpenRouter: {'✅ Ready' if readiness['providers']['openrouter']['available'] else '❌ Not Ready'}")
    
    print("\n🤖 Agent Type Support:")
    for agent_type, support in readiness["agent_types_supported"].items():
        status = "✅" if support["supported"] else "❌"
        print(f"   {agent_type.capitalize()}: {status}")
    
    print("\n💡 Recommendations:")
    for rec in readiness["recommendations"]:
        print(f"   {rec}")


async def main():
    """Main demonstration"""
    print("🚀 PyGent Factory Agent Creation Examples")
    print("=" * 60)
    
    # Show system status first
    await demonstrate_system_check()
    
    print("\n" + "=" * 60)
    print("Creating Example Agents")
    print("=" * 60)
    
    # Try to create agents with different providers
    agents = []
    
    # Try Ollama-based reasoning agent
    agent1 = await create_reasoning_agent_ollama()
    if agent1:
        agents.append(agent1)
    
    print()
    
    # Try OpenRouter-based reasoning agent
    agent2 = await create_reasoning_agent_openrouter()
    if agent2:
        agents.append(agent2)
    
    print()
    
    # Try research agent
    agent3 = await create_research_agent()
    if agent3:
        agents.append(agent3)
    
    print("\n" + "=" * 60)
    print(f"Created {len(agents)} agents successfully")
    
    if agents:
        print("\n🎯 Next Steps:")
        print("   • Use these agents for your tasks")
        print("   • Check agent.process_message() for interaction")
        print("   • Monitor agent.get_status() for health")
        print("   • Call agent_factory.destroy_agent() when done")
    else:
        print("\n❌ No agents created. Check system requirements:")
        print("   • Run: python system_check.py")
        print("   • Install Ollama or configure OpenRouter")


if __name__ == "__main__":
    asyncio.run(main())
