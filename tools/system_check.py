#!/usr/bin/env python3
"""
PyGent Factory System Check Utility

This script helps users understand what's available and what's needed
to create different types of agents without requiring a UI.

Usage:
    python system_check.py                    # General system overview
    python system_check.py --agent reasoning # Check specific agent requirements
    python system_check.py --models          # Show available models
    python system_check.py --setup           # Setup recommendations
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.agent_factory import AgentFactory
from src.core.ollama_manager import get_ollama_manager
from src.ai.providers.openrouter_provider import get_openrouter_manager
from src.config.settings import get_settings


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_status(status: str, description: str):
    """Print a status line with emoji"""
    status_icons = {
        "ready": "✅",
        "not_ready": "❌", 
        "warning": "⚠️",
        "info": "ℹ️",
        "not_configured": "⚪"
    }
    icon = status_icons.get(status, "❓")
    print(f"{icon} {description}")


async def check_system_overview():
    """Check overall system status"""
    print_header("PyGent Factory System Overview")
    
    # Initialize managers
    settings = get_settings()
    ollama_manager = get_ollama_manager(settings)
    openrouter_manager = get_openrouter_manager(settings)
    
    # Create agent factory
    agent_factory = AgentFactory(
        settings=settings,
        ollama_manager=ollama_manager,
        openrouter_manager=openrouter_manager
    )
    
    # Start services
    print("🔍 Checking services...")
    
    ollama_ready = await ollama_manager.start()
    openrouter_ready = await openrouter_manager.start()
    
    # Get system readiness
    readiness = await agent_factory.get_system_readiness()
    
    # Print provider status
    print_header("LLM Providers")
    
    if ollama_ready:
        models = await ollama_manager.get_available_models()
        print_status("ready", f"Ollama: Ready ({len(models)} models available)")
        if models:
            print(f"   Popular models: {', '.join(models[:3])}")
    else:
        print_status("not_ready", "Ollama: Not available")
        print("   💡 Install Ollama from https://ollama.ai/")
    
    if openrouter_ready:
        models = await openrouter_manager.get_available_models()
        print_status("ready", f"OpenRouter: Ready ({len(models)} models available)")
        popular = openrouter_manager.get_popular_models()
        print(f"   Popular models: {', '.join(list(popular.keys())[:3])}")
    else:
        print_status("not_ready", "OpenRouter: Not available")
        print("   💡 Set OPENROUTER_API_KEY environment variable")
    
    # Print agent support
    print_header("Agent Type Support")
    
    for agent_type, support_info in readiness["agent_types_supported"].items():
        if support_info["supported"]:
            print_status("ready", f"{agent_type.capitalize()}: Supported")
        else:
            print_status("not_ready", f"{agent_type.capitalize()}: Missing requirements")
            requirements = support_info["requirements"]
            missing = []
            if requirements.get("llm_provider") and not (ollama_ready or openrouter_ready):
                missing.append("LLM provider")
            if requirements.get("memory") and not readiness["services"]["memory_manager"]:
                missing.append("memory manager")
            if requirements.get("mcp_tools") and not readiness["services"]["mcp_servers"]:
                missing.append("MCP servers")
            if missing:
                print(f"   Missing: {', '.join(missing)}")
    
    # Print recommendations
    print_header("Setup Recommendations")
    
    for recommendation in readiness["recommendations"]:
        print(f"   {recommendation}")
    
    return readiness


async def check_agent_requirements(agent_type: str):
    """Check requirements for a specific agent type"""
    print_header(f"{agent_type.capitalize()} Agent Requirements")
    
    # Initialize managers
    settings = get_settings()
    ollama_manager = get_ollama_manager(settings)
    openrouter_manager = get_openrouter_manager(settings)
    
    # Create agent factory
    agent_factory = AgentFactory(
        settings=settings,
        ollama_manager=ollama_manager,
        openrouter_manager=openrouter_manager
    )
    
    # Start services
    await ollama_manager.start()
    await openrouter_manager.start()
    
    # Get agent creation guide
    guide = await agent_factory.get_agent_creation_guide(agent_type)
    
    if guide["system_ready"]:
        print_status("ready", f"System is ready to create {agent_type} agents")
    else:
        print_status("not_ready", f"System is NOT ready to create {agent_type} agents")
    
    print(f"\n📋 Requirements:")
    for req, needed in guide["requirements"].items():
        if needed:
            print(f"   ✓ {req.replace('_', ' ').title()}")
    
    if guide["missing_requirements"]:
        print(f"\n❌ Missing:")
        for missing in guide["missing_requirements"]:
            print(f"   • {missing}")
    
    print(f"\n🔧 Configuration Examples:")
    
    if "ollama" in guide["configuration_examples"]:
        print(f"   Ollama:")
        config = guide["configuration_examples"]["ollama"]
        print(f"     {json.dumps(config, indent=6)}")
    
    if "openrouter" in guide["configuration_examples"]:
        print(f"   OpenRouter:")
        config = guide["configuration_examples"]["openrouter"]
        print(f"     {json.dumps(config, indent=6)}")


async def show_available_models():
    """Show all available models"""
    print_header("Available Models")
    
    # Initialize managers
    settings = get_settings()
    ollama_manager = get_ollama_manager(settings)
    openrouter_manager = get_openrouter_manager(settings)
    
    # Create agent factory
    agent_factory = AgentFactory(
        settings=settings,
        ollama_manager=ollama_manager,
        openrouter_manager=openrouter_manager
    )
    
    # Start services
    await ollama_manager.start()
    await openrouter_manager.start()
    
    # Get models
    models = await agent_factory.get_available_models()
    
    if models["ollama"]:
        print(f"\n🦙 Ollama Models ({len(models['ollama'])}):")
        for model in models["ollama"]:
            print(f"   • {model}")
    else:
        print(f"\n🦙 Ollama Models: None available")
        print("   💡 Install models with: ollama pull <model-name>")
    
    if models["openrouter"]:
        print(f"\n🌐 OpenRouter Models ({len(models['openrouter'])}):")
        popular = openrouter_manager.get_popular_models()
        for model_id, description in popular.items():
            print(f"   • {model_id} - {description}")
        print(f"   ... and {len(models['openrouter']) - len(popular)} more")
    else:
        print(f"\n🌐 OpenRouter Models: Not available")
        print("   💡 Set OPENROUTER_API_KEY to access cloud models")

async def show_setup_guide():
    """Show detailed setup guide"""
    print_header("PyGent Factory Setup Guide")
    
    print("""
📖 Quick Setup Guide:

1. 🦙 Local LLM with Ollama (Recommended for privacy):
   • Install Ollama: https://ollama.ai/
   • Download a model: ollama pull phi4-fast
   • Verify: ollama list

2. 🌐 Cloud LLM with OpenRouter (Recommended for performance):
   • Sign up: https://openrouter.ai/
   • Get API key from dashboard
   • Set environment variable: OPENROUTER_API_KEY=your_key_here

3. 🧠 Optional: Memory & Tools:
   • Vector database for agent memory
   • MCP servers for external tools

4. 🚀 Create your first agent:
   • Use system_check.py --agent reasoning to see requirements
   • Configure model and provider in agent config
""")
    
    # Show current system state
    await check_system_overview()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PyGent Factory System Check")
    parser.add_argument("--agent", help="Check requirements for specific agent type")
    parser.add_argument("--models", action="store_true", help="Show available models")
    parser.add_argument("--setup", action="store_true", help="Show setup guide")
    
    args = parser.parse_args()
    
    try:
        if args.agent:
            await check_agent_requirements(args.agent)
        elif args.models:
            await show_available_models()
        elif args.setup:
            await show_setup_guide()
        else:
            await check_system_overview()
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
