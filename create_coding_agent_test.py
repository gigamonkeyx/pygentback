#!/usr/bin/env python3
"""
Create and Test a Coding Agent using OpenRouter/Ollama
Demonstrates the 3 research agents and creates a new coding agent
"""

import asyncio
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def investigate_research_agents():
    """Investigate the 3 research agents that are being initialized"""
    
    logger.info("🔍 INVESTIGATING THE 3 RESEARCH AGENTS")
    logger.info("=" * 60)
    
    try:
        # Import the agent orchestrator
        from src.core.agent_orchestrator import AgentOrchestrator, AgentType
        
        # Create orchestrator instance
        orchestrator = AgentOrchestrator()
        
        logger.info(f"📋 Found {len(orchestrator.agents)} research agents:")
        
        for agent_type, agent in orchestrator.agents.items():
            logger.info(f"  🤖 {agent_type.value}: {agent.__class__.__name__}")
            logger.info(f"     ID: {agent.agent_id}")
            logger.info(f"     Type: {agent.agent_type.value}")
            
            # Check what tasks each agent can handle
            test_tasks = ["research_planning", "analyze_document", "fact_check", "coding", "web_search"]
            capabilities = []
            for task in test_tasks:
                if agent.can_handle_task(task):
                    capabilities.append(task)
            
            logger.info(f"     Capabilities: {capabilities}")
            logger.info("")
        
        return orchestrator
        
    except Exception as e:
        logger.error(f"Failed to investigate research agents: {e}")
        return None

async def create_coding_agent():
    """Create a coding agent using the agent factory"""
    
    logger.info("🛠️ CREATING CODING AGENT")
    logger.info("=" * 40)
    
    try:
        # Import agent factory
        from src.core.agent_factory import AgentFactory
        
        # Initialize agent factory
        agent_factory = AgentFactory()
        await agent_factory.initialize()
        
        logger.info("✅ Agent factory initialized")
        
        # Create a coding agent
        coding_agent = await agent_factory.create_agent(
            agent_type="coding",
            name="Python Coding Assistant",
            capabilities=["code_generation", "code_review", "debugging", "testing"],
            custom_config={
                "programming_languages": ["python", "javascript", "html", "css"],
                "specialization": "web_development",
                "code_style": "clean_and_documented"
            }
        )
        
        logger.info(f"✅ Created coding agent: {coding_agent.config.agent_id}")
        logger.info(f"   Name: {coding_agent.config.name}")
        logger.info(f"   Type: {coding_agent.config.agent_type}")
        logger.info(f"   Capabilities: {coding_agent.config.enabled_capabilities}")
        
        return coding_agent
        
    except Exception as e:
        logger.error(f"Failed to create coding agent: {e}")
        return None

async def test_coding_agent(coding_agent):
    """Test the coding agent with a simple task"""
    
    logger.info("🧪 TESTING CODING AGENT")
    logger.info("=" * 30)
    
    try:
        # Test task: Create a simple HTML page with a flying dragon
        task = {
            "task_type": "code_generation",
            "requirements": "Create an HTML page with a flying dragon animation using CSS",
            "language": "html",
            "style": "modern and animated"
        }
        
        logger.info("📝 Task: Create HTML page with flying dragon animation")
        
        # Execute the task
        result = await coding_agent.execute_task(task)
        
        logger.info("✅ Coding agent task completed!")
        logger.info(f"   Result type: {type(result)}")
        
        if isinstance(result, dict):
            if "code" in result:
                logger.info("   Generated code preview:")
                code_preview = result["code"][:200] + "..." if len(result["code"]) > 200 else result["code"]
                logger.info(f"   {code_preview}")
            
            if "explanation" in result:
                logger.info(f"   Explanation: {result['explanation'][:100]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to test coding agent: {e}")
        return None

async def check_model_usage():
    """Check what models are being used by the agents"""
    
    logger.info("🔍 CHECKING MODEL USAGE")
    logger.info("=" * 30)
    
    try:
        # Check OpenRouter status
        import requests
        
        # Check backend health
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            
            logger.info("🏥 Backend Health Status:")
            logger.info(f"   Overall Status: {health_data.get('status', 'unknown')}")
            
            components = health_data.get('components', {})
            
            # Check OpenRouter
            if 'openrouter' in components:
                openrouter = components['openrouter']
                logger.info(f"   OpenRouter: {openrouter.get('status', 'unknown')}")
                if 'details' in openrouter:
                    models = openrouter['details'].get('available_models', [])
                    logger.info(f"   Available Models: {len(models)}")
            
            # Check Ollama
            if 'ollama' in components:
                ollama = components['ollama']
                logger.info(f"   Ollama: {ollama.get('status', 'unknown')}")
                if 'details' in ollama:
                    models = ollama['details'].get('available_models', [])
                    logger.info(f"   Available Models: {len(models)}")
                    if models:
                        logger.info(f"   Models: {models}")
        
    except Exception as e:
        logger.error(f"Failed to check model usage: {e}")

async def main():
    """Main execution function"""
    
    logger.info("🚀 STARTING CODING AGENT INVESTIGATION")
    logger.info("=" * 70)
    
    # Step 1: Investigate the 3 research agents
    orchestrator = await investigate_research_agents()
    
    # Step 2: Check model usage
    await check_model_usage()
    
    # Step 3: Create a coding agent
    coding_agent = await create_coding_agent()
    
    # Step 4: Test the coding agent
    if coding_agent:
        result = await test_coding_agent(coding_agent)
        
        if result:
            logger.info("🎉 SUCCESS! Coding agent created and tested successfully!")
        else:
            logger.warning("⚠️ Coding agent created but test failed")
    else:
        logger.error("❌ Failed to create coding agent")
    
    logger.info("🏁 INVESTIGATION COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())
