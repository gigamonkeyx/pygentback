#!/usr/bin/env python3
"""
Test script for the Research-to-Analysis Workflow

This script tests the automated Research-to-Analysis workflow without
requiring the full server to be running.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.workflows.research_analysis_orchestrator import (
    ResearchAnalysisOrchestrator,
    WorkflowStatus,
    WorkflowProgress
)
from src.core.agent_factory import AgentFactory
from src.config.config_manager import ConfigManager
from src.config.settings import Settings
from src.memory.memory_manager import MemoryManager
from src.mcp.server_registry import MCPServerManager
from src.storage.vector_store import VectorStoreManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def progress_callback(workflow_id: str, progress: WorkflowProgress):
    """Progress callback for workflow updates"""
    print(f"\n🔄 Workflow Progress ({workflow_id}):")
    print(f"   Status: {progress.status.value}")
    print(f"   Step: {progress.current_step}")
    print(f"   Progress: {progress.progress_percentage:.1f}%")
    
    if progress.research_papers_found > 0:
        print(f"   Papers Found: {progress.research_papers_found}")
    
    if progress.analysis_confidence > 0:
        print(f"   Analysis Confidence: {progress.analysis_confidence:.2f}")
    
    if progress.error_message:
        print(f"   ❌ Error: {progress.error_message}")


async def test_research_analysis_workflow():
    """Test the complete Research-to-Analysis workflow"""
    
    print("🧪 Testing Research-to-Analysis Workflow")
    print("=" * 50)
    
    try:
        # Initialize configuration
        print("\n1️⃣ Initializing configuration...")
        config_manager = ConfigManager()
        settings = Settings()

        # Initialize required managers
        print("\n2️⃣ Initializing managers...")
        # Create vector store manager for memory
        vector_store_manager = VectorStoreManager(settings)
        await vector_store_manager.initialize()
        
        memory_manager = MemoryManager(vector_store_manager, settings)
        mcp_manager = MCPServerManager(config_manager)
        
        # Initialize Ollama manager (required for reasoning agents)
        from src.core.ollama_manager import OllamaManager
        ollama_manager = OllamaManager(settings)
        try:
            await ollama_manager.initialize()
            print("   ✅ Ollama manager initialized")
        except Exception as e:
            print(f"   ❌ Ollama manager failed to initialize: {e}")
            print("   💡 Make sure Ollama is running: 'ollama serve'")
            raise Exception("Ollama is required for the research analysis workflow") from e

        # Initialize agent factory
        print("\n3️⃣ Initializing agent factory...")
        agent_factory = AgentFactory(
            mcp_manager=mcp_manager,
            memory_manager=memory_manager,
            settings=settings,
            ollama_manager=ollama_manager
        )
        
        # Create orchestrator
        print("\n4️⃣ Creating workflow orchestrator...")
        orchestrator = ResearchAnalysisOrchestrator(
            agent_factory=agent_factory,
            progress_callback=progress_callback
        )

        # Test query
        test_query = "quantum computing feasibility using larger qubits on silicon"

        print(f"\n5️⃣ Starting workflow for query: '{test_query}'")
        print("   This may take 2-5 minutes...")
        
        # Execute workflow
        start_time = datetime.now()
        
        result = await orchestrator.execute_workflow(
            query=test_query,
            analysis_model="deepseek-r1:8b",
            max_papers=10,
            analysis_depth=2,  # Reduced for testing
            workflow_id="test_workflow_001"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Display results
        print(f"\n✅ Workflow completed in {execution_time:.2f} seconds")
        print("=" * 50)
        
        if result.success:
            print("🎉 SUCCESS! Research-to-Analysis workflow is working!")
            print(f"\n📊 Results Summary:")
            print(f"   Query: {result.query}")
            print(f"   Papers Analyzed: {result.metadata.get('papers_analyzed', 0)}")
            print(f"   Analysis Model: {result.metadata.get('analysis_model', 'Unknown')}")
            print(f"   Analysis Confidence: {result.metadata.get('analysis_confidence', 0):.2f}")
            print(f"   Citations Found: {len(result.citations)}")
            print(f"   Execution Time: {result.execution_time:.2f}s")
            
            print(f"\n📝 Research Summary (first 200 chars):")
            research_summary = result.research_data.get('response', 'No research data')[:200]
            print(f"   {research_summary}...")
            
            print(f"\n🧠 Analysis Summary (first 200 chars):")
            analysis_summary = str(result.analysis_data.get('response', 'No analysis data'))[:200]
            print(f"   {analysis_summary}...")
            
            print(f"\n📄 Formatted Output Preview (first 300 chars):")
            print(f"   {result.formatted_output[:300]}...")
            
            # Save results to file
            output_file = f"test_workflow_results_{int(datetime.now().timestamp())}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.formatted_output)
            print(f"\n💾 Full results saved to: {output_file}")
            
        else:
            print("❌ FAILED! Workflow encountered errors:")
            print(f"   Error: {result.error_message}")
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


async def test_individual_components():
    """Test individual components of the workflow"""
    
    print("\n🔧 Testing Individual Components")
    print("=" * 40)
    
    try:
        # Test 1: Configuration
        print("\n🧪 Test 1: Configuration Manager")
        config_manager = ConfigManager()
        settings = Settings()
        print("   ✅ Configuration manager initialized")

        # Test 2: Required Managers
        print("\n🧪 Test 2: Required Managers")
        vector_store_manager = VectorStoreManager(settings)
        await vector_store_manager.initialize()
        
        memory_manager = MemoryManager(vector_store_manager, settings)
        mcp_manager = MCPServerManager(config_manager)
        
        # Initialize Ollama manager for testing reasoning agents
        from src.core.ollama_manager import OllamaManager
        ollama_manager = OllamaManager(settings)
        try:
            await ollama_manager.initialize()
            print("   ✅ Memory, MCP, and Ollama managers initialized")
        except Exception as e:
            print(f"   ⚠️  Ollama manager failed: {e}")
            print("   💡 Some tests may be skipped")
            ollama_manager = None

        # Test 3: Agent Factory
        print("\n🧪 Test 3: Agent Factory")
        agent_factory = AgentFactory(
            mcp_manager=mcp_manager,
            memory_manager=memory_manager,
            settings=settings,
            ollama_manager=ollama_manager
        )
        print("   ✅ Agent factory initialized")
        
        # Test 4: Research Agent Creation
        print("\n🧪 Test 4: Research Agent Creation")
        try:
            research_agent = await agent_factory.create_agent(
                agent_type="research",
                name="test_research_agent",
                custom_config={
                    "enabled_capabilities": ["academic_research"],
                    "max_papers": 5
                }
            )
            print("   ✅ Research agent created successfully")
            print(f"   Agent ID: {research_agent.agent_id}")
        except Exception as e:
            print(f"   ❌ Research agent creation failed: {e}")

        # Test 5: Reasoning Agent Creation
        print("\n🧪 Test 5: Reasoning Agent Creation")
        try:
            reasoning_agent = await agent_factory.create_agent(
                agent_type="reasoning",
                name="test_reasoning_agent",
                custom_config={
                    "model": "deepseek-r1:8b",
                    "enabled_capabilities": ["tree_of_thought"],
                    "reasoning_depth": 2
                }
            )
            print("   ✅ Reasoning agent created successfully")
            print(f"   Agent ID: {reasoning_agent.agent_id}")
        except Exception as e:
            print(f"   ❌ Reasoning agent creation failed: {e}")
        
        print("\n✅ Component tests completed!")
        
    except Exception as e:
        print(f"\n❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function"""
    
    print("🚀 Research-to-Analysis Workflow Test Suite")
    print("=" * 60)
    print("This test validates the automated Research-to-Analysis workflow")
    print("that combines real research data with AI-powered analysis.")
    print("=" * 60)
    
    # Test individual components first
    await test_individual_components()
    
    # Ask user if they want to run the full workflow test
    print("\n" + "=" * 60)
    response = input("🤔 Run full workflow test? This may take 2-5 minutes (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        await test_research_analysis_workflow()
    else:
        print("⏭️  Skipping full workflow test")
    
    print("\n🏁 Test suite completed!")
    print("\nNext steps:")
    print("1. Start the PyGent Factory server: python main.py server")
    print("2. Open the UI at http://localhost:8000")
    print("3. Navigate to 'Research & Analysis' page")
    print("4. Test the automated workflow through the web interface")


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
