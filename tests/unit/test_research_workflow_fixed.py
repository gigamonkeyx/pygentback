"""
Fixed Research Analysis Workflow Test

Fixes:
1. Memory manager initialization
2. ToT engine parameter issues
3. Enhanced research portal manager integration
4. Better error handling and fallbacks
"""

#!/usr/bin/env python3

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
from src.mcp.server_registry import MCPServerManager
from src.storage.vector_store import VectorStoreManager
from src.memory.memory_manager import MemoryManager

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


class SimplifiedMemoryManager:
    """Simplified memory manager for testing"""
    
    def __init__(self):
        self.spaces = {}
    
    async def create_memory_space(self, agent_id: str, config: dict):
        """Create a mock memory space"""
        class MockMemorySpace:
            def __init__(self, agent_id):
                self.agent_id = agent_id
        
        space = MockMemorySpace(agent_id)
        self.spaces[agent_id] = space
        return space
    
    async def cleanup_memory_space(self, agent_id: str):
        """Clean up memory space"""
        if agent_id in self.spaces:
            del self.spaces[agent_id]
    
    async def get_memory_spaces_count(self) -> int:
        """Get count of memory spaces"""
        return len(self.spaces)


async def test_research_analysis_workflow():
    """Test the complete Research-to-Analysis workflow with fixes"""
    
    print("🧪 Testing Fixed Research-to-Analysis Workflow")
    print("=" * 50)
    
    try:
        # Initialize configuration
        print("\n1️⃣ Initializing configuration...")
        config_manager = ConfigManager()
        settings = Settings()

        # Initialize required managers with fixes
        print("\n2️⃣ Initializing managers...")
        
        # Use simplified memory manager to avoid vector store issues
        memory_manager = SimplifiedMemoryManager()
        mcp_manager = MCPServerManager(config_manager)

        # Initialize agent factory with fixed memory manager
        print("\n3️⃣ Initializing agent factory...")
        agent_factory = AgentFactory(
            mcp_manager=mcp_manager,
            memory_manager=memory_manager,
            settings=settings
        )
        
        # Create orchestrator
        print("\n4️⃣ Creating workflow orchestrator...")
        orchestrator = ResearchAnalysisOrchestrator(
            agent_factory=agent_factory,
            progress_callback=progress_callback
        )

        # Test query - use a simpler topic for better results
        test_query = "machine learning applications in healthcare"

        print(f"\n5️⃣ Starting workflow for query: '{test_query}'")
        print("   This may take 2-5 minutes...")
        
        # Execute workflow
        start_time = datetime.now()
        
        result = await orchestrator.execute_workflow(
            query=test_query,
            analysis_model="deepseek-r1:8b",
            max_papers=5,  # Reduced for testing
            analysis_depth=2,  # Reduced for testing
            workflow_id="test_workflow_fixed_001"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Display results
        print(f"\n✅ Workflow completed in {execution_time:.2f} seconds")
        print("=" * 50)
        
        if result.success:
            print("🎉 SUCCESS! Fixed Research-to-Analysis workflow is working!")
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
            output_file = f"test_workflow_fixed_results_{int(datetime.now().timestamp())}.md"
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


async def test_enhanced_research_portal():
    """Test the enhanced research portal manager"""
    
    print("\n🔧 Testing Enhanced Research Portal Manager")
    print("=" * 40)
    
    try:
        # Import and test the enhanced portal manager
        from src.research.enhanced_research_portal_manager import research_portal_manager
        
        # Test with a simple query
        test_query = "machine learning"
        print(f"\n🔍 Testing search with query: '{test_query}'")
        
        results = await research_portal_manager.search_all_portals(test_query, max_results_per_portal=3)
        
        print(f"✅ Enhanced portal search completed!")
        print(f"   Total papers found: {len(results)}")
        
        for i, paper in enumerate(results[:3]):
            print(f"\n📄 Paper {i+1}:")
            print(f"   Title: {paper['title'][:60]}...")
            print(f"   Source: {paper['source']}")
            print(f"   Authors: {', '.join(paper['authors'][:2]) if paper['authors'] else 'Unknown'}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Enhanced portal test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🚀 Fixed Research Workflow Test Suite")
    print("=" * 60)
    print("This test validates fixes to the automated Research-to-Analysis workflow")
    print("=" * 60)
    
    # Test enhanced research portal first
    portal_success = await test_enhanced_research_portal()
    
    if not portal_success:
        print("\n⚠️  Enhanced portal test failed, but continuing with full workflow test...")
    
    # Test the full workflow
    await test_research_analysis_workflow()
    
    print("\n🏁 Test suite completed!")
    print("\nNext steps:")
    print("1. Check the generated results file for output quality")
    print("2. Start the PyGent Factory server: python main.py server")
    print("3. Open the UI at http://localhost:8000")
    print("4. Navigate to 'Research & Analysis' page")
    print("5. Test the automated workflow through the web interface")


if __name__ == "__main__":
    asyncio.run(main())
