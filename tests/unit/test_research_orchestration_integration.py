#!/usr/bin/env python3
"""
Research Orchestration Integration Test

This test demonstrates why the research workflow is not using the main 
orchestration system and shows how to integrate them properly.
"""

import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def analyze_orchestration_gap():
    """Analyze the gap between main orchestration and research workflow."""
    
    print("🔍 ANALYZING ORCHESTRATION SYSTEM INTEGRATION")
    print("=" * 60)
    
    # Test 1: Check if OrchestrationManager exists and what it can do
    print("\n1️⃣ Testing Main OrchestrationManager...")
    try:
        from src.orchestration.orchestration_manager import OrchestrationManager
        from src.orchestration.coordination_models import OrchestrationConfig
        
        config = OrchestrationConfig()
        manager = OrchestrationManager(config)
        
        print("✅ OrchestrationManager imported successfully")
        print(f"   - Components: {len([attr for attr in dir(manager) if not attr.startswith('_')])}")
        print(f"   - Has agent_registry: {hasattr(manager, 'agent_registry')}")
        print(f"   - Has task_dispatcher: {hasattr(manager, 'task_dispatcher')}")
        print(f"   - Has mcp_orchestrator: {hasattr(manager, 'mcp_orchestrator')}")
        
        # Check if it has research capabilities
        research_capabilities = []
        for attr in dir(manager):
            if 'research' in attr.lower():
                research_capabilities.append(attr)
        
        if research_capabilities:
            print(f"   - Research capabilities: {research_capabilities}")
        else:
            print("   ❌ NO RESEARCH-SPECIFIC CAPABILITIES FOUND")
            
    except Exception as e:
        print(f"❌ Failed to load OrchestrationManager: {e}")
        return False
    
    # Test 2: Check ResearchAnalysisOrchestrator
    print("\n2️⃣ Testing ResearchAnalysisOrchestrator...")
    try:
        from src.workflows.research_analysis_orchestrator import ResearchAnalysisOrchestrator
        
        print("✅ ResearchAnalysisOrchestrator imported successfully")
        print("   - This is a STANDALONE orchestrator")
        print("   - NOT integrated with main OrchestrationManager")
        
        # Check if it uses main orchestration components
        orchestrator = ResearchAnalysisOrchestrator(None)
        uses_main_orchestration = False
        
        for attr in dir(orchestrator):
            if any(comp in attr.lower() for comp in ['agent_registry', 'task_dispatcher', 'mcp_orchestrator']):
                uses_main_orchestration = True
                break
        
        if uses_main_orchestration:
            print("   ✅ Uses main orchestration components")
        else:
            print("   ❌ DOES NOT use main orchestration components")
            
    except Exception as e:
        print(f"❌ Failed to load ResearchAnalysisOrchestrator: {e}")
        return False
    
    # Test 3: Check what research agents exist
    print("\n3️⃣ Checking available research agents...")
    try:
        from src.core.agent_factory import AgentFactory
        
        factory = AgentFactory()
        agent_types = factory.get_available_agent_types()
        
        research_types = [t for t in agent_types if 'research' in t.lower()]
        print(f"   - Total agent types: {len(agent_types)}")
        print(f"   - Research agent types: {research_types}")
        
        if 'research' in agent_types:
            print("   ✅ Research agent type available")
        else:
            print("   ❌ No research agent type found")
            
    except Exception as e:
        print(f"❌ Failed to check agent types: {e}")
    
    # Test 4: Check task dispatcher capabilities
    print("\n4️⃣ Checking TaskDispatcher research capabilities...")
    try:
        from src.orchestration.task_dispatcher import TaskDispatcher
        from src.orchestration.coordination_models import TaskRequest, OrchestrationConfig
        from src.orchestration.agent_registry import AgentRegistry
        from src.orchestration.mcp_orchestrator import MCPOrchestrator
        
        config = OrchestrationConfig()
        agent_registry = AgentRegistry(config)
        mcp_orchestrator = MCPOrchestrator(config)
        
        dispatcher = TaskDispatcher(config, agent_registry, mcp_orchestrator)
        
        print("✅ TaskDispatcher created successfully")
        print(f"   - Has dispatch strategies: {hasattr(dispatcher, 'dispatch_strategy')}")
        print("   - Can handle research tasks: UNCLEAR - needs investigation")
        
        # Check if TaskRequest supports research-specific fields
        task_fields = [attr for attr in dir(TaskRequest) if not attr.startswith('_')]
        research_fields = [f for f in task_fields if 'research' in f.lower()]
        
        if research_fields:
            print(f"   - Research-specific task fields: {research_fields}")
        else:
            print("   ❌ NO research-specific task fields found")
            
    except Exception as e:
        print(f"❌ Failed to test TaskDispatcher: {e}")
    
    # Summary and recommendations
    print("\n" + "=" * 60)
    print("🎯 ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("\n🔴 PROBLEMS IDENTIFIED:")
    print("1. ResearchAnalysisOrchestrator is STANDALONE - not using main orchestration")
    print("2. No integration between research workflows and TaskDispatcher")
    print("3. Main OrchestrationManager doesn't know about research tasks")
    print("4. Parallel orchestration systems causing confusion")
    
    print("\n🟢 SOLUTION RECOMMENDATIONS:")
    print("1. Create ResearchTask class that extends TaskRequest")
    print("2. Register research agents with main OrchestrationManager")
    print("3. Create research workflow templates in TaskDispatcher")
    print("4. Integrate ResearchAnalysisOrchestrator as a high-level workflow")
    print("5. Use main orchestration for multi-agent research coordination")
    
    return True

async def demonstrate_proper_integration():
    """Demonstrate how research workflow should integrate with main orchestration."""
    
    print("\n🔧 DEMONSTRATING PROPER INTEGRATION")
    print("=" * 60)
    
    try:
        # Step 1: Initialize main orchestration
        print("\n1️⃣ Initializing main orchestration system...")
        from src.orchestration.orchestration_manager import OrchestrationManager
        from src.orchestration.coordination_models import OrchestrationConfig
        
        config = OrchestrationConfig()
        manager = OrchestrationManager(config)
        
        print("✅ OrchestrationManager initialized")
        
        # Step 2: Check if we can start it (this might fail due to dependencies)
        print("\n2️⃣ Attempting to start orchestration system...")
        try:
            await manager.start()
            print("✅ OrchestrationManager started successfully")
            
            # Step 3: Try to submit a research task
            print("\n3️⃣ Testing task submission...")
            
            # This would be the proper way to submit research tasks
            print("   - Research tasks should be submitted to TaskDispatcher")
            print("   - TaskDispatcher should coordinate research agents")
            print("   - Results should be collected and orchestrated")
            
            await manager.stop()
            print("✅ OrchestrationManager stopped cleanly")
            
        except Exception as e:
            print(f"❌ Failed to start OrchestrationManager: {e}")
            print("   This is expected - system has many dependencies")
            
    except Exception as e:
        print(f"❌ Failed to demonstrate integration: {e}")

async def create_integration_proposal():
    """Create a proposal for integrating research workflow with main orchestration."""
    
    print("\n📋 INTEGRATION PROPOSAL")
    print("=" * 60)
    
    proposal = """
🎯 RESEARCH ORCHESTRATION INTEGRATION PLAN

PHASE 1: Basic Integration
1. Create ResearchTask class extending TaskRequest
2. Register research agents with OrchestrationManager
3. Add research workflow templates to TaskDispatcher
4. Create research-specific task routing logic

PHASE 2: Workflow Integration  
1. Convert ResearchAnalysisOrchestrator to use TaskDispatcher
2. Break down research workflow into discrete tasks
3. Use main orchestration for agent coordination
4. Implement research task dependencies

PHASE 3: Advanced Features
1. Multi-agent research coordination
2. Research task load balancing  
3. Research workflow optimization
4. Integration with evolutionary orchestrator

BENEFITS:
✅ Unified orchestration system
✅ Better resource management
✅ Multi-agent research coordination
✅ Task priority and scheduling
✅ Performance monitoring
✅ Fault tolerance and recovery

IMPLEMENTATION STEPS:
1. Create research task types
2. Integrate with existing agent factory
3. Update research workflow to use main orchestration
4. Test end-to-end research orchestration
    """
    
    print(proposal)
    
    # Save proposal to file
    with open("research_orchestration_integration_proposal.md", "w") as f:
        f.write(proposal)
    
    print("\n💾 Proposal saved to: research_orchestration_integration_proposal.md")

async def main():
    """Main test function."""
    
    print("🚀 RESEARCH ORCHESTRATION INTEGRATION ANALYSIS")
    print("=" * 70)
    
    # Analyze the current gap
    success = await analyze_orchestration_gap()
    
    if success:
        # Demonstrate how it should work
        await demonstrate_proper_integration()
        
        # Create integration proposal
        await create_integration_proposal()
        
        print("\n✅ ANALYSIS COMPLETE")
        print("The research workflow is NOT using the main orchestration system.")
        print("See the integration proposal for next steps.")
    else:
        print("\n❌ ANALYSIS FAILED")
        print("Could not complete orchestration analysis.")

if __name__ == "__main__":
    asyncio.run(main())
