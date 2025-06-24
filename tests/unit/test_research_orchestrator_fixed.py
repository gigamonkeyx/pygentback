#!/usr/bin/env python3
"""
Test Research Orchestrator Integration - Fixed Version
Validates the new research orchestrator and its integration with PyGent Factory
"""

import asyncio
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, MagicMock, AsyncMock

# Add src to path for imports
sys.path.insert(0, Path(__file__).parent / "src")

# Configure logging to avoid Unicode issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_research_orchestrator_fixed.log')
    ]
)

logger = logging.getLogger(__name__)


class MockOrchestrationConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.research_enabled = True
        self.max_concurrent_research = 5
        self.default_timeout = 300


class MockAgentRegistry:
    """Mock agent registry for testing"""
    def __init__(self):
        self.agents = {}
    
    async def register_agent(self, agent_type, capabilities, description):
        self.agents[agent_type] = {"capabilities": capabilities, "description": description}
    
    async def get_agent(self, agent_type):
        return self.agents.get(agent_type)


class MockTaskDispatcher:
    """Mock task dispatcher for testing"""
    def __init__(self):
        self.tasks = []
    
    async def submit_task(self, task):
        self.tasks.append(task)
        return f"task_{len(self.tasks)}"
    
    async def submit_research_task(self, query):
        return await self.submit_task(query)


class MockMCPOrchestrator:
    """Mock MCP orchestrator for testing"""
    def __init__(self):
        self.running = False
    
    async def start(self):
        self.running = True
    
    async def stop(self):
        self.running = False


async def test_research_orchestrator_imports():
    """Test that all research orchestrator components can be imported"""
    
    logger.info("Testing Research Orchestrator imports...")
    
    try:
        # Test core research orchestrator import
        from src.orchestration.research_orchestrator import (
            ResearchOrchestrator,
            ResearchQuery,
            ResearchOutput,
            OutputFormat,
            ResearchPhase,
            QualityAssessment
        )
        logger.info("[PASS] Core ResearchOrchestrator components imported successfully")
        
        # Test integration module import
        from src.orchestration.research_integration import (
            ResearchOrchestrationManager,
            ResearchTaskType,
            ResearchAgentType,
            initialize_research_system
        )
        logger.info("[PASS] ResearchIntegration components imported successfully")
        
        # Test orchestration module includes research components
        from src.orchestration import (
            ResearchOrchestrator as OrchestratorFromModule,
            ResearchOrchestrationManager as ManagerFromModule
        )
        logger.info("[PASS] Research components available from orchestration module")
        
        return True
        
    except ImportError as e:
        logger.error(f"[FAIL] Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Unexpected error during imports: {e}")
        return False


async def test_research_query_creation():
    """Test creating research queries"""
    
    logger.info("Testing ResearchQuery creation...")
    
    try:
        from src.orchestration.research_orchestrator import ResearchQuery, OutputFormat
        
        # Test basic query creation
        query = ResearchQuery(
            topic="Artificial Intelligence in Healthcare",
            research_questions=[
                "What are the current applications of AI in healthcare?",
                "What are the main challenges in implementing AI healthcare solutions?",
                "What are the future trends in AI healthcare technology?"
            ],
            domain="healthcare",
            output_format=OutputFormat.RESEARCH_SUMMARY
        )
        
        logger.info(f"[PASS] ResearchQuery created with ID: {query.query_id}")
        logger.info(f"  Topic: {query.topic}")
        logger.info(f"  Domain: {query.domain}")
        logger.info(f"  Questions: {len(query.research_questions)}")
        
        # Validate query properties
        assert query.topic == "Artificial Intelligence in Healthcare"
        assert query.domain == "healthcare"
        assert len(query.research_questions) == 3
        assert query.output_format == OutputFormat.RESEARCH_SUMMARY
        
        logger.info("[PASS] ResearchQuery validation passed")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] ResearchQuery creation failed: {e}")
        return False


async def test_research_orchestrator_initialization():
    """Test ResearchOrchestrator can be initialized with mock dependencies"""
    
    logger.info("Testing ResearchOrchestrator initialization...")
    
    try:
        from src.orchestration.research_orchestrator import ResearchOrchestrator
        
        # Create mock dependencies
        config = MockOrchestrationConfig()
        agent_registry = MockAgentRegistry()
        task_dispatcher = MockTaskDispatcher()
        mcp_orchestrator = MockMCPOrchestrator()
        
        # Initialize ResearchOrchestrator with all required arguments
        orchestrator = ResearchOrchestrator(
            config=config,
            agent_registry=agent_registry,
            task_dispatcher=task_dispatcher,
            mcp_orchestrator=mcp_orchestrator
        )
        
        # Validate initialization
        assert orchestrator.config == config
        assert orchestrator.agent_registry == agent_registry
        assert orchestrator.task_dispatcher == task_dispatcher
        assert orchestrator.mcp_orchestrator == mcp_orchestrator
        
        # Check internal components are initialized
        assert hasattr(orchestrator, 'planning_engine')
        assert hasattr(orchestrator, 'knowledge_engine')
        assert hasattr(orchestrator, 'analysis_engine')
        assert hasattr(orchestrator, 'output_engine')
        assert hasattr(orchestrator, 'active_research_sessions')
        assert hasattr(orchestrator, 'research_history')
        
        logger.info("[PASS] ResearchOrchestrator initialized successfully")
        logger.info(f"  Config: {type(orchestrator.config).__name__}")
        logger.info(f"  Active sessions: {len(orchestrator.active_research_sessions)}")
        logger.info(f"  Research history: {len(orchestrator.research_history)}")
        
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] ResearchOrchestrator initialization failed: {e}")
        return False


async def test_research_integration_manager():
    """Test ResearchOrchestrationManager initialization and basic functionality"""
    
    logger.info("Testing ResearchOrchestrationManager...")
    
    try:
        from src.orchestration.research_integration import ResearchOrchestrationManager
        
        # Create mock dependencies
        config = MockOrchestrationConfig()
        agent_registry = MockAgentRegistry()
        task_dispatcher = MockTaskDispatcher()
        mcp_orchestrator = MockMCPOrchestrator()
        
        # Initialize ResearchOrchestrationManager with all required arguments
        manager = ResearchOrchestrationManager(
            config=config,
            agent_registry=agent_registry,
            task_dispatcher=task_dispatcher,
            mcp_orchestrator=mcp_orchestrator
        )
        
        # Validate initialization
        assert manager.config == config
        assert manager.agent_registry == agent_registry
        assert manager.task_dispatcher == task_dispatcher
        assert manager.mcp_orchestrator == mcp_orchestrator
        assert hasattr(manager, 'research_orchestrator')
          # Test basic functionality
        assert hasattr(manager, 'submit_research_request')
        assert hasattr(manager, 'get_research_status')
        assert hasattr(manager, 'cancel_research')
        
        logger.info("[PASS] ResearchOrchestrationManager initialized successfully")
        logger.info(f"  Config: {type(manager.config).__name__}")
        logger.info(f"  Internal orchestrator: {type(manager.research_orchestrator).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] ResearchOrchestrationManager test failed: {e}")
        return False


async def test_output_formats():
    """Test all expected output formats are available"""
    
    logger.info("Testing research output formats...")
    
    try:
        from src.orchestration.research_orchestrator import OutputFormat
        
        # Test all expected formats
        expected_formats = [
            OutputFormat.ACADEMIC_PAPER,
            OutputFormat.LITERATURE_REVIEW,
            OutputFormat.RESEARCH_SUMMARY,
            OutputFormat.AI_OPTIMIZED,
            OutputFormat.EXECUTIVE_BRIEF,
            OutputFormat.CITATION_REPORT,
            OutputFormat.TECHNICAL_REPORT
        ]
        
        logger.info(f"[PASS] Output formats available: {len(expected_formats)}")
        for fmt in expected_formats:
            logger.info(f"  - {fmt.name}: {fmt.value}")
        
        # Test format usage
        test_format = OutputFormat.TECHNICAL_REPORT
        assert test_format.value == "technical_report"
        
        logger.info("[PASS] Output format validation passed")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Output formats test failed: {e}")
        return False


async def test_research_phases():
    """Test research phases enum"""
    
    logger.info("Testing research phases...")
    
    try:
        from src.orchestration.research_orchestrator import ResearchPhase
        
        # Get all phases
        phases = list(ResearchPhase)
        
        logger.info(f"[PASS] Research phases available: {len(phases)}")
        for phase in phases:
            logger.info(f"  - {phase.value}")
        
        # Validate expected phases
        expected_phases = [
            "topic_discovery",
            "hypothesis_generation", 
            "literature_review",
            "data_collection",
            "analysis",
            "synthesis",
            "output_generation",
            "validation"
        ]
        
        phase_values = [phase.value for phase in phases]
        for expected in expected_phases:
            assert expected in phase_values, f"Missing phase: {expected}"
        
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Research phases test failed: {e}")
        return False


async def test_mock_research_workflow():
    """Test a complete mock research workflow"""
    
    logger.info("Testing mock research workflow...")
    
    try:
        from src.orchestration.research_orchestrator import ResearchQuery, OutputFormat
        from src.orchestration.research_integration import ResearchOrchestrationManager
        
        # Create mock dependencies
        config = MockOrchestrationConfig()
        agent_registry = MockAgentRegistry()
        task_dispatcher = MockTaskDispatcher()
        mcp_orchestrator = MockMCPOrchestrator()
        
        # Initialize manager
        manager = ResearchOrchestrationManager(
            config=config,
            agent_registry=agent_registry,
            task_dispatcher=task_dispatcher,
            mcp_orchestrator=mcp_orchestrator
        )
        
        # Create a research query
        query = ResearchQuery(
            topic="Machine Learning in Autonomous Vehicles",
            research_questions=[
                "What ML algorithms are used in self-driving cars?",
                "What are the safety considerations for ML in autonomous vehicles?"
            ],
            domain="automotive_ai",
            output_format=OutputFormat.TECHNICAL_REPORT
        )
          # Submit query (this will be mocked)
        query_id = await manager.submit_research_request(query.__dict__)
        
        logger.info(f"[PASS] Mock research query submitted: {query_id}")
        logger.info(f"  Topic: {query.topic}")
        logger.info(f"  Domain: {query.domain}")
        logger.info(f"  Output format: {query.output_format.value}")
        
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Mock research workflow test failed: {e}")
        return False


async def run_all_tests():
    """Run all research orchestrator tests"""
    
    logger.info("=" * 60)
    logger.info("RUNNING RESEARCH ORCHESTRATOR INTEGRATION TESTS")
    logger.info("=" * 60)
    
    test_functions = [
        ("Import Tests", test_research_orchestrator_imports),
        ("Research Query Creation", test_research_query_creation),
        ("Research Orchestrator Init", test_research_orchestrator_initialization),
        ("Integration Manager", test_research_integration_manager),
        ("Output Formats", test_output_formats),
        ("Research Phases", test_research_phases),
        ("Mock Workflow", test_mock_research_workflow)
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_name, test_func in test_functions:
        logger.info("")
        logger.info("=" * 20 + f" {test_name} " + "=" * 20)
        
        try:
            success = await test_func()
            if success:
                logger.info(f"[PASS] {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"[FAIL] {test_name} FAILED")
        except Exception as e:
            logger.error(f"[FAIL] {test_name} FAILED with exception: {e}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for i, (test_name, _) in enumerate(test_functions):
        status = "PASS" if i < passed else "FAIL"
        logger.info(f"  {status} | {test_name}")
    
    logger.info(f"\nTOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("[SUCCESS] All tests passed! Research Orchestrator integration is working.")
        return True
    else:
        logger.error(f"[ERROR] {total - passed} tests failed. Research Orchestrator integration needs fixes.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
