#!/usr/bin/env python3
"""
Test Research Orchestrator Integration
Validates the new research orchestrator and its integration with PyGent Factory
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, Path(__file__).parent / "src")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_research_orchestrator.log')
    ]
)

logger = logging.getLogger(__name__)


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
        logger.info("✓ Core ResearchOrchestrator components imported successfully")
        
        # Test integration module import
        from src.orchestration.research_integration import (
            ResearchOrchestrationManager,
            ResearchTaskType,
            ResearchAgentType,
            initialize_research_system
        )
        logger.info("✓ ResearchIntegration components imported successfully")
        
        # Test orchestration module includes research components
        from src.orchestration import (
            ResearchOrchestrator as OrchestratorFromModule,
            ResearchOrchestrationManager as ManagerFromModule
        )
        logger.info("✓ Research components available from orchestration module")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error during imports: {e}")
        return False


async def test_research_query_creation():
    """Test creation and validation of research queries"""
    
    logger.info("Testing ResearchQuery creation...")
    
    try:
        from src.orchestration.research_orchestrator import ResearchQuery, OutputFormat
        
        # Test basic query creation
        query = ResearchQuery(
            topic="Artificial Intelligence in Healthcare",
            research_questions=[
                "How is AI being used in medical diagnosis?",
                "What are the ethical considerations?",
                "What are the current limitations?"
            ],
            domain="healthcare",
            depth_level="comprehensive",
            output_format=OutputFormat.RESEARCH_SUMMARY,
            quality_threshold=0.85
        )
        
        logger.info(f"✓ ResearchQuery created with ID: {query.query_id}")
        logger.info(f"  Topic: {query.topic}")
        logger.info(f"  Domain: {query.domain}")
        logger.info(f"  Questions: {len(query.research_questions)}")
        
        # Test query validation
        if query.topic and query.domain and query.research_questions:
            logger.info("✓ ResearchQuery validation passed")
        else:
            logger.error("✗ ResearchQuery validation failed - missing required fields")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ResearchQuery creation failed: {e}")
        return False


async def test_research_orchestrator_initialization():
    """Test research orchestrator initialization"""
    
    logger.info("Testing ResearchOrchestrator initialization...")
    
    try:
        from src.orchestration.research_orchestrator import ResearchOrchestrator
        from src.orchestration.coordination_models import OrchestrationConfig
          # Create minimal config for testing
        config = OrchestrationConfig()
        config.max_concurrent_tasks = 5
        config.task_timeout = 3600
          # Initialize orchestrator
        orchestrator = ResearchOrchestrator(config)
        
        logger.info("✓ ResearchOrchestrator initialized successfully")
        logger.info(f"  Config: {config}")
        logger.info(f"  Research sessions: {len(orchestrator.research_sessions)}")
        
        # Test orchestrator methods exist
        methods_to_check = [
            'conduct_research',
            'get_research_status',
            'cancel_research',
            'get_research_metrics'
        ]
        
        for method_name in methods_to_check:
            if hasattr(orchestrator, method_name):
                logger.info(f"✓ Method {method_name} available")
            else:
                logger.error(f"✗ Method {method_name} missing")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ResearchOrchestrator initialization failed: {e}")
        return False


async def test_research_integration_manager():
    """Test research integration manager initialization"""
    
    logger.info("Testing ResearchOrchestrationManager...")
    
    try:
        from src.orchestration.research_integration import ResearchOrchestrationManager
        from src.orchestration.coordination_models import OrchestrationConfig
          # Create minimal config
        config = OrchestrationConfig()
        config.max_concurrent_tasks = 5
        config.task_timeout = 3600
          # Initialize manager
        manager = ResearchOrchestrationManager(config)
        
        logger.info("✓ ResearchOrchestrationManager initialized successfully") 
        
        # Test manager methods exist
        methods_to_check = [
            'submit_research_request',
            'conduct_research',
            'get_research_status',
            'cancel_research',
            'get_research_metrics',
            'create_research_template',
            'execute_research_template'
        ]
        
        for method_name in methods_to_check:
            if hasattr(manager, method_name):
                logger.info(f"✓ Method {method_name} available")
            else:
                logger.error(f"✗ Method {method_name} missing")
                return False
        
        # Test research task types
        from src.orchestration.research_integration import ResearchTaskType, ResearchAgentType
        
        task_types = [
            ResearchTaskType.RESEARCH_QUERY,
            ResearchTaskType.LITERATURE_REVIEW,
            ResearchTaskType.DATA_ANALYSIS,
            ResearchTaskType.REPORT_GENERATION
        ]
        
        agent_types = [
            ResearchAgentType.RESEARCH_PLANNER,
            ResearchAgentType.WEB_RESEARCHER,
            ResearchAgentType.ACADEMIC_ANALYZER,
            ResearchAgentType.CITATION_SPECIALIST,
            ResearchAgentType.OUTPUT_GENERATOR
        ]
        
        logger.info(f"✓ Research task types defined: {len(task_types)}")
        logger.info(f"✓ Research agent types defined: {len(agent_types)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ResearchOrchestrationManager test failed: {e}")
        return False


async def test_output_formats():
    """Test research output format definitions"""
    
    logger.info("Testing research output formats...")
    
    try:
        from src.orchestration.research_orchestrator import OutputFormat, QualityAssessment
        
        # Test all output formats are available
        formats = [
            OutputFormat.RESEARCH_SUMMARY,
            OutputFormat.ACADEMIC_PAPER,
            OutputFormat.LITERATURE_REVIEW,
            OutputFormat.EXECUTIVE_BRIEF,
            OutputFormat.TECHNICAL_REPORT,
            OutputFormat.PRESENTATION_SLIDES,
            OutputFormat.BIBLIOGRAPHY,
            OutputFormat.RAW_DATA
        ]
        
        logger.info(f"✓ Output formats available: {len(formats)}")
        for fmt in formats:
            logger.info(f"  - {fmt.value}")
        
        # Test quality assessment structure
        quality = QualityAssessment(
            credibility_score=0.85,
            completeness_score=0.90,
            relevance_score=0.88,
            citation_quality=0.92,
            overall_score=0.89,
            issues_found=["Minor citation formatting", "One source needs verification"],
            recommendations=["Review citations", "Add one more primary source"]
        )
        
        logger.info("✓ QualityAssessment created successfully")
        logger.info(f"  Overall score: {quality.overall_score}")
        logger.info(f"  Issues: {len(quality.issues_found)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Output formats test failed: {e}")
        return False


async def test_research_phases():
    """Test research phase definitions"""
    
    logger.info("Testing research phases...")
    
    try:
        from src.orchestration.research_orchestrator import ResearchPhase
        
        # Test all phases are available
        phases = [
            ResearchPhase.TOPIC_DISCOVERY,
            ResearchPhase.HYPOTHESIS_GENERATION,
            ResearchPhase.LITERATURE_REVIEW,
            ResearchPhase.DATA_COLLECTION,
            ResearchPhase.ANALYSIS,
            ResearchPhase.SYNTHESIS,
            ResearchPhase.OUTPUT_GENERATION,
            ResearchPhase.VALIDATION
        ]
        
        logger.info(f"✓ Research phases available: {len(phases)}")
        for phase in phases:
            logger.info(f"  - {phase.value}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Research phases test failed: {e}")
        return False


async def test_mock_research_workflow():
    """Test a mock research workflow (without external dependencies)"""
    
    logger.info("Testing mock research workflow...")
    
    try:
        from src.orchestration.research_orchestrator import ResearchQuery, OutputFormat
        from src.orchestration.research_integration import ResearchOrchestrationManager
        from src.orchestration.coordination_models import OrchestrationConfig
          # Create config and manager
        config = OrchestrationConfig()
        config.max_concurrent_tasks = 5
        config.task_timeout = 3600
        
        manager = ResearchOrchestrationManager(config)
        
        # Create a research request
        research_request = {
            "topic": "Machine Learning Ethics",
            "research_questions": [
                "What are the main ethical concerns in ML?",
                "How can bias be mitigated?",
                "What are best practices for ethical AI?"
            ],
            "domain": "technology",
            "depth_level": "comprehensive",
            "output_format": "research_summary",
            "quality_threshold": 0.8,
            "citation_style": "APA"
        }
        
        # Test query creation from request
        query = manager._create_research_query(research_request)
        
        logger.info("✓ Mock research query created from request")
        logger.info(f"  Query ID: {query.query_id}")
        logger.info(f"  Topic: {query.topic}")
        logger.info(f"  Questions: {len(query.research_questions)}")
        logger.info(f"  Domain: {query.domain}")
        logger.info(f"  Output format: {query.output_format.value}")
        
        # Test template creation
        template_config = {
            "name": "Ethics Research Template",
            "description": "Template for researching ethical issues in technology",
            "defaults": research_request,
            "parameters": {
                "domain": ["technology", "healthcare", "finance"],
                "depth_level": ["basic", "comprehensive", "expert"]
            }
        }
        
        template_id = await manager.create_research_template(template_config)
        logger.info(f"✓ Research template created: {template_id}")
        
        # Test metrics collection
        metrics = manager.get_research_metrics()
        logger.info("✓ Research metrics collected")
        logger.info(f"  System health: {metrics.get('system_health', {}).get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Mock research workflow test failed: {e}")
        return False


async def run_all_tests():
    """Run all research orchestrator tests"""
    
    logger.info("=" * 60)
    logger.info("RUNNING RESEARCH ORCHESTRATOR INTEGRATION TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Import Tests", test_research_orchestrator_imports),
        ("Research Query Creation", test_research_query_creation),
        ("Research Orchestrator Init", test_research_orchestrator_initialization),
        ("Integration Manager", test_research_integration_manager),
        ("Output Formats", test_output_formats),
        ("Research Phases", test_research_phases),
        ("Mock Workflow", test_mock_research_workflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{status:>6} | {test_name}")
    
    logger.info(f"\nTOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Research Orchestrator integration is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Research Orchestrator integration needs fixes.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
