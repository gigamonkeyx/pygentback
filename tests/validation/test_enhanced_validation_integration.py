#!/usr/bin/env python3
"""
Test Enhanced Research Validation Integration
Tests the integration of HathiTrust and cross-platform validation
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestration.research_models import ResearchQuery, ResearchSource, SourceType
from orchestration.coordination_models import OrchestrationConfig, AgentCapability, AgentType
from orchestration.historical_research_agent import HistoricalResearchAgent
from orchestration.hathitrust_integration import HathiTrustBibliographicAPI
from orchestration.cross_platform_validator import CrossPlatformValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_hathitrust_integration():
    """Test HathiTrust integration"""
    logger.info("=== Testing HathiTrust Integration ===")
    
    hathitrust = HathiTrustBibliographicAPI()
    
    # Test enhanced historical search
    from orchestration.research_models import ResearchQuery
    query = ResearchQuery(
        topic="French Revolution",
        domain="historical",
        depth_level="comprehensive"
    )
    
    results = await hathitrust.enhanced_historical_search(query)
    logger.info(f"Enhanced historical search results: {len(results)} sources")
    
    # Test source verification
    if results:
        verification = await hathitrust.verify_source(results[0])
        logger.info(f"Source verification result: {verification}")
    
    return True


async def test_cross_platform_validation():
    """Test cross-platform validation"""
    logger.info("=== Testing Cross-Platform Validation ===")
    
    validator = CrossPlatformValidator()
    
    # Create a test source
    test_source = ResearchSource(
        title="The Declaration of Independence",
        authors=["Thomas Jefferson"],
        source_type=SourceType.GOVERNMENT_DOCUMENT,
        abstract="The founding document of the United States",
        url="https://www.archives.gov/founding-docs/declaration-transcript",
        publication_date=datetime(1776, 7, 4),
        publisher="Continental Congress"
    )
    
    # Test single source validation
    validation_result = await validator.validate_single_source(test_source)
    logger.info(f"Single source validation result: {validation_result}")
    
    # Test multiple source validation
    validation_results = await validator.validate_sources([test_source])
    logger.info(f"Multiple source validation results: {len(validation_results)} results")
    
    # Test validation summary
    summary = await validator.get_validation_summary(validation_results)
    logger.info(f"Validation summary: {summary}")
    
    return True


async def test_historical_research_agent_integration():
    """Test historical research agent with enhanced validation"""
    logger.info("=== Testing Historical Research Agent Integration ===")    # Create configuration
    config = OrchestrationConfig(
        max_concurrent_tasks=3,
        task_timeout=300
    )
    
    # Create historical research agent
    agent = HistoricalResearchAgent(config)    # Create test query
    query = ResearchQuery(
        topic="American Revolution causes and consequences",
        domain="historical",
        depth_level="comprehensive",
        citation_style="Chicago",
        metadata={
            "time_period": "1760-1783",
            "geographic_scope": ["North America", "Britain"],
            "author": "George Washington"
        }
    )
    
    # Test enhanced source gathering
    logger.info("Testing enhanced source gathering...")
    sources = await agent._gather_historical_sources(query)
    logger.info(f"Gathered {len(sources)} sources from all databases")
      # Test enhanced source validation
    logger.info("Testing enhanced source validation...")
    validated_sources = await agent._validate_sources(sources)
    logger.info(f"Validated {len(validated_sources)}/{len(sources)} sources")
    
    # Check for enhanced metadata
    for source in validated_sources[:3]:  # Check first 3 sources
        if hasattr(source, 'metadata') and 'enhanced_validation' in source.metadata:
            logger.info(f"Source '{source.title}' has enhanced validation metadata")
            logger.info(f"  - Cross-platform validation: {source.metadata.get('cross_platform_validation', {}).get('found_in_platforms', 0)} platforms")
            logger.info(f"  - HathiTrust results: {source.metadata.get('hathitrust_results', {}).get('total_results', 0)} items")
            logger.info(f"  - Credibility score: {source.credibility_score:.2f}")
        else:
            logger.info(f"Source '{source.title}' validated with credibility score: {source.credibility_score:.2f}")
    
    return True


async def test_research_orchestrator_integration():
    """Test research orchestrator with enhanced validation"""
    logger.info("=== Testing Research Orchestrator Integration ===")
    
    try:
        # Test that our validation components can be initialized
        config = OrchestrationConfig(
            max_concurrent_tasks=3,
            task_timeout=300
        )
        
        # Test HathiTrust integration initialization
        hathitrust = HathiTrustBibliographicAPI(config)
        assert hasattr(hathitrust, 'enhanced_historical_search'), "HathiTrust integration missing methods"
        
        # Test cross-platform validator initialization
        validator = CrossPlatformValidator(config)
        assert hasattr(validator, 'validate_single_source'), "Cross-platform validator missing methods"
        
        # Test historical research agent initialization with enhanced validation
        agent = HistoricalResearchAgent(config)
        assert hasattr(agent, 'hathitrust_integration'), "Historical agent missing HathiTrust integration"
        assert hasattr(agent, 'cross_platform_validator'), "Historical agent missing cross-platform validator"
        
        logger.info("Research orchestrator components successfully initialized with enhanced validation")
        return True
        
    except Exception as e:
        logger.error(f"Research orchestrator integration test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    logger.info("Starting Enhanced Research Validation Integration Tests")
    
    tests = [
        ("HathiTrust Integration", test_hathitrust_integration),
        ("Cross-Platform Validation", test_cross_platform_validation),
        ("Historical Research Agent Integration", test_historical_research_agent_integration),
        ("Research Orchestrator Integration", test_research_orchestrator_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")
            
            result = await test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"PASS: {test_name}")
            else:
                logger.error(f"FAIL: {test_name}")
                
        except Exception as e:
            logger.error(f"FAIL: {test_name} with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All enhanced validation integration tests passed!")
        return True
    else:
        logger.error(f"{total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
