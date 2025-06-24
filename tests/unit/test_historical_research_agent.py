"""
Test the Historical Research Agent integration with PyGent Factory Research Orchestrator

This test validates:
1. Historical research agent initialization 
2. Historical query detection and routing
3. Historical event analysis and timeline construction
4. Source validation and bias detection
5. Integration with the research orchestrator
"""

import asyncio
import logging
from datetime import datetime
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestration.research_orchestrator import ResearchOrchestrator
from src.orchestration.research_models import ResearchQuery, ResearchSource, SourceType
from src.orchestration.historical_research_agent import (
    HistoricalResearchAgent, HistoricalEvent, HistoricalTimeline,
    HistoricalPeriod, HistoricalEventType, create_historical_research_capability
)
from src.orchestration.coordination_models import OrchestrationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestHistoricalResearchAgent:
    """Test suite for Historical Research Agent"""
    
    def __init__(self):
        self.config = OrchestrationConfig(
            max_concurrent_tasks=5,
            task_timeout=180.0,
            agent_timeout=300.0,
            agent_health_check_interval=30.0,
            server_health_check_interval=60.0        )
        self.orchestrator = None
        self.historical_agent = None
        
    async def setup(self):
        """Set up test environment"""
        try:
            logger.info("Setting up Historical Research Agent test environment")
            
            # Create mock dependencies for ResearchOrchestrator
            from src.orchestration.agent_registry import AgentRegistry
            from src.orchestration.task_dispatcher import TaskDispatcher
            from src.orchestration.mcp_orchestrator import MCPOrchestrator
              # Initialize mock dependencies
            mock_agent_registry = AgentRegistry(self.config)
            mock_mcp_orchestrator = MCPOrchestrator(self.config)
            mock_task_dispatcher = TaskDispatcher(self.config, mock_agent_registry, mock_mcp_orchestrator)
            
            # Initialize orchestrator with dependencies
            self.orchestrator = ResearchOrchestrator(
                self.config, 
                mock_agent_registry, 
                mock_task_dispatcher, 
                mock_mcp_orchestrator
            )
            await self.orchestrator.initialize_research_orchestrator()
            
            # Initialize historical agent
            self.historical_agent = HistoricalResearchAgent(self.config)
            
            logger.info("Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Test setup failed: {e}")
            return False
    
    async def test_historical_query_detection(self):
        """Test detection of historical research queries"""
        try:
            logger.info("Testing historical query detection...")
            
            # Test historical queries
            historical_queries = [
                ResearchQuery(
                    topic="The Fall of the Roman Empire",
                    description="Analyze the factors that led to the collapse of the Western Roman Empire",
                    domain="history"
                ),
                ResearchQuery(
                    topic="World War II Timeline",
                    description="Create a comprehensive timeline of major World War II events",
                    domain="military history"
                ),
                ResearchQuery(
                    topic="Ancient Egyptian Civilization",
                    description="Research the political and social structures of ancient Egypt",
                    domain="archaeology"
                )
            ]
            
            # Test non-historical queries
            non_historical_queries = [
                ResearchQuery(
                    topic="Machine Learning Algorithms",
                    description="Compare different machine learning approaches",
                    domain="computer science"
                ),
                ResearchQuery(
                    topic="Climate Change Impact",
                    description="Analyze current climate change effects",
                    domain="environmental science"
                )
            ]
            
            # Test detection
            historical_detected = 0
            for query in historical_queries:
                if await self.orchestrator._is_historical_research_query(query):
                    historical_detected += 1
                    logger.info(f"✓ Correctly identified historical query: {query.topic}")
                else:
                    logger.warning(f"✗ Missed historical query: {query.topic}")
            
            non_historical_detected = 0
            for query in non_historical_queries:
                if not await self.orchestrator._is_historical_research_query(query):
                    non_historical_detected += 1
                    logger.info(f"✓ Correctly identified non-historical query: {query.topic}")
                else:
                    logger.warning(f"✗ Incorrectly classified as historical: {query.topic}")
            
            success_rate = (historical_detected + non_historical_detected) / (len(historical_queries) + len(non_historical_queries))
            logger.info(f"Historical query detection accuracy: {success_rate:.2%}")
            
            return success_rate >= 0.8
            
        except Exception as e:
            logger.error(f"Historical query detection test failed: {e}")
            return False
    
    async def test_historical_event_analysis(self):
        """Test historical event analysis and timeline construction"""
        try:
            logger.info("Testing historical event analysis...")
            
            # Create a test query
            query = ResearchQuery(
                topic="Napoleon's Military Campaigns",
                description="Analyze Napoleon Bonaparte's major military campaigns and their outcomes",
                domain="military history"
            )
            
            # Test historical research
            analysis = await self.historical_agent.conduct_historical_research(query)
            
            # Validate analysis results
            validation_results = {
                "events_extracted": len(analysis.events) > 0,
                "timeline_created": analysis.timeline is not None,
                "themes_identified": len(analysis.key_themes) > 0,
                "causal_analysis": len(analysis.causal_relationships) > 0,
                "context_provided": len(analysis.historical_context) > 0,
                "confidence_calculated": len(analysis.confidence_metrics) > 0
            }
            
            for test, result in validation_results.items():
                if result:
                    logger.info(f"✓ {test}: PASSED")
                else:
                    logger.warning(f"✗ {test}: FAILED")
            
            success_rate = sum(validation_results.values()) / len(validation_results)
            logger.info(f"Historical event analysis success rate: {success_rate:.2%}")
            
            return success_rate >= 0.7
            
        except Exception as e:
            logger.error(f"Historical event analysis test failed: {e}")
            return False
    
    async def test_source_validation(self):
        """Test historical source validation and bias detection"""
        try:
            logger.info("Testing source validation...")
              # Create mock sources for testing
            from src.orchestration.research_models import ResearchSource, SourceType
            
            test_sources = [
                ResearchSource(
                    title="Primary Account of the Battle of Waterloo",
                    authors=["Duke of Wellington"],
                    source_type=SourceType.WEB_SOURCE,
                    abstract="First-hand account of the famous battle",
                    content="Detailed description of the battle from Wellington's perspective...",
                    publication_date=datetime(1815, 6, 20),
                    publisher="Military Archives",
                    peer_reviewed=False,
                    credibility_score=0.0
                ),
                ResearchSource(
                    title="Scholarly Analysis of Napoleonic Wars",
                    authors=["Dr. Historical Scholar"],
                    source_type=SourceType.ACADEMIC_PAPER,
                    abstract="Academic analysis of Napoleon's military strategies",
                    content="Comprehensive scholarly examination of Napoleonic military tactics...",
                    publication_date=datetime(2020, 1, 1),
                    publisher="Historical Journal",
                    peer_reviewed=True,
                    credibility_score=0.0
                )
            ]
            
            # Test source validation
            validated_sources = []
            for source in test_sources:
                validation = await self.historical_agent.source_validator.validate_historical_source(source)
                source.credibility_score = validation.get("overall_credibility", 0.0)
                validated_sources.append(source)
                
                logger.info(f"Source '{source.title}' credibility: {source.credibility_score:.2f}")
            
            # Check validation results
            avg_credibility = sum(s.credibility_score for s in validated_sources) / len(validated_sources)
            logger.info(f"Average source credibility: {avg_credibility:.2f}")
            
            return avg_credibility > 0.5
            
        except Exception as e:
            logger.error(f"Source validation test failed: {e}")
            return False
    
    async def test_timeline_construction(self):
        """Test historical timeline construction"""
        try:
            logger.info("Testing timeline construction...")
            
            # Create test events
            test_events = [
                HistoricalEvent(
                    name="Battle of Austerlitz",
                    description="Napoleon's decisive victory over Austrian and Russian forces",
                    date_start=datetime(1805, 12, 2),
                    location="Austerlitz, Austria",
                    event_type=HistoricalEventType.MILITARY,
                    period=HistoricalPeriod.EARLY_MODERN,
                    key_figures=["Napoleon Bonaparte", "Emperor Francis II", "Tsar Alexander I"],
                    confidence_level=0.95
                ),
                HistoricalEvent(
                    name="Battle of Waterloo",
                    description="Final defeat of Napoleon Bonaparte",
                    date_start=datetime(1815, 6, 18),
                    location="Waterloo, Belgium",
                    event_type=HistoricalEventType.MILITARY,
                    period=HistoricalPeriod.EARLY_MODERN,
                    key_figures=["Napoleon Bonaparte", "Duke of Wellington"],
                    confidence_level=0.98
                )
            ]
            
            # Test timeline creation
            timeline = await self.historical_agent.timeline_analyzer.create_historical_timeline(test_events)
            
            # Validate timeline
            validation_results = {
                "timeline_created": timeline is not None,
                "events_included": len(timeline.events) == len(test_events),
                "title_generated": len(timeline.title) > 0,
                "description_generated": len(timeline.description) > 0,
                "themes_identified": len(timeline.themes) > 0,
                "chronological_order": self._check_chronological_order(timeline.events)
            }
            
            for test, result in validation_results.items():
                if result:
                    logger.info(f"✓ {test}: PASSED")
                else:
                    logger.warning(f"✗ {test}: FAILED")
            
            logger.info(f"Timeline title: {timeline.title}")
            logger.info(f"Timeline themes: {', '.join(timeline.themes)}")
            
            success_rate = sum(validation_results.values()) / len(validation_results)
            return success_rate >= 0.8
            
        except Exception as e:
            logger.error(f"Timeline construction test failed: {e}")
            return False
    
    def _check_chronological_order(self, events):
        """Check if events are in chronological order"""
        try:
            dated_events = [e for e in events if e.date_start]
            if len(dated_events) < 2:
                return True
            
            for i in range(len(dated_events) - 1):
                if dated_events[i].date_start > dated_events[i + 1].date_start:
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def test_orchestrator_integration(self):
        """Test integration with the research orchestrator"""
        try:
            logger.info("Testing orchestrator integration...")
            
            # Create historical research query
            query = ResearchQuery(
                topic="The Renaissance Period",
                description="Analyze the cultural and artistic developments during the Renaissance",
                domain="cultural history"
            )
            
            # Test orchestrator routing
            output = await self.orchestrator.conduct_research(query)
            
            # Validate integration
            validation_results = {
                "output_generated": output is not None,
                "findings_present": len(output.findings) > 0,
                "metadata_included": len(output.metadata) > 0,
                "historical_type": output.metadata.get("research_type") == "historical_analysis"
            }
            
            for test, result in validation_results.items():
                if result:
                    logger.info(f"✓ {test}: PASSED")
                else:
                    logger.warning(f"✗ {test}: FAILED")
            
            logger.info(f"Research output generated with {len(output.findings)} findings")
            
            success_rate = sum(validation_results.values()) / len(validation_results)
            return success_rate >= 0.7
            
        except Exception as e:
            logger.error(f"Orchestrator integration test failed: {e}")
            return False
    
    async def test_agent_capability(self):
        """Test agent capability definition"""
        try:
            logger.info("Testing agent capability definition...")
            
            capability = create_historical_research_capability()
            
            validation_results = {
                "capability_created": capability is not None,
                "has_agent_id": len(capability.agent_id) > 0,
                "has_capabilities": len(capability.capabilities) > 0,
                "has_specializations": len(capability.specializations) > 0,
                "has_performance_metrics": len(capability.performance_metrics) > 0,
                "has_resource_requirements": len(capability.resource_requirements) > 0
            }
            
            for test, result in validation_results.items():
                if result:
                    logger.info(f"✓ {test}: PASSED")
                else:
                    logger.warning(f"✗ {test}: FAILED")
            
            logger.info(f"Agent capabilities: {', '.join(capability.capabilities)}")
            logger.info(f"Agent specializations: {', '.join(capability.specializations)}")
            
            success_rate = sum(validation_results.values()) / len(validation_results)
            return success_rate >= 0.8
            
        except Exception as e:
            logger.error(f"Agent capability test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all historical research agent tests"""
        try:
            logger.info("=" * 50)
            logger.info("STARTING HISTORICAL RESEARCH AGENT TESTS")
            logger.info("=" * 50)
            
            # Setup test environment
            if not await self.setup():
                logger.error("Test setup failed - aborting tests")
                return False
            
            # Run individual tests
            tests = [
                ("Historical Query Detection", self.test_historical_query_detection),
                ("Historical Event Analysis", self.test_historical_event_analysis),
                ("Source Validation", self.test_source_validation),
                ("Timeline Construction", self.test_timeline_construction),
                ("Orchestrator Integration", self.test_orchestrator_integration),
                ("Agent Capability", self.test_agent_capability)
            ]
            
            test_results = {}
            for test_name, test_func in tests:
                logger.info(f"\n--- Running {test_name} Test ---")
                try:
                    result = await test_func()
                    test_results[test_name] = result
                    status = "PASSED" if result else "FAILED"
                    logger.info(f"{test_name}: {status}")
                except Exception as e:
                    logger.error(f"{test_name} failed with error: {e}")
                    test_results[test_name] = False
            
            # Summary
            logger.info("\n" + "=" * 50)
            logger.info("HISTORICAL RESEARCH AGENT TEST SUMMARY")
            logger.info("=" * 50)
            
            passed_tests = sum(test_results.values())
            total_tests = len(test_results)
            
            for test_name, result in test_results.items():
                status = "✓ PASSED" if result else "✗ FAILED"
                logger.info(f"{test_name}: {status}")
            
            success_rate = passed_tests / total_tests
            logger.info(f"\nOverall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
            
            if success_rate >= 0.7:
                logger.info("🎉 Historical Research Agent tests PASSED!")
                return True
            else:
                logger.warning("⚠️ Historical Research Agent tests FAILED!")
                return False
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up test resources"""
        try:
            if self.orchestrator:
                await self.orchestrator.stop()
            logger.info("Test cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


async def main():
    """Main test execution function"""
    test_suite = TestHistoricalResearchAgent()
    
    try:
        success = await test_suite.run_all_tests()
        await test_suite.cleanup()
        
        if success:
            print("\n🎉 All Historical Research Agent tests completed successfully!")
            return 0
        else:
            print("\n❌ Some Historical Research Agent tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        await test_suite.cleanup()
        return 1


if __name__ == "__main__":
    import sys
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
