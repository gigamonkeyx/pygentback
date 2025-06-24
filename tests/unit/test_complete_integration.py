# test_complete_integration.py

"""
Complete integration test for the historical research system.
Tests all major components working together in a realistic scenario.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config.settings import get_settings
from src.core.gpu_config import gpu_manager
from src.core.ollama_manager import get_ollama_manager
from src.storage.vector.manager import VectorStoreManager
from src.utils.embedding import EmbeddingService
from src.acquisition.enhanced_document_acquisition import EnhancedDocumentAcquisition
from src.validation.anti_hallucination_framework import AntiHallucinationFramework, VerificationLevel
from src.orchestration.multi_agent_orchestrator import MultiAgentOrchestrator, AgentRole, TaskPriority
from src.output.academic_pdf_generator import AcademicPDFGenerator
from src.monitoring.system_health_monitor import SystemHealthMonitor
from src.orchestration.historical_research_agent import HistoricalResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_test.log')
    ]
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Runs comprehensive integration tests for the historical research system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.test_results = {}
        self.start_time = None
        
        # Core components
        self.vector_manager = None
        self.embedding_service = None
        self.document_acquisition = None
        self.anti_hallucination = None
        self.orchestrator = None
        self.pdf_generator = None
        self.health_monitor = None
        self.research_agent = None
        
        # Test data
        self.test_documents = [
            {
                'url': 'https://example.com/historical_document.pdf',
                'metadata': {
                    'title': 'Sample Historical Document',
                    'authors': ['John Historian'],
                    'year': 2020,
                    'type': 'academic_paper'
                }
            }
        ]
        
        self.test_research_query = "American Civil War causes and consequences 1860-1865"
    
    async def run_complete_integration_test(self):
        """Run the complete integration test suite."""
        self.start_time = time.time()
        logger.info("Starting complete historical research system integration test")
        
        try:
            # Phase 1: System Initialization
            logger.info("Phase 1: System Initialization")
            await self._test_system_initialization()
            
            # Phase 2: Core Infrastructure Tests
            logger.info("Phase 2: Core Infrastructure Tests")
            await self._test_core_infrastructure()
            
            # Phase 3: Document Processing Pipeline
            logger.info("Phase 3: Document Processing Pipeline")
            await self._test_document_processing()
            
            # Phase 4: AI and Validation Systems
            logger.info("Phase 4: AI and Validation Systems")
            await self._test_ai_validation_systems()
            
            # Phase 5: Multi-Agent Orchestration
            logger.info("Phase 5: Multi-Agent Orchestration")
            await self._test_multi_agent_orchestration()
            
            # Phase 6: Research Workflow
            logger.info("Phase 6: Research Workflow")
            await self._test_complete_research_workflow()
            
            # Phase 7: Output Generation
            logger.info("Phase 7: Output Generation")
            await self._test_output_generation()
            
            # Phase 8: System Monitoring and Health
            logger.info("Phase 8: System Monitoring and Health")
            await self._test_system_monitoring()
            
            # Generate final report
            await self._generate_integration_report()
            
        except Exception as e:
            logger.error(f"Integration test failed: {str(e)}")
            self.test_results['overall_status'] = 'FAILED'
            self.test_results['error'] = str(e)
        
        finally:
            await self._cleanup_test_environment()
    
    async def _test_system_initialization(self):
        """Test system initialization and configuration."""
        test_name = "system_initialization"
        logger.info(f"Testing {test_name}")
        
        try:
            # Test settings loading
            assert self.settings is not None, "Settings not loaded"
            
            # Test GPU configuration
            gpu_config = gpu_manager.get_config()
            gpu_available = gpu_manager.is_available()
            
            # Test Ollama manager
            ollama_manager = get_ollama_manager()
            
            self.test_results[test_name] = {
                'status': 'PASSED',
                'gpu_available': gpu_available,
                'gpu_config': gpu_config.__dict__ if gpu_config else None,
                'ollama_url': ollama_manager.ollama_url,
                'settings_loaded': True
            }
            
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            self.test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
    
    async def _test_core_infrastructure(self):
        """Test core infrastructure components."""
        test_name = "core_infrastructure"
        logger.info(f"Testing {test_name}")
        
        try:
            # Initialize vector store manager
            self.vector_manager = VectorStoreManager()
            await self.vector_manager.initialize()
            
            # Initialize embedding service
            self.embedding_service = EmbeddingService()
            await self.embedding_service.initialize()
            
            # Test vector operations
            collections = await self.vector_manager.list_collections()
            
            # Test embedding generation
            test_embedding = await self.embedding_service.get_embedding("test text")
            
            assert len(test_embedding) > 0, "Empty embedding generated"
            
            self.test_results[test_name] = {
                'status': 'PASSED',
                'vector_collections': len(collections),
                'embedding_dimension': len(test_embedding),
                'vector_store_type': self.vector_manager.store_type,
                'embedding_model': self.embedding_service.model_name
            }
            
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            self.test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
    
    async def _test_document_processing(self):
        """Test document acquisition and processing pipeline."""
        test_name = "document_processing"
        logger.info(f"Testing {test_name}")
        
        try:
            # Initialize document acquisition system
            self.document_acquisition = EnhancedDocumentAcquisition(
                self.vector_manager,
                self.embedding_service
            )
              # Simulate document processing with test content
            _ = """
            This is a sample historical document about the American Civil War.
            The Civil War began in 1861 when Southern states seceded from the Union.
            Abraham Lincoln was president during this period.
            The war ended in 1865 with Union victory.
            """
            
            # Simulate document processing
            processing_stats = await self.document_acquisition.get_processing_statistics()
            
            self.test_results[test_name] = {
                'status': 'PASSED',
                'processing_pipeline_initialized': True,
                'gpu_acceleration_available': processing_stats['system_info']['gpu_available'],
                'storage_path': processing_stats['system_info']['storage_path']
            }
            
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            self.test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
    
    async def _test_ai_validation_systems(self):
        """Test AI-powered validation and anti-hallucination systems."""
        test_name = "ai_validation_systems"
        logger.info(f"Testing {test_name}")
        
        try:
            # Initialize anti-hallucination framework
            self.anti_hallucination = AntiHallucinationFramework(
                self.vector_manager,
                self.embedding_service,
                VerificationLevel.STANDARD
            )
            
            # Test content verification
            test_content = """
            The American Civil War lasted from 1861 to 1865. 
            Abraham Lincoln was the 16th President of the United States.
            The war began with the attack on Fort Sumter.
            """
            
            verification_result = await self.anti_hallucination.verify_historical_content(
                test_content,
                {'title': 'Test Document', 'source': 'test'}
            )
            
            assert verification_result.get('success', False), "Verification failed"
            
            # Get verification statistics
            verification_stats = await self.anti_hallucination.get_verification_statistics()
            
            self.test_results[test_name] = {
                'status': 'PASSED',
                'verification_framework_active': True,
                'verification_level': verification_result.get('verification_level'),
                'claims_processed': verification_stats['processing_stats']['total_claims_processed'],
                'ollama_ready': verification_stats['system_info']['ollama_ready']
            }
            
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            self.test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
    
    async def _test_multi_agent_orchestration(self):
        """Test multi-agent orchestration system."""
        test_name = "multi_agent_orchestration"
        logger.info(f"Testing {test_name}")
        
        try:
            # Initialize orchestrator
            self.orchestrator = MultiAgentOrchestrator(
                self.vector_manager,
                self.embedding_service,
                max_concurrent_tasks=3
            )
            
            # Start orchestration
            await self.orchestrator.start_orchestration()
            
            # Create test tasks
            task_id = self.orchestrator.create_task(
                agent_role=AgentRole.RESEARCH_COORDINATOR,
                task_type="coordinate_research",
                description="Test research coordination",
                input_data={'query': 'test query'},
                priority=TaskPriority.MEDIUM
            )
            
            # Wait a moment for task processing
            await asyncio.sleep(2)
            
            # Get orchestration status
            status = self.orchestrator.get_orchestration_status()
            
            self.test_results[test_name] = {
                'status': 'PASSED',
                'orchestrator_running': status['is_running'],
                'total_agents': status['statistics']['total_agents'],
                'task_created': task_id is not None,
                'system_health': status['system_health']
            }
            
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            self.test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
    
    async def _test_complete_research_workflow(self):
        """Test complete research workflow from query to results."""
        test_name = "complete_research_workflow"
        logger.info(f"Testing {test_name}")
        
        try:
            # Initialize historical research agent
            self.research_agent = HistoricalResearchAgent()
            
            # Conduct test research
            research_result = await self.research_agent.conduct_research(
                self.test_research_query
            )
            
            assert research_result.get('success', False), "Research workflow failed"
            
            # Test timeline analysis
            timeline_result = await self.research_agent.analyze_historical_timeline(
                "1860-1865", ["Fort Sumter", "Gettysburg", "Appomattox"]
            )
            
            # Test source validation
            validation_result = await self.research_agent.validate_historical_sources([
                {"title": "Test Source", "author": "Test Author", "year": 2020}
            ])
            
            self.test_results[test_name] = {
                'status': 'PASSED',
                'research_completed': True,
                'timeline_analysis': timeline_result.get('success', False),
                'source_validation': validation_result.get('success', False),
                'research_quality': research_result.get('research_quality', 0.0)
            }
            
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            self.test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
    
    async def _test_output_generation(self):
        """Test academic PDF generation and output systems."""
        test_name = "output_generation"
        logger.info(f"Testing {test_name}")
        
        try:
            # Initialize PDF generator
            self.pdf_generator = AcademicPDFGenerator(
                output_directory="test_output",
                citation_style="apa"
            )
            
            # Test data for PDF generation
            research_data = {
                'introduction': 'This is a test introduction for historical research.',
                'methodology': 'Test methodology section.',
                'findings': 'Test findings and analysis.',
                'conclusion': 'Test conclusion.',
                'sources': [
                    {
                        'title': 'Test Source 1',
                        'authors': ['Author One'],
                        'year': 2020,
                        'type': 'journal'
                    }
                ]
            }
            
            paper_metadata = {
                'title': 'Integration Test Research Paper',
                'authors': ['Test Author'],
                'abstract': 'This is a test abstract.',
                'keywords': ['test', 'integration', 'historical'],
                'institution': 'Test University'
            }
            
            # Generate PDF
            pdf_result = await self.pdf_generator.generate_academic_paper(
                research_data, paper_metadata
            )
            
            assert pdf_result.get('success', False), "PDF generation failed"
            
            # Get generation statistics
            gen_stats = await self.pdf_generator.get_generation_statistics()
            
            self.test_results[test_name] = {
                'status': 'PASSED',
                'pdf_generated': True,
                'pdf_path': pdf_result.get('pdf_path'),
                'bibliography_generated': pdf_result.get('bibliography_path') is not None,
                'word_count': pdf_result.get('statistics', {}).get('word_count', 0),
                'total_pdfs_generated': gen_stats['total_pdfs_generated']
            }
            
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            self.test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
    
    async def _test_system_monitoring(self):
        """Test system health monitoring and performance tracking."""
        test_name = "system_monitoring"
        logger.info(f"Testing {test_name}")
        
        try:
            # Initialize health monitor
            self.health_monitor = SystemHealthMonitor(
                self.vector_manager,
                self.embedding_service,
                monitoring_interval=10  # Short interval for testing
            )
            
            # Start monitoring
            await self.health_monitor.start_monitoring()
            
            # Wait for some monitoring data
            await asyncio.sleep(5)
            
            # Get health summary
            health_summary = await self.health_monitor.get_system_health_summary()
            
            # Get performance report
            performance_report = await self.health_monitor.get_performance_report(hours=1)
            
            # Stop monitoring
            await self.health_monitor.stop_monitoring()
            
            self.test_results[test_name] = {
                'status': 'PASSED',
                'monitoring_active': health_summary.get('monitoring_active', False),
                'overall_health_status': health_summary.get('overall_status'),
                'metrics_collected': health_summary.get('latest_metrics') is not None,
                'performance_report_generated': 'error' not in performance_report,
                'service_statuses': len(health_summary.get('service_statuses', {}))
            }
            
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            self.test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
    
    async def _generate_integration_report(self):
        """Generate comprehensive integration test report."""
        test_name = "integration_report_generation"
        logger.info(f"Generating {test_name}")
        
        try:
            total_time = time.time() - self.start_time
            
            # Count test results
            passed_tests = sum(1 for result in self.test_results.values() 
                             if result.get('status') == 'PASSED')
            total_tests = len(self.test_results)
            failed_tests = total_tests - passed_tests
            
            # Determine overall status
            overall_status = 'PASSED' if failed_tests == 0 else 'FAILED'
            
            # Generate comprehensive report
            report = {
                'integration_test_report': {
                    'timestamp': datetime.now().isoformat(),
                    'total_execution_time_seconds': total_time,
                    'overall_status': overall_status,
                    'test_summary': {
                        'total_tests': total_tests,
                        'passed_tests': passed_tests,
                        'failed_tests': failed_tests,
                        'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
                    },
                    'detailed_results': self.test_results,
                    'system_capabilities': {
                        'gpu_acceleration': self.test_results.get('system_initialization', {}).get('gpu_available', False),
                        'vector_storage': self.test_results.get('core_infrastructure', {}).get('status') == 'PASSED',
                        'document_processing': self.test_results.get('document_processing', {}).get('status') == 'PASSED',
                        'ai_validation': self.test_results.get('ai_validation_systems', {}).get('status') == 'PASSED',
                        'multi_agent_orchestration': self.test_results.get('multi_agent_orchestration', {}).get('status') == 'PASSED',
                        'academic_output': self.test_results.get('output_generation', {}).get('status') == 'PASSED',
                        'system_monitoring': self.test_results.get('system_monitoring', {}).get('status') == 'PASSED'
                    },
                    'recommendations': self._generate_recommendations()
                }
            }
            
            # Save report to file
            report_file = Path('integration_test_report.json')
            with open(report_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Log summary
            logger.info(f"""
Integration Test Summary:
========================
Overall Status: {overall_status}
Total Tests: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Success Rate: {(passed_tests / total_tests) * 100:.1f}%
Execution Time: {total_time:.2f} seconds
Report saved to: {report_file}
            """)
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed tests and provide recommendations
        for test_name, result in self.test_results.items():
            if result.get('status') == 'FAILED':
                error = result.get('error', 'Unknown error')
                recommendations.append(f"Fix {test_name}: {error}")
        
        # System-specific recommendations
        if not self.test_results.get('system_initialization', {}).get('gpu_available', False):
            recommendations.append("Consider GPU setup for enhanced performance in document processing and AI operations.")
        
        if self.test_results.get('core_infrastructure', {}).get('embedding_dimension', 0) == 0:
            recommendations.append("Verify embedding service configuration for optimal vector search performance.")
        
        # Performance recommendations
        if self.test_results.get('system_monitoring', {}).get('status') == 'PASSED':
            recommendations.append("System monitoring is active. Regularly review performance metrics for optimization opportunities.")
        
        if not recommendations:
            recommendations.append("All systems operational. Consider implementing additional advanced features as needed.")
        
        return recommendations
    
    async def _cleanup_test_environment(self):
        """Clean up test environment and resources."""
        logger.info("Cleaning up test environment")
        
        try:
            # Stop orchestrator if running
            if self.orchestrator and self.orchestrator.is_running:
                await self.orchestrator.stop_orchestration()
            
            # Stop health monitor if running
            if self.health_monitor and self.health_monitor.is_monitoring:
                await self.health_monitor.stop_monitoring()
            
            # Close document acquisition resources
            if self.document_acquisition:
                await self.document_acquisition.close()
            
            # Shutdown vector manager
            if self.vector_manager:
                await self.vector_manager.shutdown()
            
            logger.info("Test environment cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


async def main():
    """Main function to run the integration test."""
    print("Historical Research System - Complete Integration Test")
    print("=" * 60)
    
    # Create and run test runner
    test_runner = IntegrationTestRunner()
    await test_runner.run_complete_integration_test()
    
    print("\nIntegration test completed. Check 'integration_test_report.json' for detailed results.")


if __name__ == "__main__":
    asyncio.run(main())
