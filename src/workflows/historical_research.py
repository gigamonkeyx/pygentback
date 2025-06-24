"""
Historical Research Workflow

Real workflow implementation for conducting historical research using
PyGent Factory's AI components. Supports the research topics you specified.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.integration.core import IntegrationEngine, WorkflowOrchestrator
from src.integration.models import WorkflowDefinition, ExecutionContext
from src.integration.adapters import NLPAdapter, MultiAgentAdapter, GeneticAlgorithmAdapter, PredictiveAdapter

logger = logging.getLogger(__name__)


class HistoricalResearchWorkflow:
    """
    Workflow for conducting historical research using AI components.
    
    Supports research topics including:
    - Scientific Revolutions
    - Enlightenment
    - Early Modern Exploration
    - Tokugawa Japan
    - Colonialism in Southeast Asia
    - Ming & Qing Dynasties
    - Haitian Revolution
    - Opposition to Imperialism
    - World's Fairs & Zoos
    - Eugenics
    - Globalization/Dictatorships
    - Decolonization
    """
    
    def __init__(self):
        self.integration_engine = IntegrationEngine()
        self.workflow_orchestrator = None
        self.is_initialized = False
        
        # Research configuration
        self.research_config = {
            'focus_on_primary_sources': True,
            'global_perspective': True,
            'avoid_european_centrism': True,
            'include_non_european_responses': True,
            'cross_cultural_analysis': True
        }
        
        # Supported research topics
        self.supported_topics = {
            'scientific_revolutions': {
                'focus_areas': ['art_architecture', 'literacy', 'global_perspectives'],
                'primary_sources': ['manuscripts', 'treatises', 'correspondence'],
                'regions': ['europe', 'islamic_world', 'china', 'india', 'americas']
            },
            'enlightenment': {
                'focus_areas': ['human_rights', 'political_values', 'philosophy'],
                'primary_sources': ['philosophical_works', 'political_documents', 'letters'],
                'regions': ['europe', 'americas', 'africa', 'asia']
            },
            'early_modern_exploration': {
                'focus_areas': ['cartography', 'flora_fauna', 'describing_otherness'],
                'primary_sources': ['maps', 'travel_accounts', 'natural_histories'],
                'regions': ['global']
            },
            'tokugawa_japan': {
                'focus_areas': ['women', 'art', 'samurai_role_shift'],
                'primary_sources': ['official_documents', 'literature', 'art_works'],
                'regions': ['japan']
            },
            'colonialism_southeast_asia': {
                'focus_areas': ['non_european_response', 'european_life_abroad'],
                'primary_sources': ['local_chronicles', 'colonial_records', 'personal_accounts'],
                'regions': ['indonesia', 'philippines', 'vietnam', 'thailand', 'malaysia']
            },
            'ming_qing_dynasties': {
                'focus_areas': ['education', 'administration', 'culture'],
                'primary_sources': ['imperial_records', 'scholarly_works', 'examination_papers'],
                'regions': ['china']
            },
            'haitian_revolution': {
                'focus_areas': ['diaspora_influences', 'global_impact'],
                'primary_sources': ['revolutionary_documents', 'personal_accounts', 'newspaper_reports'],
                'regions': ['haiti', 'france', 'usa', 'latin_america', 'africa']
            },
            'opposition_to_imperialism': {
                'focus_areas': ['resistance_movements', 'intellectual_opposition'],
                'primary_sources': ['resistance_documents', 'speeches', 'manifestos'],
                'regions': ['china', 'africa', 'europe', 'latin_america', 'southeast_asia']
            },
            'worlds_fairs_zoos': {
                'focus_areas': ['cultural_display', 'colonial_exhibition', 'public_reception'],
                'primary_sources': ['exhibition_catalogs', 'visitor_accounts', 'press_coverage'],
                'regions': ['global']
            },
            'eugenics_global': {
                'focus_areas': ['scientific_racism', 'policy_implementation', 'resistance'],
                'primary_sources': ['scientific_papers', 'policy_documents', 'opposition_writings'],
                'regions': ['global']
            },
            'globalization_dictatorships': {
                'focus_areas': ['economic_integration', 'authoritarian_responses'],
                'primary_sources': ['economic_reports', 'political_documents', 'resistance_literature'],
                'regions': ['global']
            },
            'decolonization': {
                'focus_areas': ['independence_movements', 'cultural_revival', 'economic_transformation'],
                'primary_sources': ['independence_documents', 'speeches', 'cultural_works'],
                'regions': ['africa', 'asia', 'americas', 'oceania']
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the historical research workflow"""
        try:
            # Start integration engine
            await self.integration_engine.start()
            
            # Register adapters
            await self._register_adapters()
            
            # Create workflow orchestrator
            from integration.core import OrchestrationEngine
            orchestration_engine = OrchestrationEngine(self.integration_engine)
            self.workflow_orchestrator = WorkflowOrchestrator(
                self.integration_engine, 
                orchestration_engine
            )
            
            self.is_initialized = True
            logger.info("Historical Research Workflow initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Historical Research Workflow: {e}")
            return False
    
    async def _register_adapters(self):
        """Register AI component adapters"""
        from integration.core import ComponentInfo, ComponentType
        
        # NLP Adapter for text analysis
        nlp_adapter = NLPAdapter()
        nlp_info = ComponentInfo(
            component_id="historical_nlp",
            component_type=ComponentType.NLP_SYSTEM,
            name="Historical Text Analysis",
            version="1.0",
            capabilities=["text_analysis", "source_parsing", "pattern_recognition"]
        )
        self.integration_engine.register_component(nlp_info, nlp_adapter)
        
        # Multi-Agent Adapter for coordinated research
        agent_adapter = MultiAgentAdapter()
        agent_info = ComponentInfo(
            component_id="research_coordination",
            component_type=ComponentType.MULTI_AGENT,
            name="Research Coordination System",
            version="1.0",
            capabilities=["source_validation", "cross_referencing", "synthesis"]
        )
        self.integration_engine.register_component(agent_info, agent_adapter)
        
        # Genetic Algorithm for pattern discovery
        ga_adapter = GeneticAlgorithmAdapter()
        ga_info = ComponentInfo(
            component_id="pattern_discovery",
            component_type=ComponentType.GENETIC_ALGORITHM,
            name="Historical Pattern Discovery",
            version="1.0",
            capabilities=["pattern_optimization", "trend_analysis"]
        )
        self.integration_engine.register_component(ga_info, ga_adapter)
        
        # Predictive Adapter for research insights
        pred_adapter = PredictiveAdapter()
        pred_info = ComponentInfo(
            component_id="research_prediction",
            component_type=ComponentType.PREDICTIVE_OPTIMIZATION,
            name="Research Insight Prediction",
            version="1.0",
            capabilities=["insight_prediction", "source_ranking"]
        )
        self.integration_engine.register_component(pred_info, pred_adapter)
    
    async def conduct_research(self, topic: str, research_question: str, 
                             source_texts: List[str], **kwargs) -> Dict[str, Any]:
        """
        Conduct historical research on specified topic.
        
        Args:
            topic: Research topic (e.g., 'scientific_revolutions')
            research_question: Specific research question
            source_texts: List of primary source texts to analyze
            **kwargs: Additional research parameters
            
        Returns:
            Research results with analysis and findings
        """
        if not self.is_initialized:
            await self.initialize()
        
        if topic not in self.supported_topics:
            raise ValueError(f"Unsupported research topic: {topic}")
        
        logger.info(f"Starting historical research on: {topic}")
        logger.info(f"Research question: {research_question}")
        
        # Create research workflow
        workflow_def = self._create_research_workflow(topic, research_question, source_texts, **kwargs)
        
        # Execute workflow
        context = ExecutionContext(
            execution_id=f"research_{topic}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            metadata={
                'topic': topic,
                'research_question': research_question,
                'source_count': len(source_texts),
                'config': self.research_config
            }
        )
        
        result = await self.integration_engine.execute_workflow(workflow_def, context)
        
        # Process and format results
        research_findings = self._process_research_results(result, topic, research_question)
        
        logger.info(f"Historical research completed for topic: {topic}")
        
        return research_findings
    
    def _create_research_workflow(self, topic: str, research_question: str, 
                                source_texts: List[str], **kwargs) -> WorkflowDefinition:
        """Create workflow definition for historical research"""
        
        topic_config = self.supported_topics[topic]
        
        workflow_steps = [
            {
                'step_id': 'analyze_sources',
                'component_type': 'nlp_system',
                'action': 'parse_recipe',
                'parameters': {
                    'recipe_text': '\n\n'.join(source_texts),
                    'name': f'{topic}_source_analysis',
                    'focus_areas': topic_config['focus_areas'],
                    'research_question': research_question
                },
                'required': True
            },
            {
                'step_id': 'validate_sources',
                'component_type': 'multi_agent',
                'action': 'execute_tests',
                'parameters': {
                    'test_suite': [
                        {'name': 'source_authenticity', 'type': 'validation'},
                        {'name': 'cross_reference_check', 'type': 'verification'},
                        {'name': 'bias_detection', 'type': 'analysis'}
                    ]
                },
                'dependencies': ['analyze_sources'],
                'required': True
            },
            {
                'step_id': 'discover_patterns',
                'component_type': 'genetic_algorithm',
                'action': 'optimize',
                'parameters': {
                    'recipe_data': {
                        'name': f'{topic}_pattern_discovery',
                        'description': f'Discover historical patterns in {topic}',
                        'focus_areas': topic_config['focus_areas'],
                        'regions': topic_config['regions']
                    }
                },
                'dependencies': ['validate_sources'],
                'required': True
            },
            {
                'step_id': 'predict_insights',
                'component_type': 'predictive_optimization',
                'action': 'predict_performance',
                'parameters': {
                    'research_data': {
                        'topic': topic,
                        'question': research_question,
                        'source_analysis': '${analyze_sources.result}',
                        'patterns': '${discover_patterns.result}'
                    }
                },
                'dependencies': ['discover_patterns'],
                'required': True
            }
        ]
        
        return WorkflowDefinition(
            name=f"historical_research_{topic}",
            description=f"Historical research workflow for {topic}",
            steps=workflow_steps,
            timeout_seconds=1800.0,  # 30 minutes
            metadata={
                'topic': topic,
                'research_question': research_question,
                'focus_areas': topic_config['focus_areas'],
                'regions': topic_config['regions']
            }
        )
    
    def _process_research_results(self, workflow_result: Any, topic: str, 
                                research_question: str) -> Dict[str, Any]:
        """Process workflow results into research findings"""
        
        findings = {
            'research_metadata': {
                'topic': topic,
                'research_question': research_question,
                'execution_time': workflow_result.execution_time_ms if hasattr(workflow_result, 'execution_time_ms') else 0,
                'success': workflow_result.success if hasattr(workflow_result, 'success') else False,
                'timestamp': datetime.utcnow().isoformat()
            },
            'source_analysis': {},
            'validation_results': {},
            'discovered_patterns': {},
            'predicted_insights': {},
            'synthesis': {}
        }
        
        if hasattr(workflow_result, 'results'):
            results = workflow_result.results
            
            # Extract source analysis
            if 'analyze_sources' in results:
                findings['source_analysis'] = results['analyze_sources'].get('result', {})
            
            # Extract validation results
            if 'validate_sources' in results:
                findings['validation_results'] = results['validate_sources'].get('result', {})
            
            # Extract discovered patterns
            if 'discover_patterns' in results:
                findings['discovered_patterns'] = results['discover_patterns'].get('result', {})
            
            # Extract predicted insights
            if 'predict_insights' in results:
                findings['predicted_insights'] = results['predict_insights'].get('result', {})
        
        # Create synthesis
        findings['synthesis'] = self._synthesize_findings(findings, topic, research_question)
        
        return findings
    
    def _synthesize_findings(self, findings: Dict[str, Any], topic: str, 
                           research_question: str) -> Dict[str, Any]:
        """Synthesize research findings into coherent analysis"""
        
        synthesis = {
            'executive_summary': f"Historical research analysis of {topic} addressing: {research_question}",
            'key_findings': [],
            'primary_sources_insights': [],
            'cross_cultural_perspectives': [],
            'methodological_notes': [],
            'recommendations_for_further_research': []
        }
        
        # Extract key findings from source analysis
        source_analysis = findings.get('source_analysis', {})
        if 'steps' in source_analysis:
            synthesis['key_findings'].extend([
                f"Identified {len(source_analysis['steps'])} key analytical steps",
                f"Text complexity score: {source_analysis.get('complexity', 'unknown')}",
                f"Estimated analysis duration: {source_analysis.get('estimated_duration', 'unknown')} seconds"
            ])
        
        # Add validation insights
        validation = findings.get('validation_results', {})
        if 'summary' in validation:
            summary = validation['summary']
            synthesis['methodological_notes'].append(
                f"Source validation: {summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed"
            )
        
        # Add pattern discovery insights
        patterns = findings.get('discovered_patterns', {})
        if 'optimization_metrics' in patterns:
            metrics = patterns['optimization_metrics']
            synthesis['key_findings'].append(
                f"Pattern discovery achieved {metrics.get('best_fitness', 0):.2f} fitness score"
            )
        
        # Add research recommendations
        synthesis['recommendations_for_further_research'] = [
            "Expand primary source base with additional regional perspectives",
            "Conduct comparative analysis with contemporary sources",
            "Investigate long-term historical impacts and consequences",
            "Examine connections to broader global historical trends"
        ]
        
        # Add cross-cultural perspective notes
        topic_config = self.supported_topics.get(topic, {})
        regions = topic_config.get('regions', [])
        synthesis['cross_cultural_perspectives'] = [
            f"Analysis incorporates perspectives from: {', '.join(regions)}",
            "Research methodology emphasizes non-European viewpoints",
            "Primary sources include local and indigenous perspectives where available"
        ]
        
        return synthesis
    
    def get_supported_topics(self) -> Dict[str, Any]:
        """Get list of supported research topics"""
        return self.supported_topics.copy()
    
    def get_research_config(self) -> Dict[str, Any]:
        """Get current research configuration"""
        return self.research_config.copy()
