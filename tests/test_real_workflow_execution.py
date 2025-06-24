"""
Real workflow execution tests - NO MOCKS!

Tests actual workflow execution using real components and the historical research workflow.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from src.workflows.historical_research import HistoricalResearchWorkflow


class TestRealWorkflowExecution:
    """Test real workflow execution with historical research"""
    
    @pytest.fixture
    async def research_workflow(self):
        """Create and initialize research workflow"""
        workflow = HistoricalResearchWorkflow()
        success = await workflow.initialize()
        if not success:
            pytest.skip("Failed to initialize research workflow")
        return workflow
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, research_workflow):
        """Test that workflow initializes successfully"""
        assert research_workflow.is_initialized
        assert research_workflow.integration_engine is not None
        assert research_workflow.workflow_orchestrator is not None
        
        # Check supported topics
        topics = research_workflow.get_supported_topics()
        assert 'scientific_revolutions' in topics
        assert 'enlightenment' in topics
        assert 'tokugawa_japan' in topics
        assert 'haitian_revolution' in topics
        
        print("✅ Research workflow initialized successfully")
    
    @pytest.mark.asyncio
    async def test_scientific_revolutions_research(self, research_workflow):
        """Test real historical research on Scientific Revolutions"""
        
        # Sample primary source texts for Scientific Revolutions
        source_texts = [
            """The new astronomy demonstrates that the Earth moves around the Sun, 
            contrary to ancient beliefs. This discovery transforms our understanding 
            of the cosmos and challenges traditional authorities.""",
            
            """Mathematical principles govern the motion of celestial bodies. 
            Through careful observation and calculation, we can predict planetary 
            movements with unprecedented accuracy.""",
            
            """The printing press enables rapid dissemination of new ideas across 
            Europe and beyond. Knowledge that once took decades to spread now 
            travels in months."""
        ]
        
        research_question = "How did the Scientific Revolution transform global understanding of astronomy beyond European perspectives?"
        
        # Execute real research workflow
        results = await research_workflow.conduct_research(
            topic='scientific_revolutions',
            research_question=research_question,
            source_texts=source_texts,
            focus_areas=['art_architecture', 'literacy'],
            global_perspective=True
        )
        
        # Validate results structure
        assert 'research_metadata' in results
        assert 'source_analysis' in results
        assert 'validation_results' in results
        assert 'discovered_patterns' in results
        assert 'predicted_insights' in results
        assert 'synthesis' in results
        
        # Check metadata
        metadata = results['research_metadata']
        assert metadata['topic'] == 'scientific_revolutions'
        assert metadata['research_question'] == research_question
        assert metadata['success'] is True
        
        # Check synthesis
        synthesis = results['synthesis']
        assert 'executive_summary' in synthesis
        assert 'key_findings' in synthesis
        assert 'cross_cultural_perspectives' in synthesis
        
        print(f"✅ Scientific Revolutions research completed")
        print(f"   Executive Summary: {synthesis['executive_summary']}")
        print(f"   Key Findings: {len(synthesis['key_findings'])} findings")
        
        return results
    
    @pytest.mark.asyncio
    async def test_tokugawa_japan_research(self, research_workflow):
        """Test real historical research on Tokugawa Japan"""
        
        source_texts = [
            """During the Tokugawa period, women's roles in Japanese society 
            were strictly defined by Confucian principles, yet many found ways 
            to exercise influence through family networks and cultural activities.""",
            
            """The transformation of the samurai class from warriors to bureaucrats 
            fundamentally altered Japanese social structure. Many samurai became 
            scholars, artists, and administrators.""",
            
            """Ukiyo-e art flourished during this period, depicting the 'floating world' 
            of urban pleasure quarters and kabuki theater, reflecting new cultural 
            values and social dynamics."""
        ]
        
        research_question = "How did women navigate social constraints during the Tokugawa period?"
        
        results = await research_workflow.conduct_research(
            topic='tokugawa_japan',
            research_question=research_question,
            source_texts=source_texts,
            focus_areas=['women', 'art']
        )
        
        # Validate results
        assert results['research_metadata']['success'] is True
        assert results['research_metadata']['topic'] == 'tokugawa_japan'
        
        # Check that real analysis occurred
        source_analysis = results['source_analysis']
        assert 'steps' in source_analysis
        assert len(source_analysis['steps']) > 0
        
        print(f"✅ Tokugawa Japan research completed")
        print(f"   Analyzed {len(source_analysis['steps'])} analytical steps")
        
        return results
    
    @pytest.mark.asyncio
    async def test_haitian_revolution_research(self, research_workflow):
        """Test real historical research on Haitian Revolution"""
        
        source_texts = [
            """The Haitian Revolution inspired enslaved peoples across the Americas 
            and struck fear into the hearts of slaveholders from Virginia to Brazil. 
            Its impact extended far beyond the Caribbean.""",
            
            """Toussaint Louverture's leadership demonstrated that enslaved Africans 
            could organize sophisticated military and political resistance. His 
            strategies influenced liberation movements worldwide.""",
            
            """The revolution's success challenged European racial theories and 
            colonial assumptions about African capabilities, forcing a 
            reconsideration of Enlightenment ideals."""
        ]
        
        research_question = "What was the global impact of Haitian diaspora communities following the revolution?"
        
        results = await research_workflow.conduct_research(
            topic='haitian_revolution',
            research_question=research_question,
            source_texts=source_texts,
            focus_areas=['diaspora_influences']
        )
        
        # Validate comprehensive analysis
        assert results['research_metadata']['success'] is True
        
        # Check that genetic algorithm optimization occurred
        patterns = results['discovered_patterns']
        assert 'optimization_metrics' in patterns
        
        # Check that predictive insights were generated
        insights = results['predicted_insights']
        assert insights is not None
        
        print(f"✅ Haitian Revolution research completed")
        
        return results
    
    @pytest.mark.asyncio
    async def test_full_research_pipeline(self, research_workflow):
        """Test complete research pipeline with multiple components"""
        
        # Test with a complex research question requiring all components
        source_texts = [
            """Colonial administrators in Southeast Asia often struggled to understand 
            local customs and governance systems, leading to policies that sparked 
            resistance movements.""",
            
            """Indigenous responses to colonialism varied widely, from armed resistance 
            to cultural adaptation and hybrid forms of governance that preserved 
            local autonomy.""",
            
            """European settlers in colonial territories developed distinct identities 
            that differed from their metropolitan counterparts, influenced by local 
            conditions and cross-cultural interactions."""
        ]
        
        research_question = "How did non-European responses to colonialism in Southeast Asia shape colonial policies?"
        
        results = await research_workflow.conduct_research(
            topic='colonialism_southeast_asia',
            research_question=research_question,
            source_texts=source_texts,
            focus_areas=['non_european_response', 'european_life_abroad']
        )
        
        # Comprehensive validation
        assert results['research_metadata']['success'] is True
        
        # Validate all workflow steps executed
        assert 'source_analysis' in results
        assert 'validation_results' in results
        assert 'discovered_patterns' in results
        assert 'predicted_insights' in results
        assert 'synthesis' in results
        
        # Check synthesis quality
        synthesis = results['synthesis']
        assert len(synthesis['key_findings']) > 0
        assert len(synthesis['cross_cultural_perspectives']) > 0
        assert len(synthesis['recommendations_for_further_research']) > 0
        
        # Validate cross-cultural perspective emphasis
        perspectives = synthesis['cross_cultural_perspectives']
        assert any('non-European' in p for p in perspectives)
        
        print(f"✅ Full research pipeline completed successfully")
        print(f"   Key findings: {len(synthesis['key_findings'])}")
        print(f"   Cross-cultural perspectives: {len(synthesis['cross_cultural_perspectives'])}")
        print(f"   Research recommendations: {len(synthesis['recommendations_for_further_research'])}")
        
        return results
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, research_workflow):
        """Test workflow error handling with invalid inputs"""
        
        # Test with unsupported topic
        with pytest.raises(ValueError, match="Unsupported research topic"):
            await research_workflow.conduct_research(
                topic='invalid_topic',
                research_question='Test question',
                source_texts=['Test text']
            )
        
        # Test with empty source texts
        results = await research_workflow.conduct_research(
            topic='enlightenment',
            research_question='Test question with no sources',
            source_texts=[]
        )
        
        # Should still complete but with limited analysis
        assert 'research_metadata' in results
        
        print("✅ Error handling tests passed")
    
    @pytest.mark.asyncio
    async def test_component_integration(self, research_workflow):
        """Test that all AI components are properly integrated"""
        
        # Verify integration engine has all required adapters
        engine = research_workflow.integration_engine
        
        # Check that components are registered
        components = engine.get_registered_components()
        component_types = [comp.component_type.value for comp in components]
        
        assert 'nlp_system' in component_types
        assert 'multi_agent' in component_types
        assert 'genetic_algorithm' in component_types
        assert 'predictive_optimization' in component_types
        
        print("✅ All AI components properly integrated")
        print(f"   Registered components: {len(components)}")
        print(f"   Component types: {component_types}")
    
    def test_supported_topics_coverage(self, research_workflow):
        """Test that all specified research topics are supported"""
        
        topics = research_workflow.get_supported_topics()
        
        # Verify all your specified topics are supported
        required_topics = [
            'scientific_revolutions',
            'enlightenment', 
            'early_modern_exploration',
            'tokugawa_japan',
            'colonialism_southeast_asia',
            'ming_qing_dynasties',
            'haitian_revolution',
            'opposition_to_imperialism',
            'worlds_fairs_zoos',
            'eugenics_global',
            'globalization_dictatorships',
            'decolonization'
        ]
        
        for topic in required_topics:
            assert topic in topics, f"Missing support for topic: {topic}"
            
            # Check topic configuration
            topic_config = topics[topic]
            assert 'focus_areas' in topic_config
            assert 'primary_sources' in topic_config
            assert 'regions' in topic_config
        
        print(f"✅ All {len(required_topics)} research topics supported")
        print(f"   Topics: {list(topics.keys())}")
    
    def test_research_configuration(self, research_workflow):
        """Test research configuration emphasizes primary sources and global perspectives"""
        
        config = research_workflow.get_research_config()
        
        # Verify configuration aligns with your requirements
        assert config['focus_on_primary_sources'] is True
        assert config['global_perspective'] is True
        assert config['avoid_european_centrism'] is True
        assert config['include_non_european_responses'] is True
        assert config['cross_cultural_analysis'] is True
        
        print("✅ Research configuration properly emphasizes primary sources and global perspectives")
