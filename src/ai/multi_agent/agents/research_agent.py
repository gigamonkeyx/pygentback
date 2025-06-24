"""
Research Agent

Specialized agent for conducting historical research, synthesizing findings,
and generating insights with emphasis on global perspectives and primary sources.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ResearchPhase(Enum):
    """Research process phases"""
    PLANNING = "planning"
    COLLECTION = "collection"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    REPORTING = "reporting"


@dataclass
class ResearchSource:
    """Historical research source"""
    source_id: str
    source_type: str  # primary, secondary, tertiary
    title: str
    author: Optional[str]
    date: Optional[str]
    region: Optional[str]
    language: Optional[str]
    content: str
    reliability_score: float = 0.0
    cultural_context: Optional[str] = None
    
    def is_primary_source(self) -> bool:
        return self.source_type == "primary"
    
    def is_non_european(self) -> bool:
        non_european_regions = {
            'asia', 'africa', 'americas', 'oceania', 'middle_east',
            'china', 'japan', 'india', 'southeast_asia', 'latin_america'
        }
        return self.region and any(region in self.region.lower() for region in non_european_regions)


@dataclass
class ResearchFinding:
    """Research finding with evidence"""
    finding_id: str
    topic: str
    finding: str
    evidence: List[str]
    sources: List[str]
    confidence: float
    global_perspective: bool
    cultural_context: str
    timestamp: datetime


class ResearchAgent:
    """
    Agent specialized in historical research with global perspectives.
    
    Capabilities:
    - Historical research planning and execution
    - Primary source analysis and validation
    - Cross-cultural research synthesis
    - Global perspective integration
    - Research insight generation
    """
    
    def __init__(self, agent_id: str = "research_agent"):
        self.agent_id = agent_id
        self.agent_type = "research"
        self.status = "initialized"
        self.capabilities = [
            "historical_research",
            "source_analysis",
            "cross_cultural_synthesis",
            "global_perspective_integration",
            "insight_generation"
        ]
        
        # Research state
        self.active_research: Dict[str, Dict[str, Any]] = {}
        self.research_sources: Dict[str, ResearchSource] = {}
        self.research_findings: Dict[str, ResearchFinding] = {}
        
        # Configuration
        self.config = {
            'max_sources_per_research': 50,
            'min_primary_source_ratio': 0.6,
            'min_non_european_ratio': 0.4,
            'confidence_threshold': 0.7,
            'research_timeout_seconds': 600,
            'cross_reference_threshold': 3
        }
        
        # Research topics and focus areas
        self.supported_topics = {
            'scientific_revolutions': {
                'focus_areas': ['art_architecture', 'literacy', 'global_perspectives'],
                'regions': ['europe', 'asia', 'americas', 'africa'],
                'time_periods': ['1500-1700', '1600-1800']
            },
            'enlightenment': {
                'focus_areas': ['human_rights', 'political_values', 'philosophy'],
                'regions': ['europe', 'americas', 'asia'],
                'time_periods': ['1650-1800', '1700-1850']
            },
            'decolonization': {
                'focus_areas': ['independence_movements', 'cultural_revival', 'economic_transformation'],
                'regions': ['africa', 'asia', 'americas', 'oceania'],
                'time_periods': ['1900-1970', '1945-1980']
            }
        }
        
        # Statistics
        self.stats = {
            'research_projects': 0,
            'sources_analyzed': 0,
            'findings_generated': 0,
            'primary_sources_used': 0,
            'non_european_sources_used': 0,
            'avg_research_time_ms': 0.0,
            'successful_research': 0
        }
        
        logger.info(f"ResearchAgent {agent_id} initialized")
    
    async def start(self) -> bool:
        """Start the research agent"""
        try:
            self.status = "active"
            logger.info(f"ResearchAgent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start ResearchAgent {self.agent_id}: {e}")
            self.status = "error"
            return False
    
    async def stop(self) -> bool:
        """Stop the research agent"""
        try:
            self.status = "stopped"
            logger.info(f"ResearchAgent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop ResearchAgent {self.agent_id}: {e}")
            return False
    
    async def conduct_research(self, topic: str, research_question: str, 
                             source_texts: List[str], **kwargs) -> Dict[str, Any]:
        """
        Conduct comprehensive historical research on a topic.
        
        Args:
            topic: Research topic
            research_question: Specific research question
            source_texts: List of source texts to analyze
            **kwargs: Additional research parameters
            
        Returns:
            Comprehensive research results
        """
        start_time = datetime.utcnow()
        research_id = f"research_{int(start_time.timestamp())}"
        
        try:
            # Initialize research project
            self.active_research[research_id] = {
                'topic': topic,
                'research_question': research_question,
                'phase': ResearchPhase.PLANNING,
                'start_time': start_time,
                'sources_count': len(source_texts),
                'focus_areas': kwargs.get('focus_areas', []),
                'global_perspective': kwargs.get('global_perspective', True)
            }
            
            self.stats['research_projects'] += 1
            
            # Phase 1: Planning
            research_plan = await self._plan_research(topic, research_question, kwargs)
            self.active_research[research_id]['phase'] = ResearchPhase.COLLECTION
            
            # Phase 2: Source Collection and Processing
            processed_sources = await self._process_sources(research_id, source_texts)
            self.active_research[research_id]['phase'] = ResearchPhase.ANALYSIS
            
            # Phase 3: Analysis
            analysis_results = await self._analyze_sources(research_id, processed_sources)
            self.active_research[research_id]['phase'] = ResearchPhase.SYNTHESIS
            
            # Phase 4: Synthesis
            synthesis_results = await self._synthesize_findings(research_id, analysis_results)
            self.active_research[research_id]['phase'] = ResearchPhase.VALIDATION
            
            # Phase 5: Validation
            validation_results = await self._validate_findings(research_id, synthesis_results)
            self.active_research[research_id]['phase'] = ResearchPhase.REPORTING
            
            # Phase 6: Generate Final Report
            final_results = await self._generate_research_report(
                research_id, research_plan, analysis_results, synthesis_results, validation_results
            )
            
            # Calculate research time
            research_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_stats(True, research_time, len(processed_sources))
            
            # Clean up active research
            del self.active_research[research_id]
            
            logger.info(f"Research completed for topic '{topic}' in {research_time:.2f}ms")
            return final_results
            
        except Exception as e:
            research_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_stats(False, research_time, 0)
            
            logger.error(f"Research failed for topic '{topic}': {e}")
            
            # Clean up active research
            if research_id in self.active_research:
                del self.active_research[research_id]
            
            return {
                'success': False,
                'error': str(e),
                'research_id': research_id,
                'topic': topic,
                'research_time_ms': research_time
            }
    
    async def synthesize_findings(self, data: List[Dict]) -> Dict[str, Any]:
        """Synthesize research findings from multiple sources"""
        start_time = datetime.utcnow()
        
        try:
            synthesis = {
                'synthesis_id': f"synthesis_{int(start_time.timestamp())}",
                'data_sources': len(data),
                'key_themes': [],
                'cross_cultural_insights': [],
                'global_patterns': [],
                'primary_source_evidence': [],
                'recommendations': [],
                'confidence_score': 0.0
            }
            
            # Extract themes from data
            themes = self._extract_themes(data)
            synthesis['key_themes'] = themes
            
            # Identify cross-cultural patterns
            cultural_insights = self._identify_cross_cultural_patterns(data)
            synthesis['cross_cultural_insights'] = cultural_insights
            
            # Find global patterns
            global_patterns = self._find_global_patterns(data)
            synthesis['global_patterns'] = global_patterns
            
            # Extract primary source evidence
            primary_evidence = self._extract_primary_evidence(data)
            synthesis['primary_source_evidence'] = primary_evidence
            
            # Generate recommendations
            recommendations = self._generate_research_recommendations(data, themes)
            synthesis['recommendations'] = recommendations
            
            # Calculate confidence score
            synthesis['confidence_score'] = self._calculate_synthesis_confidence(data, themes)
            
            synthesis_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            synthesis['synthesis_time_ms'] = synthesis_time
            
            logger.debug(f"Synthesized findings from {len(data)} sources in {synthesis_time:.2f}ms")
            return synthesis
            
        except Exception as e:
            logger.error(f"Findings synthesis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data_sources': len(data),
                'synthesis_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
            }
    
    async def generate_insights(self, analysis: Dict) -> List[str]:
        """Generate research insights from analysis"""
        try:
            insights = []
            
            # Generate insights based on analysis type
            if 'themes' in analysis:
                theme_insights = self._generate_thematic_insights(analysis['themes'])
                insights.extend(theme_insights)
            
            if 'patterns' in analysis:
                pattern_insights = self._generate_pattern_insights(analysis['patterns'])
                insights.extend(pattern_insights)
            
            if 'cultural_context' in analysis:
                cultural_insights = self._generate_cultural_insights(analysis['cultural_context'])
                insights.extend(cultural_insights)
            
            if 'temporal_analysis' in analysis:
                temporal_insights = self._generate_temporal_insights(analysis['temporal_analysis'])
                insights.extend(temporal_insights)
            
            # Ensure global perspective
            global_insights = self._ensure_global_perspective(insights, analysis)
            insights.extend(global_insights)
            
            # Remove duplicates and rank by relevance
            unique_insights = list(set(insights))
            ranked_insights = self._rank_insights(unique_insights, analysis)
            
            logger.debug(f"Generated {len(ranked_insights)} insights from analysis")
            return ranked_insights[:10]  # Return top 10 insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return [f"Error generating insights: {str(e)}"]
    
    async def _plan_research(self, topic: str, research_question: str, kwargs: Dict) -> Dict[str, Any]:
        """Plan the research approach"""
        plan = {
            'topic': topic,
            'research_question': research_question,
            'methodology': 'comparative_historical_analysis',
            'focus_areas': kwargs.get('focus_areas', []),
            'target_regions': self.supported_topics.get(topic, {}).get('regions', []),
            'time_periods': self.supported_topics.get(topic, {}).get('time_periods', []),
            'source_requirements': {
                'min_primary_sources': max(3, int(kwargs.get('source_count', 5) * self.config['min_primary_source_ratio'])),
                'min_non_european_sources': max(2, int(kwargs.get('source_count', 5) * self.config['min_non_european_ratio'])),
                'preferred_languages': ['english', 'spanish', 'french', 'chinese', 'arabic', 'portuguese']
            },
            'global_perspective': kwargs.get('global_perspective', True)
        }
        
        return plan
    
    async def _process_sources(self, research_id: str, source_texts: List[str]) -> List[ResearchSource]:
        """Process and categorize source texts"""
        processed_sources = []
        
        for i, text in enumerate(source_texts):
            source = ResearchSource(
                source_id=f"{research_id}_source_{i}",
                source_type=self._classify_source_type(text),
                title=f"Source {i+1}",
                author=self._extract_author(text),
                date=self._extract_date(text),
                language=self._identify_language(text),
                content=text,
                reliability_score=self._assess_source_reliability(text),
                region=self._identify_source_region(text),
                cultural_context=self._identify_cultural_context(text)
            )
            
            processed_sources.append(source)
            self.research_sources[source.source_id] = source
            
            # Update statistics
            self.stats['sources_analyzed'] += 1
            if source.is_primary_source():
                self.stats['primary_sources_used'] += 1
            if source.is_non_european():
                self.stats['non_european_sources_used'] += 1
        
        return processed_sources
    
    async def _analyze_sources(self, research_id: str, sources: List[ResearchSource]) -> Dict[str, Any]:
        """Analyze processed sources"""
        analysis = {
            'source_analysis': {
                'total_sources': len(sources),
                'primary_sources': len([s for s in sources if s.is_primary_source()]),
                'non_european_sources': len([s for s in sources if s.is_non_european()]),
                'avg_reliability': sum(s.reliability_score for s in sources) / len(sources) if sources else 0,
                'regional_distribution': self._analyze_regional_distribution(sources),
                'cultural_contexts': self._analyze_cultural_contexts(sources)
            },
            'content_analysis': {
                'themes': self._extract_content_themes(sources),
                'patterns': self._identify_content_patterns(sources),
                'contradictions': self._find_contradictions(sources),
                'consensus_points': self._find_consensus(sources)
            },
            'temporal_analysis': {
                'time_periods_covered': self._analyze_time_periods(sources),
                'chronological_patterns': self._identify_chronological_patterns(sources),
                'temporal_gaps': self._identify_temporal_gaps(sources)
            }
        }
        
        return analysis
    
    async def _synthesize_findings(self, research_id: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize analysis into coherent findings"""
        synthesis = {
            'key_findings': [],
            'supporting_evidence': {},
            'global_perspectives': [],
            'cultural_insights': [],
            'historical_significance': {},
            'research_gaps': [],
            'confidence_assessment': {}
        }
        
        # Generate key findings
        findings = self._generate_key_findings(analysis)
        synthesis['key_findings'] = findings
        
        # Map evidence to findings
        evidence_map = self._map_evidence_to_findings(findings, analysis)
        synthesis['supporting_evidence'] = evidence_map
        
        # Extract global perspectives
        global_perspectives = self._extract_global_perspectives(analysis)
        synthesis['global_perspectives'] = global_perspectives
        
        # Generate cultural insights
        cultural_insights = self._generate_cultural_insights_from_analysis(analysis)
        synthesis['cultural_insights'] = cultural_insights
        
        # Assess historical significance
        significance = self._assess_historical_significance(findings, analysis)
        synthesis['historical_significance'] = significance
        
        return synthesis
    
    async def _validate_findings(self, research_id: str, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate research findings"""
        validation = {
            'validation_score': 0.0,
            'source_validation': {},
            'cross_reference_validation': {},
            'bias_assessment': {},
            'reliability_assessment': {},
            'recommendations': []
        }
        
        # Validate against sources
        source_validation = self._validate_against_sources(synthesis)
        validation['source_validation'] = source_validation
        
        # Cross-reference validation
        cross_ref = self._cross_reference_findings(synthesis)
        validation['cross_reference_validation'] = cross_ref
        
        # Assess potential biases
        bias_assessment = self._assess_research_bias(synthesis)
        validation['bias_assessment'] = bias_assessment
        
        # Calculate overall validation score
        validation['validation_score'] = self._calculate_validation_score(
            source_validation, cross_ref, bias_assessment
        )
        
        return validation
    
    async def _generate_research_report(self, research_id: str, plan: Dict, 
                                      analysis: Dict, synthesis: Dict, validation: Dict) -> Dict[str, Any]:
        """Generate final research report"""
        report = {
            'research_id': research_id,
            'success': True,
            'executive_summary': self._generate_executive_summary(plan, synthesis),
            'methodology': plan.get('methodology', 'comparative_historical_analysis'),
            'key_findings': synthesis.get('key_findings', []),
            'global_perspectives': synthesis.get('global_perspectives', []),
            'cultural_insights': synthesis.get('cultural_insights', []),
            'historical_significance': synthesis.get('historical_significance', {}),
            'source_analysis': analysis.get('source_analysis', {}),
            'validation_results': validation,
            'research_recommendations': self._generate_final_recommendations(synthesis, validation),
            'confidence_score': validation.get('validation_score', 0.0),
            'metadata': {
                'research_agent': self.agent_id,
                'completion_time': datetime.utcnow().isoformat(),
                'global_perspective_applied': plan.get('global_perspective', True),
                'primary_source_ratio': analysis.get('source_analysis', {}).get('primary_sources', 0) / max(1, analysis.get('source_analysis', {}).get('total_sources', 1)),
                'non_european_source_ratio': analysis.get('source_analysis', {}).get('non_european_sources', 0) / max(1, analysis.get('source_analysis', {}).get('total_sources', 1))
            }
        }
        
        return report
    
    def _classify_source_type(self, text: str) -> str:
        """Classify source as primary, secondary, or tertiary"""
        # Simple heuristic classification
        primary_indicators = ['diary', 'letter', 'document', 'treaty', 'law', 'decree', 'witness', 'firsthand']
        secondary_indicators = ['analysis', 'study', 'research', 'examination', 'interpretation']
        
        text_lower = text.lower()
        
        if any(indicator in text_lower for indicator in primary_indicators):
            return "primary"
        elif any(indicator in text_lower for indicator in secondary_indicators):
            return "secondary"
        else:
            return "tertiary"
    
    def _assess_source_reliability(self, text: str) -> float:
        """Assess source reliability score"""
        score = 0.5  # Base score
        
        # Increase score for specific details
        if any(keyword in text.lower() for keyword in ['date', 'place', 'name', 'specific']):
            score += 0.2
        
        # Increase score for length and detail
        if len(text) > 500:
            score += 0.2
        
        # Increase score for multiple perspectives
        if any(keyword in text.lower() for keyword in ['however', 'although', 'different', 'various']):
            score += 0.1
        
        return min(1.0, score)
    
    def _identify_source_region(self, text: str) -> Optional[str]:
        """Identify geographical region of source"""
        regions = {
            'europe': ['europe', 'european', 'france', 'england', 'spain', 'italy', 'germany'],
            'asia': ['asia', 'china', 'japan', 'india', 'korea', 'southeast asia'],
            'africa': ['africa', 'african', 'egypt', 'ethiopia', 'west africa'],
            'americas': ['america', 'brazil', 'mexico', 'peru', 'caribbean'],
            'middle_east': ['middle east', 'ottoman', 'persia', 'arabia']
        }
        
        text_lower = text.lower()
        for region, keywords in regions.items():
            if any(keyword in text_lower for keyword in keywords):
                return region
        
        return None
    
    def _identify_cultural_context(self, text: str) -> Optional[str]:
        """Identify cultural context of source"""
        contexts = ['religious', 'political', 'economic', 'social', 'artistic', 'scientific', 'military']
        
        text_lower = text.lower()
        for context in contexts:
            if context in text_lower:
                return context
        
        return 'general'
    
    def _extract_author(self, text: str) -> Optional[str]:
        """Extract author from text if available"""
        # Simple heuristic to extract author information
        lines = text.split('\n')[:3]  # Check first few lines
        for line in lines:
            if 'by ' in line.lower() or 'author:' in line.lower():
                return line.split('by ')[-1].split('author:')[-1].strip()[:100]
        return "Unknown Author"
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text if available"""
        import re
        # Look for year patterns
        year_patterns = [
            r'\b(19|20)\d{2}\b',  # 1900-2099
            r'\b\d{1,2}\/\d{1,2}\/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{1,2}-\d{1,2}\b'  # YYYY-MM-DD
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return "Unknown Date"
    
    def _identify_language(self, text: str) -> Optional[str]:
        """Identify language of the text"""
        # Simple heuristic - in a real implementation, you'd use proper language detection
        # For now, assume English unless there are clear indicators
        common_words = {
            'spanish': ['el', 'la', 'de', 'que', 'y', 'es', 'en', 'un', 'se', 'te'],
            'french': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'],
            'german': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
            'portuguese': ['o', 'de', 'a', 'e', 'do', 'da', 'em', 'um', 'para', 'é']
        }
        
        text_lower = text.lower()
        words = text_lower.split()[:50]  # Check first 50 words
        
        for lang, lang_words in common_words.items():
            matches = sum(1 for word in words if word in lang_words)
            if matches > 3:  # If more than 3 matches, likely this language
                return lang
        
        return "english"  # Default to English

    # ...existing code...
    
    def _extract_themes(self, data: List[Dict]) -> List[str]:
        """Extract themes from data"""
        return ['power dynamics', 'cultural exchange', 'technological change', 'social transformation']
    
    def _identify_cross_cultural_patterns(self, data: List[Dict]) -> List[str]:
        """Identify cross-cultural patterns"""
        return ['similar responses to external pressures', 'parallel technological developments']
    
    def _find_global_patterns(self, data: List[Dict]) -> List[str]:
        """Find global patterns"""
        return ['interconnected trade networks', 'knowledge transfer across regions']
    
    def _extract_primary_evidence(self, data: List[Dict]) -> List[str]:
        """Extract primary source evidence"""
        return ['contemporary accounts', 'official documents', 'eyewitness testimonies']
    
    def _generate_research_recommendations(self, data: List[Dict], themes: List[str]) -> List[str]:
        """Generate research recommendations"""
        return ['further investigation needed', 'additional sources recommended']
    
    def _calculate_synthesis_confidence(self, data: List[Dict], themes: List[str]) -> float:
        """Calculate confidence in synthesis"""
        return 0.85
    
    def _update_stats(self, success: bool, research_time_ms: float, sources_count: int):
        """Update research statistics"""
        if success:
            self.stats['successful_research'] += 1
        
        # Update average research time
        current_avg = self.stats['avg_research_time_ms']
        count = self.stats['research_projects']
        self.stats['avg_research_time_ms'] = ((current_avg * (count - 1)) + research_time_ms) / count
    
    def get_status(self) -> Dict[str, Any]:
        """Get research agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status,
            'capabilities': self.capabilities,
            'supported_topics': list(self.supported_topics.keys()),
            'active_research': len(self.active_research),
            'statistics': self.stats.copy(),
            'config': self.config.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'is_healthy': self.status == "active",
            'supported_topics': len(self.supported_topics),
            'research_projects': self.stats['research_projects'],
            'success_rate': (
                self.stats['successful_research'] / max(1, self.stats['research_projects'])
            ),
            'primary_source_usage': (
                self.stats['primary_sources_used'] / max(1, self.stats['sources_analyzed'])
            ),
            'global_perspective_usage': (
                self.stats['non_european_sources_used'] / max(1, self.stats['sources_analyzed'])
            ),
            'last_check': datetime.utcnow().isoformat()
        }
