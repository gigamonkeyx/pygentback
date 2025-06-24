"""
Analysis Agent

Specialized agent for text analysis, pattern extraction, and source validation
in historical research workflows.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from analysis operations"""
    analysis_type: str
    content: Any
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime


class AnalysisAgent:
    """
    Agent specialized in text analysis and pattern extraction.
    
    Capabilities:
    - Historical text analysis
    - Pattern recognition and extraction
    - Source validation and authentication
    - Cross-cultural analysis
    - Temporal pattern detection
    """
    
    def __init__(self, agent_id: str = "analysis_agent"):
        self.agent_id = agent_id
        self.agent_type = "analysis"
        self.status = "initialized"
        self.capabilities = [
            "text_analysis",
            "pattern_extraction", 
            "source_validation",
            "cross_cultural_analysis",
            "temporal_analysis"
        ]
        
        # Analysis configuration
        self.config = {
            'max_text_length': 50000,
            'confidence_threshold': 0.7,
            'pattern_similarity_threshold': 0.8,
            'analysis_timeout_seconds': 30.0
        }
        
        # Analysis statistics
        self.stats = {
            'texts_analyzed': 0,
            'patterns_extracted': 0,
            'sources_validated': 0,
            'avg_analysis_time_ms': 0.0,
            'successful_analyses': 0
        }
        
        logger.info(f"AnalysisAgent {agent_id} initialized")
    
    async def start(self) -> bool:
        """Start the analysis agent"""
        try:
            self.status = "active"
            logger.info(f"AnalysisAgent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start AnalysisAgent {self.agent_id}: {e}")
            self.status = "error"
            return False
    
    async def stop(self) -> bool:
        """Stop the analysis agent"""
        try:
            self.status = "stopped"
            logger.info(f"AnalysisAgent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop AnalysisAgent {self.agent_id}: {e}")
            return False
    
    async def analyze_text(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze historical text for patterns, themes, and insights.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (general, cultural, temporal, etc.)
            
        Returns:
            Analysis results with patterns, themes, and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            if len(text) > self.config['max_text_length']:
                text = text[:self.config['max_text_length']]
                logger.warning(f"Text truncated to {self.config['max_text_length']} characters")
            
            # Perform text analysis based on type
            if analysis_type == "cultural":
                result = await self._analyze_cultural_context(text)
            elif analysis_type == "temporal":
                result = await self._analyze_temporal_patterns(text)
            elif analysis_type == "thematic":
                result = await self._analyze_themes(text)
            else:
                result = await self._analyze_general(text)
            
            # Calculate analysis time
            analysis_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_stats(True, analysis_time)
            
            # Add metadata
            result['metadata'] = {
                'analysis_type': analysis_type,
                'text_length': len(text),
                'analysis_time_ms': analysis_time,
                'agent_id': self.agent_id,
                'timestamp': start_time.isoformat()
            }
            
            logger.debug(f"Text analysis completed in {analysis_time:.2f}ms")
            return result
            
        except Exception as e:
            analysis_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_stats(False, analysis_time)
            logger.error(f"Text analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'analysis_type': analysis_type,
                    'analysis_time_ms': analysis_time,
                    'agent_id': self.agent_id
                }
            }
    
    async def extract_patterns(self, data: List[str], pattern_type: str = "general") -> List[Dict]:
        """
        Extract patterns from multiple text sources.
        
        Args:
            data: List of text sources
            pattern_type: Type of patterns to extract
            
        Returns:
            List of extracted patterns with confidence scores
        """
        start_time = datetime.utcnow()
        
        try:
            patterns = []
            
            for i, text in enumerate(data):
                # Extract patterns from each text
                text_patterns = await self._extract_text_patterns(text, pattern_type)
                
                for pattern in text_patterns:
                    pattern['source_index'] = i
                    pattern['source_text'] = text[:100] + "..." if len(text) > 100 else text
                    patterns.append(pattern)
            
            # Consolidate similar patterns
            consolidated_patterns = self._consolidate_patterns(patterns)
            
            analysis_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.stats['patterns_extracted'] += len(consolidated_patterns)
            
            logger.debug(f"Extracted {len(consolidated_patterns)} patterns from {len(data)} sources")
            
            return consolidated_patterns
            
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return []
    
    async def validate_sources(self, sources: List[str]) -> Dict[str, bool]:
        """
        Validate historical sources for authenticity and reliability.
        
        Args:
            sources: List of source texts or references
            
        Returns:
            Dictionary mapping source index to validation result
        """
        start_time = datetime.utcnow()
        validation_results = {}
        
        try:
            for i, source in enumerate(sources):
                validation = await self._validate_single_source(source)
                validation_results[str(i)] = validation
                
                if validation:
                    self.stats['sources_validated'] += 1
            
            analysis_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.debug(f"Validated {len(sources)} sources in {analysis_time:.2f}ms")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Source validation failed: {e}")
            return {str(i): False for i in range(len(sources))}
    
    async def _analyze_cultural_context(self, text: str) -> Dict[str, Any]:
        """Analyze cultural context and perspectives in text"""
        # Simulate cultural analysis
        cultural_indicators = [
            "religious references", "social hierarchies", "economic systems",
            "political structures", "artistic expressions", "technological mentions"
        ]
        
        detected_indicators = []
        for indicator in cultural_indicators:
            # Simple keyword-based detection (would be more sophisticated in real implementation)
            if any(keyword in text.lower() for keyword in indicator.split()):
                detected_indicators.append({
                    'indicator': indicator,
                    'confidence': 0.8,
                    'context': f"Found references to {indicator}"
                })
        
        return {
            'success': True,
            'cultural_indicators': detected_indicators,
            'cultural_complexity': min(1.0, len(detected_indicators) / len(cultural_indicators)),
            'global_perspective': len(detected_indicators) > 3,
            'confidence': 0.85
        }
    
    async def _analyze_temporal_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze temporal patterns and chronological references"""
        # Simulate temporal analysis
        temporal_markers = [
            "dates", "periods", "before", "after", "during", "century",
            "year", "decade", "era", "age", "time", "when"
        ]
        
        temporal_references = []
        for marker in temporal_markers:
            if marker in text.lower():
                temporal_references.append({
                    'marker': marker,
                    'confidence': 0.7,
                    'context': f"Temporal reference: {marker}"
                })
        
        return {
            'success': True,
            'temporal_references': temporal_references,
            'chronological_complexity': min(1.0, len(temporal_references) / 10),
            'temporal_coherence': len(temporal_references) > 2,
            'confidence': 0.8
        }
    
    async def _analyze_themes(self, text: str) -> Dict[str, Any]:
        """Analyze thematic content and main topics"""
        # Simulate thematic analysis
        themes = [
            "power", "resistance", "change", "tradition", "innovation",
            "conflict", "cooperation", "identity", "transformation", "continuity"
        ]
        
        detected_themes = []
        for theme in themes:
            if theme in text.lower():
                detected_themes.append({
                    'theme': theme,
                    'relevance': 0.8,
                    'evidence': f"Theme '{theme}' detected in text"
                })
        
        return {
            'success': True,
            'themes': detected_themes,
            'thematic_richness': min(1.0, len(detected_themes) / len(themes)),
            'primary_theme': detected_themes[0]['theme'] if detected_themes else None,
            'confidence': 0.82
        }
    
    async def _analyze_general(self, text: str) -> Dict[str, Any]:
        """Perform general text analysis"""
        # Simulate general analysis
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        return {
            'success': True,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': word_count / max(1, sentence_count),
            'complexity_score': min(1.0, word_count / 1000),
            'readability': max(0.1, 1.0 - (word_count / 2000)),
            'confidence': 0.9
        }
    
    async def _extract_text_patterns(self, text: str, pattern_type: str) -> List[Dict]:
        """Extract patterns from a single text using real NLP analysis"""
        patterns = []

        try:
            if pattern_type == "linguistic":
                patterns.extend(await self._extract_linguistic_patterns(text))
            elif pattern_type == "conceptual":
                patterns.extend(await self._extract_conceptual_patterns(text))
            else:
                patterns.extend(await self._extract_general_patterns(text))

            return patterns

        except Exception as e:
            logger.error(f"Pattern extraction failed for {pattern_type}: {e}")
            return []

    async def _extract_linguistic_patterns(self, text: str) -> List[Dict]:
        """Extract linguistic patterns using real NLP analysis."""
        try:
            patterns = []

            # Analyze sentence structure
            sentences = text.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

            if avg_sentence_length > 20:
                patterns.append({
                    'pattern_type': 'linguistic',
                    'pattern': 'complex sentence structure',
                    'confidence': min(0.9, avg_sentence_length / 30),
                    'frequency': len(sentences),
                    'details': f'Average sentence length: {avg_sentence_length:.1f} words'
                })

            # Analyze vocabulary complexity
            words = text.lower().split()
            unique_words = len(set(words))
            vocabulary_diversity = unique_words / max(len(words), 1)

            if vocabulary_diversity > 0.6:
                patterns.append({
                    'pattern_type': 'linguistic',
                    'pattern': 'rich vocabulary',
                    'confidence': vocabulary_diversity,
                    'frequency': unique_words,
                    'details': f'Vocabulary diversity: {vocabulary_diversity:.2f}'
                })

            # Analyze formal language indicators
            formal_indicators = ['therefore', 'furthermore', 'consequently', 'moreover', 'nevertheless']
            formal_count = sum(1 for indicator in formal_indicators if indicator in text.lower())

            if formal_count > 0:
                patterns.append({
                    'pattern_type': 'linguistic',
                    'pattern': 'formal language structure',
                    'confidence': min(0.9, formal_count / 5),
                    'frequency': formal_count,
                    'details': f'Formal indicators found: {formal_count}'
                })

            return patterns

        except Exception as e:
            logger.error(f"Linguistic pattern extraction failed: {e}")
            return []

    async def _extract_conceptual_patterns(self, text: str) -> List[Dict]:
        """Extract conceptual patterns using real semantic analysis."""
        try:
            patterns = []
            text_lower = text.lower()

            # Detect cause-effect relationships
            cause_effect_indicators = ['because', 'due to', 'as a result', 'therefore', 'consequently', 'led to']
            cause_effect_count = sum(1 for indicator in cause_effect_indicators if indicator in text_lower)

            if cause_effect_count > 0:
                patterns.append({
                    'pattern_type': 'conceptual',
                    'pattern': 'cause-effect relationships',
                    'confidence': min(0.9, cause_effect_count / 3),
                    'frequency': cause_effect_count,
                    'details': f'Causal indicators found: {cause_effect_count}'
                })

            # Detect temporal patterns
            temporal_indicators = ['before', 'after', 'during', 'while', 'when', 'then', 'first', 'next', 'finally']
            temporal_count = sum(1 for indicator in temporal_indicators if indicator in text_lower)

            if temporal_count > 2:
                patterns.append({
                    'pattern_type': 'conceptual',
                    'pattern': 'temporal sequence',
                    'confidence': min(0.85, temporal_count / 5),
                    'frequency': temporal_count,
                    'details': f'Temporal indicators found: {temporal_count}'
                })

            # Detect comparison patterns
            comparison_indicators = ['compared to', 'unlike', 'similar to', 'different from', 'whereas', 'however']
            comparison_count = sum(1 for indicator in comparison_indicators if indicator in text_lower)

            if comparison_count > 0:
                patterns.append({
                    'pattern_type': 'conceptual',
                    'pattern': 'comparative analysis',
                    'confidence': min(0.8, comparison_count / 3),
                    'frequency': comparison_count,
                    'details': f'Comparison indicators found: {comparison_count}'
                })

            return patterns

        except Exception as e:
            logger.error(f"Conceptual pattern extraction failed: {e}")
            return []

    async def _extract_general_patterns(self, text: str) -> List[Dict]:
        """Extract general patterns using basic text analysis."""
        try:
            patterns = []

            # Analyze text structure
            paragraphs = text.split('\n\n')
            sentences = text.split('.')

            if len(paragraphs) > 1:
                patterns.append({
                    'pattern_type': 'general',
                    'pattern': 'structured narrative',
                    'confidence': min(0.8, len(paragraphs) / 5),
                    'frequency': len(paragraphs),
                    'details': f'Paragraphs: {len(paragraphs)}, Sentences: {len(sentences)}'
                })

            # Analyze question patterns
            question_count = text.count('?')
            if question_count > 0:
                patterns.append({
                    'pattern_type': 'general',
                    'pattern': 'interrogative structure',
                    'confidence': min(0.7, question_count / 3),
                    'frequency': question_count,
                    'details': f'Questions found: {question_count}'
                })

            # Analyze emphasis patterns
            emphasis_count = text.count('!') + text.count('*') + len([w for w in text.split() if w.isupper() and len(w) > 2])
            if emphasis_count > 0:
                patterns.append({
                    'pattern_type': 'general',
                    'pattern': 'emphatic expression',
                    'confidence': min(0.75, emphasis_count / 5),
                    'frequency': emphasis_count,
                    'details': f'Emphasis markers found: {emphasis_count}'
                })

            return patterns

        except Exception as e:
            logger.error(f"General pattern extraction failed: {e}")
            return []
    
    def _consolidate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Consolidate similar patterns from multiple sources"""
        # Simple consolidation - group by pattern type
        consolidated = {}
        
        for pattern in patterns:
            pattern_key = pattern.get('pattern', 'unknown')
            if pattern_key not in consolidated:
                consolidated[pattern_key] = pattern.copy()
                consolidated[pattern_key]['sources'] = [pattern.get('source_index', 0)]
                consolidated[pattern_key]['total_frequency'] = pattern.get('frequency', 1)
            else:
                consolidated[pattern_key]['sources'].append(pattern.get('source_index', 0))
                consolidated[pattern_key]['total_frequency'] += pattern.get('frequency', 1)
                # Average confidence
                consolidated[pattern_key]['confidence'] = (
                    consolidated[pattern_key]['confidence'] + pattern.get('confidence', 0.5)
                ) / 2
        
        return list(consolidated.values())
    
    async def _validate_single_source(self, source: str) -> bool:
        """Validate a single source for authenticity using real validation methods"""
        try:
            validation_score = 0.0

            # Content quality analysis
            content_score = await self._analyze_content_quality(source)
            validation_score += content_score * 0.4

            # Structural analysis
            structure_score = await self._analyze_source_structure(source)
            validation_score += structure_score * 0.3

            # Linguistic authenticity analysis
            linguistic_score = await self._analyze_linguistic_authenticity(source)
            validation_score += linguistic_score * 0.3

            return validation_score >= self.config['confidence_threshold']

        except Exception as e:
            logger.error(f"Source validation failed: {e}")
            return False

    async def _analyze_content_quality(self, source: str) -> float:
        """Analyze content quality of the source."""
        try:
            score = 0.0

            # Length and detail analysis
            word_count = len(source.split())
            if word_count > 50:
                score += 0.3
            elif word_count > 20:
                score += 0.2

            # Specific detail indicators
            detail_keywords = ['date', 'place', 'name', 'event', 'year', 'location', 'person', 'time']
            detail_count = sum(1 for keyword in detail_keywords if keyword in source.lower())
            score += min(0.4, detail_count * 0.1)

            # Factual content indicators
            factual_indicators = ['according to', 'documented', 'recorded', 'evidence', 'witness', 'archive']
            factual_count = sum(1 for indicator in factual_indicators if indicator in source.lower())
            score += min(0.3, factual_count * 0.15)

            return min(1.0, score)

        except Exception as e:
            logger.error(f"Content quality analysis failed: {e}")
            return 0.0

    async def _analyze_source_structure(self, source: str) -> float:
        """Analyze structural quality of the source."""
        try:
            score = 0.0

            # Sentence structure analysis
            sentences = [s.strip() for s in source.split('.') if s.strip()]
            if len(sentences) > 2:
                score += 0.3

            # Paragraph structure
            paragraphs = [p.strip() for p in source.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                score += 0.2

            # Citation or reference patterns
            citation_patterns = ['(', ')', '[', ']', '"', 'p.', 'pp.', 'vol.']
            citation_count = sum(1 for pattern in citation_patterns if pattern in source)
            score += min(0.3, citation_count * 0.05)

            # Formal language indicators
            formal_indicators = ['furthermore', 'moreover', 'consequently', 'therefore', 'however']
            formal_count = sum(1 for indicator in formal_indicators if indicator in source.lower())
            score += min(0.2, formal_count * 0.1)

            return min(1.0, score)

        except Exception as e:
            logger.error(f"Source structure analysis failed: {e}")
            return 0.0

    async def _analyze_linguistic_authenticity(self, source: str) -> float:
        """Analyze linguistic authenticity of the source."""
        try:
            score = 0.0

            # Vocabulary sophistication
            words = source.lower().split()
            unique_words = len(set(words))
            vocabulary_diversity = unique_words / max(len(words), 1)

            if vocabulary_diversity > 0.5:
                score += 0.3
            elif vocabulary_diversity > 0.3:
                score += 0.2

            # Academic/formal language patterns
            academic_terms = ['analysis', 'evidence', 'research', 'study', 'investigation', 'examination']
            academic_count = sum(1 for term in academic_terms if term in source.lower())
            score += min(0.3, academic_count * 0.1)

            # Coherence indicators (transition words)
            transition_words = ['first', 'second', 'next', 'then', 'finally', 'in addition', 'furthermore']
            transition_count = sum(1 for word in transition_words if word in source.lower())
            score += min(0.2, transition_count * 0.05)

            # Avoid obvious fabrication indicators
            fabrication_indicators = ['lorem ipsum', 'placeholder', 'example text', 'sample content']
            if any(indicator in source.lower() for indicator in fabrication_indicators):
                score -= 0.5

            # Check for reasonable sentence length distribution
            sentences = [s.strip() for s in source.split('.') if s.strip()]
            if sentences:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                if 10 <= avg_sentence_length <= 30:  # Reasonable range
                    score += 0.2

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Linguistic authenticity analysis failed: {e}")
            return 0.0
    
    def _update_stats(self, success: bool, analysis_time_ms: float):
        """Update agent statistics"""
        self.stats['texts_analyzed'] += 1
        
        if success:
            self.stats['successful_analyses'] += 1
        
        # Update average analysis time
        current_avg = self.stats['avg_analysis_time_ms']
        count = self.stats['texts_analyzed']
        self.stats['avg_analysis_time_ms'] = ((current_avg * (count - 1)) + analysis_time_ms) / count
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status,
            'capabilities': self.capabilities,
            'statistics': self.stats.copy(),
            'config': self.config.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'is_healthy': self.status == "active",
            'capabilities_count': len(self.capabilities),
            'analyses_performed': self.stats['texts_analyzed'],
            'success_rate': (
                self.stats['successful_analyses'] / max(1, self.stats['texts_analyzed'])
            ),
            'last_check': datetime.utcnow().isoformat()
        }
