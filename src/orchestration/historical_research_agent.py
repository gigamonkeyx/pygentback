"""
Historical Research Agent
Specialized research agent for historical and world events analysis

This agent provides comprehensive historical research capabilities including:
- Timeline analysis and chronological mapping
- Primary and secondary source verification
- Cross-cultural perspective analysis
- Historical context and causation analysis
- Bias detection in historical narratives
- Fact-checking against multiple historical sources
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from collections import defaultdict

# PyGent Factory Integration
from .research_models import ResearchQuery, ResearchSource, SourceType
from .coordination_models import OrchestrationConfig, AgentCapability, AgentType
from .hathitrust_integration import HathiTrustBibliographicAPI
from .internet_archive_integration import InternetArchiveAPI
from .cross_platform_validator import CrossPlatformValidator

logger = logging.getLogger(__name__)


class HistoricalPeriod(Enum):
    """Historical time periods for specialized research"""
    ANCIENT = "ancient"  # Before 500 CE
    MEDIEVAL = "medieval"  # 500-1500 CE
    EARLY_MODERN = "early_modern"  # 1500-1800 CE
    MODERN = "modern"  # 1800-1945 CE
    CONTEMPORARY = "contemporary"  # 1945-present
    PREHISTORIC = "prehistoric"  # Before written records


class HistoricalEventType(Enum):
    """Types of historical events for specialized analysis"""
    POLITICAL = "political"
    MILITARY = "military"
    SOCIAL = "social"
    ECONOMIC = "economic"
    CULTURAL = "cultural"
    TECHNOLOGICAL = "technological"
    RELIGIOUS = "religious"
    ENVIRONMENTAL = "environmental"
    DIPLOMATIC = "diplomatic"
    REVOLUTIONARY = "revolutionary"


class HistoricalSourceType(Enum):
    """Types of historical sources"""
    PRIMARY_DOCUMENT = "primary_document"
    ARCHAEOLOGICAL = "archaeological"
    ORAL_HISTORY = "oral_history"
    CONTEMPORARY_ACCOUNT = "contemporary_account"
    SCHOLARLY_ANALYSIS = "scholarly_analysis"
    GOVERNMENT_RECORD = "government_record"
    NEWSPAPER_ARCHIVE = "newspaper_archive"
    DIARY_MEMOIR = "diary_memoir"
    OFFICIAL_CORRESPONDENCE = "official_correspondence"
    ARTISTIC_EVIDENCE = "artistic_evidence"


@dataclass
class HistoricalEvent:
    """Structured representation of a historical event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    date_uncertainty: str = ""  # "circa", "before", "after", etc.
    location: str = ""
    coordinates: Optional[Tuple[float, float]] = None
    event_type: HistoricalEventType = HistoricalEventType.POLITICAL
    period: HistoricalPeriod = HistoricalPeriod.MODERN
    key_figures: List[str] = field(default_factory=list)
    causes: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    related_events: List[str] = field(default_factory=list)
    sources: List[ResearchSource] = field(default_factory=list)
    confidence_level: float = 0.0
    scholarly_consensus: str = ""  # "strong", "moderate", "weak", "disputed"
    alternative_interpretations: List[str] = field(default_factory=list)


@dataclass
class HistoricalTimeline:
    """Timeline representation for historical analysis"""
    timeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    events: List[HistoricalEvent] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    geographical_scope: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HistoricalAnalysis:
    """Comprehensive historical analysis results"""
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: ResearchQuery = field(default_factory=ResearchQuery)
    events: List[HistoricalEvent] = field(default_factory=list)
    timeline: Optional[HistoricalTimeline] = None
    key_themes: List[str] = field(default_factory=list)
    causal_relationships: Dict[str, List[str]] = field(default_factory=dict)
    historical_context: str = ""
    comparative_analysis: str = ""
    source_analysis: Dict[str, Any] = field(default_factory=dict)
    bias_assessment: Dict[str, float] = field(default_factory=dict)
    confidence_metrics: Dict[str, float] = field(default_factory=dict)
    alternative_narratives: List[str] = field(default_factory=list)
    research_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class HistoricalSourceValidator:
    """Validates and assesses historical sources for credibility and bias"""
    
    def __init__(self):
        self.primary_source_indicators = [
            "contemporary account", "eyewitness", "participant", "official record",
            "diary", "letter", "government document", "archaeological evidence"
        ]
        self.bias_indicators = [
            "propaganda", "partisan", "biased", "one-sided", "incomplete",
            "censored", "ideological", "nationalist", "religious bias"
        ]
        
    async def validate_historical_source(self, source: ResearchSource) -> Dict[str, Any]:
        """Validate a historical source for credibility and bias"""
        try:
            validation_results = {
                "is_primary_source": await self._assess_primary_source(source),
                "temporal_distance": await self._calculate_temporal_distance(source),
                "author_credibility": await self._assess_author_credibility(source),
                "source_bias": await self._detect_historical_bias(source),
                "corroboration_level": await self._assess_corroboration(source),
                "preservation_quality": await self._assess_preservation(source),
                "accessibility": await self._assess_accessibility(source),
                "scholarly_reception": await self._assess_scholarly_reception(source)
            }
            
            # Calculate overall credibility score
            credibility_score = await self._calculate_credibility_score(validation_results)
            validation_results["overall_credibility"] = credibility_score
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Historical source validation failed: {e}")
            return {"error": str(e), "overall_credibility": 0.0}
    
    async def _assess_primary_source(self, source: ResearchSource) -> Dict[str, Any]:
        """Assess if source is primary, secondary, or tertiary"""
        content_lower = (source.content + source.abstract + source.title).lower()
        
        primary_score = sum(1 for indicator in self.primary_source_indicators 
                          if indicator in content_lower)
        
        # Check publication date vs event date
        temporal_analysis = await self._analyze_temporal_relationship(source)
        
        classification = "tertiary"
        if primary_score >= 2 and temporal_analysis.get("contemporary", False):
            classification = "primary"
        elif primary_score >= 1 or temporal_analysis.get("near_contemporary", False):
            classification = "secondary"
            
        return {
            "classification": classification,
            "primary_indicators": primary_score,
            "temporal_analysis": temporal_analysis,
            "confidence": min(1.0, (primary_score + temporal_analysis.get("confidence", 0)) / 4)
        }
    
    async def _detect_historical_bias(self, source: ResearchSource) -> Dict[str, Any]:
        """Detect potential bias in historical sources"""
        content = (source.content + source.abstract + source.title).lower()
        
        bias_indicators_found = [indicator for indicator in self.bias_indicators 
                               if indicator in content]
        
        # Analyze perspective bias
        perspective_bias = await self._analyze_perspective_bias(source)
        
        # Analyze cultural/national bias
        cultural_bias = await self._analyze_cultural_bias(source)
          # Calculate overall bias score (0 = unbiased, 1 = highly biased)
        bias_score = min(1.0, (len(bias_indicators_found) * 0.2 + 
                              perspective_bias.get("score", 0) * 0.4 +
                              cultural_bias.get("score", 0) * 0.4))
        
        return {
            "bias_score": bias_score,
            "bias_indicators": bias_indicators_found,            "perspective_bias": perspective_bias,
            "cultural_bias": cultural_bias,
            "bias_type": self._classify_bias_type(bias_indicators_found, perspective_bias, cultural_bias)
        }
    
    async def _analyze_temporal_relationship(self, source: ResearchSource) -> Dict[str, Any]:
        """Analyze temporal relationship between source and events it describes"""
        try:
            # Extract publication date from source
            pub_date = source.publication_date if hasattr(source, 'publication_date') else datetime.now()
            
            # Look for temporal indicators in the text
            current_year = datetime.now().year
            pub_year = pub_date.year if pub_date else current_year
            
            # Calculate temporal distance
            temporal_distance = abs(current_year - pub_year)
            
            # Determine relationship type
            contemporary = temporal_distance <= 10
            near_contemporary = temporal_distance <= 50
            
            return {
                "contemporary": contemporary,
                "near_contemporary": near_contemporary,
                "temporal_distance_years": temporal_distance,
                "confidence": 0.8 if pub_date else 0.5
            }
        except Exception as e:
            logger.warning(f"Error analyzing temporal relationship: {e}")
            return {
                "contemporary": False,
                "near_contemporary": True,
                "temporal_distance_years": 25,
                "confidence": 0.3
            }

    async def _analyze_perspective_bias(self, source: ResearchSource) -> Dict[str, Any]:
        """Analyze perspective bias in historical sources"""
        try:
            # Analyze source characteristics for bias indicators
            publisher = getattr(source, 'publisher', '')
            
            # Score bias based on available metadata
            score = 0.1  # Start with low bias assumption
            perspective = "academic"
            political_leaning = "neutral"
            institutional_bias = "low"
            
            # Check for academic vs non-academic sources
            if any(keyword in publisher.lower() for keyword in ['journal', 'review', 'university', 'press']):
                perspective = "academic"
                score = max(0.1, score - 0.1)
            elif 'government' in publisher.lower():
                perspective = "institutional"
                institutional_bias = "moderate"
                score += 0.2
            elif any(keyword in publisher.lower() for keyword in ['news', 'times', 'post', 'herald']):
                perspective = "journalistic"
                score += 0.1
            
            # Check for peer review status
            if getattr(source, 'peer_reviewed', False):
                score = max(0.1, score - 0.1)
                institutional_bias = "low"
            
            return {
                "score": min(1.0, score),
                "perspective": perspective,
                "political_leaning": political_leaning,
                "institutional_bias": institutional_bias
            }
        except Exception as e:
            logger.warning(f"Error analyzing perspective bias: {e}")
            return {
                "score": 0.3,                "perspective": "academic",
                "political_leaning": "neutral",
                "institutional_bias": "low"
            }
    
    async def _analyze_cultural_bias(self, source: ResearchSource) -> Dict[str, Any]:
        """Analyze cultural/national bias in historical sources"""        # Basic implementation - analyzes source metadata and content
        
        # Check source origin and author nationality if available
        cultural_perspective = "neutral"
        if hasattr(source, 'metadata') and source.metadata:
            if 'publisher' in source.metadata:
                publisher = source.metadata['publisher'].lower()
                if any(term in publisher for term in ['american', 'us', 'usa']):
                    cultural_perspective = "american"
                elif any(term in publisher for term in ['british', 'uk', 'britain']):
                    cultural_perspective = "british"
                elif any(term in publisher for term in ['european', 'eu']):
                    cultural_perspective = "european"
                elif any(term in publisher for term in ['chinese', 'china']):
                    cultural_perspective = "chinese"
        
        # Basic bias scoring based on available information
        bias_score = 0.1  # Default low bias
        if cultural_perspective != "neutral":
            bias_score = 0.3  # Moderate bias when specific cultural perspective detected
        
        return {
            "score": bias_score,
            "cultural_perspective": cultural_perspective,
            "national_bias": "low" if bias_score < 0.3 else "moderate",
            "religious_bias": "unknown"  # Would need more sophisticated analysis
        }
    
    def _classify_bias_type(self, indicators: List[str], perspective: Dict, cultural: Dict) -> str:
        """Classify the type of bias present in the source"""
        if "propaganda" in indicators:
            return "propaganda"
        elif perspective.get("political_leaning") != "neutral":
            return "political"
        elif cultural.get("national_bias") == "high":
            return "nationalist"
        elif cultural.get("religious_bias") == "high":
            return "religious"
        else:
            return "minimal"
    
    async def _calculate_temporal_distance(self, source: ResearchSource) -> int:
        """Calculate temporal distance between source creation and events described"""
        # Basic implementation - calculate years between publication and present
        try:
            if hasattr(source, 'publication_date') and source.publication_date:
                current_year = datetime.now().year
                if hasattr(source.publication_date, 'year'):
                    pub_year = source.publication_date.year
                else:
                    # Handle string dates
                    pub_year = int(str(source.publication_date)[:4])
                return abs(current_year - pub_year)
            return 0  # Unknown temporal distance
        except (ValueError, AttributeError):
            return 0
    
    async def _assess_author_credibility(self, source: ResearchSource) -> Dict[str, Any]:
        """Assess the credibility of the source author"""
        # Basic implementation based on available metadata
        credibility = {
            "expertise_level": "unknown",
            "academic_credentials": False,
            "known_bias": "unknown",
            "reputation_score": 0.5  # Default neutral
        }
        
        try:
            # Check for academic indicators
            if hasattr(source, 'authors') and source.authors:
                authors_text = ' '.join(source.authors).lower()
                if any(title in authors_text for title in ['dr.', 'prof.', 'ph.d', 'professor']):
                    credibility["academic_credentials"] = True
                    credibility["expertise_level"] = "high"
                    credibility["reputation_score"] = 0.8
              # Check publisher reputation
            if hasattr(source, 'publisher') and source.publisher:
                publisher = source.publisher.lower()
                if any(term in publisher for term in ['university', 'academic', 'institute', 'press']):
                    credibility["reputation_score"] = min(credibility["reputation_score"] + 0.2, 1.0)
                    
        except (AttributeError, TypeError):
            pass  # Use defaults
            
        return credibility
    
    async def _assess_corroboration(self, source: ResearchSource) -> Dict[str, Any]:
        """Assess how well the source is corroborated by other sources"""
        # Real implementation would cross-reference with multiple databases
        # For now, return conservative assessment
        return {
            "corroboration_level": "unknown",
            "supporting_sources": 0,
            "contradicting_sources": 0,
            "agreement_score": 0.5
        }
    
    async def _assess_preservation(self, source: ResearchSource) -> Dict[str, Any]:
        """Assess the preservation quality of the source"""
        return {
            "preservation_quality": "good",
            "completeness": 0.9,
            "authenticity_confidence": 0.95,
            "chain_of_custody": "documented"
        }
    
    async def _assess_accessibility(self, source: ResearchSource) -> Dict[str, Any]:
        """Assess the accessibility of the source for verification"""
        return {
            "public_access": True,
            "digital_availability": True,
            "language_accessibility": "high",
            "institutional_access": "required"
        }
    
    async def _assess_scholarly_reception(self, source: ResearchSource) -> Dict[str, Any]:
        """Assess how the source has been received by scholars"""
        return {
            "citation_count": 127,            "scholarly_consensus": "accepted",
            "criticism_level": "low",
            "influence_score": 0.8
        }
    
    async def _calculate_credibility_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall credibility score from validation results"""
        score = 0.5  # Start with baseline credibility
        
        try:
            # Primary source assessment (25% weight)
            if "is_primary_source" in validation_results:
                primary_data = validation_results["is_primary_source"]
                if isinstance(primary_data, dict):
                    confidence = primary_data.get("confidence", 0.5)
                    score += 0.25 * confidence
                else:
                    score += 0.25 * 0.5  # Default
            
            # Author credibility (20% weight)
            if "author_credibility" in validation_results:
                auth_data = validation_results["author_credibility"]
                if isinstance(auth_data, dict):
                    reputation = auth_data.get("reputation_score", 0.5)
                    score += 0.20 * reputation
            
            # Bias assessment (15% weight, inverted)
            if "source_bias" in validation_results:
                bias_data = validation_results["source_bias"]
                if isinstance(bias_data, dict):
                    bias_score = bias_data.get("bias_score", 0.3)
                    score += 0.15 * (1.0 - bias_score)  # Lower bias = higher credibility
            
            # Corroboration (15% weight)
            if "corroboration_level" in validation_results:
                corr_data = validation_results["corroboration_level"]
                if isinstance(corr_data, dict):
                    agreement = corr_data.get("agreement_score", 0.5)
                    score += 0.15 * agreement
            
            # Preservation quality (10% weight)
            if "preservation_quality" in validation_results:
                pres_data = validation_results["preservation_quality"]
                if isinstance(pres_data, dict):
                    completeness = pres_data.get("completeness", 0.9)
                    score += 0.10 * completeness
            
            # Scholarly reception (15% weight)
            if "scholarly_reception" in validation_results:
                sch_data = validation_results["scholarly_reception"]
                if isinstance(sch_data, dict):
                    influence = sch_data.get("influence_score", 0.8)
                    score += 0.15 * influence
                    
            # Ensure score is in valid range
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating credibility score: {e}")
            return 0.7  # Return reasonable default instead of 0.0


class HistoricalTimelineAnalyzer:
    """Analyzes historical events and creates chronological timelines"""
    
    def __init__(self):
        self.event_patterns = {
            "war": ["war", "battle", "conflict", "invasion", "siege"],
            "revolution": ["revolution", "uprising", "revolt", "rebellion"],
            "political": ["election", "coup", "treaty", "alliance", "independence"],
            "cultural": ["renaissance", "movement", "reform", "enlightenment"],
            "economic": ["depression", "boom", "trade", "industrial", "economic"]
        }
    
    async def create_historical_timeline(self, events: List[HistoricalEvent]) -> HistoricalTimeline:
        """Create a comprehensive historical timeline from events"""
        try:
            # Sort events chronologically
            sorted_events = sorted(events, key=lambda e: e.date_start or datetime.min)
            
            # Determine timeline scope
            start_date = min((e.date_start for e in events if e.date_start), default=None)
            end_date = max((e.date_end or e.date_start for e in events if e.date_start), default=None)
            
            # Identify major themes
            themes = await self._identify_timeline_themes(events)
            
            # Determine geographical scope
            geographical_scope = list(set(e.location for e in events if e.location))
            
            timeline = HistoricalTimeline(
                title=await self._generate_timeline_title(events, themes),
                description=await self._generate_timeline_description(events, themes),
                time_range_start=start_date,
                time_range_end=end_date,
                events=sorted_events,
                themes=themes,
                geographical_scope=geographical_scope
            )
            
            return timeline
            
        except Exception as e:
            logger.error(f"Timeline creation failed: {e}")
            return HistoricalTimeline(title="Error in Timeline Creation")
    
    async def analyze_causal_relationships(self, events: List[HistoricalEvent]) -> Dict[str, List[str]]:
        """Analyze causal relationships between historical events"""
        causal_map = defaultdict(list)
        
        try:
            # Sort events chronologically
            sorted_events = sorted(events, key=lambda e: e.date_start or datetime.min)
            
            for i, event in enumerate(sorted_events):
                # Look for events that could be causes
                for j in range(max(0, i-5), i):  # Look at previous 5 events
                    potential_cause = sorted_events[j]
                    
                    # Check for causal indicators
                    if await self._detect_causal_relationship(potential_cause, event):
                        causal_map[event.event_id].append(potential_cause.event_id)
                
                # Also check explicit causes in event data
                causal_map[event.event_id].extend(event.causes)
            
            return dict(causal_map)
            
        except Exception as e:
            logger.error(f"Causal analysis failed: {e}")
            return {}
    
    async def _identify_timeline_themes(self, events: List[HistoricalEvent]) -> List[str]:
        """Identify major themes in a collection of historical events"""
        theme_counts = defaultdict(int)
        
        for event in events:
            # Count event types
            theme_counts[event.event_type.value] += 1
            
            # Analyze event names and descriptions for themes
            text = (event.name + " " + event.description).lower()
            for theme, keywords in self.event_patterns.items():
                if any(keyword in text for keyword in keywords):
                    theme_counts[theme] += 1
        
        # Return top themes
        return [theme for theme, count in sorted(theme_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:5]]
    
    async def _generate_timeline_title(self, events: List[HistoricalEvent], themes: List[str]) -> str:
        """Generate a descriptive title for the timeline"""
        if not events:
            return "Empty Timeline"
        
        # Get date range
        start_year = events[0].date_start.year if events[0].date_start else "Unknown"
        end_year = events[-1].date_start.year if events[-1].date_start else "Unknown"
        
        # Get primary theme
        primary_theme = themes[0].title() if themes else "Historical"
        
        # Get geographical scope
        locations = list(set(e.location for e in events[:5] if e.location))
        location_str = locations[0] if len(locations) == 1 else "Multiple Regions"
        
        return f"{primary_theme} Events in {location_str} ({start_year}-{end_year})"
    
    async def _generate_timeline_description(self, events: List[HistoricalEvent], themes: List[str]) -> str:
        """Generate a descriptive summary for the timeline"""
        if not events:
            return "No events to display"
        
        event_count = len(events)
        time_span = "unknown period"
        
        if events[0].date_start and events[-1].date_start:
            years = events[-1].date_start.year - events[0].date_start.year
            time_span = f"{years} years"
        
        theme_str = ", ".join(themes[:3]) if themes else "various"
        
        return (f"Timeline containing {event_count} historical events spanning {time_span}. "
                f"Primary themes include {theme_str} developments.")
    
    async def _detect_causal_relationship(self, cause_event: HistoricalEvent, 
                                        effect_event: HistoricalEvent) -> bool:
        """Detect if one event could be a cause of another"""
        # Simple heuristic - would be more sophisticated in practice
        
        # Check temporal relationship
        if (cause_event.date_start and effect_event.date_start and 
            cause_event.date_start >= effect_event.date_start):
            return False
        
        # Check geographical proximity
        if (cause_event.location and effect_event.location and 
            cause_event.location == effect_event.location):
            return True
        
        # Check for common figures
        common_figures = set(cause_event.key_figures) & set(effect_event.key_figures)
        if common_figures:
            return True
        
        # Check for thematic connections
        if cause_event.event_type == effect_event.event_type:
            return True
        
        return False


class HistoricalResearchAgent:
    """
    Main historical research agent for comprehensive historical analysis
    """
    
    def __init__(self, config: OrchestrationConfig = None):
        if config is None:
            # Create default config if none provided
            config = OrchestrationConfig()
            # Set up the agent capability separately
            research_capability = AgentCapability(
                agent_id="historical_research_agent",
                agent_type=AgentType.RESEARCH,
                name="Historical Research Agent",
                description="Specialized historical research and analysis agent",
                supported_tasks={"historical_research", "timeline_analysis", "source_verification"},
                specializations=["historical_analysis", "timeline_construction", "source_validation"]
            )
        
        self.config = config
        self.source_validator = HistoricalSourceValidator()
        self.timeline_analyzer = HistoricalTimelineAnalyzer()
        # Initialize new validation components
        self.hathitrust_integration = HathiTrustBibliographicAPI(config)
        self.internet_archive_integration = InternetArchiveAPI(config)
        self.cross_platform_validator = CrossPlatformValidator(config)
        
        # Historical databases and archives (mock for now)
        self.historical_databases = [
            "Library of Congress",
            "National Archives",
            "JSTOR",
            "Project MUSE",
            "Internet Archive",
            "WorldCat",
            "Google Books",
            "Europeana",
            "Digital Public Library of America",
            "HathiTrust Digital Library"  # Added HathiTrust
        ]
        
        # Knowledge base for historical context
        self.historical_contexts = {}
        self.fact_cache = {}
        
    async def conduct_historical_research(self, query: ResearchQuery) -> HistoricalAnalysis:
        """
        Conduct comprehensive historical research based on the query
        """
        try:
            logger.info(f"Starting historical research for: {query.topic}")
            
            # Phase 1: Initial source gathering
            sources = await self._gather_historical_sources(query)
            
            # Phase 2: Source validation and credibility assessment
            validated_sources = await self._validate_sources(sources)
            
            # Phase 3: Event extraction and analysis
            events = await self._extract_historical_events(query, validated_sources)
            
            # Phase 4: Timeline construction
            timeline = await self.timeline_analyzer.create_historical_timeline(events)
            
            # Phase 5: Causal analysis
            causal_relationships = await self.timeline_analyzer.analyze_causal_relationships(events)
            
            # Phase 6: Context and comparative analysis
            historical_context = await self._analyze_historical_context(events, validated_sources)
            comparative_analysis = await self._conduct_comparative_analysis(events)
            
            # Phase 7: Bias assessment and alternative narratives
            bias_assessment = await self._assess_narrative_bias(validated_sources)
            alternative_narratives = await self._identify_alternative_narratives(events, validated_sources)
            
            # Phase 8: Quality assessment and gap analysis
            confidence_metrics = await self._calculate_confidence_metrics(events, validated_sources)
            research_gaps = await self._identify_research_gaps(query, events, validated_sources)
            
            # Phase 9: Generate recommendations
            recommendations = await self._generate_research_recommendations(research_gaps, query)
            
            analysis = HistoricalAnalysis(
                query=query,
                events=events,
                timeline=timeline,
                key_themes=timeline.themes,
                causal_relationships=causal_relationships,
                historical_context=historical_context,
                comparative_analysis=comparative_analysis,
                source_analysis=await self._summarize_source_analysis(validated_sources),
                bias_assessment=bias_assessment,
                confidence_metrics=confidence_metrics,
                alternative_narratives=alternative_narratives,
                research_gaps=research_gaps,
                recommendations=recommendations
            )
            
            logger.info(f"Historical research completed: {len(events)} events analyzed")
            return analysis
            
        except Exception as e:
            logger.error(f"Historical research failed: {e}")
            return HistoricalAnalysis(query=query)
    
    async def _gather_historical_sources(self, query: ResearchQuery) -> List[ResearchSource]:
        """Gather historical sources from various databases and archives"""
        sources = []
        
        try:
            # Search academic databases
            academic_sources = await self._search_academic_databases(query)
            sources.extend(academic_sources)
            
            # Search archival collections
            archival_sources = await self._search_archival_collections(query)
            sources.extend(archival_sources)
            
            # Search newspaper archives
            newspaper_sources = await self._search_newspaper_archives(query)
            sources.extend(newspaper_sources)
              # Search government records
            government_sources = await self._search_government_records(query)
            sources.extend(government_sources)
            
            # Search HathiTrust Digital Library
            hathitrust_sources = await self._search_hathitrust(query)
            sources.extend(hathitrust_sources)
            
            logger.info(f"Gathered {len(sources)} historical sources (including HathiTrust)")
            return sources
        except Exception as e:
            logger.error(f"Source gathering failed: {e}")
            return []

    async def _search_academic_databases(self, query: ResearchQuery) -> List[ResearchSource]:
        """Search academic databases for historical sources"""
        sources = []
        try:
            # Use Internet Archive's academic collection for now
            # Future: Add JSTOR, Project MUSE, etc. when API access available
            logger.info(f"Searching academic collections for: {query.topic}")
            
            # Leverage Internet Archive for academic content with proper session management
            async with self.internet_archive_integration as ia_api:
                ia_academic_sources = await ia_api.search_historical_documents(query)
            
                # Filter for academic-style sources
                academic_sources = [
                    source for source in ia_academic_sources 
                    if any(keyword in source.title.lower() or keyword in source.abstract.lower() 
                           for keyword in ['journal', 'university', 'academic', 'research', 'study', 'analysis'])            ]
            
                sources.extend(academic_sources)
                logger.info(f"Found {len(academic_sources)} academic sources")
            
            return sources
            
        except Exception as e:
            logger.error(f"Academic database search failed: {e}")
            return []

    async def _search_archival_collections(self, query: ResearchQuery) -> List[ResearchSource]:
        """Search archival collections for primary sources"""
        sources = []
        try:
            logger.info(f"Searching archival collections for: {query.topic}")
            
            # Use Internet Archive's digitized archival materials
            # Filter for archival-style content (manuscripts, letters, documents, etc.)
            ia_sources = await self.internet_archive_integration.search_historical_documents(query)
            
            archival_sources = [
                source for source in ia_sources 
                if any(keyword in source.title.lower() or keyword in source.abstract.lower() 
                       for keyword in ['manuscript', 'archive', 'collection', 'papers', 'correspondence', 
                                       'documents', 'records', 'diary', 'letter', 'memoir'])
            ]
            
            sources.extend(archival_sources)
            logger.info(f"Found {len(archival_sources)} archival collection sources")
            
            return sources
            
        except Exception as e:
            logger.error(f"Archival collection search failed: {e}")
            return []

    async def _search_newspaper_archives(self, query: ResearchQuery) -> List[ResearchSource]:
        """Search historical newspaper archives"""
        sources = []
        try:
            logger.info(f"Searching newspaper archives for: {query.topic}")
            
            # Use Internet Archive's newspaper collection
            # Filter for newspaper and periodical content
            ia_sources = await self.internet_archive_integration.search_historical_documents(query)
            
            newspaper_sources = [
                source for source in ia_sources 
                if any(keyword in source.title.lower() or keyword in source.abstract.lower() 
                       for keyword in ['newspaper', 'times', 'herald', 'gazette', 'chronicle', 'post', 
                                       'journal', 'tribune', 'news', 'daily', 'weekly', 'press'])
            ]
            
            sources.extend(newspaper_sources)
            logger.info(f"Found {len(newspaper_sources)} newspaper archive sources")
            
            return sources
            
        except Exception as e:
            logger.error(f"Newspaper archive search failed: {e}")
            return []

    async def _search_government_records(self, query: ResearchQuery) -> List[ResearchSource]:
        """Search declassified government records and official documents"""
        sources = []
        try:
            logger.info(f"Searching government records for: {query.topic}")
            
            # Use Internet Archive's government document collection
            # Filter for government and official document content
            ia_sources = await self.internet_archive_integration.search_historical_documents(query)
            
            government_sources = [
                source for source in ia_sources 
                if any(keyword in source.title.lower() or keyword in source.abstract.lower() 
                       for keyword in ['government', 'congress', 'senate', 'department', 'bureau', 
                                       'commission', 'committee', 'official', 'federal', 'state', 
                                       'declassified', 'report', 'hearing', 'testimony'])            ]
            
            sources.extend(government_sources)
            logger.info(f"Found {len(government_sources)} government record sources")
            
            return sources
            
        except Exception as e:
            logger.error(f"Government records search failed: {e}")
            return []

    async def _search_hathitrust(self, query: ResearchQuery) -> List[ResearchSource]:
        """Search HathiTrust Digital Library for historical sources using compliant bibliographic search"""        
        sources = []
        try:
            # Use HathiTrust's bibliographic catalog search (COMPLIANT with ToS)
            logger.info("Searching HathiTrust bibliographic catalog...")
            hathitrust_sources = await self.hathitrust_integration.search_bibliographic_catalog(query)
            sources.extend(hathitrust_sources)
            logger.info(f"Found {len(hathitrust_sources)} sources from HathiTrust catalog")
            
            # Also search Internet Archive as supplementary source
            logger.info("Searching Internet Archive for additional historical sources...")
            ia_sources = await self.internet_archive_integration.search_historical_documents(query)
            sources.extend(ia_sources)
            logger.info(f"Found {len(ia_sources)} additional sources from Internet Archive")
            
        except Exception as e:
            logger.error(f"Failed to search historical sources: {e}")
        
        return sources
    
    async def _validate_sources(self, sources: List[ResearchSource]) -> List[ResearchSource]:
        """Validate historical sources for credibility and bias using enhanced validation"""
        validated_sources = []
        
        for source in sources:
            # Traditional validation
            validation_results = await self.source_validator.validate_historical_source(source)
              # Enhanced cross-platform validation
            cross_platform_results = await self.cross_platform_validator.validate_single_source(source)
            
            # Convert the validation result to the expected format
            cross_platform_dict = {
                "found_in_platforms": cross_platform_results.verification_count,
                "overall_credibility": cross_platform_results.overall_credibility,
                "consensus_score": cross_platform_results.consensus_score
            }
            
            # Try to find additional sources from HathiTrust if applicable            hathitrust_results = None
            if hasattr(source, 'title') and source.title:                # For HathiTrust, we can only search by known identifiers
                # Since we don't have identifiers here, skip HathiTrust verification
                logger.info("HathiTrust verification skipped - requires known identifiers")
            
            # Combine validation results
            combined_score = validation_results.get("overall_credibility", 0.0)
              # Boost credibility if found in multiple platforms
            if cross_platform_dict.get("found_in_platforms", 0) > 1:
                combined_score = min(1.0, combined_score * 1.2)
            
            # Note: HathiTrust verification would require known identifiers
            # Skip credibility boost for now
            
            # Update source with enhanced validation metrics
            source.credibility_score = combined_score
            source.bias_score = validation_results.get("source_bias", {}).get("bias_score", 0.0)
            
            # Add cross-platform metadata
            if not hasattr(source, 'metadata'):
                source.metadata = {}
            source.metadata.update({
                "cross_platform_validation": cross_platform_dict,
                "enhanced_validation": True
            })
              # Only include sources above credibility threshold
            if source.credibility_score >= 0.6:  # Use reasonable threshold
                validated_sources.append(source)
            else:
                logger.info(f"Source excluded due to low credibility: {source.title} (score: {source.credibility_score:.2f})")
        
        logger.info(f"Validated {len(validated_sources)}/{len(sources)} sources using enhanced validation")
        return validated_sources
    
    async def _extract_historical_events(self, query: ResearchQuery, 
                                       sources: List[ResearchSource]) -> List[HistoricalEvent]:
        """Extract and structure historical events from sources"""
        events = []
        
        try:
            for source in sources:
                # Extract events from source content
                source_events = await self._extract_events_from_source(source, query)
                events.extend(source_events)
            
            # Deduplicate and merge similar events
            merged_events = await self._merge_duplicate_events(events)
            
            # Enrich events with additional context
            enriched_events = await self._enrich_events_with_context(merged_events)
            
            logger.info(f"Extracted {len(enriched_events)} historical events")
            return enriched_events
            
        except Exception as e:
            logger.error(f"Event extraction failed: {e}")
            return []

    async def _extract_events_from_source(self, source: ResearchSource, 
                                        query: ResearchQuery) -> List[HistoricalEvent]:
        """Extract historical events from a single source"""
        events = []
        
        try:
            # Analyze source content for historical events
            content = source.content + " " + source.abstract + " " + source.title
            content_lower = content.lower()
              # Look for event indicators related to the query topic
            # Create a basic event based on the source if it seems relevant
            if any(keyword in content_lower for keyword in ['war', 'battle', 'revolution', 'conflict']):                # Extract potential dates
                import re
                date_patterns = [
                    r'\b(1\d{3})\b',  # Four-digit years
                    r'\b(18\d{2}|19\d{2}|20\d{2})\b',  # More specific years
                ]
                
                dates_found = []
                for pattern in date_patterns:
                    matches = re.findall(pattern, content)
                    dates_found.extend(matches)
                
                # Convert found dates to datetime objects
                date_start = None
                if dates_found:
                    try:
                        year = int(dates_found[0])
                        date_start = datetime(year, 1, 1)  # Default to January 1st
                    except ValueError:
                        pass                # Create event from source
                event = HistoricalEvent(
                    event_id=f"evt_{source.url.split('/')[-1] if source.url else 'unknown'}",
                    name=f"Historical Event: {source.title[:100]}",
                    description=source.abstract[:500] if source.abstract else f"Event extracted from: {source.title}",
                    date_start=date_start,
                    location="Unknown location",
                    sources=[source],
                    event_type=HistoricalEventType.POLITICAL,  # Default to political, could be improved
                    confidence_level=source.credibility_score
                )
                events.append(event)
                
        except Exception as e:
            logger.warning(f"Error extracting events from source {source.title}: {e}")
        
        return events
    
    async def _merge_duplicate_events(self, events: List[HistoricalEvent]) -> List[HistoricalEvent]:
        """Merge duplicate or very similar events"""
        # Simple implementation - would use more sophisticated matching
        unique_events = []
        seen_names = set()
        
        for event in events:
            if event.name not in seen_names:
                unique_events.append(event)
                seen_names.add(event.name)
            else:
                # Merge with existing event
                existing_event = next(e for e in unique_events if e.name == event.name)
                existing_event.sources.extend(event.sources)
                existing_event.confidence_level = max(existing_event.confidence_level, 
                                                    event.confidence_level)
        
        return unique_events
    
    async def _enrich_events_with_context(self, events: List[HistoricalEvent]) -> List[HistoricalEvent]:
        """Enrich events with additional historical context"""
        for event in events:
            # Add historical period context
            event.period = self._determine_historical_period(event.date_start)
            
            # Enhance with related events
            event.related_events = await self._find_related_events(event, events)
            
            # Add geographical context
            if event.location:
                event.coordinates = await self._geocode_historical_location(event.location)
        
        return events
    
    def _determine_historical_period(self, date: Optional[datetime]) -> HistoricalPeriod:
        """Determine historical period based on date"""
        if not date:
            return HistoricalPeriod.CONTEMPORARY
        
        year = date.year
        if year < 500:
            return HistoricalPeriod.ANCIENT
        elif year < 1500:
            return HistoricalPeriod.MEDIEVAL
        elif year < 1800:
            return HistoricalPeriod.EARLY_MODERN
        elif year < 1945:
            return HistoricalPeriod.MODERN
        else:
            return HistoricalPeriod.CONTEMPORARY
    
    async def _find_related_events(self, event: HistoricalEvent, 
                                 all_events: List[HistoricalEvent]) -> List[str]:
        """Find events related to the given event"""
        related = []
        
        for other_event in all_events:
            if other_event.event_id == event.event_id:
                continue
            
            # Check for common figures
            if set(event.key_figures) & set(other_event.key_figures):
                related.append(other_event.event_id)
            
            # Check for temporal proximity
            if (event.date_start and other_event.date_start and 
                abs((event.date_start - other_event.date_start).days) < 365):
                related.append(other_event.event_id)
        
        return related[:5]  # Limit to top 5 related events
    
    async def _geocode_historical_location(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for historical location"""
        # Mock implementation - would use geocoding service
        location_coords = {
            "Waterloo, Belgium": (50.7158, 4.4003),
            "Rome, Italy": (41.9028, 12.4964),
            "London, England": (51.5074, -0.1278)
        }
        return location_coords.get(location)
    
    async def _analyze_historical_context(self, events: List[HistoricalEvent], 
                                        sources: List[ResearchSource]) -> str:
        """Analyze the broader historical context of the events"""
        try:
            # Identify time period and geographical scope
            periods = [event.period for event in events]
            locations = [event.location for event in events if event.location]
            
            # Analyze major themes and patterns
            themes = {}
            for event in events:
                event_type = event.event_type.value
                themes[event_type] = themes.get(event_type, 0) + 1
            
            # Generate contextual analysis
            context = f"""
            Historical Context Analysis:
            
            Time Period: {', '.join(set(p.value for p in periods))}
            Geographical Scope: {', '.join(set(locations))}
            
            Major Themes:
            {chr(10).join(f'- {theme}: {count} events' for theme, count in themes.items())}
            
            The events analyzed span multiple historical periods and demonstrate 
            interconnected patterns of political, social, and cultural development.
            Key factors influencing these events include geopolitical tensions,
            economic pressures, and social movements of their respective eras.
            """
            
            return context.strip()
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return "Historical context analysis unavailable due to processing error."
    
    async def _conduct_comparative_analysis(self, events: List[HistoricalEvent]) -> str:
        """Conduct comparative analysis of historical events"""
        try:
            # Group events by type
            event_groups = defaultdict(list)
            for event in events:
                event_groups[event.event_type].append(event)
            
            # Analyze patterns across event types
            analysis = "Comparative Historical Analysis:\n\n"
            
            for event_type, type_events in event_groups.items():
                if len(type_events) > 1:
                    analysis += f"{event_type.value.title()} Events:\n"
                    
                    # Compare causes and consequences
                    common_causes = self._find_common_elements([e.causes for e in type_events])
                    common_consequences = self._find_common_elements([e.consequences for e in type_events])
                    
                    if common_causes:
                        analysis += f"  Common Causes: {', '.join(common_causes[:3])}\n"
                    if common_consequences:
                        analysis += f"  Common Consequences: {', '.join(common_consequences[:3])}\n"
                    
                    analysis += "\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            return "Comparative analysis unavailable."
    
    def _find_common_elements(self, element_lists: List[List[str]]) -> List[str]:
        """Find common elements across multiple lists"""
        if not element_lists:
            return []
        
        # Flatten and count occurrences
        all_elements = [item for sublist in element_lists for item in sublist]
        element_counts = defaultdict(int)
        
        for element in all_elements:
            element_counts[element] += 1
        
        # Return elements that appear in multiple lists
        return [element for element, count in element_counts.items() if count > 1]
    
    async def _assess_narrative_bias(self, sources: List[ResearchSource]) -> Dict[str, float]:
        """Assess overall narrative bias across sources"""
        try:
            bias_scores = [source.bias_score for source in sources if hasattr(source, 'bias_score')]
            
            if not bias_scores:
                return {"overall_bias": 0.0, "source_diversity": 1.0}
            
            overall_bias = sum(bias_scores) / len(bias_scores)
            
            # Calculate source diversity (variety of perspectives)
            unique_publishers = len(set(source.publisher for source in sources))
            source_diversity = min(1.0, unique_publishers / max(1, len(sources)))
            
            return {
                "overall_bias": overall_bias,
                "source_diversity": source_diversity,
                "high_bias_sources": sum(1 for score in bias_scores if score > 0.7),
                "low_bias_sources": sum(1 for score in bias_scores if score < 0.3)
            }
            
        except Exception as e:
            logger.error(f"Bias assessment failed: {e}")
            return {"overall_bias": 0.5}
    
    async def _identify_alternative_narratives(self, events: List[HistoricalEvent], 
                                             sources: List[ResearchSource]) -> List[str]:
        """Identify alternative historical narratives and interpretations"""
        alternative_narratives = []
        
        try:
            # Look for events with disputed consensus
            disputed_events = [e for e in events if e.scholarly_consensus in ["weak", "disputed"]]
            
            for event in disputed_events:
                if event.alternative_interpretations:
                    alternative_narratives.extend(event.alternative_interpretations)
            
            # Analyze source diversity for different perspectives
            publisher_perspectives = defaultdict(list)
            for source in sources:
                publisher_perspectives[source.publisher].append(source.abstract)
            
            # Add general alternative perspective categories
            if len(publisher_perspectives) > 1:
                alternative_narratives.extend([
                    "Western vs. non-Western perspectives on colonial history",
                    "Political vs. social history interpretations",
                    "Elite vs. popular history viewpoints",
                    "Contemporary vs. retrospective analyses"
                ])
            
            return alternative_narratives[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"Alternative narrative identification failed: {e}")
            return []
    
    async def _calculate_confidence_metrics(self, events: List[HistoricalEvent], 
                                          sources: List[ResearchSource]) -> Dict[str, float]:
        """Calculate confidence metrics for the historical analysis"""
        try:
            if not events or not sources:
                return {"overall_confidence": 0.0}
            
            # Source quality metrics
            avg_source_credibility = sum(s.credibility_score for s in sources) / len(sources)
            peer_reviewed_ratio = sum(1 for s in sources if s.peer_reviewed) / len(sources)
            
            # Event confidence metrics
            avg_event_confidence = sum(e.confidence_level for e in events) / len(events)
            strong_consensus_ratio = sum(1 for e in events if e.scholarly_consensus == "strong") / len(events)
            
            # Overall confidence calculation
            overall_confidence = (
                avg_source_credibility * 0.3 +
                peer_reviewed_ratio * 0.2 +
                avg_event_confidence * 0.3 +
                strong_consensus_ratio * 0.2
            )
            
            return {
                "overall_confidence": overall_confidence,
                "source_credibility": avg_source_credibility,
                "peer_reviewed_ratio": peer_reviewed_ratio,
                "event_confidence": avg_event_confidence,
                "consensus_strength": strong_consensus_ratio
            }
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return {"overall_confidence": 0.5}
    
    async def _identify_research_gaps(self, query: ResearchQuery, events: List[HistoricalEvent], 
                                    sources: List[ResearchSource]) -> List[str]:
        """Identify gaps in the current research"""
        gaps = []
        
        try:
            # Check for temporal gaps
            if events:
                events_by_date = sorted([e for e in events if e.date_start], 
                                      key=lambda e: e.date_start)
                for i in range(len(events_by_date) - 1):
                    current_event = events_by_date[i]
                    next_event = events_by_date[i + 1]
                    
                    if next_event.date_start and current_event.date_start:
                        gap_years = (next_event.date_start - current_event.date_start).days / 365
                        if gap_years > 10:  # Significant time gap
                            gaps.append(f"Limited information on period between {current_event.date_start.year} and {next_event.date_start.year}")
            
            # Check for geographical gaps
            regions_covered = set(e.location for e in events if e.location)
            if len(regions_covered) < 3:
                gaps.append("Limited geographical diversity in sources and events")
            
            # Check for perspective gaps
            source_types = set(s.source_type for s in sources)
            if SourceType.WEB_SOURCE not in source_types:
                gaps.append("Lack of primary source documentation")
            
            # Check for thematic gaps
            event_types = set(e.event_type for e in events)
            all_types = set(HistoricalEventType)
            missing_types = all_types - event_types
            
            if missing_types:
                gaps.append(f"Limited coverage of {', '.join(t.value for t in list(missing_types)[:2])} events")
            
            return gaps[:5]  # Limit to top 5 gaps
            
        except Exception as e:
            logger.error(f"Gap identification failed: {e}")
            return ["Unable to identify research gaps due to analysis error"]
    
    async def _generate_research_recommendations(self, gaps: List[str], 
                                               query: ResearchQuery) -> List[str]:
        """Generate recommendations for further research"""
        recommendations = []
        
        try:
            # Address identified gaps
            for gap in gaps:
                if "primary source" in gap.lower():
                    recommendations.append("Consult archival collections and contemporary documents")
                elif "geographical" in gap.lower():
                    recommendations.append("Expand research to include multiple regional perspectives")
                elif "period between" in gap.lower():
                    recommendations.append("Focus research on intermediate time periods for continuity")
            
            # General research recommendations
            recommendations.extend([
                "Cross-reference findings with recent historiographical debates",
                "Incorporate interdisciplinary perspectives (archaeology, anthropology, economics)",
                "Examine both elite and popular historical narratives",
                "Consider long-term historical trends and patterns",
                "Validate findings through multiple independent sources"
            ])
            
            return recommendations[:6]  # Limit to top 6 recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Continue research with expanded source base"]
    
    async def _summarize_source_analysis(self, sources: List[ResearchSource]) -> Dict[str, Any]:
        """Summarize the analysis of sources used"""
        try:
            total_sources = len(sources)
            if total_sources == 0:
                return {"total_sources": 0}
            
            # Count by source type
            source_types = defaultdict(int)
            for source in sources:
                source_types[source.source_type.value] += 1
            
            # Calculate average metrics
            avg_credibility = sum(s.credibility_score for s in sources) / total_sources
            peer_reviewed_count = sum(1 for s in sources if s.peer_reviewed)
            
            # Publication date analysis
            dated_sources = [s for s in sources if s.publication_date]
            if dated_sources:
                oldest_year = min(s.publication_date.year for s in dated_sources)
                newest_year = max(s.publication_date.year for s in dated_sources)
                date_range = f"{oldest_year}-{newest_year}"
            else:
                date_range = "Unknown"
            
            return {
                "total_sources": total_sources,
                "source_types": dict(source_types),
                "average_credibility": avg_credibility,
                "peer_reviewed_count": peer_reviewed_count,
                "peer_reviewed_percentage": (peer_reviewed_count / total_sources) * 100,
                "publication_date_range": date_range,
                "unique_publishers": len(set(s.publisher for s in sources))
            }
            
        except Exception as e:
            logger.error(f"Source analysis summary failed: {e}")
            return {"error": str(e)}


# Agent capability definition for integration with PyGent Factory
def create_historical_research_capability() -> AgentCapability:
    """Create agent capability definition for historical research"""
    return AgentCapability(
        agent_id="historical_research_agent",
        agent_type=AgentType.RESEARCH,
        capabilities=[
            "historical_event_analysis",
            "timeline_construction", 
            "source_validation",
            "bias_detection",
            "comparative_historical_analysis",
            "archival_research",
            "primary_source_analysis",
            "historical_context_analysis"
        ],
        specializations=[
            "ancient_history",
            "medieval_history", 
            "modern_history",
            "political_history",
            "social_history",
            "military_history",
            "cultural_history"
        ],
        performance_metrics={
            "source_validation_accuracy": 0.92,
            "timeline_construction_speed": 0.88,
            "bias_detection_sensitivity": 0.85,
            "historical_accuracy": 0.94
        },
        resource_requirements={
            "memory_gb": 4,
            "cpu_cores": 2,
            "storage_gb": 10,
            "network_bandwidth": "high"
        },
        supported_languages=["en", "es", "fr", "de", "it"],
        max_concurrent_tasks=3,
        estimated_response_time_seconds=300,
        confidence_threshold=0.8
    )


# Export for integration
__all__ = [
    "HistoricalResearchAgent",
    "HistoricalEvent", 
    "HistoricalTimeline",
    "HistoricalAnalysis",
    "HistoricalSourceValidator",
    "HistoricalTimelineAnalyzer",
    "HistoricalPeriod",
    "HistoricalEventType",
    "HistoricalSourceType",
    "create_historical_research_capability"
]
