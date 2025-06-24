"""
Zero-Cost Research Orchestrator

Coordinates multiple free academic sources to provide comprehensive
historical research with global perspectives and primary source integration.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# Import MCP servers
from .mcp_servers.google_scholar_mcp import GoogleScholarMCP, ScholarPaper, ScholarSearchResult

logger = logging.getLogger(__name__)


@dataclass
class ResearchTopic:
    """Historical research topic configuration"""
    topic_id: str
    title: str
    research_question: str
    focus_areas: List[str]
    target_regions: List[str]
    historical_periods: List[str]
    max_papers: int = 50
    include_primary_sources: bool = True
    cultural_perspectives: List[str] = field(default_factory=list)
    methodology_preferences: List[str] = field(default_factory=list)


@dataclass
class ComprehensiveResearchResult:
    """Comprehensive research result from multiple sources"""
    topic: ResearchTopic
    total_papers: int
    papers_by_source: Dict[str, List[ScholarPaper]]
    global_analysis: Dict[str, Any]
    primary_source_recommendations: List[Dict[str, Any]]
    research_gaps: List[str]
    methodology_recommendations: List[str]
    cross_cultural_insights: List[str]
    quality_assessment: Dict[str, Any]
    next_steps: List[str]
    search_metadata: Dict[str, Any]


class ZeroCostResearchOrchestrator:
    """
    Orchestrates zero-cost academic research across multiple sources
    with focus on historical research and global perspectives.
    """
    
    def __init__(self, orchestrator_id: str = "zero_cost_orchestrator"):
        self.orchestrator_id = orchestrator_id
        self.status = "initialized"
        
        # MCP servers
        self.mcp_servers = {}
        
        # Research configuration
        self.research_config = {
            'max_concurrent_searches': 3,
            'search_timeout_seconds': 300,
            'min_papers_per_source': 5,
            'quality_threshold': 0.7,
            'global_coverage_target': 0.8,
            'primary_source_target': 0.3
        }
        
        # Historical research topics (your specific topics)
        self.predefined_topics = {
            'scientific_revolutions_global': ResearchTopic(
                topic_id='scientific_revolutions_global',
                title='Scientific Revolutions (Global Perspectives)',
                research_question='How did scientific revolutions manifest in art, architecture, and literacy beyond European contexts?',
                focus_areas=['art_architecture', 'literacy', 'global_perspectives', 'knowledge_transfer'],
                target_regions=['China', 'Middle East', 'India', 'Americas'],
                historical_periods=['early_modern', 'modern'],
                cultural_perspectives=['Non-Western', 'Global'],
                methodology_preferences=['Archival', 'Comparative']
            ),
            'enlightenment_cross_cultural': ResearchTopic(
                topic_id='enlightenment_cross_cultural',
                title='Enlightenment (Cross-Cultural)',
                research_question='How did Enlightenment ideas about human rights and political values develop across different cultural contexts?',
                focus_areas=['human_rights', 'political_values', 'cross_cultural_analysis'],
                target_regions=['Europe', 'Americas', 'Global'],
                historical_periods=['early_modern', 'modern'],
                cultural_perspectives=['Global', 'Postcolonial'],
                methodology_preferences=['Comparative', 'Archival']
            ),
            'tokugawa_social_transformation': ResearchTopic(
                topic_id='tokugawa_social_transformation',
                title='Tokugawa Japan Social Transformation',
                research_question='How did women\'s roles, artistic expression, and samurai identity transform during the Tokugawa period?',
                focus_areas=['women_roles', 'artistic_expression', 'social_transformation'],
                target_regions=['Japan'],
                historical_periods=['early_modern'],
                cultural_perspectives=['Non-Western'],
                methodology_preferences=['Archival', 'Qualitative']
            ),
            'haitian_revolution_global_impact': ResearchTopic(
                topic_id='haitian_revolution_global_impact',
                title='Haitian Revolution Global Impact',
                research_question='What were the global influences and impacts of Haitian diasporas following the revolution?',
                focus_areas=['diaspora_influences', 'global_impact', 'revolutionary_ideas'],
                target_regions=['Americas', 'Global'],
                historical_periods=['modern'],
                cultural_perspectives=['Postcolonial', 'Global'],
                methodology_preferences=['Archival', 'Oral History']
            ),
            'decolonization_comparative': ResearchTopic(
                topic_id='decolonization_comparative',
                title='Decolonization (Global Case Studies)',
                research_question='How did decolonization movements develop across different regions with varying strategies and outcomes?',
                focus_areas=['independence_movements', 'cultural_revival', 'global_perspectives'],
                target_regions=['Africa', 'Asia', 'Americas'],
                historical_periods=['contemporary'],
                cultural_perspectives=['Postcolonial', 'Global'],
                methodology_preferences=['Comparative', 'Oral History']
            )
        }
        
        # Statistics
        self.stats = {
            'research_sessions': 0,
            'topics_researched': 0,
            'total_papers_retrieved': 0,
            'primary_sources_identified': 0,
            'cross_cultural_connections': 0,
            'research_gaps_identified': 0
        }
        
        logger.info(f"ZeroCostResearchOrchestrator {orchestrator_id} initialized")
    
    async def start(self) -> bool:
        """Start the research orchestrator and all MCP servers"""
        try:
            self.status = "starting"
            
            # Initialize Google Scholar MCP server
            self.mcp_servers['google_scholar'] = GoogleScholarMCP("scholar_server")
            await self.mcp_servers['google_scholar'].start()
            
            # TODO: Add other MCP servers (HathiTrust, Internet Archive, etc.)
            # self.mcp_servers['hathitrust'] = HathiTrustMCP("hathitrust_server")
            # self.mcp_servers['internet_archive'] = InternetArchiveMCP("ia_server")
            # self.mcp_servers['europeana'] = EuropeanaMCP("europeana_server")
            # self.mcp_servers['doaj'] = DOAJMCP("doaj_server")
            
            self.status = "active"
            logger.info(f"ZeroCostResearchOrchestrator {self.orchestrator_id} started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ZeroCostResearchOrchestrator: {e}")
            self.status = "error"
            return False
    
    async def conduct_comprehensive_historical_research(self, topic_ids: List[str] = None) -> List[ComprehensiveResearchResult]:
        """
        Conduct comprehensive historical research on specified topics.
        
        Args:
            topic_ids: List of topic IDs to research. If None, research all predefined topics.
            
        Returns:
            List of comprehensive research results
        """
        if self.status != "active":
            raise RuntimeError("Orchestrator not active. Call start() first.")
        
        # Use all predefined topics if none specified
        if topic_ids is None:
            topic_ids = list(self.predefined_topics.keys())
        
        session_id = f"session_{int(datetime.utcnow().timestamp())}"
        self.stats['research_sessions'] += 1
        
        logger.info(f"Starting comprehensive research session {session_id} with {len(topic_ids)} topics")
        
        results = []
        
        for topic_id in topic_ids:
            if topic_id not in self.predefined_topics:
                logger.warning(f"Unknown topic ID: {topic_id}")
                continue
            
            topic = self.predefined_topics[topic_id]
            
            try:
                logger.info(f"Researching topic: {topic.title}")
                
                # Conduct comprehensive research for this topic
                result = await self._research_topic_comprehensive(topic, session_id)
                results.append(result)
                
                self.stats['topics_researched'] += 1
                self.stats['total_papers_retrieved'] += result.total_papers
                
                # Delay between topics to be respectful
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Research failed for topic {topic.title}: {e}")
                continue
        
        logger.info(f"Research session {session_id} completed: {len(results)} topics researched")
        return results
    
    async def _research_topic_comprehensive(self, topic: ResearchTopic, session_id: str) -> ComprehensiveResearchResult:
        """Conduct comprehensive research on a single topic"""
        
        start_time = datetime.utcnow()
        papers_by_source = {}
        
        # Search Google Scholar
        if 'google_scholar' in self.mcp_servers:
            try:
                scholar_result = await self.mcp_servers['google_scholar'].search_historical_research(
                    query=topic.research_question,
                    max_results=topic.max_papers,
                    focus_regions=topic.target_regions,
                    include_primary_sources=topic.include_primary_sources,
                    cultural_perspective=topic.cultural_perspectives[0] if topic.cultural_perspectives else None
                )
                papers_by_source['google_scholar'] = scholar_result.papers
                
            except Exception as e:
                logger.error(f"Google Scholar search failed for {topic.title}: {e}")
                papers_by_source['google_scholar'] = []
        
        # TODO: Add searches for other sources
        # HathiTrust search for primary sources
        # Internet Archive search for historical documents
        # Europeana search for European archives
        # DOAJ search for open access journals
        
        # Combine all papers
        all_papers = []
        for source_papers in papers_by_source.values():
            all_papers.extend(source_papers)
        
        # Analyze results comprehensively
        global_analysis = await self._analyze_global_research_results(all_papers, topic)
        primary_source_recs = await self._generate_primary_source_recommendations(all_papers, topic)
        research_gaps = await self._identify_research_gaps(all_papers, topic)
        methodology_recs = await self._generate_methodology_recommendations(all_papers, topic)
        cross_cultural_insights = await self._extract_cross_cultural_insights(all_papers, topic)
        quality_assessment = await self._assess_research_quality(all_papers, topic)
        next_steps = await self._generate_next_steps(all_papers, topic, research_gaps)
        
        # Create comprehensive result
        result = ComprehensiveResearchResult(
            topic=topic,
            total_papers=len(all_papers),
            papers_by_source=papers_by_source,
            global_analysis=global_analysis,
            primary_source_recommendations=primary_source_recs,
            research_gaps=research_gaps,
            methodology_recommendations=methodology_recs,
            cross_cultural_insights=cross_cultural_insights,
            quality_assessment=quality_assessment,
            next_steps=next_steps,
            search_metadata={
                'session_id': session_id,
                'search_time': (datetime.utcnow() - start_time).total_seconds(),
                'sources_searched': len(papers_by_source),
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        return result
    
    async def research_custom_topic(self, title: str, research_question: str, **kwargs) -> ComprehensiveResearchResult:
        """Research a custom topic not in predefined topics"""
        
        custom_topic = ResearchTopic(
            topic_id=f"custom_{int(datetime.utcnow().timestamp())}",
            title=title,
            research_question=research_question,
            focus_areas=kwargs.get('focus_areas', []),
            target_regions=kwargs.get('target_regions', []),
            historical_periods=kwargs.get('historical_periods', []),
            max_papers=kwargs.get('max_papers', 50),
            include_primary_sources=kwargs.get('include_primary_sources', True),
            cultural_perspectives=kwargs.get('cultural_perspectives', []),
            methodology_preferences=kwargs.get('methodology_preferences', [])
        )
        
        session_id = f"custom_session_{int(datetime.utcnow().timestamp())}"
        
        return await self._research_topic_comprehensive(custom_topic, session_id)

    async def _analyze_global_research_results(self, papers: List[ScholarPaper], topic: ResearchTopic) -> Dict[str, Any]:
        """Analyze research results from global perspective"""

        if not papers:
            return {'error': 'No papers to analyze'}

        # Regional representation
        region_counts = {}
        for paper in papers:
            for country in paper.countries:
                region_counts[country] = region_counts.get(country, 0) + 1

        # Temporal coverage
        temporal_coverage = {}
        for paper in papers:
            if paper.publication_year:
                decade = (paper.publication_year // 10) * 10
                temporal_coverage[f"{decade}s"] = temporal_coverage.get(f"{decade}s", 0) + 1

        # Cultural perspective diversity
        perspective_counts = {}
        for paper in papers:
            if paper.cultural_perspective:
                perspective_counts[paper.cultural_perspective] = perspective_counts.get(paper.cultural_perspective, 0) + 1

        # Subject area coverage
        subject_coverage = {}
        for paper in papers:
            for subject in paper.subject_areas:
                subject_coverage[subject] = subject_coverage.get(subject, 0) + 1

        # Calculate diversity scores
        total_regions = len(region_counts)
        target_regions_covered = sum(1 for region in topic.target_regions if region in region_counts)
        regional_coverage_score = target_regions_covered / max(len(topic.target_regions), 1)

        # Global perspective score
        non_western_papers = sum(count for perspective, count in perspective_counts.items()
                                if perspective in ['Non-Western', 'Global', 'Postcolonial'])
        global_perspective_score = non_western_papers / len(papers) if papers else 0

        return {
            'regional_representation': region_counts,
            'temporal_coverage': temporal_coverage,
            'cultural_perspectives': perspective_counts,
            'subject_coverage': subject_coverage,
            'diversity_metrics': {
                'total_regions_represented': total_regions,
                'target_regions_covered': target_regions_covered,
                'regional_coverage_score': regional_coverage_score,
                'global_perspective_score': global_perspective_score,
                'cultural_diversity_index': len(perspective_counts) / 4.0  # Max 4 perspectives
            }
        }

    async def _generate_primary_source_recommendations(self, papers: List[ScholarPaper], topic: ResearchTopic) -> List[Dict[str, Any]]:
        """Generate primary source recommendations based on research"""

        recommendations = []

        # Extract primary source mentions from papers
        source_mentions = {}
        for paper in papers:
            for source_type in paper.primary_source_references:
                source_mentions[source_type] = source_mentions.get(source_type, 0) + 1

        # Topic-specific archive recommendations
        topic_archives = {
            'scientific_revolutions_global': [
                {
                    'name': 'House of Wisdom Digital Archive',
                    'location': 'Baghdad, Iraq (digitized)',
                    'focus': 'Islamic scientific manuscripts',
                    'accessibility': 'Open access online',
                    'relevance': 'High - Islamic Golden Age scientific texts'
                },
                {
                    'name': 'Chinese Academy of Sciences Historical Archive',
                    'location': 'Beijing, China',
                    'focus': 'Chinese scientific and medical texts',
                    'accessibility': 'Researcher access required',
                    'relevance': 'High - Traditional Chinese knowledge systems'
                }
            ],
            'enlightenment_cross_cultural': [
                {
                    'name': 'Archives Nationales de France',
                    'location': 'Paris, France',
                    'focus': 'Enlightenment correspondence and documents',
                    'accessibility': 'Public access with registration',
                    'relevance': 'High - Core Enlightenment texts'
                },
                {
                    'name': 'Haitian National Archives',
                    'location': 'Port-au-Prince, Haiti',
                    'focus': 'Revolutionary and constitutional documents',
                    'accessibility': 'Limited access, digitization needed',
                    'relevance': 'Critical - Non-European Enlightenment responses'
                }
            ],
            'tokugawa_social_transformation': [
                {
                    'name': 'National Diet Library Japan',
                    'location': 'Tokyo, Japan',
                    'focus': 'Tokugawa period documents and diaries',
                    'accessibility': 'Researcher access available',
                    'relevance': 'Essential - Primary Tokugawa sources'
                }
            ],
            'haitian_revolution_global_impact': [
                {
                    'name': 'Archives Nationales d\'Haïti',
                    'location': 'Port-au-Prince, Haiti',
                    'focus': 'Revolutionary documents and diaspora records',
                    'accessibility': 'Limited access',
                    'relevance': 'Critical - Core revolutionary sources'
                },
                {
                    'name': 'New Orleans Public Library Louisiana Division',
                    'location': 'New Orleans, USA',
                    'focus': 'Haitian refugee community records',
                    'accessibility': 'Public access',
                    'relevance': 'High - Diaspora documentation'
                }
            ],
            'decolonization_comparative': [
                {
                    'name': 'National Archives of India',
                    'location': 'New Delhi, India',
                    'focus': 'Independence movement documents',
                    'accessibility': 'Researcher access',
                    'relevance': 'High - Indian independence model'
                },
                {
                    'name': 'Ghana National Archives',
                    'location': 'Accra, Ghana',
                    'focus': 'African independence movement records',
                    'accessibility': 'Public access with restrictions',
                    'relevance': 'High - African decolonization'
                }
            ]
        }

        # Add topic-specific recommendations
        if topic.topic_id in topic_archives:
            recommendations.extend(topic_archives[topic.topic_id])

        # Add general recommendations based on source mentions
        if 'manuscript' in source_mentions:
            recommendations.append({
                'name': 'Digital Manuscript Collections',
                'location': 'Various online repositories',
                'focus': 'Digitized historical manuscripts',
                'accessibility': 'Open access',
                'relevance': f'Medium - {source_mentions["manuscript"]} papers mention manuscripts'
            })

        return recommendations

    async def _identify_research_gaps(self, papers: List[ScholarPaper], topic: ResearchTopic) -> List[str]:
        """Identify research gaps based on analysis"""

        gaps = []

        # Check regional representation
        represented_regions = set()
        for paper in papers:
            represented_regions.update(paper.countries)

        missing_regions = set(topic.target_regions) - represented_regions
        if missing_regions:
            gaps.append(f"Limited research on {', '.join(missing_regions)} perspectives")

        # Check cultural perspectives
        western_bias = sum(1 for paper in papers if paper.cultural_perspective == 'Western')
        if western_bias > len(papers) * 0.7:
            gaps.append("Overrepresentation of Western perspectives, need more diverse viewpoints")

        # Check primary source usage
        primary_source_papers = sum(1 for paper in papers if paper.primary_source_references)
        if primary_source_papers < len(papers) * 0.3:
            gaps.append("Limited primary source analysis, more archival research needed")

        # Check methodological diversity
        methodologies = set()
        for paper in papers:
            if paper.research_methodology:
                methodologies.add(paper.research_methodology)

        if len(methodologies) < 3:
            gaps.append("Limited methodological diversity, consider interdisciplinary approaches")

        # Topic-specific gaps
        if topic.topic_id == 'scientific_revolutions_global':
            if 'China' not in represented_regions:
                gaps.append("Missing Chinese scientific tradition perspectives")
            if 'Middle East' not in represented_regions:
                gaps.append("Insufficient coverage of Islamic scientific contributions")

        elif topic.topic_id == 'enlightenment_cross_cultural':
            if not any('women' in paper.title.lower() or 'gender' in paper.title.lower() for paper in papers):
                gaps.append("Limited analysis of women's roles in Enlightenment")

        elif topic.topic_id == 'tokugawa_social_transformation':
            if not any('art' in paper.title.lower() for paper in papers):
                gaps.append("Insufficient coverage of artistic transformation")

        return gaps

    async def _generate_methodology_recommendations(self, papers: List[ScholarPaper], topic: ResearchTopic) -> List[str]:
        """Generate methodology recommendations based on research analysis"""

        recommendations = []

        # Analyze current methodologies
        used_methodologies = set()
        for paper in papers:
            if paper.research_methodology:
                used_methodologies.add(paper.research_methodology)

        # Recommend missing methodologies
        all_methodologies = {'Archival', 'Oral History', 'Comparative', 'Quantitative', 'Qualitative', 'Digital Humanities'}
        missing_methodologies = all_methodologies - used_methodologies

        if 'Archival' in missing_methodologies:
            recommendations.append("Incorporate archival research from multiple cultural contexts")

        if 'Oral History' in missing_methodologies:
            recommendations.append("Conduct oral history interviews where applicable")

        if 'Comparative' in missing_methodologies:
            recommendations.append("Utilize comparative historical methodology across regions")

        if 'Digital Humanities' in missing_methodologies:
            recommendations.append("Apply digital humanities tools for large-scale text analysis")

        # Topic-specific recommendations
        if topic.topic_id == 'scientific_revolutions_global':
            recommendations.append("Use multilingual source analysis for non-European texts")
            recommendations.append("Apply network analysis to trace knowledge transfer")

        elif topic.topic_id == 'enlightenment_cross_cultural':
            recommendations.append("Employ gender analysis framework for women's contributions")
            recommendations.append("Use postcolonial theory to examine non-European responses")

        elif topic.topic_id == 'tokugawa_social_transformation':
            recommendations.append("Integrate art historical analysis with social history")
            recommendations.append("Use microhistory approach for individual case studies")

        elif topic.topic_id == 'haitian_revolution_global_impact':
            recommendations.append("Map diaspora networks using geographic information systems")
            recommendations.append("Analyze revolutionary discourse across multiple languages")

        elif topic.topic_id == 'decolonization_comparative':
            recommendations.append("Develop systematic comparison framework across regions")
            recommendations.append("Use political science theories of state formation")

        return recommendations

    async def _extract_cross_cultural_insights(self, papers: List[ScholarPaper], topic: ResearchTopic) -> List[str]:
        """Extract cross-cultural insights from research"""

        insights = []

        # Analyze cultural perspectives represented
        perspectives = {}
        for paper in papers:
            if paper.cultural_perspective:
                perspectives[paper.cultural_perspective] = perspectives.get(paper.cultural_perspective, 0) + 1

        # Generate insights based on representation
        if 'Global' in perspectives and 'Non-Western' in perspectives:
            insights.append("Research shows increasing attention to global and non-Western perspectives")

        if 'Postcolonial' in perspectives:
            insights.append("Postcolonial analysis reveals alternative narratives to traditional historical accounts")

        # Topic-specific insights
        if topic.topic_id == 'scientific_revolutions_global':
            insights.append("Scientific knowledge transfer occurred through multiple cultural networks, not just European")
            insights.append("Islamic, Chinese, and Indian scientific traditions contributed significantly to global knowledge")

        elif topic.topic_id == 'enlightenment_cross_cultural':
            insights.append("Enlightenment ideas were adapted and transformed in different cultural contexts")
            insights.append("Non-European societies developed their own versions of human rights concepts")

        elif topic.topic_id == 'tokugawa_social_transformation':
            insights.append("Social transformation in Japan followed unique patterns distinct from European models")
            insights.append("Women's roles evolved within specifically Japanese cultural frameworks")

        elif topic.topic_id == 'haitian_revolution_global_impact':
            insights.append("Haitian Revolution had global impact on concepts of freedom and citizenship")
            insights.append("Diaspora communities maintained revolutionary ideals across multiple continents")

        elif topic.topic_id == 'decolonization_comparative':
            insights.append("Decolonization strategies varied significantly based on local cultural and political contexts")
            insights.append("Indigenous knowledge systems played crucial roles in independence movements")

        return insights

    async def _assess_research_quality(self, papers: List[ScholarPaper], topic: ResearchTopic) -> Dict[str, Any]:
        """Assess the quality of research results"""

        if not papers:
            return {'overall_score': 0, 'assessment': 'No papers to assess'}

        # Calculate quality metrics
        total_citations = sum(paper.citations for paper in papers)
        avg_citations = total_citations / len(papers)

        # Open access ratio
        open_access_count = sum(1 for paper in papers if paper.is_open_access)
        open_access_ratio = open_access_count / len(papers)

        # Primary source ratio
        primary_source_count = sum(1 for paper in papers if paper.primary_source_references)
        primary_source_ratio = primary_source_count / len(papers)

        # Methodological diversity
        methodologies = set()
        for paper in papers:
            if paper.research_methodology:
                methodologies.add(paper.research_methodology)
        methodology_diversity = len(methodologies) / 6.0  # Max 6 methodologies

        # Cultural diversity
        perspectives = set()
        for paper in papers:
            if paper.cultural_perspective:
                perspectives.add(paper.cultural_perspective)
        cultural_diversity = len(perspectives) / 4.0  # Max 4 perspectives

        # Regional diversity
        regions = set()
        for paper in papers:
            regions.update(paper.countries)
        regional_diversity = len(regions) / len(topic.target_regions) if topic.target_regions else 0

        # Calculate overall score
        citation_score = min(avg_citations / 50.0, 1.0)  # Normalize to max 50 citations
        overall_score = (
            citation_score * 0.2 +
            open_access_ratio * 0.15 +
            primary_source_ratio * 0.25 +
            methodology_diversity * 0.15 +
            cultural_diversity * 0.15 +
            regional_diversity * 0.1
        ) * 10  # Scale to 10

        return {
            'overall_score': overall_score,
            'citation_metrics': {
                'total_citations': total_citations,
                'average_citations': avg_citations,
                'citation_score': citation_score
            },
            'accessibility_metrics': {
                'open_access_ratio': open_access_ratio,
                'open_access_count': open_access_count
            },
            'source_quality': {
                'primary_source_ratio': primary_source_ratio,
                'primary_source_count': primary_source_count
            },
            'diversity_metrics': {
                'methodology_diversity': methodology_diversity,
                'cultural_diversity': cultural_diversity,
                'regional_diversity': regional_diversity,
                'methodologies_used': list(methodologies),
                'perspectives_represented': list(perspectives),
                'regions_covered': list(regions)
            }
        }

    async def _generate_next_steps(self, papers: List[ScholarPaper], topic: ResearchTopic, research_gaps: List[str]) -> List[str]:
        """Generate actionable next steps for research"""

        next_steps = []

        # Based on research gaps
        if any('primary source' in gap.lower() for gap in research_gaps):
            next_steps.append("Begin systematic archival research in recommended collections")

        if any('perspective' in gap.lower() for gap in research_gaps):
            next_steps.append("Seek out scholars and sources from underrepresented regions")

        if any('methodological' in gap.lower() for gap in research_gaps):
            next_steps.append("Develop interdisciplinary research approach")

        # Based on primary source recommendations
        if len([p for p in papers if p.primary_source_references]) > 0:
            next_steps.append("Contact archives for access to primary source collections")

        # Topic-specific next steps
        if topic.topic_id == 'scientific_revolutions_global':
            next_steps.extend([
                "Collaborate with historians of science in China and Middle East",
                "Develop multilingual research capabilities",
                "Plan archival visits to Islamic manuscript collections"
            ])

        elif topic.topic_id == 'enlightenment_cross_cultural':
            next_steps.extend([
                "Research women's correspondence networks across cultures",
                "Examine constitutional documents from non-European contexts",
                "Study indigenous responses to Enlightenment ideas"
            ])

        elif topic.topic_id == 'tokugawa_social_transformation':
            next_steps.extend([
                "Access Japanese court archives for women's writings",
                "Study ukiyo-e art collections for social commentary",
                "Interview descendants of samurai families"
            ])

        elif topic.topic_id == 'haitian_revolution_global_impact':
            next_steps.extend([
                "Map Haitian diaspora communities globally",
                "Collect oral histories from diaspora descendants",
                "Research diplomatic archives for international responses"
            ])

        elif topic.topic_id == 'decolonization_comparative':
            next_steps.extend([
                "Develop systematic comparison framework",
                "Study independence movement archives across regions",
                "Interview independence movement participants"
            ])

        # General next steps
        next_steps.extend([
            "Develop research proposal based on identified gaps",
            "Apply for research grants for archival travel",
            "Establish international scholarly collaborations",
            "Plan primary source digitization projects"
        ])

        return next_steps

    async def stop(self) -> bool:
        """Stop the orchestrator and all MCP servers"""
        try:
            for server in self.mcp_servers.values():
                await server.stop()

            self.status = "stopped"
            logger.info(f"ZeroCostResearchOrchestrator {self.orchestrator_id} stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop ZeroCostResearchOrchestrator: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'orchestrator_id': self.orchestrator_id,
            'status': self.status,
            'mcp_servers': {name: server.get_status() for name, server in self.mcp_servers.items()},
            'statistics': self.stats.copy(),
            'predefined_topics': list(self.predefined_topics.keys())
        }
