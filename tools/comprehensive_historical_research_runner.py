#!/usr/bin/env python3
"""
Comprehensive Historical Research Runner
Processes multiple historical research topics and generates a complete analysis document
with primary sources, summaries, and detailed findings for each topic.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Any

# PyGent Factory imports
from src.orchestration.historical_research_agent import HistoricalResearchAgent
from src.orchestration.research_models import ResearchQuery
from src.orchestration.coordination_models import OrchestrationConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveHistoricalResearcher:
    """
    Comprehensive historical research system that processes multiple topics
    and generates detailed analysis documents with primary sources.
    """
    
    def __init__(self):
        self.config = OrchestrationConfig()
        self.research_agent = HistoricalResearchAgent(self.config)
        self.research_topics = self._define_research_topics()
        self.results = {}
        
    def _define_research_topics(self) -> List[Dict[str, Any]]:
        """Define all research topics with their specific focus areas"""
        return [
            {
                "topic": "Scientific Revolutions (Beyond European Perspective)",
                "subtopics": ["Art & Architecture", "Literacy"],
                "time_period": "1400-1800",
                "geographic_scope": ["Global", "Asia", "Africa", "Americas", "Middle East"],
                "focus": "primary_sources",
                "research_type": "cultural"
            },
            {
                "topic": "Enlightenment",
                "subtopics": ["Shifting notions of human rights", "Shifting political values"],
                "time_period": "1650-1800",
                "geographic_scope": ["Europe", "Americas", "Global"],
                "focus": "primary_sources",
                "research_type": "political"
            },
            {
                "topic": "Early Modern Exploration",
                "subtopics": ["Cartography", "Flora & Fauna", "Describing Otherness"],
                "time_period": "1450-1700",
                "geographic_scope": ["Global", "Africa", "Asia", "Americas"],
                "focus": "primary_sources",
                "research_type": "cultural"
            },
            {
                "topic": "Tokugawa Japan",
                "subtopics": ["Women", "Art", "Shifting Role of Samurai"],
                "time_period": "1603-1868",
                "geographic_scope": ["Japan"],
                "focus": "primary_sources",
                "research_type": "social"
            },
            {
                "topic": "Colonialism in Southeast Asia",
                "subtopics": ["Non-European Response", "European life abroad"],
                "time_period": "1500-1900",
                "geographic_scope": ["Indonesia", "Philippines", "Vietnam", "Thailand", "Malaysia"],
                "focus": "primary_sources",
                "research_type": "political"
            },
            {
                "topic": "Ming & Qing Dynasties",
                "subtopics": ["Education"],
                "time_period": "1368-1912",
                "geographic_scope": ["China"],
                "focus": "primary_sources",
                "research_type": "social"
            },
            {
                "topic": "Haitian Revolution",
                "subtopics": ["Influences of Haitian Diasporas"],
                "time_period": "1791-1804",
                "geographic_scope": ["Haiti", "France", "United States", "Caribbean"],
                "focus": "primary_sources",
                "research_type": "revolutionary"
            },
            {
                "topic": "Opposition to Imperialism (China, African nations)",
                "subtopics": ["Resistance movements", "Anti-colonial thought"],
                "time_period": "1800-1950",
                "geographic_scope": ["China", "Africa", "Global"],
                "focus": "primary_sources",
                "research_type": "political"
            },
            {
                "topic": "Opposition to Imperialism (European, Latin American, Southeast Asian, African)",
                "subtopics": ["Anti-imperial movements", "Decolonization thought"],
                "time_period": "1800-1960",
                "geographic_scope": ["Global", "Europe", "Latin America", "Southeast Asia", "Africa"],
                "focus": "primary_sources",
                "research_type": "political"
            },
            {
                "topic": "World's Fairs & Zoos",
                "subtopics": ["Cultural exhibitions", "Imperial displays", "Public reception"],
                "time_period": "1850-1950",
                "geographic_scope": ["Global", "Europe", "Americas"],
                "focus": "primary_sources",
                "research_type": "cultural"
            },
            {
                "topic": "Eugenics in Global Perspective",
                "subtopics": ["Scientific racism", "Policy implementation", "Resistance"],
                "time_period": "1880-1945",
                "geographic_scope": ["Global", "Europe", "Americas", "Asia"],
                "focus": "primary_sources",
                "research_type": "social"
            },
            {
                "topic": "Globalization, Imperialism, and Modern Dictatorships",
                "subtopics": ["Economic integration", "Political control", "Social transformation"],
                "time_period": "1870-1950",
                "geographic_scope": ["Global"],
                "focus": "primary_sources",
                "research_type": "political"
            },
            {
                "topic": "Decolonization (Global Perspectives)",
                "subtopics": ["Independence movements", "Post-colonial transitions", "Global impact"],
                "time_period": "1945-1975",
                "geographic_scope": ["Global", "Africa", "Asia", "Americas"],
                "focus": "primary_sources",
                "research_type": "political"
            }
        ]
    
    async def conduct_comprehensive_research(self) -> Dict[str, Any]:
        """Conduct research on all topics and compile comprehensive results"""
        logger.info("Starting comprehensive historical research on all topics...")
        
        for topic_data in self.research_topics:
            logger.info(f"Researching: {topic_data['topic']}")
            
            # Create research query for main topic
            main_query = ResearchQuery(
                topic=topic_data["topic"],
                domain="historical",
                metadata={
                    "time_period": topic_data["time_period"],
                    "geographic_scope": topic_data["geographic_scope"],
                    "focus": topic_data["focus"],
                    "research_type": topic_data["research_type"],
                    "subtopics": topic_data["subtopics"]
                }
            )
            
            # Conduct main research
            main_analysis = await self.research_agent.conduct_historical_research(main_query)
            
            # Research each subtopic
            subtopic_analyses = {}
            for subtopic in topic_data["subtopics"]:
                logger.info(f"  - Researching subtopic: {subtopic}")
                subtopic_query = ResearchQuery(
                    topic=f"{topic_data['topic']}: {subtopic}",
                    domain="historical",
                    metadata={
                        "time_period": topic_data["time_period"],
                        "geographic_scope": topic_data["geographic_scope"],
                        "focus": "primary_sources",
                        "research_type": topic_data["research_type"],
                        "parent_topic": topic_data["topic"]
                    }
                )
                subtopic_analysis = await self.research_agent.conduct_historical_research(subtopic_query)
                subtopic_analyses[subtopic] = subtopic_analysis
            
            # Store comprehensive results
            self.results[topic_data["topic"]] = {
                "main_analysis": main_analysis,
                "subtopic_analyses": subtopic_analyses,
                "topic_metadata": topic_data,
                "research_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Completed research for: {topic_data['topic']}")
        
        logger.info("Comprehensive research completed for all topics!")
        return self.results
    
    def generate_comprehensive_document(self, output_path: str = "comprehensive_historical_research_analysis.md"):
        """Generate a comprehensive markdown document with all research findings"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Document header
            f.write("# Comprehensive Historical Research Analysis\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n\n")
            f.write("**Research Focus:** Primary Sources and Global Perspectives\n\n")
            f.write("---\n\n")
            
            # Table of contents
            f.write("## Table of Contents\n\n")
            for i, topic in enumerate(self.results.keys(), 1):
                f.write(f"{i}. [{topic}](#{topic.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(',', '').replace('&', 'and')})\n")
            f.write("\n---\n\n")
            
            # Research findings for each topic
            for topic_name, topic_data in self.results.items():
                self._write_topic_section(f, topic_name, topic_data)
        
        logger.info(f"Comprehensive document generated: {output_path}")
        return output_path
    
    def _write_topic_section(self, file_handle, topic_name: str, topic_data: Dict[str, Any]):
        """Write a complete section for a research topic"""
        
        main_analysis = topic_data["main_analysis"]
        subtopic_analyses = topic_data["subtopic_analyses"]
        metadata = topic_data["topic_metadata"]
        
        # Topic header
        file_handle.write(f"## {topic_name}\n\n")
        
        # Topic overview
        file_handle.write("### Overview\n\n")
        file_handle.write(f"**Time Period:** {metadata['time_period']}\n\n")
        file_handle.write(f"**Geographic Scope:** {', '.join(metadata['geographic_scope'])}\n\n")
        file_handle.write(f"**Research Type:** {metadata['research_type'].title()}\n\n")
        file_handle.write(f"**Subtopics Analyzed:** {', '.join(metadata['subtopics'])}\n\n")
        
        # Historical context
        if main_analysis.historical_context:
            file_handle.write("### Historical Context\n\n")
            file_handle.write(f"{main_analysis.historical_context}\n\n")
        
        # Primary sources section
        file_handle.write("### Primary Sources\n\n")
        self._write_primary_sources_section(file_handle, main_analysis, subtopic_analyses)
        
        # Research summary
        file_handle.write("### Research Summary\n\n")
        self._write_research_summary(file_handle, main_analysis, subtopic_analyses)
        
        # Subtopic analyses
        if subtopic_analyses:
            file_handle.write("### Detailed Subtopic Analysis\n\n")
            for subtopic, analysis in subtopic_analyses.items():
                file_handle.write(f"#### {subtopic}\n\n")
                
                # Subtopic context
                if analysis.historical_context:
                    file_handle.write(f"**Context:** {analysis.historical_context[:500]}...\n\n")
                
                # Subtopic primary sources
                self._write_subtopic_primary_sources(file_handle, analysis)
                
                # Key findings
                if analysis.key_themes:
                    file_handle.write(f"**Key Themes:** {', '.join(analysis.key_themes)}\n\n")
        
        # Timeline
        if main_analysis.timeline and main_analysis.timeline.events:
            file_handle.write("### Timeline\n\n")
            file_handle.write(f"**{main_analysis.timeline.title}**\n\n")
            file_handle.write(f"{main_analysis.timeline.description}\n\n")
            
            for event in main_analysis.timeline.events[:10]:  # Limit to top 10 events
                date_str = event.date_start.strftime("%Y") if event.date_start else "Unknown"
                file_handle.write(f"- **{date_str}:** {event.name}\n")
            file_handle.write("\n")
        
        # Comparative analysis
        if main_analysis.comparative_analysis:
            file_handle.write("### Comparative Analysis\n\n")
            file_handle.write(f"{main_analysis.comparative_analysis}\n\n")
        
        # Alternative narratives
        if main_analysis.alternative_narratives:
            file_handle.write("### Alternative Narratives\n\n")
            for narrative in main_analysis.alternative_narratives:
                file_handle.write(f"- {narrative}\n")
            file_handle.write("\n")
        
        # Research gaps and recommendations
        file_handle.write("### Research Gaps & Recommendations\n\n")
        
        if main_analysis.research_gaps:
            file_handle.write("**Identified Gaps:**\n")
            for gap in main_analysis.research_gaps:
                file_handle.write(f"- {gap}\n")
            file_handle.write("\n")
        
        if main_analysis.recommendations:
            file_handle.write("**Recommendations for Further Research:**\n")
            for rec in main_analysis.recommendations:
                file_handle.write(f"- {rec}\n")
            file_handle.write("\n")
        
        # Confidence metrics
        if main_analysis.confidence_metrics:
            file_handle.write("### Research Quality Metrics\n\n")
            metrics = main_analysis.confidence_metrics
            file_handle.write(f"- **Overall Confidence:** {metrics.get('overall_confidence', 0):.2f}\n")
            file_handle.write(f"- **Source Credibility:** {metrics.get('source_credibility', 0):.2f}\n")
            file_handle.write(f"- **Peer-Reviewed Sources:** {metrics.get('peer_reviewed_ratio', 0):.1%}\n")
            file_handle.write("\n")
        
        file_handle.write("---\n\n")    
    def _write_primary_sources_section(self, file_handle, main_analysis, subtopic_analyses):
        """Write the primary sources section with links when available"""
        
        # No fake primary source generation - system is honest about limitations
        file_handle.write("*Primary sources research in progress - additional database integrations needed for direct links.*\n\n")    
    def _write_subtopic_primary_sources(self, file_handle, analysis):
        """Write primary sources for a subtopic"""
        # No fake source generation - system is honest about current limitations
        file_handle.write("*Primary source integration pending - real API implementations needed.*\n\n")
    
    def _write_research_summary(self, file_handle, main_analysis, subtopic_analyses):
        """Write a comprehensive research summary"""
        
        # Main findings
        file_handle.write("**Key Findings:**\n\n")
        
        if main_analysis.key_themes:
            file_handle.write(f"- **Major Themes:** {', '.join(main_analysis.key_themes)}\n")
        
        # Event count and timeline span
        event_count = len(main_analysis.events) if main_analysis.events else 0
        file_handle.write(f"- **Historical Events Analyzed:** {event_count}\n")
        
        if main_analysis.timeline:
            timeline_span = "Unknown"
            if main_analysis.timeline.time_range_start and main_analysis.timeline.time_range_end:
                start_year = main_analysis.timeline.time_range_start.year
                end_year = main_analysis.timeline.time_range_end.year
                timeline_span = f"{start_year}-{end_year}"
            file_handle.write(f"- **Timeline Span:** {timeline_span}\n")
        
        # Geographic coverage
        if main_analysis.timeline and main_analysis.timeline.geographical_scope:
            file_handle.write(f"- **Geographic Coverage:** {', '.join(main_analysis.timeline.geographical_scope)}\n")
        
        file_handle.write("\n")
          # Subtopic summaries
        if subtopic_analyses:
            file_handle.write("**Subtopic Summary:**\n\n")
            for subtopic, analysis in subtopic_analyses.items():
                themes = ', '.join(analysis.key_themes[:3]) if analysis.key_themes else "Various themes"
                file_handle.write(f"- **{subtopic}:** {themes}\n")
            file_handle.write("\n")
    
    def save_raw_results(self, output_path: str = "raw_research_results.json"):
        """Save raw research results as JSON for further analysis"""
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for topic, data in self.results.items():
            serializable_results[topic] = {
                "topic_metadata": data["topic_metadata"],
                "research_timestamp": data["research_timestamp"],
                "main_analysis_summary": {
                    "key_themes": data["main_analysis"].key_themes,
                    "event_count": len(data["main_analysis"].events),
                    "historical_context": data["main_analysis"].historical_context[:500] + "..." if data["main_analysis"].historical_context else "",
                    "confidence_metrics": data["main_analysis"].confidence_metrics,
                    "research_gaps": data["main_analysis"].research_gaps,
                    "recommendations": data["main_analysis"].recommendations
                },
                "subtopic_summaries": {
                    subtopic: {
                        "key_themes": analysis.key_themes,
                        "event_count": len(analysis.events)
                    }
                    for subtopic, analysis in data["subtopic_analyses"].items()
                }
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Raw results saved: {output_path}")
        return output_path

async def main():
    """Main execution function"""
    print("Starting Comprehensive Historical Research Analysis...")
    print("This will analyze all your research topics with focus on primary sources")
    print("Estimated time: 10-15 minutes for complete analysis\n")
    
    researcher = ComprehensiveHistoricalResearcher()
    
    # Conduct comprehensive research
    results = await researcher.conduct_comprehensive_research()
    
    # Generate comprehensive document    print("\nGenerating comprehensive research document...")
    doc_path = researcher.generate_comprehensive_document()
    
    # Save raw results
    json_path = researcher.save_raw_results()
    
    print("\nResearch Complete!")
    print(f"Comprehensive Document: {doc_path}")
    print(f"Raw Data: {json_path}")
    print(f"Topics Analyzed: {len(results)}")
    
    return doc_path, json_path

if __name__ == "__main__":
    asyncio.run(main())
