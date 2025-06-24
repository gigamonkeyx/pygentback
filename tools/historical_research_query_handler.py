"""
Historical Research Query Handler
Specialized handler for comprehensive multi-sectional historical research

This script processes complex historical research queries with:
- Primary source validation and documentation
- Global perspectives beyond Eurocentric narratives
- Comprehensive coverage of multiple historical periods and topics
- Source quality assessment and cross-referencing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HistoricalResearchTopic:
    """Represents a historical research topic with validation requirements"""
    title: str
    description: str
    period: str
    geographic_focus: List[str]
    subtopics: List[Dict[str, str]]
    primary_source_types: List[str]
    validation_criteria: List[str]


class HistoricalResearchQueryHandler:
    """
    Handler for comprehensive historical research queries
    Focuses on primary source retrieval and validation
    """
    
    def __init__(self):
        self.research_topics = self._initialize_research_topics()
        self.execution_timestamp = datetime.now()
    
    def _initialize_research_topics(self) -> List[HistoricalResearchTopic]:
        """Initialize all research topics from the user's query"""
        
        topics = [
            # Scientific Revolutions (Global Perspective)
            HistoricalResearchTopic(
                title="Scientific Revolutions: Beyond European Perspectives",
                description="Global perspectives on scientific advancement and knowledge exchange",
                period="1400-1800 CE",
                geographic_focus=["Global", "Asia", "Middle East", "Africa", "Americas", "Europe"],
                subtopics=[
                    {
                        "topic": "Art & Architecture", 
                        "focus": "Scientific influence on artistic expression and architectural innovation globally"
                    },
                    {
                        "topic": "Literacy", 
                        "focus": "Scientific knowledge dissemination and changing literacy patterns"
                    }
                ],
                primary_source_types=[
                    "Architectural treatises and building plans",
                    "Scientific manuscripts and translations",
                    "Artist correspondence and workshop records",
                    "Educational curriculum documents",
                    "Publishing records and book inventories",
                    "Scientific society membership records"
                ],
                validation_criteria=[
                    "Verify manuscript authenticity and dating",
                    "Cross-reference with multiple cultural sources",
                    "Check translation accuracy for non-European texts",
                    "Validate against established historical timeline",
                    "Assess potential cultural bias in documentation"
                ]
            ),
            
            # Enlightenment
            HistoricalResearchTopic(
                title="Enlightenment: Transforming Ideas and Values",
                description="Evolution of human rights concepts and political values",
                period="1650-1800 CE",
                geographic_focus=["Europe", "Americas", "Global"],
                subtopics=[
                    {
                        "topic": "Shifting Notions of Human Rights",
                        "focus": "Development and spread of human rights concepts across cultures"
                    },
                    {
                        "topic": "Shifting Political Values",
                        "focus": "Changes in governance concepts, popular sovereignty, and political legitimacy"
                    }
                ],
                primary_source_types=[
                    "Philosophical treatises and essays",
                    "Constitutional documents and legal codes",
                    "Political pamphlets and broadsides",
                    "Personal correspondence of key thinkers",
                    "Government records and legislative proceedings",
                    "Revolutionary documents and declarations"
                ],
                validation_criteria=[
                    "Verify authorship and publication circumstances",
                    "Cross-reference ideas across multiple thinkers",
                    "Check for contemporary critiques and responses",
                    "Validate historical impact and influence",
                    "Assess translation quality for non-English sources"
                ]
            ),
            
            # Early Modern Exploration
            HistoricalResearchTopic(
                title="Early Modern Exploration: Knowledge and Cultural Encounter",
                description="Exploration, cartography, natural history, and cultural description",
                period="1450-1750 CE",
                geographic_focus=["Global", "Americas", "Asia", "Africa", "Pacific"],
                subtopics=[
                    {
                        "topic": "Cartography",
                        "focus": "Evolution of mapmaking, geographic knowledge, and spatial understanding"
                    },
                    {
                        "topic": "Flora & Fauna",
                        "focus": "Documentation, classification, and study of new species and ecosystems"
                    },
                    {
                        "topic": "Describing Otherness",
                        "focus": "European perceptions, descriptions, and representations of other cultures"
                    }
                ],
                primary_source_types=[
                    "Original maps, atlases, and navigation charts",
                    "Navigation logs and expedition journals",
                    "Botanical and zoological illustrations",
                    "Naturalist field notes and specimen catalogs",
                    "Travel accounts and exploration memoirs",
                    "Missionary reports and cultural observations",
                    "Diplomatic correspondence and trade records"
                ],
                validation_criteria=[
                    "Verify map authenticity and cartographic accuracy",
                    "Cross-reference multiple expedition accounts",
                    "Assess bias in cultural descriptions and observations",
                    "Check scientific accuracy of natural history records",
                    "Validate geographic and temporal details"
                ]
            ),
            
            # Tokugawa Japan
            HistoricalResearchTopic(
                title="Tokugawa Japan: Society and Culture in Isolation",
                description="Social structures, cultural development, and change in Edo period Japan",
                period="1603-1868 CE",
                geographic_focus=["Japan"],
                subtopics=[
                    {
                        "topic": "Women",
                        "focus": "Women's roles, rights, experiences, and agency in Tokugawa society"
                    },
                    {
                        "topic": "Art",
                        "focus": "Artistic development, cultural expression, and aesthetic evolution"
                    },
                    {
                        "topic": "Shifting Role of Samurai",
                        "focus": "Transformation from military warriors to administrative bureaucrats"
                    }
                ],
                primary_source_types=[
                    "Women's diaries and personal writings",
                    "Family records and genealogical documents",
                    "Legal documents and court records",
                    "Literary works and poetry by women",
                    "Original artworks and their documentation",
                    "Artist biographies and workshop records",
                    "Patronage records and artistic contracts",
                    "Samurai family records and service documents",
                    "Government edicts and administrative records"
                ],
                validation_criteria=[
                    "Verify Japanese text authenticity and translation accuracy",
                    "Cross-reference with multiple Japanese archives",
                    "Check temporal consistency with Tokugawa chronology",
                    "Validate social and cultural context",
                    "Assess preservation status and accessibility"
                ]
            ),
            
            # Southeast Asian Colonialism
            HistoricalResearchTopic(
                title="Colonialism in Southeast Asia: Responses and Experiences",
                description="Colonial encounters, indigenous responses, and European experiences",
                period="1500-1945 CE",
                geographic_focus=["Indonesia", "Philippines", "Malaysia", "Vietnam", "Thailand", "Burma"],
                subtopics=[
                    {
                        "topic": "Non-European Response",
                        "focus": "Indigenous and local responses, resistance, and adaptation to colonial rule"
                    },
                    {
                        "topic": "European Life Abroad",
                        "focus": "Daily life, experiences, and perspectives of European colonists and administrators"
                    }
                ],
                primary_source_types=[
                    "Resistance movement documents and manifestos",
                    "Local administrative and court records",
                    "Personal accounts of indigenous leaders",
                    "Traditional literature and oral histories",
                    "European colonial diaries and letters",
                    "Colonial administration official records",
                    "Trading company documents and reports",
                    "Missionary accounts and correspondence"
                ],
                validation_criteria=[
                    "Verify authenticity of indigenous sources",
                    "Check translation quality for local languages",
                    "Cross-reference colonial and local perspectives",
                    "Assess colonial bias in European sources",
                    "Validate dates and geographic specificity"
                ]
            ),
            
            # Ming & Qing Dynasties
            HistoricalResearchTopic(
                title="Ming & Qing Dynasties: Imperial China's Educational Evolution",
                description="Educational systems, examinations, and intellectual development",
                period="1368-1912 CE",
                geographic_focus=["China"],
                subtopics=[
                    {
                        "topic": "Education",
                        "focus": "Educational systems, imperial examinations, scholarly culture, and intellectual life"
                    }
                ],
                primary_source_types=[
                    "Imperial examination records and essays",
                    "Educational treatises and textbooks",
                    "Scholar personal writings and correspondence",
                    "Government edicts on education policy",
                    "Academy records and curricula",
                    "Student writings and examination responses"
                ],
                validation_criteria=[
                    "Verify Chinese text authenticity and translation quality",
                    "Cross-reference with multiple Chinese historical sources",
                    "Check chronological consistency across dynasties",
                    "Validate against established imperial records",
                    "Assess scholarly consensus on interpretations"
                ]
            ),
            
            # Haitian Revolution
            HistoricalResearchTopic(
                title="Haitian Revolution: Revolution and Global Diaspora",
                description="Revolutionary processes and international influence of Haitian diaspora",
                period="1791-1825 CE",
                geographic_focus=["Haiti", "Caribbean", "USA", "France", "Latin America"],
                subtopics=[
                    {
                        "topic": "Influences of Haitian Diasporas",
                        "focus": "Impact and influence of Haitian refugees and emigrants in specific locations"
                    }
                ],
                primary_source_types=[
                    "Revolutionary documents and proclamations",
                    "Personal accounts from revolution participants",
                    "Immigration and refugee records",
                    "Diaspora community documents",
                    "Contemporary newspaper coverage",
                    "Diplomatic correspondence",
                    "Personal memoirs and letters"
                ],
                validation_criteria=[
                    "Verify authenticity of revolutionary documents",
                    "Cross-reference multiple contemporary accounts",
                    "Check for revolutionary and colonial bias",
                    "Validate geographic spread of diaspora influence",
                    "Assess impact on receiving communities"
                ]
            ),
            
            # Opposition to Imperialism
            HistoricalResearchTopic(
                title="Opposition to Imperialism: Global Resistance Movements",
                description="Anti-imperial resistance across China, Africa, Europe, Latin America, and Southeast Asia",
                period="1800-1950 CE",
                geographic_focus=["China", "Africa", "Europe", "Latin America", "Southeast Asia"],
                subtopics=[
                    {
                        "topic": "Chinese Anti-Imperial Resistance",
                        "focus": "Chinese responses to foreign imperialism and territorial encroachment"
                    },
                    {
                        "topic": "African Anti-Imperial Resistance",
                        "focus": "African responses to European colonialism and resistance movements"
                    },
                    {
                        "topic": "Global Anti-Imperial Networks",
                        "focus": "International connections and coordination between resistance movements"
                    }
                ],
                primary_source_types=[
                    "Resistance movement documents and manifestos",
                    "Anti-imperial speeches and writings",
                    "Government suppression records",
                    "International correspondence between movements",
                    "Anti-imperial publications and newspapers",
                    "Conference proceedings and meeting records",
                    "Personal networks and correspondence"
                ],
                validation_criteria=[
                    "Verify authenticity of resistance documents",
                    "Cross-reference colonial and resistance perspectives",
                    "Check for propaganda and political bias",
                    "Validate international connections and timing",
                    "Assess impact and effectiveness of movements"
                ]
            ),
            
            # World's Fair & Zoos
            HistoricalResearchTopic(
                title="World's Fairs & Zoos: Spectacle and Imperial Display",
                description="Imperial exhibitions, cultural display, and public spectacle",
                period="1850-1950 CE",
                geographic_focus=["Global", "Europe", "USA", "Colonial territories"],
                subtopics=[
                    {
                        "topic": "Imperial Exhibitions",
                        "focus": "Display of colonial peoples, cultures, and imperial achievements"
                    },
                    {
                        "topic": "Zoological and Human Displays",
                        "focus": "Exhibition of people, animals, and 'exotic' cultures for public consumption"
                    }
                ],
                primary_source_types=[
                    "Exhibition catalogs and official programs",
                    "Visitor accounts and contemporary reviews",
                    "Official exhibition planning records",
                    "Photographic and visual documentation",
                    "Zoo records and acquisition documents",
                    "Scientific society proceedings",
                    "Press coverage and publicity materials"
                ],
                validation_criteria=[
                    "Verify authenticity of exhibition materials",
                    "Cross-reference multiple visitor perspectives",
                    "Check for promotional bias in official records",
                    "Validate photographic evidence and context",
                    "Assess ethical implications and contemporary critiques"
                ]
            ),
            
            # Eugenics in Global Perspective
            HistoricalResearchTopic(
                title="Eugenics: Global Movement and Local Adaptations",
                description="International eugenics movement and its adaptation across cultures",
                period="1880-1945 CE",
                geographic_focus=["Global", "USA", "Europe", "Asia", "Latin America", "Australia"],
                subtopics=[
                    {
                        "topic": "International Eugenics Movement",
                        "focus": "Global spread, adaptation, and implementation of eugenic ideas and policies"
                    }
                ],
                primary_source_types=[
                    "Scientific publications and eugenic journals",
                    "International conference proceedings",
                    "Government policy documents and legislation",
                    "Personal correspondence of eugenicists",
                    "University and research institution records",
                    "Public health and medical records"
                ],
                validation_criteria=[
                    "Verify scientific publication authenticity",
                    "Cross-reference policies across nations",
                    "Check for scientific bias and flawed methodology",
                    "Validate international connections and influence",
                    "Assess ethical implications and later condemnation"
                ]
            ),
            
            # Globalization, Imperialism, and Modern Dictatorships
            HistoricalResearchTopic(
                title="Globalization, Imperialism, and Rise of Modern Dictatorships",
                description="Connections between imperial collapse, globalization, and authoritarian emergence",
                period="1900-1950 CE",
                geographic_focus=["Global", "Europe", "Asia", "Latin America", "Africa"],
                subtopics=[
                    {
                        "topic": "Rise of Modern Dictatorships",
                        "focus": "Connection between imperial collapse, economic crisis, and authoritarian rise"
                    }
                ],
                primary_source_types=[
                    "Government documents and official records",
                    "Dictatorial speeches and propaganda materials",
                    "International diplomatic correspondence",
                    "Personal memoirs and contemporary accounts",
                    "Economic records and policy documents",
                    "Opposition writings and resistance materials"
                ],
                validation_criteria=[
                    "Verify authenticity of government documents",
                    "Cross-reference multiple international sources",
                    "Check for propaganda and authoritarian bias",
                    "Validate chronological connections and causation",
                    "Assess multiple perspectives on historical events"
                ]
            ),
            
            # Decolonization
            HistoricalResearchTopic(
                title="Decolonization: Global Perspectives and Case Studies",
                description="Independence movements, decolonization processes, and their global impact",
                period="1945-1975 CE",
                geographic_focus=["Global", "Africa", "Asia", "Caribbean", "Pacific", "Middle East"],
                subtopics=[
                    {
                        "topic": "Decolonization Movements",
                        "focus": "Independence movements, their strategies, and international connections"
                    },
                    {
                        "topic": "Case Studies",
                        "focus": "Detailed analysis of specific decolonization processes and outcomes"
                    }
                ],
                primary_source_types=[
                    "Independence movement documents and manifestos",
                    "Leader speeches and political writings",
                    "International negotiation and mediation records",
                    "Colonial administration final records",
                    "United Nations proceedings and documents",
                    "Personal accounts from independence leaders",
                    "Contemporary press coverage and analysis"
                ],
                validation_criteria=[
                    "Verify authenticity of independence documents",
                    "Cross-reference colonial and independence perspectives",
                    "Check for political bias and propaganda",
                    "Validate international involvement and mediation",
                    "Assess long-term outcomes and impact"
                ]
            )
        ]
        
        return topics
    
    def generate_research_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of all research topics"""
        
        summary = {
            "query_metadata": {
                "execution_time": self.execution_timestamp.isoformat(),
                "total_topics": len(self.research_topics),
                "total_subtopics": sum(len(topic.subtopics) for topic in self.research_topics),
                "geographic_coverage": self._analyze_geographic_coverage(),
                "temporal_coverage": self._analyze_temporal_coverage()
            },
            "research_topics": [],
            "primary_source_requirements": self._compile_primary_source_requirements(),
            "validation_framework": self._compile_validation_framework(),
            "research_recommendations": self._generate_research_recommendations()
        }
        
        # Process each research topic
        for topic in self.research_topics:
            topic_summary = {
                "title": topic.title,
                "description": topic.description,
                "period": topic.period,
                "geographic_focus": topic.geographic_focus,
                "subtopics": topic.subtopics,
                "primary_source_types": topic.primary_source_types,
                "primary_source_count": len(topic.primary_source_types),
                "validation_criteria": topic.validation_criteria,
                "validation_criteria_count": len(topic.validation_criteria),
                "research_complexity": self._assess_research_complexity(topic)
            }
            summary["research_topics"].append(topic_summary)
        
        return summary
    
    def _analyze_geographic_coverage(self) -> Dict[str, int]:
        """Analyze geographic coverage across all topics"""
        geo_coverage = {}
        for topic in self.research_topics:
            for geo_area in topic.geographic_focus:
                geo_coverage[geo_area] = geo_coverage.get(geo_area, 0) + 1
        return dict(sorted(geo_coverage.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_temporal_coverage(self) -> Dict[str, str]:
        """Analyze temporal coverage across all topics"""
        periods = [topic.period for topic in self.research_topics]
        return {
            "earliest_period": min(periods),
            "latest_period": max(periods),
            "total_periods": len(set(periods)),
            "period_distribution": {period: periods.count(period) for period in set(periods)}
        }
    
    def _compile_primary_source_requirements(self) -> Dict[str, Any]:
        """Compile all primary source requirements"""
        all_source_types = []
        for topic in self.research_topics:
            all_source_types.extend(topic.primary_source_types)
        
        unique_source_types = list(set(all_source_types))
        source_frequency = {source: all_source_types.count(source) for source in unique_source_types}
        
        return {
            "total_unique_source_types": len(unique_source_types),
            "most_common_sources": dict(sorted(source_frequency.items(), key=lambda x: x[1], reverse=True)[:10]),
            "source_categories": self._categorize_sources(unique_source_types),
            "archive_requirements": self._identify_archive_requirements()
        }
    
    def _categorize_sources(self, source_types: List[str]) -> Dict[str, List[str]]:
        """Categorize sources by type"""
        categories = {
            "Official Documents": [],
            "Personal Writings": [],
            "Visual Materials": [],
            "Scientific/Academic": [],
            "Literary/Cultural": [],
            "Commercial/Economic": []
        }
        
        for source in source_types:
            source_lower = source.lower()
            if any(term in source_lower for term in ["government", "official", "administrative", "legal", "constitutional"]):
                categories["Official Documents"].append(source)
            elif any(term in source_lower for term in ["diary", "letter", "correspondence", "memoir", "personal"]):
                categories["Personal Writings"].append(source)
            elif any(term in source_lower for term in ["map", "illustration", "photograph", "visual", "artwork"]):
                categories["Visual Materials"].append(source)
            elif any(term in source_lower for term in ["scientific", "academic", "treatise", "journal", "research"]):
                categories["Scientific/Academic"].append(source)
            elif any(term in source_lower for term in ["literature", "poetry", "cultural", "artistic"]):
                categories["Literary/Cultural"].append(source)
            elif any(term in source_lower for term in ["trade", "commercial", "economic", "business"]):
                categories["Commercial/Economic"].append(source)
            else:
                categories["Official Documents"].append(source)  # Default category
        
        return {k: v for k, v in categories.items() if v}  # Remove empty categories
    
    def _identify_archive_requirements(self) -> List[str]:
        """Identify key archives and collections needed"""
        return [
            "National Archives (multiple countries)",
            "University manuscript collections",
            "Museum archives and special collections",
            "Religious institution archives",
            "Corporate and trading company archives",
            "Personal papers and family collections",
            "Government ministry archives",
            "Military and diplomatic archives",
            "Colonial administration archives",
            "Indigenous community archives",
            "International organization archives (UN, etc.)",
            "Newspaper and media archives"
        ]
    
    def _compile_validation_framework(self) -> Dict[str, Any]:
        """Compile comprehensive validation framework"""
        all_criteria = []
        for topic in self.research_topics:
            all_criteria.extend(topic.validation_criteria)
        
        unique_criteria = list(set(all_criteria))
        criteria_frequency = {criterion: all_criteria.count(criterion) for criterion in unique_criteria}
        
        return {
            "total_validation_criteria": len(unique_criteria),
            "most_critical_criteria": dict(sorted(criteria_frequency.items(), key=lambda x: x[1], reverse=True)[:5]),
            "validation_categories": self._categorize_validation_criteria(unique_criteria),
            "quality_assurance_requirements": [
                "Cross-reference multiple independent sources",
                "Verify provenance and chain of custody",
                "Check translation accuracy for non-English sources",
                "Assess potential bias and perspective limitations",
                "Validate chronological and geographic details",
                "Confirm accessibility and citation information",
                "Rate source reliability and scholarly consensus"
            ]
        }
    
    def _categorize_validation_criteria(self, criteria: List[str]) -> Dict[str, List[str]]:
        """Categorize validation criteria by type"""
        categories = {
            "Authenticity Verification": [],
            "Translation and Language": [],
            "Bias Assessment": [],
            "Cross-referencing": [],
            "Temporal Validation": [],
            "Scholarly Consensus": []
        }
        
        for criterion in criteria:
            criterion_lower = criterion.lower()
            if any(term in criterion_lower for term in ["authenticity", "verify", "validation"]):
                categories["Authenticity Verification"].append(criterion)
            elif any(term in criterion_lower for term in ["translation", "language", "text"]):
                categories["Translation and Language"].append(criterion)
            elif any(term in criterion_lower for term in ["bias", "perspective", "propaganda"]):
                categories["Bias Assessment"].append(criterion)
            elif any(term in criterion_lower for term in ["cross-reference", "multiple", "sources"]):
                categories["Cross-referencing"].append(criterion)
            elif any(term in criterion_lower for term in ["chronological", "temporal", "dating", "timeline"]):
                categories["Temporal Validation"].append(criterion)
            elif any(term in criterion_lower for term in ["scholarly", "consensus", "interpretation"]):
                categories["Scholarly Consensus"].append(criterion)
            else:
                categories["Authenticity Verification"].append(criterion)  # Default
        
        return {k: v for k, v in categories.items() if v}
    
    def _assess_research_complexity(self, topic: HistoricalResearchTopic) -> str:
        """Assess the research complexity of a topic"""
        complexity_score = 0
        
        # Geographic scope complexity
        if len(topic.geographic_focus) > 4:
            complexity_score += 2
        elif len(topic.geographic_focus) > 2:
            complexity_score += 1
        
        # Subtopic complexity
        complexity_score += len(topic.subtopics)
        
        # Source type complexity
        if len(topic.primary_source_types) > 6:
            complexity_score += 2
        elif len(topic.primary_source_types) > 3:
            complexity_score += 1
        
        # Validation complexity
        if len(topic.validation_criteria) > 4:
            complexity_score += 2
        elif len(topic.validation_criteria) > 2:
            complexity_score += 1
        
        if complexity_score >= 8:
            return "Very High"
        elif complexity_score >= 6:
            return "High"
        elif complexity_score >= 4:
            return "Medium"
        else:
            return "Low"
    
    def _generate_research_recommendations(self) -> List[str]:
        """Generate research recommendations based on the query"""
        return [
            "Prioritize digitization of identified primary sources for accessibility",
            "Develop partnerships with international archives and libraries",
            "Create collaborative translation projects for non-English sources",
            "Establish cross-cultural research teams with local expertise",
            "Focus on underrepresented voices and perspectives",
            "Develop multimedia presentations combining text, visual, and audio sources",
            "Create educational resources for public engagement",
            "Establish quality control protocols for source validation",
            "Build comprehensive citation and bibliography databases",
            "Plan for long-term preservation of digital research materials",
            "Consider ethical implications of displaying sensitive historical materials",
            "Engage with descendant communities for culturally sensitive research",
            "Develop interdisciplinary approaches combining history with other fields",
            "Create public access portals for research findings",
            "Plan for ongoing updates as new sources are discovered"
        ]
    
    def save_summary(self, summary: Dict[str, Any], filename: str = None) -> str:
        """Save research summary to file"""
        if filename is None:
            timestamp = self.execution_timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"historical_research_summary_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Research summary saved to {filename}")
        return filename
    
    def generate_bibliography_framework(self) -> Dict[str, Any]:
        """Generate framework for comprehensive bibliography"""
        return {
            "primary_sources": {
                "archival_materials": [
                    "Government archives and official documents",
                    "Personal papers and correspondence collections",
                    "Institutional records and administrative documents"
                ],
                "published_primary_sources": [
                    "Contemporary newspapers and periodicals",
                    "Published memoirs and autobiographies",
                    "Official government publications and reports"
                ],
                "visual_and_material_sources": [
                    "Maps, charts, and cartographic materials",
                    "Photographs and visual documentation",
                    "Artifacts and material culture items"
                ]
            },
            "secondary_sources": {
                "scholarly_books": [],
                "academic_articles": [],
                "dissertation_research": []
            },
            "digital_resources": {
                "online_archives": [],
                "digital_collections": [],
                "databases": []
            },
            "citation_style": "Chicago Manual of Style (17th edition)",
            "language_considerations": [
                "Original language sources with translation notes",
                "Translator information and translation quality assessment",
                "Multiple translation comparison where available"
            ]
        }


def main():
    """Main execution function"""
    print("Historical Research Query Handler")
    print("=" * 50)
    print()
    print("Processing comprehensive multi-sectional historical research query...")
    print("Focus: Primary sources with global perspectives beyond Eurocentric narratives")
    print()
    
    # Initialize handler
    handler = HistoricalResearchQueryHandler()
    
    # Generate comprehensive summary
    summary = handler.generate_research_summary()
    
    # Save summary
    filename = handler.save_summary(summary)
    
    # Display results
    print("RESEARCH QUERY ANALYSIS COMPLETE")
    print("-" * 40)
    print(f"Total Research Topics: {summary['query_metadata']['total_topics']}")
    print(f"Total Subtopics: {summary['query_metadata']['total_subtopics']}")
    print(f"Unique Primary Source Types: {summary['primary_source_requirements']['total_unique_source_types']}")
    print(f"Validation Criteria: {summary['validation_framework']['total_validation_criteria']}")
    print()
    
    print("GEOGRAPHIC COVERAGE:")
    for geo, count in list(summary['query_metadata']['geographic_coverage'].items())[:5]:
        print(f"  {geo}: {count} topics")
    print()
    
    print("TEMPORAL COVERAGE:")
    print(f"  Earliest Period: {summary['query_metadata']['temporal_coverage']['earliest_period']}")
    print(f"  Latest Period: {summary['query_metadata']['temporal_coverage']['latest_period']}")
    print(f"  Total Unique Periods: {summary['query_metadata']['temporal_coverage']['total_periods']}")
    print()
    
    print("MOST COMMON PRIMARY SOURCE TYPES:")
    for source, count in list(summary['primary_source_requirements']['most_common_sources'].items())[:5]:
        print(f"  {source}: {count} topics")
    print()
    
    print("RESEARCH COMPLEXITY ANALYSIS:")
    complexity_dist = {}
    for topic in summary['research_topics']:
        complexity = topic['research_complexity']
        complexity_dist[complexity] = complexity_dist.get(complexity, 0) + 1
    
    for complexity, count in sorted(complexity_dist.items(), key=lambda x: ["Low", "Medium", "High", "Very High"].index(x[0])):
        print(f"  {complexity} Complexity: {count} topics")
    print()
    
    print("KEY RECOMMENDATIONS:")
    for i, rec in enumerate(summary['research_recommendations'][:5], 1):
        print(f"  {i}. {rec}")
    print()
    
    print(f"Full detailed analysis saved to: {filename}")
    print()
    print("This framework provides the foundation for executing comprehensive")
    print("historical research with validated primary sources and global perspectives.")
    
    # Generate bibliography framework
    bib_framework = handler.generate_bibliography_framework()
    bib_filename = f"bibliography_framework_{handler.execution_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(bib_filename, 'w', encoding='utf-8') as f:
        json.dump(bib_framework, f, indent=2, ensure_ascii=False)
    
    print(f"Bibliography framework saved to: {bib_filename}")


if __name__ == "__main__":
    main()
