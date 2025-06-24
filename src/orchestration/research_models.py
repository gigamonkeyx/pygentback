"""
Shared research models for PyGent Factory Research System

This module contains shared classes and data structures used across
the research orchestrator and specialized research agents to avoid circular imports.
"""

import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class SourceType(Enum):
    """Types of research sources"""
    ACADEMIC_PAPER = "academic_paper"
    WEB_SOURCE = "web_source"
    BOOK = "book"
    REPORT = "report"
    NEWS_ARTICLE = "news_article"
    GOVERNMENT_DOCUMENT = "government_document"
    DATASET = "dataset"
    INTERVIEW = "interview"
    SURVEY = "survey"
    OTHER = "other"


class OutputFormat(Enum):
    """Output formats for research results"""
    REPORT = "report"
    SUMMARY = "summary"
    PRESENTATION = "presentation"
    ACADEMIC_PAPER = "academic_paper"
    EXECUTIVE_SUMMARY = "executive_summary"
    INFOGRAPHIC = "infographic"
    TIMELINE = "timeline"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


class ResearchPhase(Enum):
    """Phases of the research process"""
    TOPIC_DISCOVERY = "topic_discovery"
    LITERATURE_REVIEW = "literature_review"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    OUTPUT_GENERATION = "output_generation"
    VALIDATION = "validation"
    PEER_REVIEW = "peer_review"


@dataclass
class ResearchQuery:
    """Research query specification"""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    description: str = ""
    domain: str = ""
    keywords: List[str] = field(default_factory=list)
    expected_duration_hours: int = 24
    output_format: OutputFormat = OutputFormat.REPORT
    citation_style: str = "APA"
    language: str = "en"
    depth_level: str = "standard"  # basic, standard, comprehensive, expert
    include_citations: bool = True
    max_sources: int = 50
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchSource:
    """Research source information"""
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    authors: List[str] = field(default_factory=list)
    source_type: SourceType = SourceType.WEB_SOURCE
    url: str = ""
    abstract: str = ""
    content: str = ""
    publication_date: Optional[datetime] = None
    publisher: str = ""
    doi: str = ""
    isbn: str = ""
    tags: List[str] = field(default_factory=list)
    language: str = "en"
    credibility_score: float = 0.0
    relevance_score: float = 0.0
    bias_score: float = 0.0
    peer_reviewed: bool = False
    open_access: bool = False
    citation_count: int = 0
    access_date: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchFindings:
    """Research findings from analysis"""
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: str = ""
    content: str = ""
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    sources: List[ResearchSource] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAssessment:
    """Quality assessment metrics for research"""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_quality: float = 0.0
    methodology_rigor: float = 0.0
    evidence_strength: float = 0.0
    bias_level: float = 0.0
    completeness: float = 0.0
    consistency: float = 0.0
    relevance: float = 0.0
    timeliness: float = 0.0
    overall_quality: float = 0.0
    quality_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    reviewer_notes: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResearchOutput:
    """Final research output"""
    output_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: ResearchQuery = field(default_factory=ResearchQuery)
    findings: List[ResearchFindings] = field(default_factory=list)
    summary: str = ""
    conclusions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    future_research: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    citations: List[str] = field(default_factory=list)
    bibliography: List[str] = field(default_factory=list)
    appendices: Dict[str, str] = field(default_factory=dict)
    format: OutputFormat = OutputFormat.REPORT
    word_count: int = 0
    page_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Export all classes
__all__ = [
    "ResearchQuery",
    "ResearchSource", 
    "ResearchFindings",
    "ResearchOutput",
    "QualityAssessment",
    "SourceType",
    "OutputFormat",
    "ResearchPhase"
]
