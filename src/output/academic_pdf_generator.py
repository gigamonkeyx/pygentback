# src/output/academic_pdf_generator.py

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from dataclasses import dataclass, field

# PDF generation libraries
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT

# For bibliography management
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a bibliographic citation"""
    id: str
    title: str
    authors: List[str]
    publication_year: int
    source_type: str  # journal, book, website, archive, etc.
    publication_venue: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    url: str = ""
    doi: str = ""
    isbn: str = ""
    publisher: str = ""
    accessed_date: Optional[datetime] = None
    notes: str = ""


@dataclass
class ResearchSection:
    """Represents a section in the research paper"""
    title: str
    content: str
    subsections: List['ResearchSection'] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)  # Citation IDs
    level: int = 1  # Heading level
    page_break_before: bool = False


@dataclass
class AcademicPaper:
    """Represents a complete academic paper structure"""
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    sections: List[ResearchSection]
    citations: Dict[str, Citation] = field(default_factory=dict)
    date_created: datetime = field(default_factory=datetime.now)
    institution: str = ""
    department: str = ""
    course: str = ""
    assignment: str = ""
    word_count: int = 0
    page_count: int = 0


class AcademicPDFGenerator:
    """Generates professional academic PDFs from research data."""
    
    def __init__(self, 
                 output_directory: str = "output/academic_papers",
                 citation_style: str = "apa"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.citation_style = citation_style
        self.styles = getSampleStyleSheet()
        
        # Initialize custom styles for academic formatting
        self._initialize_academic_styles()
        
        # Citation counters and tracking
        self.citation_counter = 0
        self.figure_counter = 0
        self.table_counter = 0
        
        logger.info(f"Academic PDF Generator initialized with {citation_style} citation style")
    
    def _initialize_academic_styles(self):
        """Initialize custom paragraph styles for academic formatting."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Author style
        self.styles.add(ParagraphStyle(
            name='Author',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Abstract style
        self.styles.add(ParagraphStyle(
            name='Abstract',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=0.5*inch,
            rightIndent=0.5*inch,
            spaceAfter=20,
            spaceBefore=10,
            alignment=TA_JUSTIFY
        ))
        
        # Keywords style
        self.styles.add(ParagraphStyle(
            name='Keywords',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=0.5*inch,
            rightIndent=0.5*inch,
            spaceAfter=20,
            fontName='Helvetica-Oblique'        ))
        
        # Section heading styles
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=self.styles['CustomHeading2'],
            fontSize=11,
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT
        ))
          # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            firstLineIndent=0.2*inch
        ))
        
        # Citation style
        self.styles.add(ParagraphStyle(
            name='Citation',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=0.5*inch,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Footnote style
        self.styles.add(ParagraphStyle(
            name='Footnote',
            parent=self.styles['Normal'],
            fontSize=9,
            spaceAfter=4,
            alignment=TA_JUSTIFY
        ))
    
    async def generate_academic_paper(self, 
                                    research_data: Dict[str, Any],
                                    paper_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete academic paper from research data."""
        try:
            # Step 1: Structure the research data into academic paper format
            paper = await self._structure_academic_paper(research_data, paper_metadata)
            
            # Step 2: Generate the PDF document
            pdf_path = await self._generate_pdf_document(paper)
            
            # Step 3: Generate bibliography file
            bibliography_path = await self._generate_bibliography(paper)
            
            # Step 4: Generate paper statistics
            statistics = self._generate_paper_statistics(paper)
            
            return {
                'success': True,
                'pdf_path': str(pdf_path),
                'bibliography_path': str(bibliography_path),
                'paper_title': paper.title,
                'statistics': statistics,
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating academic paper: {str(e)}")
            return {'error': f"PDF generation failed: {str(e)}"}
    
    async def _structure_academic_paper(self, 
                                      research_data: Dict[str, Any],
                                      paper_metadata: Dict[str, Any]) -> AcademicPaper:
        """Structure research data into academic paper format."""
        # Extract metadata
        title = paper_metadata.get('title', 'Historical Research Analysis')
        authors = paper_metadata.get('authors', ['Anonymous'])
        abstract = paper_metadata.get('abstract', '')
        keywords = paper_metadata.get('keywords', [])
        
        # Create sections from research data
        sections = []
        
        # Introduction section
        if research_data.get('introduction'):
            sections.append(ResearchSection(
                title="1. Introduction",
                content=research_data['introduction'],
                level=1
            ))
        
        # Literature Review section
        if research_data.get('literature_review'):
            sections.append(ResearchSection(
                title="2. Literature Review",
                content=research_data['literature_review'],
                level=1
            ))
        
        # Methodology section
        if research_data.get('methodology'):
            sections.append(ResearchSection(
                title="3. Methodology",
                content=research_data['methodology'],
                level=1
            ))
        
        # Findings/Results section
        if research_data.get('findings'):
            findings_section = ResearchSection(
                title="4. Findings and Analysis",
                content="",
                level=1
            )
            
            # Add subsections for different types of findings
            if isinstance(research_data['findings'], dict):
                for key, value in research_data['findings'].items():
                    if isinstance(value, str) and value.strip():
                        subsection = ResearchSection(
                            title=f"4.{len(findings_section.subsections) + 1}. {key.replace('_', ' ').title()}",
                            content=value,
                            level=2
                        )
                        findings_section.subsections.append(subsection)
            else:
                findings_section.content = str(research_data['findings'])
            
            sections.append(findings_section)
        
        # Discussion section
        if research_data.get('discussion'):
            sections.append(ResearchSection(
                title="5. Discussion",
                content=research_data['discussion'],
                level=1
            ))
        
        # Conclusion section
        if research_data.get('conclusion'):
            sections.append(ResearchSection(
                title="6. Conclusion",
                content=research_data['conclusion'],
                level=1
            ))
        
        # Extract and process citations
        citations = {}
        if research_data.get('sources'):
            citations = await self._process_citations(research_data['sources'])
        
        # Create the paper object
        paper = AcademicPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            sections=sections,
            citations=citations,
            institution=paper_metadata.get('institution', ''),
            department=paper_metadata.get('department', ''),
            course=paper_metadata.get('course', ''),
            assignment=paper_metadata.get('assignment', '')
        )
        
        # Calculate word count
        paper.word_count = self._calculate_word_count(paper)
        
        return paper
    
    async def _process_citations(self, sources: List[Dict[str, Any]]) -> Dict[str, Citation]:
        """Process source data into Citation objects."""
        citations = {}
        
        for i, source in enumerate(sources):
            citation_id = f"ref{i+1}"
            
            # Extract author names
            authors = []
            if isinstance(source.get('authors'), list):
                authors = source['authors']
            elif isinstance(source.get('authors'), str):
                authors = [source['authors']]
            elif source.get('author'):
                authors = [source['author']]
            
            # Extract publication year
            pub_year = source.get('year', source.get('publication_year', datetime.now().year))
            if isinstance(pub_year, str):
                try:
                    pub_year = int(re.search(r'\d{4}', pub_year).group())
                except (AttributeError, ValueError):
                    pub_year = datetime.now().year
            
            citation = Citation(
                id=citation_id,
                title=source.get('title', 'Untitled'),
                authors=authors,
                publication_year=int(pub_year),
                source_type=source.get('type', 'unknown'),
                publication_venue=source.get('venue', source.get('journal', source.get('publisher', ''))),
                volume=str(source.get('volume', '')),
                issue=str(source.get('issue', '')),
                pages=str(source.get('pages', '')),
                url=source.get('url', ''),
                doi=source.get('doi', ''),
                isbn=source.get('isbn', ''),
                publisher=source.get('publisher', ''),
                accessed_date=datetime.now() if source.get('url') else None,
                notes=source.get('notes', '')
            )
            
            citations[citation_id] = citation
        
        return citations
    
    async def _generate_pdf_document(self, paper: AcademicPaper) -> Path:
        """Generate the PDF document from the structured paper."""
        # Create filename
        safe_title = re.sub(r'[^\w\s-]', '', paper.title).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = self.output_directory / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build document content
        story = []
        
        # Title page
        story.extend(self._build_title_page(paper))
        story.append(PageBreak())
        
        # Abstract and keywords
        if paper.abstract:
            story.extend(self._build_abstract_section(paper))
            story.append(Spacer(1, 12))
        
        # Main content sections
        for section in paper.sections:
            story.extend(self._build_section(section))
            story.append(Spacer(1, 12))
        
        # References/Bibliography
        if paper.citations:
            story.append(PageBreak())
            story.extend(self._build_bibliography_section(paper))
        
        # Build the PDF
        doc.build(story)
        
        # Update page count
        paper.page_count = len(story) // 30  # Rough estimate
        
        logger.info(f"Generated academic PDF: {pdf_path}")
        return pdf_path
    
    def _build_title_page(self, paper: AcademicPaper) -> List:
        """Build the title page elements."""
        elements = []
        
        # Title
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(paper.title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Authors
        authors_text = ", ".join(paper.authors)
        elements.append(Paragraph(authors_text, self.styles['Author']))
        elements.append(Spacer(1, 1*inch))
        
        # Institution info
        if paper.institution:
            elements.append(Paragraph(paper.institution, self.styles['Author']))
        if paper.department:
            elements.append(Paragraph(paper.department, self.styles['Author']))
        
        elements.append(Spacer(1, 1*inch))
        
        # Course/Assignment info
        if paper.course:
            elements.append(Paragraph(f"Course: {paper.course}", self.styles['Author']))
        if paper.assignment:
            elements.append(Paragraph(f"Assignment: {paper.assignment}", self.styles['Author']))
        
        elements.append(Spacer(1, 1*inch))
        
        # Date
        date_str = paper.date_created.strftime("%B %d, %Y")
        elements.append(Paragraph(date_str, self.styles['Author']))
        
        return elements
    
    def _build_abstract_section(self, paper: AcademicPaper) -> List:
        """Build abstract and keywords section."""
        elements = []
        
        # Abstract heading
        elements.append(Paragraph("<b>Abstract</b>", self.styles['CustomHeading2']))
        elements.append(Spacer(1, 6))
        
        # Abstract content
        elements.append(Paragraph(paper.abstract, self.styles['Abstract']))
        elements.append(Spacer(1, 12))
        
        # Keywords
        if paper.keywords:
            keywords_text = f"<b>Keywords:</b> {', '.join(paper.keywords)}"
            elements.append(Paragraph(keywords_text, self.styles['Keywords']))
        
        return elements
    
    def _build_section(self, section: ResearchSection) -> List:
        """Build a content section."""
        elements = []
        
        # Section heading
        heading_style = f'Heading{min(section.level, 3)}'
        elements.append(Paragraph(section.title, self.styles[heading_style]))
        elements.append(Spacer(1, 6))
        
        # Section content
        if section.content:
            # Split content into paragraphs
            paragraphs = section.content.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    # Process citations in the text
                    processed_paragraph = self._process_inline_citations(paragraph.strip())
                    elements.append(Paragraph(processed_paragraph, self.styles['CustomBodyText']))
                    elements.append(Spacer(1, 6))
        
        # Subsections
        for subsection in section.subsections:
            elements.extend(self._build_section(subsection))
            elements.append(Spacer(1, 6))
        
        return elements
    
    def _process_inline_citations(self, text: str) -> str:
        """Process inline citations in the text."""
        # Look for citation patterns like [Author, Year] or [ref1]
        citation_pattern = r'\[([^\]]+)\]'
        
        def replace_citation(match):
            citation_text = match.group(1)
            # For now, keep the citation as is but format it properly
            return f"({citation_text})"
        
        return re.sub(citation_pattern, replace_citation, text)
    
    def _build_bibliography_section(self, paper: AcademicPaper) -> List:
        """Build the bibliography/references section."""
        elements = []
        
        # Bibliography heading
        elements.append(Paragraph("References", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 12))
        
        # Sort citations alphabetically by first author's last name
        sorted_citations = sorted(
            paper.citations.values(),
            key=lambda c: c.authors[0].split()[-1] if c.authors else c.title
        )
        
        # Generate citation entries
        for citation in sorted_citations:
            citation_text = self._format_citation(citation)
            elements.append(Paragraph(citation_text, self.styles['Citation']))
            elements.append(Spacer(1, 6))
        
        return elements
    
    def _format_citation(self, citation: Citation) -> str:
        """Format a citation according to the specified style."""
        if self.citation_style.lower() == 'apa':
            return self._format_apa_citation(citation)
        elif self.citation_style.lower() == 'mla':
            return self._format_mla_citation(citation)
        elif self.citation_style.lower() == 'chicago':
            return self._format_chicago_citation(citation)
        else:
            return self._format_apa_citation(citation)  # Default to APA
    
    def _format_apa_citation(self, citation: Citation) -> str:
        """Format citation in APA style."""
        parts = []
        
        # Authors
        if citation.authors:
            if len(citation.authors) == 1:
                author_text = citation.authors[0]
            elif len(citation.authors) == 2:
                author_text = f"{citation.authors[0]} & {citation.authors[1]}"
            else:
                author_text = f"{citation.authors[0]} et al."
            parts.append(author_text)
        
        # Year
        parts.append(f"({citation.publication_year})")
        
        # Title
        if citation.source_type in ['journal', 'article']:
            parts.append(f"{citation.title}.")
        else:
            parts.append(f"<i>{citation.title}</i>.")
        
        # Publication info
        if citation.publication_venue:
            if citation.source_type in ['journal', 'article']:
                venue_text = f"<i>{citation.publication_venue}</i>"
                if citation.volume:
                    venue_text += f", {citation.volume}"
                if citation.issue:
                    venue_text += f"({citation.issue})"
                if citation.pages:
                    venue_text += f", {citation.pages}"
                parts.append(f"{venue_text}.")
            else:
                parts.append(f"{citation.publication_venue}.")
        
        # URL/DOI
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}")
        elif citation.url:
            parts.append(citation.url)
        
        return " ".join(parts)
    
    def _format_mla_citation(self, citation: Citation) -> str:
        """Format citation in MLA style."""
        parts = []
        
        # Author
        if citation.authors:
            parts.append(f"{citation.authors[0]}.")
        
        # Title
        if citation.source_type in ['journal', 'article']:
            parts.append(f'"{citation.title}."')
        else:
            parts.append(f"<i>{citation.title}</i>.")
        
        # Publication info
        if citation.publication_venue:
            parts.append(f"<i>{citation.publication_venue}</i>,")
        
        # Date
        parts.append(f"{citation.publication_year}.")
        
        # URL
        if citation.url:
            parts.append(f"Web. {datetime.now().strftime('%d %b %Y')}.")
        
        return " ".join(parts)
    
    def _format_chicago_citation(self, citation: Citation) -> str:
        """Format citation in Chicago style."""
        parts = []
        
        # Author
        if citation.authors:
            parts.append(f"{citation.authors[0]}.")
        
        # Title
        if citation.source_type in ['journal', 'article']:
            parts.append(f'"{citation.title}."')
        else:
            parts.append(f"<i>{citation.title}</i>.")
        
        # Publication info
        if citation.publication_venue:
            if citation.source_type in ['journal', 'article']:
                parts.append(f"<i>{citation.publication_venue}</i>")
                if citation.volume:
                    parts.append(f"no. {citation.volume}")
                parts.append(f"({citation.publication_year}):")
                if citation.pages:
                    parts.append(f"{citation.pages}.")
            else:
                parts.append(f"{citation.publication_venue}, {citation.publication_year}.")
        
        return " ".join(parts)
    
    async def _generate_bibliography(self, paper: AcademicPaper) -> Path:
        """Generate a separate bibliography file in BibTeX format."""
        # Create filename
        safe_title = re.sub(r'[^\w\s-]', '', paper.title).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        filename = f"{safe_title}_bibliography.bib"
        bib_path = self.output_directory / filename
        
        # Create BibTeX database
        db = BibDatabase()
        db.entries = []
        
        for citation in paper.citations.values():
            entry = {
                'ID': citation.id,
                'title': citation.title,
                'year': str(citation.publication_year),
                'author': ' and '.join(citation.authors) if citation.authors else 'Anonymous'
            }
            
            # Add type-specific fields
            if citation.source_type in ['journal', 'article']:
                entry['ENTRYTYPE'] = 'article'
                if citation.publication_venue:
                    entry['journal'] = citation.publication_venue
                if citation.volume:
                    entry['volume'] = citation.volume
                if citation.issue:
                    entry['number'] = citation.issue
                if citation.pages:
                    entry['pages'] = citation.pages
            elif citation.source_type == 'book':
                entry['ENTRYTYPE'] = 'book'
                if citation.publisher:
                    entry['publisher'] = citation.publisher
                if citation.isbn:
                    entry['isbn'] = citation.isbn
            else:
                entry['ENTRYTYPE'] = 'misc'
            
            if citation.url:
                entry['url'] = citation.url
            if citation.doi:
                entry['doi'] = citation.doi
            if citation.notes:
                entry['note'] = citation.notes
            
            db.entries.append(entry)
        
        # Write BibTeX file
        writer = BibTexWriter()
        with open(bib_path, 'w', encoding='utf-8') as bibfile:
            bibfile.write(writer.write(db))
        
        logger.info(f"Generated bibliography file: {bib_path}")
        return bib_path
    
    def _calculate_word_count(self, paper: AcademicPaper) -> int:
        """Calculate the total word count of the paper."""
        word_count = 0
        
        # Count words in abstract
        if paper.abstract:
            word_count += len(paper.abstract.split())
        
        # Count words in sections
        for section in paper.sections:
            word_count += self._count_section_words(section)
        
        return word_count
    
    def _count_section_words(self, section: ResearchSection) -> int:
        """Count words in a section and its subsections."""
        word_count = 0
        
        if section.content:
            word_count += len(section.content.split())
        
        for subsection in section.subsections:
            word_count += self._count_section_words(subsection)
        
        return word_count
    
    def _generate_paper_statistics(self, paper: AcademicPaper) -> Dict[str, Any]:
        """Generate comprehensive statistics about the paper."""
        return {
            'word_count': paper.word_count,
            'estimated_pages': max(1, paper.word_count // 250),  # Rough estimate
            'section_count': len(paper.sections),
            'citation_count': len(paper.citations),
            'author_count': len(paper.authors),
            'keyword_count': len(paper.keywords),
            'subsection_count': sum(len(section.subsections) for section in paper.sections),
            'character_count': sum(len(section.content) for section in paper.sections),
            'estimated_reading_time': max(1, paper.word_count // 200)  # Minutes
        }
    
    async def generate_citation_report(self, citations: Dict[str, Citation]) -> Dict[str, Any]:
        """Generate a comprehensive citation analysis report."""
        try:
            citation_stats = {
                'total_citations': len(citations),
                'source_types': {},
                'publication_years': {},
                'author_analysis': {},
                'venue_analysis': {}
            }
            
            # Analyze source types
            for citation in citations.values():
                source_type = citation.source_type
                citation_stats['source_types'][source_type] = citation_stats['source_types'].get(source_type, 0) + 1
            
            # Analyze publication years
            for citation in citations.values():
                year = citation.publication_year
                citation_stats['publication_years'][year] = citation_stats['publication_years'].get(year, 0) + 1
            
            # Analyze authors
            all_authors = []
            for citation in citations.values():
                all_authors.extend(citation.authors)
            
            author_counts = {}
            for author in all_authors:
                author_counts[author] = author_counts.get(author, 0) + 1
            
            citation_stats['author_analysis'] = {
                'total_unique_authors': len(set(all_authors)),
                'most_cited_authors': sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
            # Analyze venues
            venues = [c.publication_venue for c in citations.values() if c.publication_venue]
            venue_counts = {}
            for venue in venues:
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
            
            citation_stats['venue_analysis'] = {
                'total_unique_venues': len(set(venues)),
                'most_common_venues': sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
            return {
                'success': True,
                'citation_statistics': citation_stats,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating citation report: {str(e)}")
            return {'error': f"Citation analysis failed: {str(e)}"}
    
    async def get_generation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about PDF generation."""
        output_files = list(self.output_directory.glob("*.pdf"))
        
        return {
            'total_pdfs_generated': len(output_files),
            'output_directory': str(self.output_directory),
            'citation_style': self.citation_style,
            'recent_files': [
                {
                    'filename': f.name,
                    'size_kb': f.stat().st_size // 1024,
                    'created': datetime.fromtimestamp(f.stat().st_ctime).isoformat()
                }
                for f in sorted(output_files, key=lambda x: x.stat().st_ctime, reverse=True)[:10]
            ],
            'supported_citation_styles': ['apa', 'mla', 'chicago'],
            'system_info': {
                'reportlab_available': True,
                'bibtex_available': True
            }
        }
