"""
Intelligent semantic chunking system for historical documents.
Optimized for maintaining context and document structure.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a semantically meaningful chunk of a document."""
    id: str
    text: str
    start_position: int
    end_position: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_type: str = "content"  # content, title, header, footer, table, etc.
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class SemanticChunker:
    """Intelligent semantic chunking for historical documents."""
    
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 2000):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Patterns for document structure detection
        self.patterns = {
            'chapter_header': re.compile(r'^(Chapter|CHAPTER)\s+\d+', re.MULTILINE),
            'section_header': re.compile(r'^[A-Z][A-Z\s]+$', re.MULTILINE),
            'numbered_section': re.compile(r'^\d+\.\s+[A-Z]', re.MULTILINE),
            'page_marker': re.compile(r'=== Page \d+ ==='),
            'date_marker': re.compile(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b'),
            'year_marker': re.compile(r'\b(18|19|20)\d{2}\b'),
            'paragraph_boundary': re.compile(r'\n\s*\n'),
            'sentence_boundary': re.compile(r'[.!?]+\s+'),
            'quote_marker': re.compile(r'["""\'\']\s*(.+?)\s*["""\'\']', re.DOTALL)
        }
    
    def chunk_document(self, 
                      document_id: str, 
                      text: str, 
                      document_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Chunk a document using semantic boundaries."""
        if not text or not text.strip():
            return []
        
        try:
            # First, identify document structure
            structure = self._analyze_document_structure(text)
            
            # Split into semantic sections
            sections = self._split_into_sections(text, structure)
            
            # Create chunks from sections
            chunks = []
            for section in sections:
                section_chunks = self._chunk_section(
                    document_id=document_id,
                    section=section,
                    document_metadata=document_metadata or {}
                )
                chunks.extend(section_chunks)
            
            # Post-process chunks
            chunks = self._post_process_chunks(chunks)
            
            logger.info(f"Created {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk document {document_id}: {e}")
            return []
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure to identify sections and boundaries."""
        structure = {
            'page_boundaries': [],
            'section_headers': [],
            'paragraph_boundaries': [],
            'date_mentions': [],
            'quotes': [],
            'total_length': len(text)
        }
        
        # Find page boundaries
        for match in self.patterns['page_marker'].finditer(text):
            page_num = re.search(r'\d+', match.group())
            structure['page_boundaries'].append({
                'position': match.start(),
                'page_number': int(page_num.group()) if page_num else None,
                'text': match.group()
            })
        
        # Find section headers
        for pattern_name in ['chapter_header', 'section_header', 'numbered_section']:
            for match in self.patterns[pattern_name].finditer(text):
                structure['section_headers'].append({
                    'position': match.start(),
                    'type': pattern_name,
                    'text': match.group().strip(),
                    'level': self._get_header_level(pattern_name)
                })
        
        # Find paragraph boundaries
        for match in self.patterns['paragraph_boundary'].finditer(text):
            structure['paragraph_boundaries'].append(match.start())
        
        # Find date mentions
        for match in self.patterns['date_marker'].finditer(text):
            structure['date_mentions'].append({
                'position': match.start(),
                'date': match.group()
            })
        
        # Find quotes
        for match in self.patterns['quote_marker'].finditer(text):
            structure['quotes'].append({
                'position': match.start(),
                'quote': match.group(1) if match.groups() else match.group()
            })
        
        return structure
    
    def _get_header_level(self, pattern_name: str) -> int:
        """Get hierarchical level of header type."""
        levels = {
            'chapter_header': 1,
            'section_header': 2,
            'numbered_section': 3
        }
        return levels.get(pattern_name, 4)
    
    def _split_into_sections(self, text: str, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into semantic sections based on structure analysis."""
        sections = []
        
        # Get all section boundaries sorted by position
        boundaries = []
        
        # Add page boundaries
        for page in structure['page_boundaries']:
            boundaries.append({
                'position': page['position'],
                'type': 'page',
                'data': page
            })
        
        # Add section headers
        for header in structure['section_headers']:
            boundaries.append({
                'position': header['position'],
                'type': 'header',
                'data': header
            })
        
        # Sort boundaries by position
        boundaries.sort(key=lambda x: x['position'])
        
        # Create sections based on boundaries
        if not boundaries:
            # No structure detected, treat as single section
            sections.append({
                'text': text,
                'start_position': 0,
                'end_position': len(text),
                'type': 'content',
                'page_number': None,
                'section_title': None
            })
        else:
            # Process each boundary
            for i, boundary in enumerate(boundaries):
                start_pos = boundary['position']
                end_pos = boundaries[i + 1]['position'] if i + 1 < len(boundaries) else len(text)
                
                section_text = text[start_pos:end_pos].strip()
                if section_text:
                    section = {
                        'text': section_text,
                        'start_position': start_pos,
                        'end_position': end_pos,
                        'type': boundary['type'],
                        'page_number': None,
                        'section_title': None
                    }
                    
                    # Extract metadata based on boundary type
                    if boundary['type'] == 'page':
                        section['page_number'] = boundary['data'].get('page_number')
                    elif boundary['type'] == 'header':
                        section['section_title'] = boundary['data'].get('text')
                        section['header_level'] = boundary['data'].get('level')
                    
                    sections.append(section)
        
        return sections
    
    def _chunk_section(self,
                      document_id: str,
                      section: Dict[str, Any],
                      document_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk a single section into optimal-sized pieces."""
        chunks = []
        text = section['text']
        
        if len(text) <= self.max_chunk_size:
            # Section fits in single chunk
            chunk = DocumentChunk(
                id=f"{document_id}_chunk_0",
                text=text,
                start_position=section['start_position'],
                end_position=section['end_position'],
                page_number=section.get('page_number'),
                section_title=section.get('section_title'),
                section_type=section.get('type', 'content'),
                metadata={
                    'document_id': document_id,
                    **document_metadata,
                    'chunk_method': 'single_section'
                }
            )
            chunks.append(chunk)
        else:
            # Split section into multiple chunks
            section_chunks = self._split_large_section(
                document_id=document_id,
                section=section,
                document_metadata=document_metadata
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _split_large_section(self,
                           document_id: str,
                           section: Dict[str, Any],
                           document_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split a large section into optimally-sized chunks."""
        chunks = []
        text = section['text']
        
        # Find paragraph boundaries within the section
        paragraphs = re.split(self.patterns['paragraph_boundary'], text)
        
        current_chunk = ""
        current_start = section['start_position']
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if len(test_chunk) > self.max_chunk_size and current_chunk:
                # Save current chunk and start new one
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        text=current_chunk,
                        start_position=current_start,
                        section=section,
                        document_metadata=document_metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap if possible
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + para
                current_start = current_start + len(current_chunk) - len(para)
            else:
                # Add paragraph to current chunk
                current_chunk = test_chunk
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk = self._create_chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                text=current_chunk,
                start_position=current_start,
                section=section,
                document_metadata=document_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self,
                     document_id: str,
                     chunk_index: int,
                     text: str,
                     start_position: int,
                     section: Dict[str, Any],
                     document_metadata: Dict[str, Any]) -> DocumentChunk:
        """Create a DocumentChunk with proper metadata."""
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        
        # Extract additional metadata from chunk text
        chunk_metadata = {
            'document_id': document_id,
            'chunk_index': chunk_index,
            **document_metadata,
            'chunk_method': 'semantic_split',
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_dates': bool(self.patterns['date_marker'].search(text)),
            'has_quotes': bool(self.patterns['quote_marker'].search(text))
        }
        
        return DocumentChunk(
            id=chunk_id,
            text=text,
            start_position=start_position,
            end_position=start_position + len(text),
            page_number=section.get('page_number'),
            section_title=section.get('section_title'),
            section_type=section.get('type', 'content'),
            metadata=chunk_metadata
        )
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of previous chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to find a good breaking point near the overlap boundary
        overlap_start = len(text) - self.chunk_overlap
        
        # Look for sentence boundaries
        sentences = list(self.patterns['sentence_boundary'].finditer(text[overlap_start:]))
        if sentences:
            # Use the first sentence boundary
            break_point = overlap_start + sentences[0].end()
            return text[break_point:] + " "
        
        # Fallback to word boundary
        words = text[overlap_start:].split()
        if len(words) > 1:
            return " ".join(words[1:]) + " "
        
        # Last resort: character boundary
        return text[overlap_start:]
    
    def _post_process_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Post-process chunks to ensure quality and consistency."""
        processed_chunks = []
        
        for chunk in chunks:
            # Clean up whitespace
            chunk.text = re.sub(r'\s+', ' ', chunk.text.strip())
            
            # Skip chunks that are too small or empty
            if len(chunk.text) < self.min_chunk_size:
                logger.debug(f"Skipping small chunk: {chunk.id}")
                continue
            
            # Update metadata
            chunk.metadata['final_text_length'] = len(chunk.text)
            chunk.metadata['final_word_count'] = len(chunk.text.split())
            
            processed_chunks.append(chunk)
        
        # Re-index chunks
        for i, chunk in enumerate(processed_chunks):
            chunk.metadata['final_chunk_index'] = i
        
        return processed_chunks
    
    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results."""
        if not chunks:
            return {}
        
        text_lengths = [len(chunk.text) for chunk in chunks]
        word_counts = [chunk.metadata.get('word_count', 0) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'average_text_length': sum(text_lengths) / len(text_lengths),
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths),
            'average_word_count': sum(word_counts) / len(word_counts),
            'chunks_with_dates': sum(1 for chunk in chunks if chunk.metadata.get('has_dates')),
            'chunks_with_quotes': sum(1 for chunk in chunks if chunk.metadata.get('has_quotes')),
            'section_types': {
                section_type: sum(1 for chunk in chunks if chunk.section_type == section_type)
                for section_type in set(chunk.section_type for chunk in chunks)
            }
        }
