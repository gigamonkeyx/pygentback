# Vector Database Integration for Historical Research

## Overview
This document outlines how to leverage PyGent Factory's existing vector store infrastructure to implement robust context building and semantic search capabilities for the historical research workflow.

## Current Vector Store Infrastructure

### Available Implementations
PyGent Factory already has a sophisticated vector store system with three implementations:

1. **FAISS Vector Store** (`src/storage/vector/faiss.py`)
   - High-performance local similarity search
   - File-based persistence
   - Support for multiple index types (IVFFlat, HNSW, Flat)
   - Perfect for our local historical research needs

2. **ChromaDB Vector Store** (`src/storage/vector/chromadb.py`)
   - Modern vector database with built-in embeddings
   - Good for development and experimentation

3. **PostgreSQL Vector Store** (`src/storage/vector/postgresql.py`)
   - Production-ready with pgvector extension
   - SQL integration for complex queries

### Vector Store Manager
The `VectorStoreManager` class provides:
- Unified interface across all implementations
- Automatic configuration based on database type
- Support for multiple concurrent stores
- Collection management

## Integration with Historical Research Pipeline

### 1. Document Embedding Pipeline

#### Enhanced Document Processor
```python
class HistoricalDocumentEmbeddingProcessor:
    """Enhanced document processor with vector storage integration."""
    
    def __init__(self, vector_manager, embedding_service):
        self.vector_manager = vector_manager
        self.embedding_service = embedding_service
        self.collection_name = "historical_documents"
        
    async def process_document_for_vector_storage(self, doc_path: str, metadata: Dict[str, Any]) -> bool:
        """Process document and store in vector database."""
        try:
            # Extract text with multiple methods
            text_data = await self._extract_comprehensive_text(doc_path)
            
            # Create semantic chunks
            chunks = await self._create_semantic_chunks(text_data, metadata)
            
            # Generate embeddings for chunks
            vector_documents = []
            for i, chunk in enumerate(chunks):
                embedding = await self.embedding_service.embed_text(chunk['text'])
                
                vector_doc = VectorDocument(
                    id=f"{metadata.get('document_id', 'unknown')}_{i}",
                    content=chunk['text'],
                    embedding=embedding,
                    metadata={
                        **metadata,
                        'chunk_index': i,
                        'chunk_type': chunk['type'],
                        'page_number': chunk.get('page', 0),
                        'total_chunks': len(chunks),
                        'extracted_at': datetime.utcnow().isoformat()
                    },
                    collection=self.collection_name
                )
                vector_documents.append(vector_doc)
            
            # Store in vector database
            store = await self.vector_manager.get_store()
            doc_ids = await store.add_documents(vector_documents)
            
            logger.info(f"Stored {len(doc_ids)} chunks for document {metadata.get('document_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process document for vector storage: {e}")
            return False
    
    async def _create_semantic_chunks(self, text_data: Dict, metadata: Dict) -> List[Dict]:
        """Create semantically meaningful chunks from extracted text."""
        chunks = []
        
        # Process different text extraction formats
        if 'structured_text' in text_data:
            # Use structured data for better chunking
            chunks.extend(self._chunk_structured_text(text_data['structured_text']))
        
        if 'plain_text' in text_data:
            # Fallback to plain text chunking
            chunks.extend(self._chunk_plain_text(text_data['plain_text']))
        
        # Add document-level metadata chunk
        if metadata.get('title') or metadata.get('abstract'):
            chunks.append({
                'text': f"Title: {metadata.get('title', 'Unknown')}\n\nAbstract: {metadata.get('abstract', 'N/A')}",
                'type': 'metadata',
                'page': 0
            })
        
        return chunks
    
    def _chunk_structured_text(self, structured_data: List[Dict]) -> List[Dict]:
        """Create chunks from structured text data."""
        chunks = []
        current_chunk = []
        current_size = 0
        max_chunk_size = 1000  # characters
        
        for item in structured_data:
            text = item['text'].strip()
            if not text:
                continue
                
            # Check if this is a natural break point
            is_heading = item.get('flags', 0) & 16  # Bold flag often indicates headings
            is_large_font = item.get('size', 12) > 14
            
            if (is_heading or is_large_font) and current_chunk:
                # Save current chunk and start new one
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'type': 'paragraph',
                    'page': item.get('page', 0)
                })
                current_chunk = []
                current_size = 0
            
            # Add text to current chunk
            current_chunk.append(text)
            current_size += len(text)
            
            # Check size limit
            if current_size > max_chunk_size:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'type': 'paragraph',
                    'page': item.get('page', 0)
                })
                current_chunk = []
                current_size = 0
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'type': 'paragraph',
                'page': structured_data[-1].get('page', 0) if structured_data else 0
            })
        
        return chunks
```

### 2. Semantic Search Integration

#### Historical Research Query Engine
```python
class HistoricalResearchQueryEngine:
    """Advanced query engine for historical research with vector search."""
    
    def __init__(self, vector_manager, embedding_service):
        self.vector_manager = vector_manager
        self.embedding_service = embedding_service
        self.collection_name = "historical_documents"
        
    async def research_query(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute comprehensive research query with vector search."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Prepare search filters
            metadata_filter = self._build_metadata_filter(filters)
            
            # Execute vector search
            vector_query = VectorQuery(
                query_vector=query_embedding,
                collection=self.collection_name,
                limit=50,  # Get more results for comprehensive research
                similarity_threshold=0.7,
                metadata_filter=metadata_filter
            )
            
            store = await self.vector_manager.get_store()
            search_results = await store.search_similar(vector_query)
            
            # Group and analyze results
            analyzed_results = await self._analyze_search_results(search_results, query)
            
            # Generate research context
            context = await self._build_research_context(analyzed_results)
            
            return {
                'query': query,
                'total_results': len(search_results),
                'relevant_documents': analyzed_results['documents'],
                'key_themes': analyzed_results['themes'],
                'temporal_analysis': analyzed_results['temporal'],
                'source_analysis': analyzed_results['sources'],
                'research_context': context,
                'confidence_score': analyzed_results['confidence']
            }
            
        except Exception as e:
            logger.error(f"Research query failed: {e}")
            return {'error': str(e)}
    
    async def _analyze_search_results(self, results: List[VectorSearchResult], query: str) -> Dict[str, Any]:
        """Analyze search results for historical research insights."""
        documents = {}
        themes = {}
        temporal_data = []
        sources = {}
        
        for result in results:
            doc_metadata = result.document.metadata
            doc_id = doc_metadata.get('document_id', 'unknown')
            
            # Group by document
            if doc_id not in documents:
                documents[doc_id] = {
                    'document_id': doc_id,
                    'title': doc_metadata.get('title', 'Unknown'),
                    'source_url': doc_metadata.get('source_url'),
                    'chunks': [],
                    'relevance_score': 0,
                    'temporal_info': doc_metadata.get('date_range'),
                    'source_type': doc_metadata.get('source_info', {}).get('type')
                }
            
            # Add chunk info
            documents[doc_id]['chunks'].append({
                'text': result.document.content,
                'similarity': result.similarity_score,
                'chunk_index': doc_metadata.get('chunk_index', 0),
                'page': doc_metadata.get('page_number', 0)
            })
            
            # Update document relevance
            documents[doc_id]['relevance_score'] = max(
                documents[doc_id]['relevance_score'],
                result.similarity_score
            )
            
            # Analyze themes (simple keyword extraction)
            text_words = result.document.content.lower().split()
            for word in text_words:
                if len(word) > 5:  # Focus on longer, more meaningful words
                    themes[word] = themes.get(word, 0) + result.similarity_score
            
            # Collect temporal data
            if doc_metadata.get('date_range'):
                temporal_data.append({
                    'document_id': doc_id,
                    'date_range': doc_metadata['date_range'],
                    'relevance': result.similarity_score
                })
            
            # Track sources
            source_type = doc_metadata.get('source_info', {}).get('type', 'unknown')
            sources[source_type] = sources.get(source_type, 0) + 1
        
        # Sort and filter results
        top_documents = sorted(documents.values(), key=lambda x: x['relevance_score'], reverse=True)[:10]
        top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Calculate confidence score
        confidence = self._calculate_confidence(results, top_documents)
        
        return {
            'documents': top_documents,
            'themes': dict(top_themes),
            'temporal': temporal_data,
            'sources': sources,
            'confidence': confidence
        }
    
    def _calculate_confidence(self, raw_results: List, analyzed_docs: List) -> float:
        """Calculate overall confidence in search results."""
        if not raw_results:
            return 0.0
        
        # Base confidence on similarity scores and result distribution
        avg_similarity = sum(r.similarity_score for r in raw_results) / len(raw_results)
        result_consistency = len(analyzed_docs) / max(len(raw_results), 1)
        
        return min(1.0, (avg_similarity + result_consistency) / 2)
```

### 3. Context Building for Content Generation

#### Research Context Builder
```python
class ResearchContextBuilder:
    """Build comprehensive research context from vector search results."""
    
    def __init__(self, vector_manager, embedding_service):
        self.vector_manager = vector_manager
        self.embedding_service = embedding_service
        
    async def build_research_context(self, topic: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive research context for content generation."""
        
        # Multiple query strategies for comprehensive coverage
        query_strategies = [
            f"historical events {topic}",
            f"primary sources {topic}",
            f"contemporary accounts {topic}",
            f"scholarly analysis {topic}",
            f"causes and effects {topic}",
            f"timeline chronology {topic}"
        ]
        
        all_results = {}
        for strategy in query_strategies:
            results = await self._execute_contextual_search(strategy, scope)
            all_results[strategy] = results
        
        # Synthesize context
        context = await self._synthesize_context(all_results, topic, scope)
        
        return context
    
    async def _execute_contextual_search(self, query: str, scope: Dict[str, Any]) -> List[VectorSearchResult]:
        """Execute contextual search with scope filters."""
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Build filters from scope
        metadata_filter = {}
        if scope.get('time_period'):
            metadata_filter['time_period'] = scope['time_period']
        if scope.get('geographical_focus'):
            metadata_filter['geography'] = scope['geographical_focus']
        if scope.get('source_types'):
            metadata_filter['source_type'] = {'$in': scope['source_types']}
        
        vector_query = VectorQuery(
            query_vector=query_embedding,
            collection="historical_documents",
            limit=30,
            similarity_threshold=0.65,
            metadata_filter=metadata_filter
        )
        
        store = await self.vector_manager.get_store()
        return await store.search_similar(vector_query)
    
    async def _synthesize_context(self, search_results: Dict, topic: str, scope: Dict) -> Dict[str, Any]:
        """Synthesize search results into comprehensive research context."""
        
        # Extract key information from all search results
        primary_sources = []
        secondary_sources = []
        key_events = []
        important_figures = []
        geographical_info = []
        temporal_info = []
        
        for strategy, results in search_results.items():
            for result in results:
                metadata = result.document.metadata
                content = result.document.content
                
                # Categorize content based on metadata and strategy
                if 'primary sources' in strategy:
                    primary_sources.append({
                        'content': content[:500] + "...",
                        'source': metadata.get('title', 'Unknown'),
                        'confidence': result.similarity_score
                    })
                elif 'scholarly analysis' in strategy:
                    secondary_sources.append({
                        'content': content[:500] + "...",
                        'source': metadata.get('title', 'Unknown'),
                        'confidence': result.similarity_score
                    })
                
                # Extract temporal information
                if metadata.get('date_range'):
                    temporal_info.append(metadata['date_range'])
                
                # Extract geographical information
                if metadata.get('geography'):
                    geographical_info.append(metadata['geography'])
        
        # Remove duplicates and sort by confidence
        primary_sources = sorted(primary_sources, key=lambda x: x['confidence'], reverse=True)[:5]
        secondary_sources = sorted(secondary_sources, key=lambda x: x['confidence'], reverse=True)[:5]
        
        return {
            'topic': topic,
            'scope': scope,
            'primary_sources': primary_sources,
            'secondary_sources': secondary_sources,
            'temporal_context': list(set(temporal_info)),
            'geographical_context': list(set(geographical_info)),
            'evidence_strength': self._assess_evidence_strength(search_results),
            'research_gaps': self._identify_research_gaps(search_results, scope),
            'context_summary': self._generate_context_summary(primary_sources, secondary_sources)
        }
```

### 4. Integration with Existing Historical Research Agent

#### Enhanced Agent with Vector Context
```python
class EnhancedHistoricalResearchAgent:
    """Enhanced historical research agent with vector search integration."""
    
    def __init__(self, vector_manager, embedding_service, original_agent):
        self.vector_manager = vector_manager
        self.embedding_service = embedding_service
        self.original_agent = original_agent
        self.query_engine = HistoricalResearchQueryEngine(vector_manager, embedding_service)
        self.context_builder = ResearchContextBuilder(vector_manager, embedding_service)
        
    async def research_with_vector_context(self, query: str, scope: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute research with enhanced vector context."""
        
        # Build comprehensive research context
        vector_context = await self.context_builder.build_research_context(query, scope or {})
        
        # Execute vector-enhanced query
        search_results = await self.query_engine.research_query(query, scope)
        
        # Combine with traditional research methods
        traditional_results = await self.original_agent.search_and_validate_sources(query)
        
        # Synthesize results
        enhanced_results = {
            'query': query,
            'vector_context': vector_context,
            'vector_search_results': search_results,
            'traditional_search_results': traditional_results,
            'synthesis': await self._synthesize_research_results(
                vector_context, search_results, traditional_results
            ),
            'confidence_assessment': await self._assess_overall_confidence(
                vector_context, search_results, traditional_results
            )
        }
        
        return enhanced_results
    
    async def _synthesize_research_results(self, vector_context, vector_results, traditional_results):
        """Synthesize results from multiple research methods."""
        synthesis = {
            'primary_evidence': [],
            'secondary_analysis': [],
            'cross_validation': [],
            'contradictions': [],
            'confidence_indicators': []
        }
        
        # Combine evidence from vector search and traditional methods
        if vector_context.get('primary_sources'):
            synthesis['primary_evidence'].extend(vector_context['primary_sources'])
        
        if traditional_results.get('validated_sources'):
            for source in traditional_results['validated_sources']:
                synthesis['primary_evidence'].append({
                    'content': source.get('content', ''),
                    'source': source.get('url', ''),
                    'confidence': source.get('credibility_score', 0),
                    'method': 'traditional_search'
                })
        
        # Cross-validate findings
        vector_sources = set(s['source'] for s in vector_context.get('primary_sources', []))
        traditional_sources = set(s.get('url', '') for s in traditional_results.get('validated_sources', []))
        
        synthesis['cross_validation'] = {
            'overlapping_sources': len(vector_sources & traditional_sources),
            'unique_vector_sources': len(vector_sources - traditional_sources),
            'unique_traditional_sources': len(traditional_sources - vector_sources),
            'total_unique_sources': len(vector_sources | traditional_sources)
        }
        
        return synthesis
```

## 5. Implementation Plan

### Phase 1: Vector Database Setup (IMMEDIATE)
1. **Configure FAISS Vector Store** for historical documents
   - Set up dedicated collection for historical research
   - Configure optimal FAISS parameters for document similarity
   - Test with sample historical documents

2. **Implement Document Embedding Pipeline**
   - Integrate with existing document acquisition system
   - Add semantic chunking for better retrieval
   - Store document metadata for filtering

### Phase 2: Enhanced Search Integration (WEEK 1)
3. **Implement Historical Research Query Engine**
   - Multi-strategy search for comprehensive coverage
   - Advanced result analysis and grouping
   - Confidence scoring for search results

4. **Build Research Context Builder**
   - Synthesize vector search results into usable context
   - Cross-reference multiple query strategies
   - Generate structured context for content generation

### Phase 3: Agent Integration (WEEK 2)
5. **Enhance Historical Research Agent**
   - Integrate vector search with existing traditional search
   - Cross-validate findings between methods
   - Provide unified research results

6. **Add Anti-hallucination Framework Integration**
   - Use vector search for source verification
   - Implement fact-checking against stored documents
   - Provide evidence trails for all claims

### Configuration for Historical Research
```python
# Add to PyGent Factory settings
VECTOR_STORE_CONFIG = {
    "type": "faiss",  # Use FAISS for local performance
    "persist_directory": "./data/historical_vectors",
    "index_type": "IVFFlat",
    "nlist": 200,  # More clusters for better precision
    "nprobe": 20,
    "default_collection": "historical_documents"
}

EMBEDDING_CONFIG = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",  # Fast, good quality
    "device": "cpu",  # or "cuda" if available
    "batch_size": 32,
    "max_length": 512
}

RESEARCH_CONFIG = {
    "similarity_threshold": 0.7,
    "max_chunks_per_document": 10,
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "context_window": 5000
}
```

This implementation leverages PyGent Factory's existing sophisticated vector store infrastructure while adding specialized functionality for historical research. The modular design allows for easy testing and gradual integration with the existing research pipeline.

## Next Steps

1. **Test existing vector store setup** with sample historical documents
2. **Implement document embedding pipeline** for our acquired PDFs  
3. **Build and test semantic search capabilities**
4. **Integrate with existing historical research agent**
5. **Add comprehensive logging and monitoring**

The existing infrastructure provides an excellent foundation - we just need to add the historical research-specific logic on top of it!
