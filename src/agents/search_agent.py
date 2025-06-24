"""
Search Agent - s3 RAG Implementation

Uses the sophisticated s3 RAG pipeline for enhanced search and retrieval tasks.
Replaces the mock "search agent" with real AI-powered search capabilities.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from core.agent import BaseAgent, AgentMessage, MessageType
from core.agent.config import AgentConfig

logger = logging.getLogger(__name__)


class RealDocumentRetriever:
    """Real document retriever using database and vector search"""

    def __init__(self):
        self.db_manager = None
        self.vector_search_available = False

        # Try to initialize database connection
        try:
            from database.production_manager import db_manager
            self.db_manager = db_manager
        except ImportError:
            logger.warning("Database manager not available for document retrieval")

    async def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Real document retrieval using database and vector search"""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        documents = []

        # Try vector search first if available
        try:
            vector_docs = await self._vector_search(query, k)
            if vector_docs:
                documents.extend(vector_docs)
                logger.info(f"Retrieved {len(vector_docs)} documents via vector search")
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        # If we don't have enough documents, try full-text search
        if len(documents) < k and self.db_manager:
            try:
                remaining = k - len(documents)
                text_docs = await self._fulltext_search(query, remaining)
                documents.extend(text_docs)
                logger.info(f"Retrieved {len(text_docs)} additional documents via full-text search")
            except Exception as e:
                logger.warning(f"Full-text search failed: {e}")

        # If still no documents, try basic keyword search
        if not documents and self.db_manager:
            try:
                keyword_docs = await self._keyword_search(query, k)
                documents.extend(keyword_docs)
                logger.info(f"Retrieved {len(keyword_docs)} documents via keyword search")
            except Exception as e:
                logger.error(f"Keyword search failed: {e}")

        # If no real documents found, raise error instead of returning mock data
        if not documents:
            raise RuntimeError(f"No documents found for query '{query}'. Real document store required.")

        return documents[:k]

    async def _vector_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        if not self.db_manager:
            return []

        try:
            # Use pgvector for similarity search if available
            search_query = """
            SELECT id, title, content, source, metadata,
                   1 - (embedding <=> $1) as similarity_score
            FROM documents
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """

            # Generate query embedding (simplified - would use real embedding model)
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []

            rows = await self.db_manager.fetch_all(search_query, query_embedding, k)

            documents = []
            for row in rows:
                documents.append({
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"],
                    "score": float(row["similarity_score"]),
                    "source": row["source"],
                    "metadata": row.get("metadata", {}),
                    "search_method": "vector_similarity"
                })

            return documents

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    async def _fulltext_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform PostgreSQL full-text search"""
        if not self.db_manager:
            return []

        try:
            search_query = """
            SELECT id, title, content, source, metadata,
                   ts_rank(search_vector, plainto_tsquery($1)) as relevance_score
            FROM documents
            WHERE search_vector @@ plainto_tsquery($1)
            ORDER BY relevance_score DESC
            LIMIT $2
            """

            rows = await self.db_manager.fetch_all(search_query, query, k)

            documents = []
            for row in rows:
                documents.append({
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"],
                    "score": float(row["relevance_score"]),
                    "source": row["source"],
                    "metadata": row.get("metadata", {}),
                    "search_method": "fulltext_search"
                })

            return documents

        except Exception as e:
            logger.error(f"Full-text search error: {e}")
            return []

    async def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform basic keyword search as fallback"""
        if not self.db_manager:
            return []

        try:
            # Simple ILIKE search as last resort
            keywords = query.split()
            where_conditions = []
            params = []

            for i, keyword in enumerate(keywords[:3]):  # Limit to 3 keywords
                param_num = i + 1
                where_conditions.append(f"(title ILIKE ${param_num} OR content ILIKE ${param_num})")
                params.append(f"%{keyword}%")

            if not where_conditions:
                return []

            search_query = f"""
            SELECT id, title, content, source, metadata,
                   0.5 as relevance_score
            FROM documents
            WHERE {' OR '.join(where_conditions)}
            ORDER BY title
            LIMIT ${len(params) + 1}
            """

            params.append(k)
            rows = await self.db_manager.fetch_all(search_query, *params)

            documents = []
            for row in rows:
                documents.append({
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"],
                    "score": float(row["relevance_score"]),
                    "source": row["source"],
                    "metadata": row.get("metadata", {}),
                    "search_method": "keyword_search"
                })

            return documents

        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text (simplified implementation)"""
        try:
            # Try to use sentence transformers if available
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(text)
            return embedding.tolist()

        except ImportError:
            logger.warning("Sentence transformers not available for embeddings")
            return None
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None


class RealResponseGenerator:
    """Real response generator using Ollama or other LLM services"""

    def __init__(self):
        self.ollama_available = False
        self.ollama_manager = None

        # Try to initialize Ollama
        try:
            from core.ollama_gpu_integration import ollama_gpu_manager
            if ollama_gpu_manager and ollama_gpu_manager.is_initialized:
                self.ollama_manager = ollama_gpu_manager
                self.ollama_available = True
                logger.info("Real response generator initialized with Ollama")
        except ImportError:
            logger.warning("Ollama not available for response generation")

    async def generate(self, query: str, context: str) -> str:
        """Real response generation using LLM"""
        if not query.strip():
            raise ValueError("Query cannot be empty for response generation")

        if not context.strip():
            raise ValueError("Context cannot be empty for response generation")

        # Try Ollama first
        if self.ollama_available and self.ollama_manager:
            try:
                return await self._generate_with_ollama(query, context)
            except Exception as e:
                logger.error(f"Ollama generation failed: {e}")
                # Don't fall back to mock - raise the error
                raise RuntimeError(f"Real LLM generation required but failed: {e}")

        # If no real LLM available, raise error instead of returning mock
        raise RuntimeError("No real LLM service available for response generation. Ollama or other LLM service required.")

    async def _generate_with_ollama(self, query: str, context: str) -> str:
        """Generate response using Ollama"""
        try:
            prompt = f"""Based on the following context documents, provide a comprehensive answer to the query.

Query: {query}

Context:
{context}

Please provide a detailed, accurate response based solely on the information provided in the context. If the context doesn't contain enough information to fully answer the query, please indicate what information is missing."""

            result = await self.ollama_manager.generate(
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.7,
                    "num_predict": 1000
                }
            )

            if result and "response" in result:
                return result["response"]
            else:
                raise RuntimeError("Invalid response from Ollama")

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise


class SearchAgent(BaseAgent):
    """
    Real Search Agent using s3 RAG pipeline
    
    This agent uses the sophisticated s3 RAG framework for enhanced search,
    iterative query refinement, and intelligent document retrieval.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Initialize s3 RAG components with real implementations
        self.retriever = RealDocumentRetriever()
        self.generator = RealResponseGenerator()
        
        # Try to import and initialize real s3 RAG
        try:
            from rag.s3.models import S3Config, SearchStrategy
            from rag.s3.s3_pipeline import S3Pipeline
            
            # Configure s3 RAG
            self.s3_config = S3Config(
                search_strategy=SearchStrategy.ITERATIVE_REFINEMENT,
                max_search_iterations=3,
                max_documents_per_iteration=5,
                similarity_threshold=0.3,
                training_episodes=10
            )
            
            # Initialize s3 pipeline
            self.s3_pipeline = S3Pipeline(
                self.s3_config,
                self.retriever,
                self.generator
            )
            
            self.s3_available = True
            logger.info("SearchAgent initialized with s3 RAG pipeline")
            
        except Exception as e:
            logger.warning(f"s3 RAG not available, using fallback search: {e}")
            self.s3_available = False
        
        # Agent state
        self.search_history = []
        self.total_search_time = 0.0

    async def _agent_initialize(self) -> None:
        """Agent-specific initialization logic"""
        logger.info("SearchAgent initialized")

    async def _agent_shutdown(self) -> None:
        """Agent-specific shutdown logic"""
        logger.info("SearchAgent shutting down")

    async def _handle_request(self, message: AgentMessage) -> AgentMessage:
        """Handle a request message"""
        return await self.process_message(message)

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process message using s3 RAG search pipeline
        
        Args:
            message: Incoming message with search query
            
        Returns:
            AgentMessage: Response with search results
        """
        start_time = datetime.now()
        
        try:
            # Extract the search query from the message
            query = message.content.get("content", str(message.content))
            
            logger.info(f"Starting s3 RAG search for: {query[:100]}...")
            
            if self.s3_available:
                # Use real s3 RAG pipeline
                result = await self.s3_pipeline.query(query, return_details=True)
                response_content = self._format_s3_response(result, query)

            else:
                # Use real document retrieval and generation directly
                result = await self._real_search(query)
                response_content = self._format_real_search_response(result, query)
            
            # Calculate search time
            search_time = (datetime.now() - start_time).total_seconds()
            self.total_search_time += search_time
            
            # Store in search history
            self.search_history.append({
                "query": query,
                "search_time": search_time,
                "timestamp": start_time.isoformat(),
                "s3_used": self.s3_available
            })
            
            logger.info(f"Search completed in {search_time:.2f}s")
            
            # Create response message
            response = AgentMessage(
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                content=response_content,
                correlation_id=message.id
            )
            
            # Update agent activity
            self.last_activity = datetime.utcnow()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in SearchAgent.process_message: {e}")
            
            # Create error response
            error_response = AgentMessage(
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                content={
                    "type": "error",
                    "message": f"Search failed: {str(e)}",
                    "agent": "search"
                },
                correlation_id=message.id
            )
            
            return error_response
    
    def _format_s3_response(self, result, query: str) -> Dict[str, Any]:
        """Format s3 RAG result into response"""
        return {
            "type": "search_response",
            "query": query,
            "response": result.response,
            "documents_found": len(result.search_state.documents) if result.search_state else 0,
            "search_iterations": result.search_state.iteration if result.search_state else 1,
            "metrics": {
                "search_time": result.search_time,
                "generation_time": result.generation_time,
                "total_time": result.search_time + result.generation_time,
                "search_efficiency": result.search_state.search_efficiency if result.search_state else 0.8
            },
            "agent": "search",
            "method": "s3_rag"
        }
    
    async def _real_search(self, query: str) -> Dict[str, Any]:
        """Real search using document retrieval and LLM generation"""
        import time

        search_start = time.time()

        # Retrieve real documents
        documents = await self.retriever.retrieve(query, k=5)
        search_time = time.time() - search_start

        # Generate real response
        generation_start = time.time()
        context = "\n\n".join([
            f"Title: {doc['title']}\nSource: {doc['source']}\nContent: {doc['content']}"
            for doc in documents
        ])
        response = await self.generator.generate(query, context)
        generation_time = time.time() - generation_start

        return {
            "response": response,
            "documents": documents,
            "search_time": search_time,
            "generation_time": generation_time
        }

    def _format_real_search_response(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Format real search result"""
        return {
            "type": "search_response",
            "query": query,
            "response": result["response"],
            "documents_found": len(result["documents"]),
            "search_iterations": 1,
            "metrics": {
                "search_time": result["search_time"],
                "generation_time": result["generation_time"],
                "total_time": result["search_time"] + result["generation_time"],
                "search_efficiency": 1.0  # Real search efficiency
            },
            "agent": "search",
            "method": "real_search",
            "documents": [
                {
                    "title": doc["title"],
                    "source": doc["source"],
                    "score": doc["score"],
                    "search_method": doc.get("search_method", "unknown")
                }
                for doc in result["documents"]
            ]
        }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics for this agent"""
        return {
            "total_searches": len(self.search_history),
            "total_search_time": self.total_search_time,
            "average_search_time": self.total_search_time / max(1, len(self.search_history)),
            "s3_available": self.s3_available,
            "recent_searches": self.search_history[-5:] if self.search_history else []
        }
