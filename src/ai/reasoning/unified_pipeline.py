"""
Unified AI Reasoning Pipeline

Combines Tree of Thought reasoning, s3 RAG retrieval, and GPU vector search
into a comprehensive AI system for complex problem solving and knowledge retrieval.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .tot.models import ToTConfig, ThoughtState, ThoughtTree, SearchResult as ToTSearchResult
from .tot.tot_engine import ToTEngine
from .tot.integrations.vector_search_integration import VectorSearchIntegration
try:
    from search.gpu_search import VectorSearchConfig, IndexType
except ImportError:
    # Fallback for when search module is not available
    from typing import Any
    VectorSearchConfig = Any
    IndexType = Any

logger = logging.getLogger(__name__)


class ReasoningMode(Enum):
    """Mode of reasoning to use"""
    TOT_ONLY = "tot_only"                    # Pure Tree of Thought
    RAG_ONLY = "rag_only"                    # Pure RAG retrieval
    S3_RAG = "s3_rag"                        # s3 RAG framework
    TOT_ENHANCED_RAG = "tot_enhanced_rag"    # ToT + traditional RAG
    TOT_S3_RAG = "tot_s3_rag"               # ToT + s3 RAG (full system)
    ADAPTIVE = "adaptive"                     # Automatically choose best mode


class TaskComplexity(Enum):
    """Complexity level of the task"""
    SIMPLE = "simple"        # Direct factual queries
    MODERATE = "moderate"    # Multi-step reasoning
    COMPLEX = "complex"      # Deep analysis and synthesis
    RESEARCH = "research"    # Academic research tasks


@dataclass
class UnifiedConfig:
    """Configuration for unified reasoning pipeline"""
    # Mode selection
    reasoning_mode: ReasoningMode = ReasoningMode.ADAPTIVE
    default_complexity: TaskComplexity = TaskComplexity.MODERATE
    
    # ToT configuration
    tot_config: Optional[ToTConfig] = None
    enable_vector_search: bool = True
    
    # RAG configuration
    enable_s3_rag: bool = True
    max_documents: int = 10
    similarity_threshold: float = 0.7
    
    # GPU search configuration
    vector_search_config: Optional[VectorSearchConfig] = None
    
    # Performance settings
    max_reasoning_time: float = 300.0  # 5 minutes max
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Quality thresholds
    min_confidence_threshold: float = 0.6
    min_relevance_threshold: float = 0.5
    
    def __post_init__(self):
        """Initialize default configurations"""
        if self.tot_config is None:
            self.tot_config = ToTConfig(
                max_depth=6,
                n_generate_sample=3,
                n_evaluate_sample=2,
                n_select_sample=3,
                temperature=0.7
            )
        
        if self.vector_search_config is None:
            self.vector_search_config = VectorSearchConfig(
                index_type=IndexType.IVF_FLAT,
                dimension=768,
                use_gpu=True,
                use_float16=True
            )


@dataclass
class UnifiedResult:
    """Result from unified reasoning pipeline"""
    query: str
    response: str
    reasoning_mode: ReasoningMode
    task_complexity: TaskComplexity
    
    # Performance metrics
    total_time: float = 0.0
    reasoning_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    
    # Quality metrics
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    
    # Component results
    tot_result: Optional[ToTSearchResult] = None
    rag_documents: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    
    # Metadata
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'query': self.query,
            'response': self.response,
            'reasoning_mode': self.reasoning_mode.value,
            'task_complexity': self.task_complexity.value,
            'total_time': self.total_time,
            'reasoning_time': self.reasoning_time,
            'retrieval_time': self.retrieval_time,
            'generation_time': self.generation_time,
            'confidence_score': self.confidence_score,
            'relevance_score': self.relevance_score,
            'coherence_score': self.coherence_score,
            'tot_result': self.tot_result.to_dict() if self.tot_result else None,
            'rag_documents': self.rag_documents,
            'reasoning_path': self.reasoning_path,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class UnifiedReasoningPipeline:
    """
    Unified AI Reasoning Pipeline
    
    Intelligently combines Tree of Thought reasoning, s3 RAG retrieval,
    and GPU vector search based on task complexity and requirements.
    """
    
    def __init__(self, config: UnifiedConfig, retriever=None, generator=None):
        self.config = config
        self.retriever = retriever
        self.generator = generator
        
        # Initialize components
        self.tot_engine = None
        self.s3_pipeline = None
        self.vector_search = None
        
        # Performance cache
        self.response_cache = {} if config.enable_caching else None
        
        # Statistics
        self.query_count = 0
        self.mode_usage = {mode: 0 for mode in ReasoningMode}
        self.complexity_distribution = {comp: 0 for comp in TaskComplexity}
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Unified reasoning pipeline initialized with mode: {config.reasoning_mode.value}")
    
    def _initialize_components(self):
        """Initialize reasoning components"""
        try:
            # Initialize ToT engine
            self.tot_engine = ToTEngine(
                self.config.tot_config,
                enable_vector_search=self.config.enable_vector_search
            )
            logger.info("ToT engine initialized")
            
            # Initialize vector search integration
            if self.config.enable_vector_search:
                self.vector_search = VectorSearchIntegration(
                    embedding_dim=self.config.vector_search_config.dimension,
                    use_gpu=self.config.vector_search_config.use_gpu
                )
                logger.info("Vector search integration initialized")
            
            # Initialize s3 RAG pipeline
            if self.config.enable_s3_rag and self.retriever and self.generator:
                try:
                    from rag.s3.models import S3Config
                    from rag.s3.s3_pipeline import S3Pipeline
                    
                    s3_config = S3Config(
                        max_documents_per_iteration=self.config.max_documents,
                        similarity_threshold=self.config.similarity_threshold
                    )
                    
                    self.s3_pipeline = S3Pipeline(s3_config, self.retriever, self.generator)
                    logger.info("S3 RAG pipeline initialized")
                    
                except ImportError:
                    logger.warning("S3 RAG components not available")
                    self.config.enable_s3_rag = False
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
    
    async def reason(self, query: str, 
                    mode: Optional[ReasoningMode] = None,
                    complexity: Optional[TaskComplexity] = None,
                    context: Optional[Dict[str, Any]] = None) -> UnifiedResult:
        """
        Process a query using unified reasoning
        
        Args:
            query: Input query or problem
            mode: Reasoning mode (overrides config)
            complexity: Task complexity (overrides auto-detection)
            context: Additional context for reasoning
            
        Returns:
            UnifiedResult with response and metrics
        """
        start_time = time.time()
        self.query_count += 1
        
        # Check cache first
        if self.response_cache and query in self.response_cache:
            cached_result = self.response_cache[query]
            if time.time() - cached_result['timestamp'] < self.config.cache_ttl:
                logger.info(f"Returning cached result for query: {query[:50]}...")
                return cached_result['result']
        
        # Determine reasoning mode and complexity
        reasoning_mode = mode or self._select_reasoning_mode(query, context)
        task_complexity = complexity or self._assess_task_complexity(query, context)
        
        # Update statistics
        self.mode_usage[reasoning_mode] += 1
        self.complexity_distribution[task_complexity] += 1
        
        # Create result object
        result = UnifiedResult(
            query=query,
            response="",
            reasoning_mode=reasoning_mode,
            task_complexity=task_complexity
        )
        
        try:
            # Execute reasoning based on selected mode
            if reasoning_mode == ReasoningMode.TOT_ONLY:
                await self._execute_tot_reasoning(query, result, context)
            elif reasoning_mode == ReasoningMode.RAG_ONLY:
                await self._execute_rag_retrieval(query, result, context)
            elif reasoning_mode == ReasoningMode.S3_RAG:
                await self._execute_s3_rag(query, result, context)
            elif reasoning_mode == ReasoningMode.TOT_ENHANCED_RAG:
                await self._execute_tot_enhanced_rag(query, result, context)
            elif reasoning_mode == ReasoningMode.TOT_S3_RAG:
                await self._execute_full_pipeline(query, result, context)
            else:  # ADAPTIVE
                await self._execute_adaptive_reasoning(query, result, context)
            
            result.total_time = time.time() - start_time
            
            # Cache successful results
            if self.response_cache and result.success:
                self.response_cache[query] = {
                    'result': result,
                    'timestamp': time.time()
                }
            
            logger.info(f"Query processed in {result.total_time:.2f}s using {reasoning_mode.value}")
            return result
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            result.success = False
            result.error_message = str(e)
            result.total_time = time.time() - start_time
            return result
    
    def _select_reasoning_mode(self, query: str, context: Optional[Dict[str, Any]]) -> ReasoningMode:
        """Select appropriate reasoning mode based on query characteristics"""
        
        if self.config.reasoning_mode != ReasoningMode.ADAPTIVE:
            return self.config.reasoning_mode
        
        # Simple heuristics for mode selection
        query_lower = query.lower()
        
        # Check for research indicators
        research_keywords = ['research', 'analyze', 'compare', 'synthesize', 'literature', 'papers']
        if any(keyword in query_lower for keyword in research_keywords):
            return ReasoningMode.S3_RAG if self.config.enable_s3_rag else ReasoningMode.RAG_ONLY
        
        # Check for complex reasoning indicators
        reasoning_keywords = ['solve', 'optimize', 'design', 'plan', 'strategy', 'approach']
        if any(keyword in query_lower for keyword in reasoning_keywords):
            return ReasoningMode.TOT_S3_RAG if self.s3_pipeline else ReasoningMode.TOT_ENHANCED_RAG
        
        # Check for factual queries
        factual_keywords = ['what is', 'who is', 'when did', 'where is', 'define']
        if any(keyword in query_lower for keyword in factual_keywords):
            return ReasoningMode.RAG_ONLY
        
        # Default to enhanced RAG for moderate complexity
        return ReasoningMode.TOT_ENHANCED_RAG
    
    def _assess_task_complexity(self, query: str, context: Optional[Dict[str, Any]]) -> TaskComplexity:
        """Assess task complexity based on query characteristics"""
        
        query_length = len(query.split())
        
        # Simple complexity assessment
        if query_length < 5:
            return TaskComplexity.SIMPLE
        elif query_length < 15:
            return TaskComplexity.MODERATE
        elif query_length < 30:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.RESEARCH
    
    async def _execute_tot_reasoning(self, query: str, result: UnifiedResult, 
                                   context: Optional[Dict[str, Any]]):
        """Execute pure ToT reasoning"""
        if not self.tot_engine:
            raise RuntimeError("ToT engine not available")
        
        reasoning_start = time.time()
        
        # Create task context for ToT
        task_context = context or {}
        task_context.update({
            'propose_prompt': "Solve: {task_description}\nCurrent: {current_thought}\nNext step:",
            'value_prompt': "Rate this step: {thought_content}\nScore (0.0-1.0):",
            'solution_prompt': "Is this complete: {thought_content}\nAnswer:"
        })
        
        # Execute ToT reasoning
        tot_result = await self.tot_engine.solve(query, task_context)
        result.tot_result = tot_result
        result.reasoning_time = time.time() - reasoning_start
        
        if tot_result.best_solution:
            result.response = tot_result.best_solution.content
            result.confidence_score = tot_result.best_solution.value_score
            result.reasoning_path = [
                state.content for state in 
                self.tot_engine.get_solution_path(tot_result.best_solution, tot_result.tree)
            ]
        else:
            result.response = "Unable to find a solution through reasoning."
            result.confidence_score = 0.0
    
    async def _execute_rag_retrieval(self, query: str, result: UnifiedResult,
                                   context: Optional[Dict[str, Any]]):
        """Execute traditional RAG retrieval"""
        if not self.retriever or not self.generator:
            raise RuntimeError("RAG components not available")
        
        retrieval_start = time.time()
        
        # Retrieve documents
        documents = await self.retriever.retrieve(query, k=self.config.max_documents)
        result.rag_documents = documents
        result.retrieval_time = time.time() - retrieval_start
        
        # Generate response
        generation_start = time.time()
        context_text = "\n".join([doc.get('content', '') for doc in documents])
        response = await self.generator.generate(query, context_text)
        result.response = response
        result.generation_time = time.time() - generation_start
        
        # Simple quality assessment
        result.relevance_score = min(1.0, len(documents) / self.config.max_documents)
        result.confidence_score = 0.7  # Default for RAG
    
    async def _execute_s3_rag(self, query: str, result: UnifiedResult,
                            context: Optional[Dict[str, Any]]):
        """Execute s3 RAG reasoning"""
        if not self.s3_pipeline:
            raise RuntimeError("S3 RAG pipeline not available")
        
        # Execute s3 RAG
        s3_result = await self.s3_pipeline.query(query, return_details=True)
        
        result.response = s3_result.response
        result.retrieval_time = s3_result.search_time
        result.generation_time = s3_result.generation_time
        result.rag_documents = s3_result.search_state.documents
        result.relevance_score = s3_result.relevance_score
        result.confidence_score = s3_result.search_state.average_relevance
        
        # Add s3-specific metadata
        result.metadata.update({
            'search_iterations': s3_result.search_state.iteration,
            'gbr_gain': s3_result.gbr_reward.normalized_gain if s3_result.gbr_reward else 0.0
        })
    
    async def _execute_tot_enhanced_rag(self, query: str, result: UnifiedResult,
                                      context: Optional[Dict[str, Any]]):
        """Execute ToT reasoning enhanced with RAG retrieval"""
        # First, retrieve relevant documents
        await self._execute_rag_retrieval(query, result, context)
        
        # Then, use ToT to reason about the retrieved information
        if result.rag_documents:
            # Create enhanced context with retrieved documents
            doc_context = "\n".join([
                f"Document {i+1}: {doc.get('content', '')}"
                for i, doc in enumerate(result.rag_documents[:5])
            ])
            
            enhanced_query = f"Based on the following information, {query}\n\nInformation:\n{doc_context}"
            
            # Execute ToT reasoning with enhanced context
            reasoning_start = time.time()
            task_context = {
                'propose_prompt': f"Using the provided information, solve: {enhanced_query}\nCurrent: {{current_thought}}\nNext step:",
                'value_prompt': "Rate this reasoning step: {thought_content}\nScore (0.0-1.0):",
                'solution_prompt': "Is this a complete answer: {thought_content}\nAnswer:"
            }
            
            tot_result = await self.tot_engine.solve(enhanced_query, task_context)
            result.tot_result = tot_result
            result.reasoning_time = time.time() - reasoning_start
            
            if tot_result.best_solution:
                result.response = tot_result.best_solution.content
                result.confidence_score = tot_result.best_solution.value_score
                result.reasoning_path = [
                    state.content for state in 
                    self.tot_engine.get_solution_path(tot_result.best_solution, tot_result.tree)
                ]
    
    async def _execute_full_pipeline(self, query: str, result: UnifiedResult,
                                   context: Optional[Dict[str, Any]]):
        """Execute full ToT + s3 RAG pipeline"""
        # Execute s3 RAG first for high-quality retrieval
        await self._execute_s3_rag(query, result, context)
        
        # Then enhance with ToT reasoning if needed
        if result.confidence_score < self.config.min_confidence_threshold:
            # Use ToT to improve the reasoning
            enhanced_query = f"Improve this analysis: {result.response}\n\nOriginal query: {query}"
            
            reasoning_start = time.time()
            task_context = {
                'propose_prompt': f"Improve the analysis: {enhanced_query}\nCurrent: {{current_thought}}\nNext step:",
                'value_prompt': "Rate this improvement: {thought_content}\nScore (0.0-1.0):",
                'solution_prompt': "Is this a better analysis: {thought_content}\nAnswer:"
            }
            
            tot_result = await self.tot_engine.solve(enhanced_query, task_context)
            result.tot_result = tot_result
            result.reasoning_time += time.time() - reasoning_start
            
            if tot_result.best_solution and tot_result.best_solution.value_score > result.confidence_score:
                result.response = tot_result.best_solution.content
                result.confidence_score = tot_result.best_solution.value_score
    
    async def _execute_adaptive_reasoning(self, query: str, result: UnifiedResult,
                                        context: Optional[Dict[str, Any]]):
        """Execute adaptive reasoning based on query characteristics"""
        # This is handled by mode selection, so redirect to selected mode
        selected_mode = self._select_reasoning_mode(query, context)
        result.reasoning_mode = selected_mode
        
        if selected_mode == ReasoningMode.TOT_ONLY:
            await self._execute_tot_reasoning(query, result, context)
        elif selected_mode == ReasoningMode.RAG_ONLY:
            await self._execute_rag_retrieval(query, result, context)
        elif selected_mode == ReasoningMode.S3_RAG:
            await self._execute_s3_rag(query, result, context)
        elif selected_mode == ReasoningMode.TOT_ENHANCED_RAG:
            await self._execute_tot_enhanced_rag(query, result, context)
        else:  # TOT_S3_RAG
            await self._execute_full_pipeline(query, result, context)
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            'query_count': self.query_count,
            'mode_usage': {mode.value: count for mode, count in self.mode_usage.items()},
            'complexity_distribution': {comp.value: count for comp, count in self.complexity_distribution.items()},
            'cache_size': len(self.response_cache) if self.response_cache else 0,
            'components_available': {
                'tot_engine': self.tot_engine is not None,
                's3_pipeline': self.s3_pipeline is not None,
                'vector_search': self.vector_search is not None,
                'retriever': self.retriever is not None,
                'generator': self.generator is not None
            }
        }
    
    def clear_cache(self):
        """Clear response cache"""
        if self.response_cache:
            self.response_cache.clear()
            logger.info("Response cache cleared")
