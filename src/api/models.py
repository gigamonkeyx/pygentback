"""
API Models

Pydantic models for API request/response validation and serialization.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
try:
    from pydantic import BaseModel, Field
except ImportError as e:
    logger.error(f"Pydantic is required for API models: {e}")
    logger.error("Please install pydantic: pip install pydantic")
    raise ImportError("Pydantic is required for API functionality. Install with: pip install pydantic") from e

from enum import Enum


class ReasoningMode(str, Enum):
    """Reasoning mode options"""
    TOT_ONLY = "tot_only"
    RAG_ONLY = "rag_only"
    S3_RAG = "s3_rag"
    TOT_ENHANCED_RAG = "tot_enhanced_rag"
    TOT_S3_RAG = "tot_s3_rag"
    ADAPTIVE = "adaptive"


class TaskComplexity(str, Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    RESEARCH = "research"


class EvolutionStrategy(str, Enum):
    """Evolution strategy options"""
    GENETIC_ONLY = "genetic_only"
    TOT_GUIDED = "tot_guided"
    RAG_INFORMED = "rag_informed"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class SpeedRating(str, Enum):
    """Model speed rating options"""
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


class ModelArchitecture(str, Enum):
    """Model architecture options"""
    QWEN3 = "qwen3"
    QWEN2_5 = "qwen2.5"
    LLAMA3_1 = "llama3.1"
    DEEPSEEK_R1 = "deepseek-r1"
    DEEPSEEK_CODER = "deepseek-coder"
    CODELLAMA = "codellama"
    PHI4 = "phi4"
    OTHER = "other"


# Reasoning API Models
class ReasoningRequest(BaseModel):
    """Request model for reasoning endpoint"""
    query: str = Field(..., description="Query or problem to reason about")
    mode: Optional[ReasoningMode] = Field(None, description="Reasoning mode to use")
    complexity: Optional[TaskComplexity] = Field(None, description="Task complexity level")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    return_details: bool = Field(False, description="Return detailed metrics")
    max_time: Optional[float] = Field(None, description="Maximum reasoning time in seconds")


class ReasoningResponse(BaseModel):
    """Response model for reasoning endpoint"""
    query: str
    response: str
    reasoning_mode: ReasoningMode
    task_complexity: TaskComplexity
    
    # Performance metrics
    total_time: float
    reasoning_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    
    # Quality metrics
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    
    # Optional details
    reasoning_path: Optional[List[str]] = None
    documents_retrieved: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    success: bool = True
    error_message: Optional[str] = None


# Evolution API Models
class RecipeData(BaseModel):
    """Recipe data model"""
    id: str
    name: str
    description: str
    steps: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EvolutionRequest(BaseModel):
    """Request model for evolution endpoint"""
    initial_recipes: List[RecipeData] = Field(..., description="Initial recipe population")
    target_objectives: List[str] = Field(..., description="Optimization objectives")
    constraints: Optional[List[str]] = Field(None, description="Evolution constraints")
    
    # Evolution parameters
    population_size: Optional[int] = Field(None, description="Population size")
    max_generations: Optional[int] = Field(None, description="Maximum generations")
    strategy: Optional[EvolutionStrategy] = Field(None, description="Evolution strategy")
    
    # Performance settings
    max_time: Optional[float] = Field(None, description="Maximum evolution time")
    return_details: bool = Field(False, description="Return detailed evolution history")


class EvolutionResponse(BaseModel):
    """Response model for evolution endpoint"""
    success: bool
    total_time: float
    generations_completed: int
    evaluations_performed: int
    
    # Results
    best_recipes: List[Dict[str, Any]]
    convergence_achieved: bool
    
    # Optional details
    fitness_history: Optional[List[Dict[str, Any]]] = None
    final_population_size: Optional[int] = None
    
    error_message: Optional[str] = None


# Search API Models
class SearchRequest(BaseModel):
    """Request model for vector search endpoint"""
    query: str = Field(..., description="Search query")
    k: int = Field(5, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(None, description="Minimum similarity threshold")
    include_metadata: bool = Field(False, description="Include document metadata")


class SearchResult(BaseModel):
    """Individual search result"""
    id: str
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    query: str
    results: List[SearchResult]
    total_time: float
    success: bool = True
    error_message: Optional[str] = None


# System API Models
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: str
    version: str
    environment: str
    
    # Component status
    components: Dict[str, str] = Field(default_factory=dict)
    
    # System metrics
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: Optional[float] = None
    
    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    average_response_time: float = 0.0


class MetricsResponse(BaseModel):
    """Metrics response"""
    timestamp: str
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    
    # Component metrics
    reasoning_requests: int = 0
    evolution_requests: int = 0
    search_requests: int = 0
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: Optional[float] = None
    
    # AI component metrics
    tot_success_rate: float = 0.0
    rag_success_rate: float = 0.0
    evolution_success_rate: float = 0.0
    
    # Quality metrics
    average_confidence: float = 0.0
    average_relevance: float = 0.0
    average_coherence: float = 0.0


class ConfigUpdateRequest(BaseModel):
    """Configuration update request"""
    section: str = Field(..., description="Configuration section to update")
    updates: Dict[str, Any] = Field(..., description="Configuration updates")
    validate_config: bool = Field(True, description="Validate configuration after update")


class ConfigUpdateResponse(BaseModel):
    """Configuration update response"""
    success: bool
    message: str
    validation_errors: Optional[List[str]] = None
    validation_warnings: Optional[List[str]] = None


# Batch processing models
class BatchReasoningRequest(BaseModel):
    """Batch reasoning request"""
    queries: List[str] = Field(..., description="List of queries to process")
    mode: Optional[ReasoningMode] = Field(None, description="Reasoning mode for all queries")
    complexity: Optional[TaskComplexity] = Field(None, description="Task complexity for all queries")
    return_details: bool = Field(False, description="Return detailed metrics")
    max_concurrent: Optional[int] = Field(None, description="Maximum concurrent processing")


class BatchReasoningResponse(BaseModel):
    """Batch reasoning response"""
    total_queries: int
    successful_queries: int
    failed_queries: int
    total_time: float
    average_time_per_query: float
    
    results: List[ReasoningResponse]
    
    success: bool = True
    error_message: Optional[str] = None


# Model Performance API Models
class ModelPerformanceData(BaseModel):
    """Model performance data"""
    model_name: str
    model_size_gb: float
    usefulness_score: float = Field(..., ge=0, le=100, description="Usefulness score 0-100")
    speed_rating: SpeedRating
    speed_seconds: Optional[float] = Field(None, description="Actual response time in seconds")
    gpu_utilization: float = Field(..., ge=0, le=100, description="GPU utilization percentage")
    gpu_layers_offloaded: int = Field(..., ge=0, description="Number of layers offloaded to GPU")
    gpu_layers_total: int = Field(..., ge=0, description="Total number of layers")
    context_window: int = Field(..., gt=0, description="Context window size")
    parameters_billions: float = Field(..., gt=0, description="Model parameters in billions")
    architecture: ModelArchitecture
    best_use_cases: List[str] = Field(default_factory=list, description="Best use cases for this model")
    cost_per_token: Optional[float] = Field(None, description="Estimated cost per token")
    test_results: Dict[str, Any] = Field(default_factory=dict, description="Detailed test results")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


class ModelPerformanceResponse(BaseModel):
    """Model performance response"""
    id: str
    model_name: str
    model_size_gb: float
    usefulness_score: float
    speed_rating: SpeedRating
    speed_seconds: Optional[float]
    gpu_utilization: float
    gpu_layers_offloaded: int
    gpu_layers_total: int
    context_window: int
    parameters_billions: float
    architecture: ModelArchitecture
    best_use_cases: List[str]
    cost_per_token: Optional[float]
    last_tested: datetime
    test_results: Dict[str, Any]
    user_ratings: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ModelPerformanceListResponse(BaseModel):
    """Model performance list response"""
    models: List[ModelPerformanceResponse]
    total_count: int
    page: int = 1
    page_size: int = 50


class ModelRecommendationRequest(BaseModel):
    """Model recommendation request"""
    task_type: str = Field(..., description="Type of task (coding, reasoning, general)")
    priority: str = Field("balanced", description="Priority: speed, quality, or balanced")
    max_size_gb: Optional[float] = Field(None, description="Maximum model size in GB")
    min_usefulness_score: Optional[float] = Field(None, description="Minimum usefulness score")


class ModelRecommendationResponse(BaseModel):
    """Model recommendation response"""
    recommended_models: List[ModelPerformanceResponse]
    reasoning: str
    task_type: str
    priority: str


class ModelRatingRequest(BaseModel):
    """Model rating request"""
    rating: float = Field(..., ge=1, le=5, description="Rating from 1-5")
    task_type: str
    feedback: Optional[str] = Field(None, description="Optional feedback text")


class ModelRatingResponse(BaseModel):
    """Model rating response"""
    success: bool
    message: str
    new_average_rating: float


# Error models
class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    message: str
    timestamp: str
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
