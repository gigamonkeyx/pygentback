"""
NLP Data Models

Data structures and models for natural language processing components.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class RecipeIntent(Enum):
    """Types of recipe intents"""
    CREATE_CONTENT = "create_content"
    ANALYZE_DATA = "analyze_data"
    PROCESS_FILES = "process_files"
    INTEGRATE_APIS = "integrate_apis"
    AUTOMATE_WORKFLOW = "automate_workflow"
    GENERATE_REPORTS = "generate_reports"
    TRANSFORM_DATA = "transform_data"
    VALIDATE_INPUT = "validate_input"
    MONITOR_SYSTEM = "monitor_system"
    COMMUNICATE = "communicate"
    SEARCH_INFORMATION = "search_information"
    EXECUTE_CODE = "execute_code"


@dataclass
class RecipeIntentResult:
    """Result from intent extraction with attributes expected by tests"""
    intent_type: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


class ActionType(Enum):
    """Types of actions in recipes"""
    INPUT = "input"
    OUTPUT = "output"
    PROCESSING = "processing"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    INTEGRATION = "integration"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"


class TestStatus(Enum):
    """Test execution status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class QueryType(Enum):
    """Types of natural language queries"""
    RECIPE_SEARCH = "recipe_search"
    SERVER_STATUS = "server_status"
    PERFORMANCE_QUERY = "performance_query"
    CAPABILITY_QUERY = "capability_query"
    HELP_REQUEST = "help_request"
    CONFIGURATION = "configuration"
    TROUBLESHOOTING = "troubleshooting"


class DocumentationType(Enum):
    """Types of generated documentation"""
    API_DOCS = "api_docs"
    USER_GUIDE = "user_guide"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    TROUBLESHOOTING = "troubleshooting"
    CHANGELOG = "changelog"
    README = "readme"


@dataclass
class ParsedAction:
    """Parsed action from natural language"""
    action_type: ActionType
    description: str
    tool_suggestions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'action_type': self.action_type.value,
            'description': self.description,
            'tool_suggestions': self.tool_suggestions,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'conditions': self.conditions,
            'confidence': self.confidence
        }


@dataclass
class ParsedRecipe:
    """Parsed recipe from natural language description"""
    name: str
    description: str
    intent: RecipeIntent
    actions: List[ParsedAction] = field(default_factory=list)
    
    # Recipe metadata
    complexity_level: str = "medium"  # simple, medium, complex
    estimated_duration: Optional[str] = None
    required_capabilities: List[str] = field(default_factory=list)
    
    # Parsing metadata
    parsing_confidence: float = 0.0
    ambiguities: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Source information
    original_text: str = ""
    parsed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'intent': self.intent.value,
            'actions': [action.to_dict() for action in self.actions],
            'complexity_level': self.complexity_level,
            'estimated_duration': self.estimated_duration,
            'required_capabilities': self.required_capabilities,
            'parsing_confidence': self.parsing_confidence,
            'ambiguities': self.ambiguities,
            'suggestions': self.suggestions,
            'original_text': self.original_text,
            'parsed_at': self.parsed_at.isoformat()
        }


@dataclass
class TestMetrics:
    """Test execution metrics"""
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_requests: int = 0
    errors_count: int = 0
    warnings_count: int = 0


@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    status: TestStatus
    message: str = ""
    
    # Detailed results
    expected_result: Any = None
    actual_result: Any = None
    error_details: Optional[str] = None
    
    # Metrics
    metrics: TestMetrics = field(default_factory=TestMetrics)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    test_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_name': self.test_name,
            'status': self.status.value,
            'message': self.message,
            'expected_result': self.expected_result,
            'actual_result': self.actual_result,
            'error_details': self.error_details,
            'metrics': {
                'execution_time_ms': self.metrics.execution_time_ms,
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'cpu_usage_percent': self.metrics.cpu_usage_percent,
                'network_requests': self.metrics.network_requests,
                'errors_count': self.metrics.errors_count,
                'warnings_count': self.metrics.warnings_count
            },
            'timestamp': self.timestamp.isoformat(),
            'test_id': self.test_id,
            'tags': self.tags
        }


@dataclass
class InterpretationResult:
    """Result of test interpretation"""
    summary: str
    success_rate: float
    performance_assessment: str
    
    # Detailed analysis
    passed_tests: List[str] = field(default_factory=list)
    failed_tests: List[str] = field(default_factory=list)
    error_patterns: List[str] = field(default_factory=list)
    performance_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Confidence and metadata
    interpretation_confidence: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QueryContext:
    """Context for natural language queries"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    previous_queries: List[str] = field(default_factory=list)
    current_state: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResponse:
    """Response to natural language query"""
    query_type: QueryType
    response_text: str
    
    # Structured data
    data: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    
    # Response metadata
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    sources: List[str] = field(default_factory=list)
    
    # Context
    requires_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'query_type': self.query_type.value,
            'response_text': self.response_text,
            'data': self.data,
            'suggestions': self.suggestions,
            'follow_up_questions': self.follow_up_questions,
            'confidence': self.confidence,
            'processing_time_ms': self.processing_time_ms,
            'sources': self.sources,
            'requires_clarification': self.requires_clarification,
            'clarification_questions': self.clarification_questions
        }


@dataclass
class DocumentationSection:
    """Section of generated documentation"""
    title: str
    content: str
    section_type: str = "content"  # content, code, example, note, warning
    level: int = 1  # heading level
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedDoc:
    """Generated documentation"""
    title: str
    doc_type: DocumentationType
    sections: List[DocumentationSection] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    template_used: Optional[str] = None
    generation_confidence: float = 0.0
    
    # Source information
    source_data: Dict[str, Any] = field(default_factory=dict)
    generation_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'doc_type': self.doc_type.value,
            'sections': [
                {
                    'title': section.title,
                    'content': section.content,
                    'section_type': section.section_type,
                    'level': section.level,
                    'metadata': section.metadata
                }
                for section in self.sections
            ],
            'generated_at': self.generated_at.isoformat(),
            'template_used': self.template_used,
            'generation_confidence': self.generation_confidence,
            'source_data': self.source_data,
            'generation_context': self.generation_context
        }


@dataclass
class SemanticEmbedding:
    """Semantic embedding representation"""
    text: str
    embedding: List[float]
    model_name: str = "default"
    embedding_dimension: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if self.embedding_dimension == 0:
            self.embedding_dimension = len(self.embedding)


@dataclass
class SimilarityScore:
    """Similarity score between texts"""
    text1: str
    text2: str
    score: float
    similarity_type: str = "cosine"  # cosine, euclidean, jaccard
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
@dataclass
class Intent:
    """Classified intent"""
    intent_type: str
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)  # Add entities field for test compatibility


@dataclass
class IntentPrediction:
    """Intent prediction result"""
    primary_intent: Intent
    alternative_intents: List[Intent] = field(default_factory=list)
    prediction_confidence: float = 0.0
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


# Utility functions for model validation
def validate_confidence_score(score: float) -> bool:
    """Validate confidence score is between 0 and 1"""
    return 0.0 <= score <= 1.0


def validate_test_result(result: TestResult) -> bool:
    """Validate test result structure"""
    if not isinstance(result.test_name, str) or not result.test_name:
        return False
    
    if not isinstance(result.status, TestStatus):
        return False
    
    if not validate_confidence_score(result.metrics.cpu_usage_percent / 100.0):
        return False
    
    return True


def validate_parsed_recipe(recipe: ParsedRecipe) -> bool:
    """Validate parsed recipe structure"""
    if not isinstance(recipe.name, str) or not recipe.name:
        return False
    
    if not isinstance(recipe.intent, RecipeIntent):
        return False
    
    if not validate_confidence_score(recipe.parsing_confidence):
        return False
    
    for action in recipe.actions:
        if not isinstance(action.action_type, ActionType):
            return False
        if not validate_confidence_score(action.confidence):
            return False
    
    return True
