"""
PyGent Factory Production Database Models

Production-ready SQLAlchemy models for PostgreSQL with comprehensive schema.
Eliminates all in-memory storage with enterprise-grade database design.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Text, JSON,
    ForeignKey, Float, LargeBinary, Index, UniqueConstraint,
    CheckConstraint, Enum
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB

# PostgreSQL-only configuration
import os
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54321/pygent_factory")

# Use PostgreSQL types exclusively
JSONType = JSONB
ArrayType = ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func

# Import pgvector for vector operations
try:
    from pgvector.sqlalchemy import Vector
    VECTOR_SUPPORT = True
except ImportError:
    VECTOR_SUPPORT = False

Base = declarative_base()


# Enums for type safety
class TaskState(PyEnum):
    """Task execution states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(PyEnum):
    """Agent types in the system"""
    ORCHESTRATOR = "orchestrator"
    DOCUMENT_PROCESSOR = "document_processor"
    VECTOR_SEARCH = "vector_search"
    A2A_AGENT = "a2a_agent"
    CODE_ANALYZER = "code_analyzer"
    CODE_GENERATOR = "code_generator"


class ProcessingStatus(PyEnum):
    """Processing status for documents and code"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                       onupdate=func.now(), nullable=False)


class User(Base, TimestampMixin):
    """User accounts and authentication"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255))
    role = Column(String(50), default="user", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime(timezone=True))

    # OAuth and preferences
    oauth_providers = Column(JSONType, default=list)
    preferences = Column(JSONType, default=dict)
    user_metadata = Column(JSONType, default=dict)

    # Relationships
    oauth_tokens = relationship("OAuthToken", back_populates="user", cascade="all, delete-orphan")
    agents = relationship("Agent", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_users_username", "username"),
        Index("idx_users_email", "email"),
        Index("idx_users_role", "role"),
        Index("idx_users_active", "is_active"),
        UniqueConstraint("username", name="uq_users_username"),
        UniqueConstraint("email", name="uq_users_email"),
    )


class OAuthToken(Base, TimestampMixin):
    """OAuth token storage"""
    __tablename__ = "oauth_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(100), nullable=False)
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text)
    token_type = Column(String(50), default="Bearer")
    expires_at = Column(DateTime(timezone=True))
    scopes = Column(JSONType, default=list)

    # Provider-specific data
    provider_user_id = Column(String(255))
    provider_user_info = Column(JSONType, default=dict)

    # Relationships
    user = relationship("User", back_populates="oauth_tokens")

    __table_args__ = (
        Index("idx_oauth_tokens_user_id", "user_id"),
        Index("idx_oauth_tokens_provider", "provider"),
        UniqueConstraint("user_id", "provider", name="uq_oauth_tokens_user_provider"),
    )


class Agent(Base, TimestampMixin):
    """Agent registry and configuration"""
    __tablename__ = "agents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    name = Column(String(255), nullable=False)
    agent_type = Column(Enum(AgentType), nullable=False)
    description = Column(Text)
    capabilities = Column(JSONType, default=list)
    configuration = Column(JSONType, default=dict)

    # Status and metrics
    status = Column(String(20), default="idle")
    is_active = Column(Boolean, default=True)
    success_rate = Column(Float, default=1.0)
    task_count = Column(Integer, default=0)

    # A2A Protocol fields
    a2a_url = Column(String(512))
    a2a_agent_card = Column(JSONType)

    @property
    def is_a2a_enabled(self) -> bool:
        """Check if agent has A2A protocol enabled"""
        return self.a2a_url is not None and self.a2a_url.strip() != ""

    @property
    def a2a_agent_id(self) -> str:
        """Get A2A agent identifier"""
        if self.a2a_agent_card and isinstance(self.a2a_agent_card, dict):
            return self.a2a_agent_card.get("id", str(self.id))
        return str(self.id)

    def get_a2a_capabilities(self) -> list:
        """Get A2A agent capabilities"""
        if self.a2a_agent_card and isinstance(self.a2a_agent_card, dict):
            return self.a2a_agent_card.get("capabilities", [])
        return []

    # Performance metrics
    avg_response_time_ms = Column(Float, default=0.0)
    last_seen_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="agents")
    tasks = relationship("Task", back_populates="agent")
    memory_entries = relationship("AgentMemory", back_populates="agent", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_agent_type_status', 'agent_type', 'status'),
        Index('idx_agent_active', 'is_active'),
        Index('idx_agent_user', 'user_id'),
    )


class Task(Base, TimestampMixin):
    """Task execution tracking"""
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_type = Column(String(100), nullable=False)
    description = Column(Text)

    # Task data
    input_data = Column(JSONType, default=dict)
    output_data = Column(JSONType, default=dict)
    task_metadata = Column(JSONType, default=dict)

    # Execution state
    state = Column(Enum(TaskState), default=TaskState.PENDING)
    priority = Column(Integer, default=5)  # 1-10 scale
    progress = Column(Float, default=0.0)  # 0.0-1.0

    # Agent assignment
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"))
    assigned_at = Column(DateTime(timezone=True))

    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    timeout_at = Column(DateTime(timezone=True))

    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # A2A Protocol fields
    a2a_context_id = Column(String(255))
    a2a_message_history = Column(JSONType, default=list)

    @property
    def is_a2a_task(self) -> bool:
        """Check if task is part of A2A protocol"""
        return self.a2a_context_id is not None and self.a2a_context_id.strip() != ""

    def add_a2a_message(self, message: dict) -> None:
        """Add message to A2A message history"""
        if not isinstance(self.a2a_message_history, list):
            self.a2a_message_history = []
        self.a2a_message_history.append(message)

    def get_a2a_message_count(self) -> int:
        """Get count of A2A messages"""
        if isinstance(self.a2a_message_history, list):
            return len(self.a2a_message_history)
        return 0

    # Relationships
    agent = relationship("Agent", back_populates="tasks")
    artifacts = relationship("TaskArtifact", back_populates="task", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_task_state_priority', 'state', 'priority'),
        Index('idx_task_agent_state', 'agent_id', 'state'),
        Index('idx_task_created', 'created_at'),
    )


class TaskArtifact(Base, TimestampMixin):
    """Task output artifacts"""
    __tablename__ = "task_artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)

    name = Column(String(255), nullable=False)
    artifact_type = Column(String(50), nullable=False)  # code, document, analysis, etc.
    content_type = Column(String(100))  # MIME type

    # Content storage
    content_text = Column(Text)  # For text content
    content_binary = Column(LargeBinary)  # For binary content
    content_json = Column(JSONType)  # For structured data
    file_path = Column(String(1024))  # For large files

    # Metadata
    size_bytes = Column(Integer)
    checksum_sha256 = Column(String(64))
    artifact_metadata = Column(JSONType, default=dict)

    # Relationships
    task = relationship("Task", back_populates="artifacts")

    __table_args__ = (
        Index('idx_artifact_task_type', 'task_id', 'artifact_type'),
    )


class AgentMemory(Base, TimestampMixin):
    """Agent memory with vector embeddings"""
    __tablename__ = "agent_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    memory_type = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    memory_metadata = Column(JSONType, default=dict)

    # Vector embedding (if pgvector available)
    embedding = Column(Vector(1536) if VECTOR_SUPPORT else JSONType)
    embedding_model = Column(String(100))

    # Relationships
    agent = relationship("Agent", back_populates="memory_entries")

    __table_args__ = (
        Index("idx_agent_memory_agent_id", "agent_id"),
        Index("idx_agent_memory_type", "memory_type"),
    )


class Document(Base, TimestampMixin):
    """Document storage and metadata"""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    title = Column(String(512), nullable=False)
    content = Column(Text)
    content_type = Column(String(100))

    # File information
    original_filename = Column(String(512))
    file_path = Column(String(1024))
    file_size = Column(Integer)
    checksum_md5 = Column(String(32))
    checksum_sha256 = Column(String(64))

    # Processing status
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    extraction_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    analysis_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)

    # Metadata and tags
    document_metadata = Column(JSONType, default=dict)
    tags = Column(ARRAY(String), default=list)

    # Source information
    source_url = Column(String(2048))
    source_type = Column(String(100))
    processed_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="documents")
    embeddings = relationship("DocumentEmbedding", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_document_status', 'processing_status'),
        Index('idx_document_type', 'content_type'),
        Index('idx_document_created', 'created_at'),
        Index('idx_document_user', 'user_id'),
    )


class DocumentChunk(Base, TimestampMixin):
    """Document chunks for RAG processing"""
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    chunk_metadata = Column(JSONType, default=dict)

    # Vector embedding
    embedding = Column(Vector(1536) if VECTOR_SUPPORT else JSONType)
    embedding_model = Column(String(100))

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_document_chunks_document_id", "document_id"),
        Index("idx_document_chunks_index", "document_id", "chunk_index"),
    )


class DocumentEmbedding(Base, TimestampMixin):
    """Vector embeddings for documents"""
    __tablename__ = "document_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)

    # Embedding data
    embedding_model = Column(String(100), nullable=False)
    embedding_vector = Column(Vector(1536) if VECTOR_SUPPORT else ARRAY(Float))
    embedding_dimension = Column(Integer, nullable=False)

    # Chunk information (for large documents)
    chunk_index = Column(Integer, default=0)
    chunk_text = Column(Text)
    chunk_start = Column(Integer)
    chunk_end = Column(Integer)

    # Metadata
    embedding_metadata = Column(JSONType, default=dict)

    # Relationships
    document = relationship("Document", back_populates="embeddings")

    __table_args__ = (
        Index('idx_embedding_document_model', 'document_id', 'embedding_model'),
        Index('idx_embedding_model_dim', 'embedding_model', 'embedding_dimension'),
    )


class CodeRepository(Base, TimestampMixin):
    """Code repository tracking"""
    __tablename__ = "code_repositories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Repository information
    repository_url = Column(String(1024))
    repository_type = Column(String(50))  # git, svn, etc.
    branch = Column(String(255), default="main")
    commit_hash = Column(String(64))

    # Local storage
    local_path = Column(String(1024))

    # Analysis status
    analysis_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    last_analyzed_at = Column(DateTime(timezone=True))

    # Metadata
    repository_metadata = Column(JSONType, default=dict)
    languages = Column(ARRAY(String), default=list)

    # Relationships
    code_files = relationship("CodeFile", back_populates="repository", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_repo_status', 'analysis_status'),
        UniqueConstraint('repository_url', 'branch', name='uq_repo_url_branch'),
    )


class CodeFile(Base, TimestampMixin):
    """Individual code files"""
    __tablename__ = "code_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("code_repositories.id"), nullable=False)

    # File information
    file_path = Column(String(1024), nullable=False)
    filename = Column(String(512), nullable=False)
    file_extension = Column(String(20))
    language = Column(String(50))

    # Content
    content = Column(Text)
    content_hash = Column(String(64))
    file_size = Column(Integer)
    line_count = Column(Integer)

    # Analysis results
    complexity_score = Column(Float)
    quality_score = Column(Float)
    maintainability_index = Column(Float)

    # Metadata
    file_metadata = Column(JSONType, default=dict)
    functions = Column(JSONType, default=list)  # Extracted function signatures
    classes = Column(JSONType, default=list)    # Extracted class definitions
    imports = Column(JSONType, default=list)    # Import statements

    # Relationships
    repository = relationship("CodeRepository", back_populates="code_files")
    embeddings = relationship("CodeEmbedding", back_populates="code_file", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_codefile_repo_path', 'repository_id', 'file_path'),
        Index('idx_codefile_language', 'language'),
        UniqueConstraint('repository_id', 'file_path', name='uq_repo_filepath'),
    )


class CodeEmbedding(Base, TimestampMixin):
    """Vector embeddings for code"""
    __tablename__ = "code_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code_file_id = Column(UUID(as_uuid=True), ForeignKey("code_files.id"), nullable=False)

    # Embedding data
    embedding_model = Column(String(100), nullable=False)
    embedding_vector = Column(Vector(1536) if VECTOR_SUPPORT else ARRAY(Float))
    embedding_dimension = Column(Integer, nullable=False)

    # Code segment information
    segment_type = Column(String(50))  # function, class, file, etc.
    segment_name = Column(String(255))
    segment_start_line = Column(Integer)
    segment_end_line = Column(Integer)
    segment_code = Column(Text)

    # Metadata
    code_metadata = Column(JSONType, default=dict)

    # Relationships
    code_file = relationship("CodeFile", back_populates="embeddings")

    __table_args__ = (
        Index('idx_code_embedding_file_type', 'code_file_id', 'segment_type'),
        Index('idx_code_embedding_model', 'embedding_model'),
    )


class UserSession(Base, TimestampMixin):
    """User session management"""
    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_token = Column(String(255), nullable=False, unique=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Session data
    session_data = Column(JSONType, default=dict)

    # Timing
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_accessed_at = Column(DateTime(timezone=True), server_default=func.now())

    # Security
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)

    # Relationships
    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        Index('idx_session_token', 'session_token'),
        Index('idx_session_user_expires', 'user_id', 'expires_at'),
    )


class SystemConfiguration(Base, TimestampMixin):
    """System-wide configuration storage"""
    __tablename__ = "system_configuration"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(255), nullable=False, unique=True)
    value = Column(JSONType, nullable=False)
    description = Column(Text)

    # Metadata
    category = Column(String(100))
    is_sensitive = Column(Boolean, default=False)

    __table_args__ = (
        Index('idx_config_category', 'category'),
    )


# Legacy models for backward compatibility (will be migrated)
class TestCase(Base, TimestampMixin):
    """Test case model - stores evaluation test cases"""
    __tablename__ = "test_cases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    input_data = Column(JSONType, nullable=False)
    expected_output = Column(JSONType)
    category = Column(String(100))
    difficulty = Column(String(50))

    __table_args__ = (
        Index("idx_test_cases_category", "category"),
        Index("idx_test_cases_difficulty", "difficulty"),
        Index("idx_test_cases_name", "name"),
    )


class KnowledgeGraph(Base, TimestampMixin):
    """Knowledge graph model - stores entity relationships"""
    __tablename__ = "knowledge_graph"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_entity = Column(String(255), nullable=False)
    predicate = Column(String(255), nullable=False)
    object_entity = Column(String(255), nullable=False)
    confidence = Column(Float, default=1.0)
    source_document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"))
    meta_data = Column(JSON, default={}, nullable=False)

    # Indexes (SQLite compatible)
    __table_args__ = (
        Index("idx_kg_subject", "subject_entity"),
        Index("idx_kg_predicate", "predicate"),
        Index("idx_kg_object", "object_entity"),
        Index("idx_kg_confidence", "confidence"),
        Index("idx_kg_triple", "subject_entity", "predicate", "object_entity"),
    )
    
    def __repr__(self):
        return f"<KnowledgeGraph(subject='{self.subject_entity}', predicate='{self.predicate}', object='{self.object_entity}')>"


class AgentSession(Base, TimestampMixin):
    """Agent session model - tracks agent interaction sessions"""
    __tablename__ = "agent_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    session_type = Column(String(100), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True))
    status = Column(String(50), default="active", nullable=False)
    meta_data = Column(JSON, default={}, nullable=False)

    # Relationships
    user = relationship("User", back_populates="sessions")
    agent = relationship("Agent")

    # Indexes (SQLite compatible)
    __table_args__ = (
        Index("idx_agent_sessions_user_id", "user_id"),
        Index("idx_agent_sessions_agent_id", "agent_id"),
        Index("idx_agent_sessions_type", "session_type"),
        Index("idx_agent_sessions_status", "status"),
        Index("idx_agent_sessions_start_time", "start_time"),
    )

    def __repr__(self):
        return f"<AgentSession(id={self.id}, agent_id={self.agent_id}, type='{self.session_type}')>"


class ModelPerformance(Base, TimestampMixin):
    """Model performance model - stores AI model performance metrics and usefulness scores"""
    __tablename__ = "model_performance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(255), nullable=False, unique=True)
    model_size_gb = Column(Float, nullable=False)
    usefulness_score = Column(Float, nullable=False)  # 0-100 scale
    speed_rating = Column(String(50), nullable=False)  # fast, medium, slow
    speed_seconds = Column(Float)  # actual response time in seconds
    gpu_utilization = Column(Float, nullable=False)  # 0-100 percentage
    gpu_layers_offloaded = Column(Integer, nullable=False)
    gpu_layers_total = Column(Integer, nullable=False)
    context_window = Column(Integer, nullable=False)  # context window size
    parameters_billions = Column(Float, nullable=False)  # model parameters in billions
    architecture = Column(String(100), nullable=False)  # qwen3, llama, deepseek, etc.
    best_use_cases = Column(JSON, default=[], nullable=False)  # array of use case strings
    cost_per_token = Column(Float)  # estimated cost per token
    last_tested = Column(DateTime(timezone=True), nullable=False)
    test_results = Column(JSON, default={}, nullable=False)  # detailed test results
    user_ratings = Column(JSON, default=[], nullable=False)  # user feedback ratings
    performance_metrics = Column(JSON, default={}, nullable=False)  # detailed metrics

    # Indexes (SQLite compatible)
    __table_args__ = (
        Index("idx_model_performance_name", "model_name"),
        Index("idx_model_performance_usefulness", "usefulness_score"),
        Index("idx_model_performance_speed", "speed_rating"),
        Index("idx_model_performance_gpu", "gpu_utilization"),
        Index("idx_model_performance_architecture", "architecture"),
        Index("idx_model_performance_last_tested", "last_tested"),
        UniqueConstraint("model_name", name="uq_model_performance_name"),
    )

    def __repr__(self):
        return f"<ModelPerformance(model='{self.model_name}', usefulness={self.usefulness_score}, speed='{self.speed_rating}')>"


# Documentation Models for user-associated documentation and research
class DocumentationFile(Base, TimestampMixin):
    """Documentation file model - stores user documentation with versioning"""
    __tablename__ = "documentation_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    meta_data = Column(JSON, default={}, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="documentation_files")
    versions = relationship("DocumentationVersion", back_populates="documentation_file", cascade="all, delete-orphan")
    tags = relationship("DocumentationTag", back_populates="documentation_file", cascade="all, delete-orphan")
    
    # Computed properties for backward compatibility
    @property
    def version_number(self) -> int:
        """Get current version number"""
        return self.meta_data.get('version_number', 1) if self.meta_data else 1
    
    @property
    def file_size(self) -> int:
        """Get file size in bytes"""
        return len(self.content.encode('utf-8')) if self.content else 0
    
    @property
    def generated_by_agent(self) -> bool:
        """Check if document was generated by an agent"""
        return self.meta_data.get('generated_by_agent', False) if self.meta_data else False
    
    @property
    def research_session_id(self) -> Optional[str]:
        """Get research session ID if available"""
        return self.meta_data.get('research_session_id') if self.meta_data else None
    
    # Indexes
    __table_args__ = (
        Index("idx_doc_files_user_id", "user_id"),
        Index("idx_doc_files_category", "category"),
        Index("idx_doc_files_path", "file_path"),
    )


class DocumentationVersion(Base, TimestampMixin):
    """Documentation version model - tracks changes to documentation files"""
    __tablename__ = "documentation_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    documentation_file_id = Column(UUID(as_uuid=True), ForeignKey("documentation_files.id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    change_summary = Column(Text)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    documentation_file = relationship("DocumentationFile", back_populates="versions")
    creator = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index("idx_doc_versions_file_id", "documentation_file_id"),
        Index("idx_doc_versions_created_by", "created_by"),
    )


class DocumentationTag(Base, TimestampMixin):
    """Documentation tag model - categorizes documentation files"""
    __tablename__ = "documentation_tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    documentation_file_id = Column(UUID(as_uuid=True), ForeignKey("documentation_files.id", ondelete="CASCADE"), nullable=False)
    tag_name = Column(String(100), nullable=False)
    
    # Relationships
    documentation_file = relationship("DocumentationFile", back_populates="tags")
    
    # Indexes
    __table_args__ = (
        Index("idx_doc_tags_file_id", "documentation_file_id"),
        Index("idx_doc_tags_name", "tag_name"),
    )


class ResearchSession(Base, TimestampMixin):
    """Research session model - tracks research agent sessions"""
    __tablename__ = "research_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    research_query = Column(Text, nullable=False)
    research_scope = Column(JSON, default={}, nullable=False)
    agent_config = Column(JSON, default={}, nullable=False)
    status = Column(String(50), default="active", nullable=False)
    results_summary = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="research_sessions")
    workflows = relationship("DocumentationWorkflow", back_populates="research_session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_research_sessions_user_id", "user_id"),
        Index("idx_research_sessions_status", "status"),
    )


class DocumentationWorkflow(Base, TimestampMixin):
    """Documentation workflow model - tracks documentation creation workflows"""
    __tablename__ = "documentation_workflows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    research_session_id = Column(UUID(as_uuid=True), ForeignKey("research_sessions.id", ondelete="SET NULL"))
    workflow_type = Column(String(100), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    parameters = Column(JSON, default={}, nullable=False)
    status = Column(String(50), default="pending", nullable=False)
    result_data = Column(JSON, default={}, nullable=False)
    
    # Relationships
    user = relationship("User")
    research_session = relationship("ResearchSession", back_populates="workflows")
    
    # Indexes
    __table_args__ = (
        Index("idx_doc_workflows_user_id", "user_id"),
        Index("idx_doc_workflows_session_id", "research_session_id"),
        Index("idx_doc_workflows_type", "workflow_type"),
        Index("idx_doc_workflows_status", "status"),
    )


# Database utility functions
def create_all_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)


def drop_all_tables(engine):
    """Drop all tables from the database"""
    Base.metadata.drop_all(bind=engine)


# Model registry for production schema
PRODUCTION_MODEL_REGISTRY = {
    "User": User,
    "OAuthToken": OAuthToken,
    "Agent": Agent,
    "Task": Task,
    "TaskArtifact": TaskArtifact,
    "AgentMemory": AgentMemory,
    "Document": Document,
    "DocumentChunk": DocumentChunk,
    "DocumentEmbedding": DocumentEmbedding,
    "CodeRepository": CodeRepository,
    "CodeFile": CodeFile,
    "CodeEmbedding": CodeEmbedding,
    "UserSession": UserSession,
    "SystemConfiguration": SystemConfiguration,
    "TestCase": TestCase,  # Legacy compatibility
}
