"""
Documentation Database Models

Database models for storing documentation metadata, content, versions,
and integration with research agents and document creation system.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DocumentationFile(Base):
    """Core documentation file metadata and content."""
    __tablename__ = "documentation_files"
    
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(500), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    content = Column(Text, nullable=False)
    html_content = Column(Text)  # Rendered HTML
    
    # Metadata
    file_size = Column(Integer, default=0)
    checksum = Column(String(64))  # SHA-256 hash for change detection
    
    # Research integration
    research_session_id = Column(String(100), nullable=True, index=True)
    generated_by_agent = Column(Boolean, default=False)
    agent_workflow_id = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    versions = relationship("DocumentationVersion", back_populates="document", cascade="all, delete-orphan")
    tags = relationship("DocumentationTag", back_populates="document", cascade="all, delete-orphan")
    links = relationship("DocumentationLink", foreign_keys="DocumentationLink.source_doc_id", back_populates="source_doc")
    
    # Indexes
    __table_args__ = (
        Index('idx_doc_category_updated', 'category', 'updated_at'),
        Index('idx_doc_research_session', 'research_session_id', 'created_at'),
        Index('idx_doc_agent_generated', 'generated_by_agent', 'created_at'),
    )


class DocumentationVersion(Base):
    """Version history for documentation files."""
    __tablename__ = "documentation_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documentation_files.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    html_content = Column(Text)
    
    # Change metadata
    change_summary = Column(String(500))
    changed_by = Column(String(100))  # User or agent identifier
    change_type = Column(String(50))  # 'manual', 'research_agent', 'documentation_agent'
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("DocumentationFile", back_populates="versions")
    
    # Indexes
    __table_args__ = (
        Index('idx_version_doc_number', 'document_id', 'version_number'),
    )


class DocumentationTag(Base):
    """Tags for categorizing and organizing documentation."""
    __tablename__ = "documentation_tags"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documentation_files.id"), nullable=False)
    tag_name = Column(String(100), nullable=False, index=True)
    tag_type = Column(String(50), default='manual')  # 'manual', 'auto_generated', 'research_derived'
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("DocumentationFile", back_populates="tags")
    
    # Indexes
    __table_args__ = (
        Index('idx_tag_name_type', 'tag_name', 'tag_type'),
    )


class DocumentationLink(Base):
    """Cross-references and links between documentation files."""
    __tablename__ = "documentation_links"
    
    id = Column(Integer, primary_key=True, index=True)
    source_doc_id = Column(Integer, ForeignKey("documentation_files.id"), nullable=False)
    target_doc_id = Column(Integer, ForeignKey("documentation_files.id"), nullable=True)
    external_url = Column(String(1000), nullable=True)
    
    link_type = Column(String(50), default='reference')  # 'reference', 'related', 'supersedes', 'implements'
    link_text = Column(String(255))
    context = Column(Text)  # Surrounding text context
    
    # Auto-detection metadata
    auto_detected = Column(Boolean, default=False)
    confidence_score = Column(Integer, default=100)  # 0-100
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    validated_at = Column(DateTime, nullable=True)
    
    # Relationships
    source_doc = relationship("DocumentationFile", foreign_keys=[source_doc_id])
    target_doc = relationship("DocumentationFile", foreign_keys=[target_doc_id])


class ResearchSession(Base):
    """Research sessions that generate documentation."""
    __tablename__ = "research_sessions"
    
    id = Column(String(100), primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Session metadata
    session_type = Column(String(50), default='research')  # 'research', 'documentation', 'analysis'
    status = Column(String(50), default='active')  # 'active', 'completed', 'failed', 'cancelled'
    
    # Research parameters
    research_query = Column(Text)
    research_scope = Column(JSON)  # Areas, topics, constraints
    agent_config = Column(JSON)  # Agent configuration used
    
    # Results
    documents_generated = Column(Integer, default=0)
    insights_generated = Column(Integer, default=0)
    quality_score = Column(Integer, nullable=True)  # 0-100
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Relationships - Note: Using string foreign key to avoid circular import
    # documents = relationship("DocumentationFile", backref="research_session")


class DocumentationWorkflow(Base):
    """Workflow tracking for document creation and updates."""
    __tablename__ = "documentation_workflows"
    
    id = Column(String(100), primary_key=True, index=True)
    workflow_type = Column(String(50), nullable=False)  # 'creation', 'update', 'reorganization'
    status = Column(String(50), default='pending')  # 'pending', 'running', 'completed', 'failed'
    
    # Workflow metadata
    title = Column(String(255), nullable=False)
    description = Column(Text)
    parameters = Column(JSON)  # Workflow-specific parameters
    
    # Progress tracking
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    current_step = Column(String(255))
    
    # Results
    documents_created = Column(Integer, default=0)
    documents_updated = Column(Integer, default=0)
    errors_encountered = Column(Integer, default=0)
    result_summary = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_workflow_type_status', 'workflow_type', 'status'),
        Index('idx_workflow_created', 'created_at'),
    )


class DocumentationAccess(Base):
    """Track access patterns for documentation."""
    __tablename__ = "documentation_access"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documentation_files.id"), nullable=False)
    
    # Access metadata
    access_type = Column(String(50), default='view')  # 'view', 'search', 'download', 'link_follow'
    user_session = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # Context
    referrer_url = Column(String(1000), nullable=True)
    search_query = Column(String(255), nullable=True)
    
    # Timestamps
    accessed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    document = relationship("DocumentationFile")
    
    # Indexes
    __table_args__ = (
        Index('idx_access_doc_time', 'document_id', 'accessed_at'),
        Index('idx_access_session_time', 'user_session', 'accessed_at'),
    )


class DocumentationSearch(Base):
    """Search queries and results for analytics."""
    __tablename__ = "documentation_searches"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Search details
    query = Column(String(255), nullable=False, index=True)
    query_normalized = Column(String(255), index=True)  # Cleaned/normalized version
    search_type = Column(String(50), default='full_text')  # 'full_text', 'semantic', 'tag_based'
    
    # Results
    results_count = Column(Integer, default=0)
    results_data = Column(JSON)  # Actual search results (limited)
    
    # User context
    user_session = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    
    # Performance
    search_duration_ms = Column(Integer, default=0)
    
    # Timestamps
    searched_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_search_query_time', 'query', 'searched_at'),
        Index('idx_search_session_time', 'user_session', 'searched_at'),
    )


class DocumentationTemplate(Base):
    """Templates for generating new documentation."""
    __tablename__ = "documentation_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    
    # Template content
    template_content = Column(Text, nullable=False)
    variables = Column(JSON)  # Template variables and their types
    
    # Metadata
    category = Column(String(100), nullable=False, index=True)
    template_type = Column(String(50), default='markdown')  # 'markdown', 'structured', 'research_report'
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# Database initialization function
def create_documentation_tables(engine):
    """Create all documentation-related database tables."""
    Base.metadata.create_all(bind=engine)


# Database utility functions
def get_documentation_schema_info():
    """Get schema information for documentation tables."""
    tables = {}
    for table_name, table in Base.metadata.tables.items():
        tables[table_name] = {
            'columns': [col.name for col in table.columns],
            'indexes': [idx.name for idx in table.indexes],
            'foreign_keys': [fk.column.name for fk in table.foreign_keys]
        }
    return tables
