"""Initial schema creation

Revision ID: 0001
Revises: 
Create Date: 2025-05-27 22:39:00.000000

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    
    # Create schemas
    op.execute("CREATE SCHEMA IF NOT EXISTS agents")
    op.execute("CREATE SCHEMA IF NOT EXISTS knowledge") 
    op.execute("CREATE SCHEMA IF NOT EXISTS mcp")
    op.execute("CREATE SCHEMA IF NOT EXISTS evaluation")
    
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create agents.agents table
    op.create_table('agents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('type', sa.String(length=100), nullable=False),
        sa.Column('capabilities', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='agents'
    )
    op.create_index('idx_agents_name', 'agents', ['name'], unique=False, schema='agents')
    op.create_index('idx_agents_status', 'agents', ['status'], unique=False, schema='agents')
    op.create_index('idx_agents_type', 'agents', ['type'], unique=False, schema='agents')
    
    # Create agents.agent_memory table
    op.create_table('agent_memory',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('memory_type', sa.String(length=100), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.agents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        schema='agents'
    )
    op.create_index('idx_agent_memory_agent_id', 'agent_memory', ['agent_id'], unique=False, schema='agents')
    op.create_index('idx_agent_memory_type', 'agent_memory', ['memory_type'], unique=False, schema='agents')
    op.create_index('idx_agent_memory_embedding', 'agent_memory', ['embedding'], 
                   unique=False, schema='agents', postgresql_using='ivfflat',
                   postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_cosine_ops'})
    
    # Create agents.agent_sessions table
    op.create_table('agent_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_type', sa.String(length=100), nullable=False),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.agents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        schema='agents'
    )
    op.create_index('idx_agent_sessions_agent_id', 'agent_sessions', ['agent_id'], unique=False, schema='agents')
    op.create_index('idx_agent_sessions_start_time', 'agent_sessions', ['start_time'], unique=False, schema='agents')
    op.create_index('idx_agent_sessions_status', 'agent_sessions', ['status'], unique=False, schema='agents')
    op.create_index('idx_agent_sessions_type', 'agent_sessions', ['session_type'], unique=False, schema='agents')
    
    # Create knowledge.documents table
    op.create_table('documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('document_type', sa.String(length=100), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='knowledge'
    )
    op.create_index('idx_documents_embedding', 'documents', ['embedding'], 
                   unique=False, schema='knowledge', postgresql_using='ivfflat',
                   postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_index('idx_documents_title', 'documents', ['title'], unique=False, schema='knowledge')
    op.create_index('idx_documents_type', 'documents', ['document_type'], unique=False, schema='knowledge')
    
    # Create knowledge.document_chunks table
    op.create_table('document_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['knowledge.documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        schema='knowledge'
    )
    op.create_index('idx_document_chunks_document_id', 'document_chunks', ['document_id'], unique=False, schema='knowledge')
    op.create_index('idx_document_chunks_embedding', 'document_chunks', ['embedding'], 
                   unique=False, schema='knowledge', postgresql_using='ivfflat',
                   postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_index('idx_document_chunks_index', 'document_chunks', ['document_id', 'chunk_index'], unique=False, schema='knowledge')
    
    # Create knowledge.knowledge_graph table
    op.create_table('knowledge_graph',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('subject_entity', sa.String(length=255), nullable=False),
        sa.Column('predicate', sa.String(length=255), nullable=False),
        sa.Column('object_entity', sa.String(length=255), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('source_document_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['source_document_id'], ['knowledge.documents.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        schema='knowledge'
    )
    op.create_index('idx_kg_confidence', 'knowledge_graph', ['confidence'], unique=False, schema='knowledge')
    op.create_index('idx_kg_object', 'knowledge_graph', ['object_entity'], unique=False, schema='knowledge')
    op.create_index('idx_kg_predicate', 'knowledge_graph', ['predicate'], unique=False, schema='knowledge')
    op.create_index('idx_kg_subject', 'knowledge_graph', ['subject_entity'], unique=False, schema='knowledge')
    op.create_index('idx_kg_triple', 'knowledge_graph', ['subject_entity', 'predicate', 'object_entity'], unique=False, schema='knowledge')
    
    # Create mcp.servers table
    op.create_table('servers',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('command', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('capabilities', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('transport', sa.String(length=50), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', name='uq_servers_name'),
        schema='mcp'
    )
    op.create_index('idx_servers_name', 'servers', ['name'], unique=False, schema='mcp')
    op.create_index('idx_servers_status', 'servers', ['status'], unique=False, schema='mcp')
    
    # Create mcp.tools table
    op.create_table('tools',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('server_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['server_id'], ['mcp.servers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('server_id', 'name', name='uq_tools_server_name'),
        schema='mcp'
    )
    op.create_index('idx_tools_name', 'tools', ['name'], unique=False, schema='mcp')
    op.create_index('idx_tools_server_id', 'tools', ['server_id'], unique=False, schema='mcp')
    
    # Create evaluation.test_cases table
    op.create_table('test_cases',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('input_data', sa.JSON(), nullable=False),
        sa.Column('expected_output', sa.JSON(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('difficulty', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='evaluation'
    )
    op.create_index('idx_test_cases_category', 'test_cases', ['category'], unique=False, schema='evaluation')
    op.create_index('idx_test_cases_difficulty', 'test_cases', ['difficulty'], unique=False, schema='evaluation')
    op.create_index('idx_test_cases_name', 'test_cases', ['name'], unique=False, schema='evaluation')
    
    # Create evaluation.test_results table
    op.create_table('test_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('test_case_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('actual_output', sa.JSON(), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['test_case_id'], ['evaluation.test_cases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        schema='evaluation'
    )
    op.create_index('idx_test_results_agent_id', 'test_results', ['agent_id'], unique=False, schema='evaluation')
    op.create_index('idx_test_results_score', 'test_results', ['score'], unique=False, schema='evaluation')
    op.create_index('idx_test_results_status', 'test_results', ['status'], unique=False, schema='evaluation')
    op.create_index('idx_test_results_test_case_id', 'test_results', ['test_case_id'], unique=False, schema='evaluation')


def downgrade() -> None:
    """Downgrade database schema."""
    
    # Drop tables in reverse order
    op.drop_table('test_results', schema='evaluation')
    op.drop_table('test_cases', schema='evaluation')
    op.drop_table('tools', schema='mcp')
    op.drop_table('servers', schema='mcp')
    op.drop_table('knowledge_graph', schema='knowledge')
    op.drop_table('document_chunks', schema='knowledge')
    op.drop_table('documents', schema='knowledge')
    op.drop_table('agent_sessions', schema='agents')
    op.drop_table('agent_memory', schema='agents')
    op.drop_table('agents', schema='agents')
    
    # Drop schemas
    op.execute("DROP SCHEMA IF EXISTS evaluation CASCADE")
    op.execute("DROP SCHEMA IF EXISTS mcp CASCADE")
    op.execute("DROP SCHEMA IF EXISTS knowledge CASCADE")
    op.execute("DROP SCHEMA IF EXISTS agents CASCADE")
