-- Initialize PyGent Factory Database
-- This script sets up the database with pgvector extension and initial schema

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS knowledge;
CREATE SCHEMA IF NOT EXISTS mcp;
CREATE SCHEMA IF NOT EXISTS evaluation;

-- Set search path
SET search_path TO public, agents, knowledge, mcp, evaluation;

-- Create agents tables
CREATE TABLE IF NOT EXISTS agents.agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    capabilities TEXT[] DEFAULT '{}',
    config JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'inactive',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agents.agent_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents.agents(id) ON DELETE CASCADE,
    memory_type VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create knowledge base tables
CREATE TABLE IF NOT EXISTS knowledge.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    document_type VARCHAR(100) NOT NULL,
    source_url TEXT,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge.document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES knowledge.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create MCP tables
CREATE TABLE IF NOT EXISTS mcp.servers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    command TEXT[] NOT NULL,
    capabilities TEXT[] DEFAULT '{}',
    transport VARCHAR(50) DEFAULT 'stdio',
    config JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'inactive',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mcp.tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    server_id UUID REFERENCES mcp.servers(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    parameters JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create evaluation tables
CREATE TABLE IF NOT EXISTS evaluation.test_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    input_data JSONB NOT NULL,
    expected_output JSONB,
    category VARCHAR(100),
    difficulty VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation.test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_case_id UUID REFERENCES evaluation.test_cases(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents.agents(id) ON DELETE CASCADE,
    actual_output JSONB,
    score DECIMAL(5,2),
    execution_time_ms INTEGER,
    status VARCHAR(50),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents.agents(type);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents.agents(status);
CREATE INDEX IF NOT EXISTS idx_agent_memory_agent_id ON agents.agent_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON agents.agent_memory(memory_type);

CREATE INDEX IF NOT EXISTS idx_documents_type ON knowledge.documents(document_type);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON knowledge.document_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_servers_name ON mcp.servers(name);
CREATE INDEX IF NOT EXISTS idx_servers_status ON mcp.servers(status);
CREATE INDEX IF NOT EXISTS idx_tools_server_id ON mcp.tools(server_id);

CREATE INDEX IF NOT EXISTS idx_test_cases_category ON evaluation.test_cases(category);
CREATE INDEX IF NOT EXISTS idx_test_results_agent_id ON evaluation.test_results(agent_id);
CREATE INDEX IF NOT EXISTS idx_test_results_test_case_id ON evaluation.test_results(test_case_id);

-- Create vector similarity search indexes
CREATE INDEX IF NOT EXISTS idx_agent_memory_embedding ON agents.agent_memory 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_documents_embedding ON knowledge.documents 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON knowledge.document_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Insert initial MCP servers configuration
INSERT INTO mcp.servers (name, command, capabilities, transport, config) VALUES
('filesystem', ARRAY['node', 'mcp-filesystem-server'], ARRAY['file-read', 'file-write', 'directory-list'], 'stdio', '{}'),
('postgres', ARRAY['node', 'mcp-postgres-server'], ARRAY['sql-execution', 'schema-management'], 'stdio', '{}'),
('github', ARRAY['node', 'mcp-github-server'], ARRAY['repository-operations', 'commit-management'], 'stdio', '{}'),
('brave-search', ARRAY['node', 'mcp-brave-search-server'], ARRAY['web-search', 'content-extraction'], 'stdio', '{}')
ON CONFLICT (name) DO NOTHING;

-- Create functions for vector similarity search
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding vector(1536),
    similarity_threshold float DEFAULT 0.7,
    max_results int DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    title VARCHAR(500),
    content TEXT,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.content,
        1 - (d.embedding <=> query_embedding) as similarity
    FROM knowledge.documents d
    WHERE 1 - (d.embedding <=> query_embedding) > similarity_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector(1536),
    similarity_threshold float DEFAULT 0.7,
    max_results int DEFAULT 20
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    content TEXT,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.document_id,
        c.content,
        1 - (c.embedding <=> query_embedding) as similarity
    FROM knowledge.document_chunks c
    WHERE 1 - (c.embedding <=> query_embedding) > similarity_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agents TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA knowledge TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA mcp TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA evaluation TO postgres;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA agents TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA knowledge TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA mcp TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA evaluation TO postgres;

-- Log completion
SELECT 'PyGent Factory database initialized successfully!' as status;
