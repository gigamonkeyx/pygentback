-- Initialize PyGent Factory Database with A2A Protocol Integration

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    content TEXT,
    content_type VARCHAR(100),
    original_filename VARCHAR(255),
    file_path VARCHAR(500),
    file_size INTEGER,
    checksum_md5 VARCHAR(32),
    checksum_sha256 VARCHAR(64),
    processing_status VARCHAR(50) DEFAULT 'pending',
    extraction_status VARCHAR(50) DEFAULT 'pending',
    analysis_status VARCHAR(50) DEFAULT 'pending',
    document_metadata JSONB,
    tags TEXT[],
    source_url VARCHAR(500),
    source_type VARCHAR(100),
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- A2A Protocol Integration Note:
-- A2A fields are integrated into main tables (agents, tasks) rather than separate tables.
-- See agents.a2a_url, agents.a2a_agent_card, tasks.a2a_context_id, tasks.a2a_message_history

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_title ON documents USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_documents_content ON documents USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
-- A2A Protocol indexes (applied via migration 0002_a2a_integration)
-- CREATE INDEX IF NOT EXISTS idx_agents_a2a_url ON agents(a2a_url);
-- CREATE INDEX IF NOT EXISTS idx_tasks_a2a_context ON tasks(a2a_context_id);

-- Insert default admin user
INSERT INTO users (id, username, email, password_hash, is_admin) 
VALUES (
    gen_random_uuid(),
    'admin',
    'admin@pygentfactory.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3QJflLxQxe', -- password: admin123
    true
) ON CONFLICT (username) DO NOTHING;

-- Insert sample documents for testing
INSERT INTO documents (user_id, title, content, source_type) 
SELECT 
    (SELECT id FROM users WHERE username = 'admin'),
    'Introduction to Machine Learning',
    'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.',
    'sample'
WHERE NOT EXISTS (SELECT 1 FROM documents WHERE title = 'Introduction to Machine Learning');

INSERT INTO documents (user_id, title, content, source_type) 
SELECT 
    (SELECT id FROM users WHERE username = 'admin'),
    'Neural Network Architectures',
    'Neural networks are computing systems inspired by biological neural networks. Common architectures include feedforward networks, convolutional neural networks (CNNs), and recurrent neural networks (RNNs).',
    'sample'
WHERE NOT EXISTS (SELECT 1 FROM documents WHERE title = 'Neural Network Architectures');

INSERT INTO documents (user_id, title, content, source_type) 
SELECT 
    (SELECT id FROM users WHERE username = 'admin'),
    'Artificial Intelligence Applications',
    'AI applications span many domains including natural language processing, computer vision, robotics, and autonomous systems. These technologies are transforming industries worldwide.',
    'sample'
WHERE NOT EXISTS (SELECT 1 FROM documents WHERE title = 'Artificial Intelligence Applications');
