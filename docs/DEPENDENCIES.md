# PyGent Factory Dependencies

## **PYTHON VERSION REQUIREMENT**
- **Minimum**: Python 3.9+
- **Recommended**: Python 3.11+
- **Reason**: Async/await support, type hints, performance improvements

## **CORE FRAMEWORK DEPENDENCIES**

### **Web Framework**
```
fastapi>=0.104.0          # Modern async web framework
uvicorn[standard]>=0.24.0 # ASGI server with performance extras
pydantic>=2.5.0          # Data validation and serialization
```

### **Database & ORM**
```
sqlalchemy>=2.0.0        # Async ORM with modern syntax
alembic>=1.12.0          # Database migrations
psycopg2-binary>=2.9.0   # PostgreSQL adapter
asyncpg>=0.29.0          # Async PostgreSQL driver
```

### **Vector Database**
```
pgvector>=0.2.0          # PostgreSQL vector extension
chromadb>=0.4.0          # Vector database for embeddings
faiss-cpu>=1.7.0         # Facebook AI Similarity Search
```

## **MCP INTEGRATION DEPENDENCIES**

### **Official MCP SDK**
```
mcp>=0.1.0               # Official Python MCP SDK
```

### **MCP Server Dependencies**
```
# Node.js for official MCP servers
# Install via: npm install -g @modelcontextprotocol/server-filesystem
# npm install -g @modelcontextprotocol/server-postgres
# npm install -g @modelcontextprotocol/server-github
# npm install -g @modelcontextprotocol/server-brave-search
```

## **AI & ML DEPENDENCIES**

### **LLM Integration**
```
openai>=1.0.0            # OpenAI API client
anthropic>=0.7.0         # Anthropic Claude API client
ollama>=0.1.0            # Local Ollama integration
```

### **Embeddings & Transformers**
```
transformers>=4.35.0     # Hugging Face transformers
sentence-transformers>=2.2.0  # Sentence embeddings
torch>=2.1.0             # PyTorch for model inference
```

### **RAG & LangChain**
```
langchain>=0.0.350       # LLM orchestration framework
langchain-community>=0.0.10  # Community integrations
langchain-openai>=0.0.5  # OpenAI integration
langchain-anthropic>=0.0.1   # Anthropic integration
```

## **DATABASE DEPENDENCIES**

### **Supabase Integration**
```
supabase>=1.0.0          # Supabase Python client
postgrest>=0.10.0        # PostgREST client
```

### **Redis (Optional)**
```
redis>=5.0.0             # Redis client for caching
aioredis>=2.0.0          # Async Redis client
```

## **DEVELOPMENT DEPENDENCIES**

### **Code Quality**
```
black>=23.0.0            # Code formatter
isort>=5.12.0            # Import sorter
mypy>=1.7.0              # Static type checker
flake8>=6.0.0            # Linting
```

### **Testing**
```
pytest>=7.4.0           # Testing framework
pytest-asyncio>=0.21.0  # Async testing support
pytest-cov>=4.1.0       # Coverage reporting
httpx>=0.25.0            # HTTP client for testing
```

### **Development Tools**
```
pre-commit>=3.5.0        # Git hooks
python-dotenv>=1.0.0     # Environment variable loading
rich>=13.0.0             # Rich terminal output
typer>=0.9.0             # CLI framework
```

## **DEPLOYMENT DEPENDENCIES**

### **Containerization**
```
# Docker requirements (no pip install needed)
# docker>=24.0.0
# docker-compose>=2.20.0
```

### **Monitoring & Logging**
```
structlog>=23.0.0        # Structured logging
prometheus-client>=0.19.0 # Metrics collection
sentry-sdk[fastapi]>=1.38.0  # Error tracking
```

### **Security**
```
python-jose[cryptography]>=3.3.0  # JWT handling
passlib[bcrypt]>=1.7.0   # Password hashing
python-multipart>=0.0.6  # Form data parsing
```

## **COMPLETE REQUIREMENTS.TXT**

```txt
# Web Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0

# Database
sqlalchemy>=2.0.0
alembic>=1.12.0
psycopg2-binary>=2.9.0
asyncpg>=0.29.0
supabase>=1.0.0
postgrest>=0.10.0

# Vector Database
pgvector>=0.2.0
chromadb>=0.4.0
faiss-cpu>=1.7.0

# MCP Integration
mcp>=0.1.0

# AI & ML
openai>=1.0.0
anthropic>=0.7.0
ollama>=0.1.0
transformers>=4.35.0
sentence-transformers>=2.2.0
torch>=2.1.0

# RAG & LangChain
langchain>=0.0.350
langchain-community>=0.0.10
langchain-openai>=0.0.5
langchain-anthropic>=0.0.1

# Development
black>=23.0.0
isort>=5.12.0
mypy>=1.7.0
flake8>=6.0.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
httpx>=0.25.0
pre-commit>=3.5.0

# Utilities
python-dotenv>=1.0.0
rich>=13.0.0
typer>=0.9.0
structlog>=23.0.0

# Security
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.0
python-multipart>=0.0.6

# Optional: Redis
redis>=5.0.0
aioredis>=2.0.0

# Optional: Monitoring
prometheus-client>=0.19.0
sentry-sdk[fastapi]>=1.38.0
```

## **INSTALLATION COMMANDS**

### **Step-by-Step Installation**
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
pip install fastapi uvicorn sqlalchemy alembic psycopg2-binary

# Install MCP SDK
pip install mcp

# Install AI/ML dependencies
pip install openai anthropic ollama transformers sentence-transformers

# Install database dependencies
pip install supabase asyncpg pgvector

# Install RAG dependencies
pip install chromadb langchain faiss-cpu

# Install development dependencies
pip install pytest black isort mypy pre-commit

# Install all at once (alternative)
pip install -r requirements.txt
```

### **Node.js Dependencies for MCP Servers**
```bash
# Install Node.js MCP servers globally
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-postgres
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-brave-search
```

## **ENVIRONMENT VARIABLES**

### **Required Environment Variables**
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:54321/pygent_factory
SUPABASE_URL=http://localhost:54321
SUPABASE_KEY=your_supabase_anon_key

# AI APIs
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# MCP Configuration
MCP_SERVER_PATH=/path/to/mcp/servers
MCP_LOG_LEVEL=INFO

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Optional: Redis
REDIS_URL=redis://localhost:6379

# Optional: Monitoring
SENTRY_DSN=your_sentry_dsn
```

This dependency configuration ensures all required packages are available for the PyGent Factory implementation.
