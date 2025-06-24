#!/bin/bash
# PyGent Factory Production Entrypoint
# Real Agent Implementation - Zero Mock Code

set -e

echo "🚀 Starting PyGent Factory Production Service"
echo "================================================"

# Validate environment
echo "📋 Validating production environment..."

# Check required environment variables
required_vars=(
    "DATABASE_URL"
    "REDIS_URL"
    "OLLAMA_URL"
    "REAL_AGENTS_ENABLED"
    "MOCK_IMPLEMENTATIONS_DISABLED"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

echo "✅ Environment validation passed"

# Verify real implementations are enabled
if [ "$REAL_AGENTS_ENABLED" != "true" ]; then
    echo "❌ ERROR: REAL_AGENTS_ENABLED must be true for production"
    exit 1
fi

if [ "$MOCK_IMPLEMENTATIONS_DISABLED" != "true" ]; then
    echo "❌ ERROR: MOCK_IMPLEMENTATIONS_DISABLED must be true for production"
    exit 1
fi

echo "✅ Real implementation configuration verified"

# Wait for dependencies
echo "📋 Waiting for dependencies..."

# Wait for PostgreSQL
echo "⏳ Waiting for PostgreSQL..."
while ! pg_isready -h postgres -p 5432 -U postgres; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 2
done
echo "✅ PostgreSQL is ready"

# Wait for Redis
echo "⏳ Waiting for Redis..."
while ! redis-cli -h redis -p 6379 ping > /dev/null 2>&1; do
    echo "Redis is unavailable - sleeping"
    sleep 2
done
echo "✅ Redis is ready"

# Wait for Ollama
echo "⏳ Waiting for Ollama..."
while ! curl -f http://ollama:11434/api/tags > /dev/null 2>&1; do
    echo "Ollama is unavailable - sleeping"
    sleep 5
done
echo "✅ Ollama is ready"

# Initialize database schema
echo "📋 Initializing database schema..."
python -c "
import sys
sys.path.insert(0, '/app/src')
import asyncio
from orchestration.real_database_client import RealDatabaseClient

async def init_schema():
    client = RealDatabaseClient('$DATABASE_URL')
    success = await client.connect()
    if success:
        await client.initialize_schema()
        await client.close()
        print('✅ Database schema initialized')
    else:
        print('❌ Database schema initialization failed')
        sys.exit(1)

asyncio.run(init_schema())
"

# Validate real agent implementations
echo "📋 Validating real agent implementations..."
python -c "
import sys
sys.path.insert(0, '/app/src')
from orchestration.real_agent_integration import RealAgentClient
import asyncio

async def validate_agents():
    try:
        from orchestration.real_agent_integration import create_real_agent_client
        client = await create_real_agent_client()
        print('✅ Real agent client validation passed')
        return True
    except Exception as e:
        print(f'❌ Real agent validation failed: {e}')
        return False

success = asyncio.run(validate_agents())
if not success:
    sys.exit(1)
"

# Pre-load Ollama models
echo "📋 Pre-loading Ollama models..."
if [ -n "$OLLAMA_MODELS" ]; then
    IFS=',' read -ra MODELS <<< "$OLLAMA_MODELS"
    for model in "${MODELS[@]}"; do
        echo "⏳ Loading model: $model"
        curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"$model\"}" || echo "⚠️ Failed to load $model"
    done
fi

# Create log directory
mkdir -p /app/logs

# Set proper permissions
chown -R pygent:pygent /app/logs

echo "🎯 Production environment ready"
echo "Real agents enabled: $REAL_AGENTS_ENABLED"
echo "Mock implementations disabled: $MOCK_IMPLEMENTATIONS_DISABLED"
echo "Database: $DATABASE_URL"
echo "Redis: $REDIS_URL"
echo "Ollama: $OLLAMA_URL"
echo "================================================"

# Execute the main command
exec "$@"
