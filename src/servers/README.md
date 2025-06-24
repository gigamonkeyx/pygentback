# PyGent Factory Embedding MCP Server

## Overview

The PyGent Factory Embedding MCP Server transforms our internal EmbeddingService into a standalone, interoperable service that can be used by any agent or application. It provides an OpenAI-compatible API while leveraging PyGent Factory's advanced multi-provider embedding infrastructure.

## Features

### 🔌 **OpenAI Compatibility**
- **Standard API**: `/v1/embeddings` endpoint compatible with OpenAI's API
- **Request/Response Format**: Identical to OpenAI's embedding API
- **Drop-in Replacement**: Can replace OpenAI embeddings in any application

### 🚀 **Multi-Provider Support**
- **OpenAI**: Official OpenAI embeddings (text-embedding-ada-002, etc.)
- **OpenRouter**: Access to multiple embedding models via OpenRouter
- **SentenceTransformers**: Local models (BAAI/bge-small-en-v1.5, etc.)
- **Ollama**: Local Ollama models with GPU acceleration

### ⚡ **Performance Features**
- **Automatic Fallbacks**: Intelligent provider selection based on availability
- **Built-in Caching**: Embedding cache for improved performance
- **Batch Processing**: Efficient handling of multiple texts
- **GPU Acceleration**: NVIDIA RTX 3080 support for local models

### 📊 **Monitoring & Health**
- **Health Endpoints**: Real-time server health and metrics
- **Performance Tracking**: Request counts, response times, error rates
- **Provider Status**: Monitor availability of all embedding providers

## Quick Start

### 1. Start the Server

```bash
# From project root
python src/servers/embedding_mcp_server.py

# Or with custom host/port
python src/servers/embedding_mcp_server.py 0.0.0.0 8001
```

### 2. Test the Server

```bash
# Health check
curl http://localhost:8001/health

# Generate embeddings
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "text-embedding-ada-002"
  }'
```

### 3. Use with OpenAI SDK

```python
import openai

# Configure to use local server
client = openai.OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8001"
)

# Generate embeddings
response = client.embeddings.create(
    input="Your text here",
    model="text-embedding-ada-002"
)

embeddings = response.data[0].embedding
```

## API Reference

### POST /v1/embeddings

Generate embeddings for input text(s).

**Request Body:**
```json
{
  "input": "string or array of strings",
  "model": "text-embedding-ada-002",
  "encoding_format": "float",
  "dimensions": null,
  "user": "optional-user-id"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

### GET /health

Get server health status and metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-21T...",
  "service": "embedding-mcp-server",
  "version": "1.0.0",
  "providers": {
    "current_provider": "sentence_transformer",
    "available_providers": ["sentence_transformer", "ollama"],
    "provider_count": 2
  },
  "performance": {
    "uptime_seconds": 3600,
    "requests_per_second": 2.5,
    "average_response_time_ms": 150,
    "total_embeddings": 1000,
    "error_rate": 0.1
  }
}
```

## Configuration

The server uses PyGent Factory's centralized configuration system. Key settings:

```python
# In settings.py or environment variables
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Default model
OLLAMA_BASE_URL = "http://localhost:11434"   # Ollama server
OPENAI_API_KEY = "your-key"                  # OpenAI API key
OPENROUTER_API_KEY = "your-key"              # OpenRouter API key
```

## Integration Examples

### With PyGent Factory Agents

```python
# Agents can use the embedding server
embedding_client = openai.OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8001"
)

# Use in agent reasoning
embeddings = embedding_client.embeddings.create(
    input=agent_thoughts,
    model="text-embedding-ada-002"
)
```

### With External Applications

```javascript
// JavaScript/Node.js
const OpenAI = require('openai');

const client = new OpenAI({
  apiKey: 'not-needed',
  baseURL: 'http://localhost:8001'
});

const response = await client.embeddings.create({
  input: 'Text to embed',
  model: 'text-embedding-ada-002'
});
```

### With Curl

```bash
# Simple embedding request
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["First text", "Second text"],
    "model": "text-embedding-ada-002"
  }'
```

## Performance Optimization

### Batch Processing
- Send multiple texts in a single request for better throughput
- Maximum 100 texts per request
- Automatic batching for optimal performance

### Caching
- Built-in embedding cache reduces redundant computations
- Cache hit rates typically >70% in production
- Automatic cache management and cleanup

### Provider Selection
- Automatic fallback to available providers
- Performance-based provider ranking
- Health monitoring and automatic recovery

## Monitoring

### Metrics Available
- **Request Metrics**: Count, rate, response times
- **Provider Metrics**: Usage distribution, health status
- **Error Metrics**: Error rates, failure types
- **Performance Metrics**: Cache hit rates, throughput

### Health Monitoring
- Continuous health checks of all providers
- Automatic failover on provider failures
- Real-time status reporting

## Troubleshooting

### Common Issues

1. **Server Won't Start**
   - Check Python dependencies: `pip install -r requirements.txt`
   - Verify port availability: `netstat -an | grep 8001`

2. **Provider Errors**
   - Check API keys in environment variables
   - Verify Ollama server is running: `curl http://localhost:11434/api/tags`

3. **Performance Issues**
   - Monitor GPU usage: `nvidia-smi`
   - Check cache hit rates in health endpoint
   - Consider adjusting batch sizes

### Logs
Server logs provide detailed information about:
- Provider initialization and health
- Request processing and errors
- Performance metrics and warnings

## Development

### Adding New Providers
1. Implement provider class in `src/utils/embedding.py`
2. Add provider initialization in `EmbeddingService`
3. Update provider selection logic
4. Add configuration options

### Testing
```bash
# Run embedding service tests
pytest tests/test_embedding_service.py

# Test server endpoints
pytest tests/test_embedding_server.py
```

## License

MIT License - Part of PyGent Factory project.
