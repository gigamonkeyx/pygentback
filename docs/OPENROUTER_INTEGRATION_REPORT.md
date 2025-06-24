# OpenRouter Integration Research Report

## Executive Summary

After conducting deep research into OpenRouter's API, documentation, and real-world implementation patterns, I have identified the exact approach needed to integrate OpenRouter as an LLM backend for the PyGent Factory system.

## Key Findings

### 1. OpenRouter API Architecture
- **API Compatibility**: OpenRouter uses OpenAI-compatible endpoints
- **Base URL**: `https://openrouter.ai/api/v1`
- **Primary Endpoint**: `/chat/completions`
- **Authentication**: Bearer token via `Authorization` header
- **Request Format**: Identical to OpenAI Chat Completions API

### 2. Proven Integration Patterns

#### Method 1: OpenAI Client Library (Recommended)
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

response = await client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```

#### Method 2: Direct HTTP Implementation
```python
import httpx

headers = {
    "Authorization": "Bearer $OPENROUTER_API_KEY",
    "Content-Type": "application/json",
    "HTTP-Referer": "YOUR_SITE_URL",  # Optional for rankings
    "X-Title": "YOUR_SITE_NAME",      # Optional for rankings
}

payload = {
    "model": "anthropic/claude-3.5-sonnet",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": True
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )
```

## 3. Implementation Architecture for PyGent Factory

### OpenRouterBackend Class
Based on the existing `OllamaBackend` pattern in `src/ai/reasoning/tot/thought_generator.py`:

```python
class OpenRouterBackend(LLMBackend):
    """OpenRouter backend for cloud LLM inference"""
    
    def __init__(self, model_name: str = "anthropic/claude-3.5-sonnet", 
                 api_key: str = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenRouter"""
        import aiohttp
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions", 
                headers=headers, 
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"OpenRouter API error: {response.status}")
                    return ""
```

### Agent Factory Integration
Update `src/core/agent_factory.py` to support OpenRouter:

```python
def __init__(self,
             mcp_manager: Optional[MCPServerManager] = None,
             memory_manager: Optional[MemoryManager] = None,
             settings: Optional[Settings] = None,
             ollama_manager=None,
             openrouter_manager=None):  # Add OpenRouter manager
    
    # Initialize both providers
    self.ollama_manager = ollama_manager
    self.openrouter_manager = openrouter_manager

async def _validate_model_availability(self, config: AgentConfig) -> None:
    """Validate that required models are available in chosen provider."""
    provider = config.custom_config.get("provider", "ollama")
    
    if provider == "openrouter":
        if not self.openrouter_manager:
            if config.agent_type in ['reasoning', 'analysis']:
                raise AgentError(f"OpenRouter manager required for {config.agent_type} agents")
        # OpenRouter models are validated via API key, not local availability
        
    elif provider == "ollama":
        # Existing Ollama validation logic
        if not self.ollama_manager or not self.ollama_manager.is_ready:
            if config.agent_type in ['reasoning', 'analysis']:
                raise AgentError(f"Ollama manager required for {config.agent_type} agents")
```

### Reasoning Agent Updates
Update `src/agents/reasoning_agent.py` to support provider selection:

```python
def __init__(self, config: AgentConfig):
    super().__init__(config)
    
    provider = config.get_custom_config("provider", "ollama")
    model_name = config.get_custom_config("model_name", "")
    
    if provider == "openrouter":
        self.llm_backend = OpenRouterBackend(
            model_name=model_name or "anthropic/claude-3.5-sonnet",
            api_key=config.get_custom_config("openrouter_api_key")
        )
    else:
        # Existing Ollama backend
        self.llm_backend = OllamaBackend(
            model_name=model_name,
            base_url=config.get_custom_config("ollama_url", "http://localhost:11434")
        )
```

## 4. Popular Model Options

OpenRouter provides access to leading models:
- **Anthropic**: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-haiku`
- **OpenAI**: `openai/gpt-4o`, `openai/gpt-4o-mini`
- **Google**: `google/gemini-pro`, `google/gemini-flash`
- **Meta**: `meta-llama/llama-3.1-405b-instruct`
- **Mistral**: `mistralai/mixtral-8x7b-instruct`

## 5. Configuration Integration

### Environment Variables
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Agent Creation with OpenRouter
```python
agent = await agent_factory.create_agent(
    agent_type="reasoning",
    custom_config={
        "provider": "openrouter",
        "model_name": "anthropic/claude-3.5-sonnet",
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY")
    }
)
```

## 6. Error Handling & Best Practices

### Rate Limiting
- OpenRouter handles rate limiting automatically
- Different models have different rate limits
- API provides detailed error responses

### Cost Management
- Pay-per-use pricing model
- Different models have different costs per token
- Usage tracking available via OpenRouter dashboard

### Fallback Strategy
```python
async def _create_llm_backend(self, config):
    provider = config.get("provider", "ollama")
    
    if provider == "openrouter":
        try:
            return OpenRouterBackend(...)
        except Exception as e:
            logger.warning(f"OpenRouter unavailable: {e}, falling back to Ollama")
            return OllamaBackend(...)
    else:
        return OllamaBackend(...)
```

## 7. Production Deployment Considerations

### Security
- Store API keys in environment variables
- Use secure key management systems in production
- Implement request signing for additional security

### Monitoring
- Track API usage and costs
- Monitor response times and success rates
- Implement logging for debugging

### Scaling
- OpenRouter handles scaling automatically
- No need for local GPU infrastructure
- Supports high concurrent request volumes

## Implementation Priority

1. **Phase 1**: Create `OpenRouterBackend` class
2. **Phase 2**: Update Agent Factory to support provider selection
3. **Phase 3**: Update reasoning agents to use selected provider
4. **Phase 4**: Add configuration management and error handling
5. **Phase 5**: Implement comprehensive testing and validation

## Conclusion

OpenRouter integration is straightforward and production-ready. The API is mature, well-documented, and follows industry standards. Using the OpenAI client library approach provides the most robust implementation with minimal custom code required.

The key advantage is that OpenRouter provides immediate access to state-of-the-art models without requiring local infrastructure, while maintaining the same programming interface as other LLM providers.
