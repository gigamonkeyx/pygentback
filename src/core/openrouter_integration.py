"""
OpenRouter Integration Framework for External LLM API Access
Provides unified API for managing external LLM models via OpenRouter.
"""
import os
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

@dataclass
class OpenRouterModel:
    """OpenRouter model configuration."""
    name: str
    provider: str
    cost_per_token: float
    context_length: int
    capabilities: List[str]
    description: str

@dataclass
class OpenRouterResponse:
    """Response from OpenRouter API."""
    content: str
    model: str
    usage: Dict[str, int]
    cost: float

class OpenRouterManager:
    """Manages OpenRouter API access and model selection."""
    
    def __init__(self):
        self.api_key = None
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = None
        self._initialized = False
        
        # Available models registry
        self.available_models = {
            "anthropic/claude-3-sonnet": OpenRouterModel(
                name="claude-3-sonnet",
                provider="anthropic",
                cost_per_token=0.000003,
                context_length=200000,
                capabilities=["reasoning", "analysis", "writing"],
                description="High-quality reasoning and analysis model"
            ),
            "openai/gpt-4o": OpenRouterModel(
                name="gpt-4o",
                provider="openai",
                cost_per_token=0.000005,
                context_length=128000,
                capabilities=["reasoning", "analysis", "multimodal"],
                description="Advanced multimodal reasoning model"
            ),
            "google/gemini-pro": OpenRouterModel(
                name="gemini-pro",
                provider="google",
                cost_per_token=0.000001,
                context_length=128000,
                capabilities=["reasoning", "analysis"],
                description="Google's advanced reasoning model"
            )
        }
        
        self.budget_used = 0.0
        self.budget_limit = 10.0  # Default $10 limit
    
    async def initialize(self) -> bool:
        """Initialize OpenRouter connection."""
        try:
            # Get API key from environment
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                logger.warning("No OpenRouter API key found. Set OPENROUTER_API_KEY environment variable.")
                return False
            
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test API connection
            if await self._test_connection():
                self._initialized = True
                logger.info("OpenRouter manager initialized successfully")
                return True
            else:
                logger.error("Failed to connect to OpenRouter API")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter manager: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test connection to OpenRouter API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Simple test request
            async with self.session.get(
                f"{self.base_url}/models",
                headers=headers
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.debug(f"OpenRouter connection test failed: {e}")
            return False
    
    async def generate(self, 
                      prompt: str,
                      model: str = "anthropic/claude-3-sonnet",
                      max_tokens: int = 4000,
                      temperature: float = 0.7,
                      **kwargs) -> OpenRouterResponse:
        """Generate response using OpenRouter model."""
        
        if not self._initialized:
            raise RuntimeError("OpenRouter manager not initialized")
        
        if self.budget_used >= self.budget_limit:
            raise RuntimeError(f"Budget limit exceeded: ${self.budget_used:.2f} / ${self.budget_limit:.2f}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://pygent-factory.local",
            "X-Title": "PyGent Factory Historical Research"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"OpenRouter API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Extract response
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                
                # Calculate cost (simplified)
                model_config = self.available_models.get(model)
                cost = 0.0
                if model_config and usage:
                    total_tokens = usage.get("total_tokens", 0)
                    cost = total_tokens * model_config.cost_per_token
                    self.budget_used += cost
                
                return OpenRouterResponse(
                    content=content,
                    model=model,
                    usage=usage,
                    cost=cost
                )
                
        except Exception as e:
            logger.error(f"Failed to generate response with OpenRouter: {e}")
            raise
    
    async def analyze_for_facts(self, content: str, topic: str) -> Dict[str, Any]:
        """Analyze content for factual claims related to topic."""
        
        prompt = f"""Analyze the following historical content about "{topic}" for factual claims:

{content[:3000]}

Identify:
1. Specific historical facts and claims
2. Dates, names, and events mentioned
3. Verifiable vs unverifiable statements
4. Potential inaccuracies or biases
5. Overall credibility assessment

Provide analysis as JSON with structured data."""
        
        response = await self.generate(
            prompt=prompt,
            model="anthropic/claude-3-sonnet",
            temperature=0.2
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"analysis": response.content}
    
    async def synthesize_research(self, documents: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        """Synthesize research from multiple documents."""
        
        # Prepare document summaries
        doc_summaries = []
        for doc in documents[:5]:  # Limit to prevent token overflow
            summary = {
                "title": doc.get("title", "Unknown"),
                "content": doc.get("content", "")[:500] + "...",
                "source": doc.get("source", "Unknown")
            }
            doc_summaries.append(summary)
        
        prompt = f"""Synthesize comprehensive research on "{topic}" based on these historical documents:

{json.dumps(doc_summaries, indent=2)}

Create a scholarly synthesis that:
1. Identifies key themes and patterns
2. Presents a coherent narrative
3. Notes areas of consensus and disagreement
4. Evaluates source reliability
5. Draws evidence-based conclusions

Provide as structured JSON with sections for introduction, main findings, conclusions, and sources."""
        
        response = await self.generate(
            prompt=prompt,
            model="anthropic/claude-3-sonnet",
            max_tokens=6000,
            temperature=0.3
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"synthesis": response.content}
    
    def get_budget_status(self) -> Dict[str, float]:
        """Get current budget usage status."""
        return {
            "used": self.budget_used,
            "limit": self.budget_limit,
            "remaining": self.budget_limit - self.budget_used,
            "usage_percentage": (self.budget_used / self.budget_limit) * 100
        }
    
    def set_budget_limit(self, limit: float):
        """Set budget limit for API usage."""
        self.budget_limit = limit
        logger.info(f"OpenRouter budget limit set to ${limit:.2f}")
    
    def get_available_models(self) -> Dict[str, OpenRouterModel]:
        """Get list of available models."""
        return self.available_models
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()

# Global OpenRouter manager instance
openrouter_manager = OpenRouterManager()

# Backward compatibility alias
OpenRouterIntegration = OpenRouterManager

# Export all classes
__all__ = [
    "OpenRouterModel",
    "OpenRouterResponse",
    "OpenRouterManager",
    "OpenRouterIntegration",  # Backward compatibility
    "openrouter_manager"
]
