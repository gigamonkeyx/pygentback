"""
Ollama Integration Framework for Local LLM Inference
Provides unified API for managing local Ollama models with fallback capabilities.
"""
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelCapability(Enum):
    """Model capability categories."""
    REASONING = "reasoning"
    FAST_RESPONSE = "fast_response"
    MULTIMODAL = "multimodal"
    CODE_ANALYSIS = "code_analysis"
    FACT_CHECKING = "fact_checking"

@dataclass
class OllamaModel:
    """Ollama model configuration."""
    name: str
    size_gb: float
    capabilities: List[ModelCapability]
    description: str
    context_length: int = 4096
    temperature: float = 0.7
    is_available: bool = False

@dataclass
class ModelResponse:
    """Response from Ollama model."""
    content: str
    model: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

class OllamaManager:
    """Manages Ollama models and provides unified API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = None
        self._initialized = False
        
        # Model registry based on available models
        self.available_models = {
            "qwen2.5:3b": OllamaModel(
                name="qwen2.5:3b",
                size_gb=2.0,
                capabilities=[ModelCapability.FAST_RESPONSE, ModelCapability.REASONING],
                description="Fast general-purpose model for quick responses",
                context_length=8192
            ),
            "qwen2.5:7b": OllamaModel(
                name="qwen2.5:7b", 
                size_gb=4.4,
                capabilities=[ModelCapability.REASONING, ModelCapability.FACT_CHECKING],
                description="Balanced model for reasoning and analysis",
                context_length=8192
            ),
            "deepseek-r1:8b": OllamaModel(
                name="deepseek-r1:8b",
                size_gb=5.2,
                capabilities=[ModelCapability.REASONING, ModelCapability.CODE_ANALYSIS],
                description="Strong reasoning model for complex analysis",
                context_length=4096
            ),
            "janus:latest": OllamaModel(
                name="janus:latest",
                size_gb=4.0,
                capabilities=[ModelCapability.MULTIMODAL],
                description="Multimodal model for document and image analysis",
                context_length=4096
            )
        }
        
        # Model assignment for different tasks
        self.task_models = {
            ModelCapability.FAST_RESPONSE: "qwen2.5:3b",
            ModelCapability.REASONING: "deepseek-r1:8b", 
            ModelCapability.FACT_CHECKING: "qwen2.5:7b",
            ModelCapability.CODE_ANALYSIS: "deepseek-r1:8b",
            ModelCapability.MULTIMODAL: "janus:latest"
        }
        
        self.current_model = None
        self.model_load_times = {}
    
    async def initialize(self) -> bool:
        """Initialize Ollama connection and check available models."""
        try:
            self.session = aiohttp.ClientSession()
            
            # Check Ollama service
            if not await self._check_ollama_service():
                logger.error("Ollama service not available")
                return False
            
            # Detect available models
            await self._detect_available_models()
            
            # Set default model
            self.current_model = self._get_best_available_model(ModelCapability.REASONING)
            
            self._initialized = True
            logger.info(f"Ollama manager initialized with {len([m for m in self.available_models.values() if m.is_available])} available models")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama manager: {e}")
            return False
    
    async def _check_ollama_service(self) -> bool:
        """Check if Ollama service is running."""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"Ollama service check failed: {e}")
            return False
    
    async def _detect_available_models(self):
        """Detect which models are actually available in Ollama."""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_names = [model["name"] for model in data.get("models", [])]
                    
                    for model_name, model_config in self.available_models.items():
                        # Check exact match or base name match
                        is_available = (model_name in available_names or 
                                      any(name.startswith(model_name.split(":")[0]) for name in available_names))
                        model_config.is_available = is_available
                        
                        if is_available:
                            logger.info(f"Found available model: {model_name}")
                        
        except Exception as e:
            logger.error(f"Failed to detect available models: {e}")
    
    def _get_best_available_model(self, capability: ModelCapability) -> Optional[str]:
        """Get the best available model for a specific capability."""
        # Try preferred model for capability
        preferred = self.task_models.get(capability)
        if preferred and self.available_models.get(preferred, {}).is_available:
            return preferred
        
        # Fallback to any available model with the capability
        for model_name, model_config in self.available_models.items():
            if model_config.is_available and capability in model_config.capabilities:
                return model_name
        
        # Last resort: any available model
        for model_name, model_config in self.available_models.items():
            if model_config.is_available:
                return model_name
        
        return None
    
    async def generate(self, 
                      prompt: str, 
                      model: Optional[str] = None,
                      capability: Optional[ModelCapability] = None,
                      stream: bool = False,
                      **kwargs) -> ModelResponse:
        """Generate response using Ollama model."""
        
        if not self._initialized:
            raise RuntimeError("Ollama manager not initialized")
        
        # Determine model to use
        if model:
            target_model = model
        elif capability:
            target_model = self._get_best_available_model(capability)
        else:
            target_model = self.current_model
        
        if not target_model:
            raise RuntimeError("No suitable model available")
        
        # Prepare request
        request_data = {
            "model": target_model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }        
        try:
            if stream:
                return await self._generate_stream(request_data)
            else:
                return await self._generate_single(request_data)
                
        except Exception as e:
            logger.error(f"Failed to generate response with model {target_model}: {e}")
            raise
    
    async def _generate_single(self, request_data: Dict[str, Any]) -> ModelResponse:
        """Generate single response."""
        async with self.session.post(
            f"{self.base_url}/api/generate",
            json=request_data
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama API error: {response.status}")
            
            data = await response.json()
            
            return ModelResponse(
                content=data.get("response", ""),
                model=data.get("model", request_data["model"]),
                done=data.get("done", True),
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_eval_count=data.get("prompt_eval_count"),
                eval_count=data.get("eval_count"),
                eval_duration=data.get("eval_duration")
            )
    
    async def _generate_stream(self, request_data: Dict[str, Any]) -> AsyncGenerator[ModelResponse, None]:
        """Generate streaming response."""
        async with self.session.post(
            f"{self.base_url}/api/generate",
            json=request_data
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama API error: {response.status}")
            
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode())
                        yield ModelResponse(
                            content=data.get("response", ""),
                            model=data.get("model", request_data["model"]),
                            done=data.get("done", False),
                            total_duration=data.get("total_duration"),
                            load_duration=data.get("load_duration"),
                            prompt_eval_count=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count"),
                            eval_duration=data.get("eval_duration")
                        )
                    except json.JSONDecodeError:
                        continue
    
    async def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze a document for historical content."""
        
        prompt = f"""Analyze the following historical document text:

{text[:2000]}...

Extract:
1. Historical period/dates mentioned
2. Key historical figures
3. Important events or developments
4. Geographic locations
5. Themes and significance

Provide response as JSON with these categories."""
        
        response = await self.generate(
            prompt=prompt,
            capability=ModelCapability.REASONING,
            temperature=0.2
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"entities": response.content}
    
    def get_available_models(self) -> Dict[str, OllamaModel]:
        """Get list of available models."""
        return {name: model for name, model in self.available_models.items() if model.is_available}
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model usage statistics."""
        available_count = len([m for m in self.available_models.values() if m.is_available])
        
        return {
            "total_models": len(self.available_models),
            "available_models": available_count,
            "current_model": self.current_model,
            "model_load_times": self.model_load_times,
            "capabilities_coverage": {
                cap.value: self._get_best_available_model(cap) is not None
                for cap in ModelCapability
            }
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()

# Global Ollama manager instance
ollama_manager = OllamaManager()

# Backward compatibility alias
OllamaIntegration = OllamaManager

# Export all classes
__all__ = [
    "ModelCapability",
    "OllamaModel",
    "ModelResponse",
    "OllamaManager",
    "OllamaIntegration",  # Backward compatibility
    "ollama_manager"
]
