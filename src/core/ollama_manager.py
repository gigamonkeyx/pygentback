"""
Ollama Service Manager

Manages Ollama service lifecycle, health checks, and model availability.
Ensures Ollama is ready before other components try to use it.
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiohttp
import json

try:
    from config.settings import get_settings
except ImportError:
    # Fallback for when config module is not available
    def get_settings():
        class Settings:
            OLLAMA_BASE_URL = "http://localhost:11434"
            OLLAMA_TIMEOUT = 30
        return Settings()
    get_settings = get_settings

logger = logging.getLogger(__name__)


class OllamaManager:
    """
    Manages Ollama service lifecycle and health monitoring.
    
    Responsibilities:
    - Start Ollama service if not running
    - Health checks and readiness verification
    - Model availability verification
    - Service monitoring and restart if needed
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.ollama_url = self.settings.ai.OLLAMA_BASE_URL
        self.ollama_executable = self._find_ollama_executable()
        self.process = None
        self.is_ready = False
        self.available_models = []
        self.health_check_interval = 30  # seconds
        self.startup_timeout = 60  # seconds
        self.max_startup_retries = 3
        
    def _find_ollama_executable(self) -> Optional[Path]:
        """Find Ollama executable in common locations"""
        possible_paths = [
            Path("D:/ollama/bin/ollama.exe"),  # User's specific path
            Path("C:/Users") / Path.home().name / "AppData/Local/Programs/Ollama/ollama.exe",
            Path("C:/Program Files/Ollama/ollama.exe"),
            Path("C:/Program Files (x86)/Ollama/ollama.exe"),
        ]
        
        # Also check PATH
        try:
            result = subprocess.run(["where", "ollama"], capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                possible_paths.insert(0, Path(result.stdout.strip().split('\n')[0]))
        except Exception:
            pass
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found Ollama executable at: {path}")
                return path
        
        logger.warning("Ollama executable not found in common locations")
        return None
    
    async def start(self) -> bool:
        """
        Start Ollama service and wait for it to be ready.
        
        Returns:
            bool: True if Ollama is ready, False otherwise
        """
        logger.info("Starting Ollama service...")
        
        # Check if already running
        if await self._check_health():
            logger.info("Ollama is already running and healthy")
            self.is_ready = True
            await self._load_available_models()
            return True
        
        # Try to start Ollama
        for attempt in range(self.max_startup_retries):
            logger.info(f"Ollama startup attempt {attempt + 1}/{self.max_startup_retries}")
            
            if await self._start_ollama_process():
                # Wait for service to be ready
                if await self._wait_for_ready():
                    logger.info("Ollama service started successfully")
                    self.is_ready = True
                    await self._load_available_models()
                    return True
            
            logger.warning(f"Ollama startup attempt {attempt + 1} failed")
            await asyncio.sleep(5)  # Wait before retry
        
        logger.error("Failed to start Ollama service after all attempts")
        return False
    
    async def _start_ollama_process(self) -> bool:
        """Start the Ollama process"""
        if not self.ollama_executable:
            logger.error("Cannot start Ollama: executable not found")
            return False
        
        try:
            # Start Ollama serve in background
            self.process = subprocess.Popen(
                [str(self.ollama_executable), "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
            )
            
            logger.info(f"Started Ollama process with PID: {self.process.pid}")
            
            # Give it a moment to start
            await asyncio.sleep(3)
            
            # Check if process is still running
            if self.process.poll() is None:
                return True
            else:
                logger.error("Ollama process exited immediately")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start Ollama process: {e}")
            return False
    
    async def _wait_for_ready(self) -> bool:
        """Wait for Ollama to be ready to accept requests"""
        start_time = time.time()
        
        while time.time() - start_time < self.startup_timeout:
            if await self._check_health():
                return True
            
            await asyncio.sleep(2)
        
        return False
    
    async def _check_health(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _load_available_models(self) -> None:
        """Load list of available models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.available_models = [model["name"] for model in data.get("models", [])]
                        logger.info(f"Available Ollama models: {self.available_models}")
                    else:
                        logger.warning("Failed to load available models")
        except Exception as e:
            logger.error(f"Error loading available models: {e}")
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        if not self.available_models:
            await self._load_available_models()
        return self.available_models
    
    async def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        available = await self.get_available_models()
        return model_name in available
    
    async def ensure_model_available(self, model_name: str) -> bool:
        """
        Ensure a model is available, pull it if necessary.
        
        Args:
            model_name: Name of the model to ensure is available
            
        Returns:
            bool: True if model is available, False otherwise
        """
        if await self.is_model_available(model_name):
            return True
        
        logger.info(f"Model {model_name} not available, attempting to pull...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/pull",
                    json={"name": model_name}
                ) as response:
                    if response.status == 200:
                        # Wait for pull to complete
                        async for line in response.content:
                            if line:
                                data = json.loads(line)
                                if data.get("status") == "success":
                                    logger.info(f"Successfully pulled model: {model_name}")
                                    await self._load_available_models()
                                    return True
                    else:
                        logger.error(f"Failed to pull model {model_name}: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop Ollama service"""
        logger.info("Stopping Ollama service...")
        
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(2)
                
                if self.process.poll() is None:
                    self.process.kill()
                    
                logger.info("Ollama process stopped")
            except Exception as e:
                logger.error(f"Error stopping Ollama process: {e}")
        
        self.is_ready = False
        self.available_models = []
    
    async def restart(self) -> bool:
        """Restart Ollama service"""
        logger.info("Restarting Ollama service...")
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()
    
    async def generate_response(self, prompt: str, model: str = "llama3.2:latest",
                              max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        """
        Generate response using Ollama model.

        Args:
            prompt: Input prompt for generation
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature

        Returns:
            Generated response text
        """
        if not self.is_ready:
            if not await self.start():
                raise RuntimeError("Ollama service is not available")

        # Ensure model is available
        if not await self.ensure_model_available(model):
            raise RuntimeError(f"Model {model} is not available")

        try:
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                }

                if max_tokens:
                    request_data["options"]["num_predict"] = max_tokens

                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama generation failed: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get current status of Ollama service"""
        return {
            "is_ready": self.is_ready,
            "url": self.ollama_url,
            "available_models": self.available_models,
            "process_running": self.process is not None and self.process.poll() is None,
            "executable_path": str(self.ollama_executable) if self.ollama_executable else None
        }


# Global instance
_ollama_manager = None


def get_ollama_manager() -> OllamaManager:
    """Get global Ollama manager instance"""
    global _ollama_manager
    if _ollama_manager is None:
        _ollama_manager = OllamaManager()
    return _ollama_manager
