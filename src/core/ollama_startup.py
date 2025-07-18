#!/usr/bin/env python3
"""
Ollama Startup Sequence - Observer-mandated integration
RIPER-Ω Protocol Compliance: Ollama MUST be validated before any agent operations
"""

import asyncio
import aiohttp
import subprocess
import time
import logging
import os
import sys
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OllamaStartupManager:
    """Observer-mandated Ollama startup and validation manager"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.startup_timeout = 120  # Extended timeout for model loading
        self.validation_timeout = 45  # Extended validation timeout
        self.model_ready_timeout = 30  # Time to wait for models to be ready
        self.is_running = False
        self.models_available = []
        self.startup_required = True  # Toggle for startup requirement
        
    async def check_ollama_running(self) -> bool:
        """Check if Ollama is already running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.models_available = [model.get('name', 'unknown') for model in data.get('models', [])]
                        self.is_running = True
                        logger.info(f"*** Ollama already running with {len(self.models_available)} models")
                        return True
                    else:
                        logger.warning(f"*** Ollama responded with status {response.status}")
                        return False
        except Exception as e:
            logger.warning(f"*** Ollama not running: {e}")
            return False
    
    def start_ollama_service(self) -> bool:
        """Start Ollama service if not running"""
        try:
            logger.info("*** Starting Ollama service...")
            
            # Try to start Ollama serve
            if sys.platform == "win32":
                # Windows
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # Unix/Linux/Mac
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # OBSERVER CRITICAL: Give Ollama proper time to initialize
            logger.info("*** Waiting for Ollama service to initialize...")
            time.sleep(10)  # Extended initial wait

            # Check if process is still running
            if process.poll() is None:
                logger.info("*** Ollama service started successfully")
                # Additional wait for model loading
                logger.info("*** Waiting for models to load...")
                time.sleep(15)  # Wait for models to be ready
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"*** Ollama failed to start: {stderr.decode()}")
                return False
                
        except FileNotFoundError:
            logger.error("*** Ollama not found in PATH. Please install Ollama first.")
            return False
        except Exception as e:
            logger.error(f"*** Failed to start Ollama: {e}")
            return False
    
    async def wait_for_ollama_ready(self) -> bool:
        """Wait for Ollama to be ready and responsive with proper model loading"""
        logger.info("*** OBSERVER CRITICAL: Waiting for Ollama to be FULLY ready...")

        start_time = time.time()
        consecutive_successes = 0
        required_successes = 3  # Need 3 consecutive successful checks

        while time.time() - start_time < self.startup_timeout:
            if await self.check_ollama_running():
                consecutive_successes += 1
                logger.info(f"*** Ollama check {consecutive_successes}/{required_successes} successful")

                if consecutive_successes >= required_successes:
                    # Additional validation: test actual model endpoint
                    if await self.validate_model_endpoint():
                        logger.info("*** OBSERVER SUCCESS: Ollama is FULLY ready and responsive")
                        return True
                    else:
                        logger.warning("*** Model endpoint not ready, continuing wait...")
                        consecutive_successes = 0

                await asyncio.sleep(3)  # Longer wait between checks
            else:
                consecutive_successes = 0
                logger.info("*** Ollama not ready yet, waiting...")
                await asyncio.sleep(5)  # Even longer wait on failure

        logger.error(f"*** OBSERVER CRITICAL: Ollama failed to become ready within {self.startup_timeout} seconds")
        return False

    async def validate_model_endpoint(self) -> bool:
        """OBSERVER CRITICAL: Validate that model endpoint is actually functional"""
        if not self.models_available:
            return False

        try:
            # Test the generate endpoint with the first available model
            primary_model = self.models_available[0]
            logger.info(f"*** Testing model endpoint with {primary_model}")

            async with aiohttp.ClientSession() as session:
                test_payload = {
                    "model": primary_model,
                    "prompt": "test",
                    "stream": False,
                    "options": {"num_predict": 1}  # Minimal generation
                }

                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=test_payload,
                    timeout=15
                ) as response:
                    if response.status == 200:
                        logger.info(f"*** Model endpoint test SUCCESS with {primary_model}")
                        return True
                    else:
                        logger.warning(f"*** Model endpoint test failed: {response.status}")
                        return False

        except Exception as e:
            logger.warning(f"*** Model endpoint test error: {e}")
            return False

    async def validate_ollama_models(self) -> bool:
        """Validate that required models are available"""
        if not self.models_available:
            logger.warning("⚠️ No models available in Ollama")
            return False
        
        logger.info(f"*** Ollama models available: {', '.join(self.models_available)}")

        # Check for at least one model
        if len(self.models_available) > 0:
            logger.info(f"*** Ollama validation successful with {len(self.models_available)} models")
            return True
        else:
            logger.error("*** No models found in Ollama")
            return False
    
    async def ensure_ollama_ready(self) -> Dict[str, Any]:
        """Observer-mandated: Ensure Ollama is ready before any operations"""
        logger.info("*** OBSERVER-MANDATED OLLAMA VALIDATION STARTING")
        
        result = {
            'success': False,
            'running': False,
            'models_count': 0,
            'models': [],
            'error': None
        }
        
        try:
            # Step 1: Check if already running
            if await self.check_ollama_running():
                result['running'] = True
                result['models'] = self.models_available
                result['models_count'] = len(self.models_available)
                
                # Validate models
                if await self.validate_ollama_models():
                    result['success'] = True
                    logger.info("*** OBSERVER VALIDATION: Ollama ready and validated")
                    return result
                else:
                    result['error'] = "No models available"
                    logger.error("*** OBSERVER VALIDATION FAILED: No models available")
                    return result
            
            # Step 2: Try to start Ollama
            logger.info("*** OBSERVER DIRECTIVE: Starting Ollama service")
            if not self.start_ollama_service():
                result['error'] = "Failed to start Ollama service"
                logger.error("*** OBSERVER VALIDATION FAILED: Could not start Ollama")
                return result
            
            # Step 3: Wait for ready
            if not await self.wait_for_ollama_ready():
                result['error'] = "Ollama failed to become ready"
                logger.error("*** OBSERVER VALIDATION FAILED: Ollama not ready")
                return result
            
            # Step 4: Final validation
            result['running'] = True
            result['models'] = self.models_available
            result['models_count'] = len(self.models_available)
            
            if await self.validate_ollama_models():
                result['success'] = True
                logger.info("*** OBSERVER VALIDATION COMPLETE: Ollama fully operational")
            else:
                result['error'] = "Model validation failed"
                logger.error("*** OBSERVER VALIDATION FAILED: Model validation failed")
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"*** OBSERVER VALIDATION CRITICAL FAILURE: {e}")
            return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current Ollama status"""
        return {
            'running': self.is_running,
            'models_count': len(self.models_available),
            'models': self.models_available,
            'url': self.ollama_url
        }

# Global instance for startup sequence
ollama_startup_manager = OllamaStartupManager()

async def ensure_ollama_startup() -> Dict[str, Any]:
    """Observer-mandated startup function - MUST be called before any agent operations"""
    return await ollama_startup_manager.ensure_ollama_ready()

def get_ollama_status() -> Dict[str, Any]:
    """Get current Ollama status"""
    return ollama_startup_manager.get_status()

# Observer-mandated validation decorator
def require_ollama(func):
    """Decorator to ensure Ollama is running before function execution"""
    async def wrapper(*args, **kwargs):
        status = await ensure_ollama_startup()
        if not status['success']:
            raise RuntimeError(f"OBSERVER VALIDATION FAILED: Ollama not ready - {status.get('error', 'Unknown error')}")
        return await func(*args, **kwargs)
    return wrapper
