#!/usr/bin/env python3
"""
Ollama Integration Test
Tests Ollama connectivity and generation capability
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, Path(__file__).parent / "src")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_ollama_connection():
    """Test basic Ollama connection"""
    try:
        from src.ai.providers.ollama_provider import OllamaProvider
        
        provider = OllamaProvider()
        success = await provider.initialize()
        
        if success:
            logger.info("✅ Ollama connection successful")
            models = await provider.get_available_models()
            logger.info(f"✅ Available models: {models}")
            return True, models
        else:
            logger.error("❌ Ollama connection failed")
            return False, []
            
    except Exception as e:
        logger.error(f"❌ Ollama connection error: {e}")
        return False, []


async def test_ollama_generation():
    """Test Ollama text generation"""
    try:
        from src.ai.providers.ollama_provider import OllamaProvider
        
        provider = OllamaProvider()
        await provider.initialize()
        
        # Get available models
        models = await provider.get_available_models()
        if not models:
            logger.error("❌ No models available for generation test")
            return False
        
        # Use qwen3:8b if available, otherwise first model
        test_model = "qwen3:8b" if "qwen3:8b" in models else models[0]
        logger.info(f"🧪 Testing generation with model: {test_model}")
          # Test generation
        prompt = "What is 2 + 2? Answer in one sentence."
        response = await provider.generate_text(
            prompt=prompt,
            model=test_model,
            max_tokens=50
        )
        
        if response and len(response.strip()) > 0:
            logger.info(f"✅ Generation successful: {response[:100]}...")
            return True
        else:
            logger.error("❌ Generation returned empty response")
            return False
            
    except Exception as e:
        logger.error(f"❌ Generation test error: {e}")
        return False


async def test_ollama_manager():
    """Test Ollama manager functionality"""
    try:
        from src.core.ollama_manager import get_ollama_manager
        
        manager = get_ollama_manager()
        success = await manager.start()
        
        if success:
            logger.info("✅ Ollama manager started successfully")
            models = await manager.get_available_models()
            logger.info(f"✅ Manager found models: {models}")
            return True
        else:
            logger.error("❌ Ollama manager failed to start")
            return False
            
    except Exception as e:
        logger.error(f"❌ Manager test error: {e}")
        return False


async def test_provider_registry():
    """Test Ollama in provider registry"""
    try:
        from src.ai.providers.provider_registry import ProviderRegistry
        
        registry = ProviderRegistry()
        await registry.initialize()
        
        if "ollama" in registry.providers:
            logger.info("✅ Ollama found in provider registry")
            ollama_provider = registry.providers["ollama"]
            
            # Test getting models through registry
            models = await ollama_provider.get_available_models()
            logger.info(f"✅ Registry models: {models}")
            return True
        else:
            logger.error("❌ Ollama not found in provider registry")
            return False
            
    except Exception as e:
        logger.error(f"❌ Provider registry test error: {e}")
        return False


async def run_all_tests():
    """Run comprehensive Ollama tests"""
    
    logger.info("=" * 60)
    logger.info("🧪 OLLAMA INTEGRATION TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Ollama Manager", test_ollama_manager),
        ("Provider Registry", test_provider_registry),
        ("Text Generation", test_ollama_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Testing {test_name}...")
        
        try:
            if test_name == "Ollama Connection":
                success, models = await test_func()
                results.append((test_name, success))
                if success:
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            else:
                success = await test_func()
                results.append((test_name, success))
                if success:
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
                    
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 OLLAMA TEST RESULTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{status:>6} | {test_name}")
    
    logger.info(f"\nTOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL OLLAMA TESTS PASSED!")
        return True
    else:
        logger.error(f"❌ {total - passed} Ollama tests failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
