#!/usr/bin/env python3
"""
Core System Health Check with Ollama Integration
Focuses on core functionality and validates Ollama integration
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

class CoreSystemHealthChecker:
    """Focused core system health checker"""
    
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
        
    async def test_core_imports(self):
        """Test basic core imports"""
        logger.info("Testing Core Imports...")
        
        try:
            # Test basic imports
            from src.orchestration.task_dispatcher import TaskDispatcher
            from src.orchestration.orchestration_manager import OrchestrationManager
            from src.orchestration.agent_factory import AgentFactory
            
            logger.info("✅ Core imports successful")
            self.passed += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Core imports failed: {e}")
            self.failed += 1
            return False
    
    async def test_ollama_direct(self):
        """Test direct Ollama connection"""
        logger.info("Testing Direct Ollama Connection...")
        
        try:
            from src.ai.providers.ollama_provider import OllamaProvider
            
            provider = OllamaProvider()
            success = await provider.initialize()
            
            if success:
                models = await provider.get_available_models()
                logger.info(f"✅ Ollama connected with {len(models)} models: {models}")
                self.passed += 1
                return True
            else:
                logger.error("❌ Ollama failed to initialize")
                self.failed += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Ollama test failed: {e}")
            self.failed += 1
            return False
    
    async def test_ollama_manager(self):
        """Test Ollama manager"""
        logger.info("Testing Ollama Manager...")
        
        try:
            from src.core.ollama_manager import get_ollama_manager
            
            manager = get_ollama_manager()
            success = await manager.start()
            
            if success:
                models = await manager.get_available_models()
                logger.info(f"✅ Ollama manager working with models: {models}")
                self.passed += 1
                return True
            else:
                logger.error("❌ Ollama manager failed to start")
                self.failed += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Ollama manager test failed: {e}")
            self.failed += 1
            return False
    
    async def test_ollama_generation(self):
        """Test Ollama text generation"""
        logger.info("Testing Ollama Generation...")
        
        try:
            from src.ai.providers.ollama_provider import OllamaProvider
            
            provider = OllamaProvider()
            await provider.initialize()
            
            models = await provider.get_available_models()
            if not models:
                logger.error("❌ No models available for generation test")
                self.failed += 1
                return False
            
            # Use qwen3:8b if available, otherwise first model
            test_model = "qwen3:8b" if "qwen3:8b" in models else models[0]
            
            response = await provider.generate_text(
                prompt="Test: What is 2+2? Answer briefly.",
                model=test_model,
                max_tokens=20
            )
            
            if response and len(response.strip()) > 0:
                logger.info(f"✅ Generation successful: {response[:50]}...")
                self.passed += 1
                return True
            else:
                logger.error("❌ Generation returned empty response")
                self.failed += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Generation test failed: {e}")
            self.failed += 1
            return False
    
    async def test_provider_registry(self):
        """Test provider registry with Ollama"""
        logger.info("Testing Provider Registry...")
        
        try:
            from src.ai.providers.provider_registry import ProviderRegistry
            
            registry = ProviderRegistry()
            await registry.initialize()
            
            if "ollama" in registry.providers:
                ollama_provider = registry.providers["ollama"]
                models = await ollama_provider.get_available_models()
                logger.info(f"✅ Registry has Ollama with models: {models}")
                self.passed += 1
                return True
            else:
                logger.error("❌ Ollama not found in provider registry")
                self.failed += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Provider registry test failed: {e}")
            self.failed += 1
            return False
    
    async def run_all_tests(self):
        """Run all core system tests"""
        logger.info("=" * 60)
        logger.info("🧪 CORE SYSTEM HEALTH CHECK WITH OLLAMA")
        logger.info("=" * 60)
        
        tests = [
            self.test_core_imports(),
            self.test_ollama_direct(),
            self.test_ollama_manager(),
            self.test_ollama_generation(),
            self.test_provider_registry()
        ]
        
        await asyncio.gather(*tests, return_exceptions=True)
        
        logger.info("=" * 60)
        logger.info("📊 TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ PASSED: {self.passed}")
        logger.info(f"❌ FAILED: {self.failed}")
        logger.info(f"📈 SUCCESS RATE: {self.passed}/{self.passed + self.failed} ({(self.passed/(self.passed + self.failed)*100):.1f}%)")
        
        if self.failed == 0:
            logger.info("🎉 ALL TESTS PASSED!")
            return True
        else:
            logger.error(f"❌ {self.failed} TESTS FAILED")
            return False

async def main():
    """Main function"""
    checker = CoreSystemHealthChecker()
    success = await checker.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
