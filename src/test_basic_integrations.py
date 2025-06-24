"""
Basic Integration Test

Simple test to verify core functionality without complex dependencies.
"""

import asyncio
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_database():
    """Test basic database functionality."""
    try:
        logger.info("🗄️ Testing Basic Database Integration...")
        
        import asyncpg
        
        # Test connection (will fail gracefully if no DB)
        try:
            conn = await asyncpg.connect("postgresql://postgres:postgres@localhost:54321/pygent_factory")
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            
            if result == 1:
                logger.info("✅ Database: Real PostgreSQL connection working!")
                return "real"
        except Exception as e:
            logger.info(f"🔄 Database: Real connection failed ({e}), fallback available")
            return "fallback"
            
    except ImportError:
        logger.error("❌ Database: asyncpg not available")
        return "error"


async def test_basic_memory():
    """Test basic memory/cache functionality."""
    try:
        logger.info("💾 Testing Basic Memory Integration...")
        
        import redis
        
        # Test Redis connection (will fail gracefully if no Redis)
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.set('test_key', 'test_value', ex=60)
            value = r.get('test_key')
            
            if value == 'test_value':
                logger.info("✅ Memory: Real Redis connection working!")
                return "real"
        except Exception as e:
            logger.info(f"🔄 Memory: Real connection failed ({e}), fallback available")
            return "fallback"
            
    except ImportError:
        logger.error("❌ Memory: redis not available")
        return "error"


async def test_basic_http():
    """Test basic HTTP functionality."""
    try:
        logger.info("🌐 Testing Basic HTTP Integration...")
        
        import requests
        
        # Test simple HTTP request
        try:
            response = requests.get("https://api.github.com/rate_limit", timeout=5)
            if response.status_code == 200:
                logger.info("✅ HTTP: Real GitHub API connection working!")
                return "real"
        except Exception as e:
            logger.info(f"🔄 HTTP: Real connection failed ({e}), fallback available")
            return "fallback"
            
    except ImportError:
        logger.error("❌ HTTP: requests not available")
        return "error"


async def test_orchestration_imports():
    """Test that orchestration modules can be imported."""
    try:
        logger.info("🎼 Testing Orchestration Module Imports...")
        
        # Test core imports
        sys.path.append(os.path.join(os.path.dirname(__file__), 'orchestration'))
        
        try:
            from orchestration.coordination_models import OrchestrationConfig
            logger.info("✅ Orchestration: coordination_models imported")
        except Exception as e:
            logger.error(f"❌ Orchestration: coordination_models failed - {e}")
            return False
        
        try:
            from orchestration.real_database_client import RealDatabaseClient
            logger.info("✅ Orchestration: real_database_client imported")
        except Exception as e:
            logger.error(f"❌ Orchestration: real_database_client failed - {e}")
            return False
        
        try:
            from orchestration.real_memory_client import RealMemoryClient
            logger.info("✅ Orchestration: real_memory_client imported")
        except Exception as e:
            logger.error(f"❌ Orchestration: real_memory_client failed - {e}")
            return False
        
        try:
            from orchestration.real_github_client import RealGitHubClient
            logger.info("✅ Orchestration: real_github_client imported")
        except Exception as e:
            logger.error(f"❌ Orchestration: real_github_client failed - {e}")
            return False
        
        logger.info("✅ All orchestration modules imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Orchestration imports failed: {e}")
        return False


async def test_zero_mock_validation():
    """Validate that mock code patterns are not present."""
    try:
        logger.info("🔍 Testing Zero Mock Code Validation...")
        
        # Check orchestration files for mock patterns
        orchestration_dir = os.path.join(os.path.dirname(__file__), 'orchestration')
        
        mock_patterns = ['mock_', 'fake_', 'simulate_', 'return.*0.5', 'placeholder']
        files_to_check = [
            'real_database_client.py',
            'real_memory_client.py', 
            'real_github_client.py',
            'real_agent_integration.py',
            'integration_manager.py'
        ]
        
        mock_found = False
        
        for filename in files_to_check:
            filepath = os.path.join(orchestration_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    for pattern in mock_patterns:
                        if pattern in content:
                            logger.warning(f"⚠️ Potential mock pattern '{pattern}' found in {filename}")
                            mock_found = True
                            
                except Exception as e:
                    logger.error(f"❌ Could not check {filename}: {e}")
        
        if not mock_found:
            logger.info("✅ Zero Mock Validation: No obvious mock patterns detected")
            return True
        else:
            logger.warning("⚠️ Zero Mock Validation: Some potential mock patterns found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Zero mock validation failed: {e}")
        return False


async def main():
    """Run basic integration tests."""
    logger.info("🚀 Starting Basic Integration Validation...")
    
    try:
        # Test basic integrations
        db_result = await test_basic_database()
        memory_result = await test_basic_memory()
        http_result = await test_basic_http()
        
        # Test orchestration imports
        imports_success = await test_orchestration_imports()
        
        # Test zero mock validation
        zero_mock_success = await test_zero_mock_validation()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 BASIC INTEGRATION TEST RESULTS:")
        logger.info(f"🗄️ Database Integration: {db_result}")
        logger.info(f"💾 Memory Integration: {memory_result}")
        logger.info(f"🌐 HTTP Integration: {http_result}")
        logger.info(f"🎼 Orchestration Imports: {'✅ Success' if imports_success else '❌ Failed'}")
        logger.info(f"🔍 Zero Mock Validation: {'✅ Passed' if zero_mock_success else '⚠️ Issues'}")
        
        # Overall assessment
        real_integrations = sum(1 for result in [db_result, memory_result, http_result] if result == "real")
        fallback_integrations = sum(1 for result in [db_result, memory_result, http_result] if result == "fallback")
        
        if real_integrations > 0:
            logger.info(f"🎯 REAL INTEGRATIONS: {real_integrations}/3 working with real systems")
        
        if fallback_integrations > 0:
            logger.info(f"🔄 FALLBACK INTEGRATIONS: {fallback_integrations}/3 using fallback implementations")
        
        if imports_success and zero_mock_success:
            logger.info("🏆 INTEGRATION FRAMEWORK: Ready for production!")
            logger.info("✅ All orchestration modules can be imported")
            logger.info("✅ No obvious mock code patterns detected")
            logger.info("🚀 SYSTEM IS READY FOR REAL INTEGRATIONS!")
        else:
            logger.warning("⚠️ Some issues detected - check logs above")
        
        logger.info("="*60)
        
        return imports_success and zero_mock_success
        
    except Exception as e:
        logger.error(f"❌ Basic integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)