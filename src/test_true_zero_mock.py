"""
True Zero Mock Test

Tests that the system operates with ZERO mock code.
If any integration fails, the test fails. No fallbacks allowed.
"""

import asyncio
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_strict_integration_manager():
    """Test strict integration manager with zero fallbacks."""
    try:
        logger.info("🎯 Testing STRICT Integration Manager (Zero Mock Code)...")
        
        sys.path.append(os.path.join(os.path.dirname(__file__), 'orchestration'))
        
        from orchestration.strict_integration_manager import get_strict_integration_manager, shutdown_strict_integration_manager
        
        # This will FAIL if any integration is not real
        try:
            manager = await get_strict_integration_manager()
            logger.info("✅ Strict Integration Manager: All real integrations connected!")
            
            # Get status
            status = await manager.get_integration_status()
            
            if status["zero_mock_code"]:
                logger.info("🎉 ZERO MOCK CODE CONFIRMED!")
                logger.info(f"✅ All {status['connected_systems']}/{status['total_systems']} integrations are REAL")
            else:
                logger.error("❌ ZERO MOCK CODE FAILED!")
                return False
            
            # Test each integration with real operations
            await test_real_database_operations(manager)
            await test_real_memory_operations(manager)
            await test_real_github_operations(manager)
            await test_real_agent_operations(manager)
            
            await shutdown_strict_integration_manager()
            return True
            
        except ConnectionError as e:
            logger.error(f"💥 STRICT INTEGRATION FAILED: {e}")
            logger.error("🚫 This is EXPECTED if real services are not available")
            logger.error("🎯 Zero mock code means: REAL INTEGRATIONS OR FAILURE")
            return False
            
    except Exception as e:
        logger.error(f"❌ Strict integration test failed: {e}")
        return False


async def test_real_database_operations(manager):
    """Test real database operations - no mocks allowed."""
    try:
        logger.info("🗄️ Testing REAL Database Operations...")
        
        # Real database query
        request = {
            "operation": "query",
            "sql": "SELECT NOW() as current_time, 'real_test' as test_type",
            "params": []
        }
        
        result = await manager.execute_database_request(request)
        
        # Verify it's real
        if result["integration_type"] != "real":
            raise ValueError(f"Expected real integration, got: {result['integration_type']}")
        
        if result["status"] != "success":
            raise ValueError(f"Real database operation failed: {result}")
        
        # Verify real data
        if not result.get("rows"):
            raise ValueError("Real database should return actual rows")
        
        logger.info("✅ Real Database: Actual PostgreSQL operations confirmed")
        
    except Exception as e:
        logger.error(f"❌ Real database test failed: {e}")
        raise


async def test_real_memory_operations(manager):
    """Test real memory operations - no mocks allowed."""
    try:
        logger.info("💾 Testing REAL Memory Operations...")
        
        # Real memory store
        store_request = {
            "operation": "store",
            "key": "zero_mock_test_key",
            "value": "real_redis_value",
            "ttl": 300
        }
        
        result = await manager.execute_memory_request(store_request)
        
        # Verify it's real
        if result["integration_type"] != "real":
            raise ValueError(f"Expected real integration, got: {result['integration_type']}")
        
        if result["status"] != "success":
            raise ValueError(f"Real memory operation failed: {result}")
        
        # Real memory retrieve
        retrieve_request = {
            "operation": "retrieve",
            "key": "zero_mock_test_key"
        }
        
        result = await manager.execute_memory_request(retrieve_request)
        
        if result["value"] != "real_redis_value":
            raise ValueError("Real Redis should return the actual stored value")
        
        logger.info("✅ Real Memory: Actual Redis operations confirmed")
        
    except Exception as e:
        logger.error(f"❌ Real memory test failed: {e}")
        raise


async def test_real_github_operations(manager):
    """Test real GitHub operations - no mocks allowed."""
    try:
        logger.info("🐙 Testing REAL GitHub Operations...")
        
        # Real GitHub API call
        request = {
            "operation": "search_repositories",
            "query": "python",
            "sort": "stars"
        }
        
        result = await manager.execute_github_request(request)
        
        # Verify it's real
        if result["integration_type"] != "real":
            raise ValueError(f"Expected real integration, got: {result['integration_type']}")
        
        if result["status"] != "success":
            raise ValueError(f"Real GitHub operation failed: {result}")
        
        # Verify real data
        if not result.get("repositories"):
            raise ValueError("Real GitHub API should return actual repositories")
        
        # Check for real repository data
        first_repo = result["repositories"][0]
        if not first_repo.get("full_name") or not first_repo.get("url"):
            raise ValueError("Real GitHub API should return actual repository data")
        
        logger.info("✅ Real GitHub: Actual GitHub API operations confirmed")
        
    except Exception as e:
        logger.error(f"❌ Real GitHub test failed: {e}")
        raise


async def test_real_agent_operations(manager):
    """Test real agent operations - no mocks allowed."""
    try:
        logger.info("🤖 Testing REAL Agent Operations...")
        
        # Create real agent executor
        agent_executor = await manager.create_real_agent_executor("zero_mock_test_agent", "tot_reasoning")
        
        # Real agent task
        task_data = {
            "task_type": "reasoning",
            "input_data": {
                "problem": "Zero mock code validation test problem"
            }
        }
        
        result = await agent_executor.execute_task(task_data)
        
        # Verify it's real
        if not result.get("is_real"):
            raise ValueError("Expected real agent execution")
        
        if result["status"] != "success":
            raise ValueError(f"Real agent operation failed: {result}")
        
        # Verify real agent response
        if "fallback" in str(result).lower():
            raise ValueError("Real agent should not use fallback implementations")
        
        logger.info("✅ Real Agents: Actual PyGent Factory agent operations confirmed")
        
    except Exception as e:
        logger.error(f"❌ Real agent test failed: {e}")
        raise


async def main():
    """Run true zero mock test."""
    logger.info("🎯 STARTING TRUE ZERO MOCK CODE TEST...")
    logger.info("🚫 NO FALLBACKS - REAL INTEGRATIONS OR FAILURE")
    
    try:
        # Test strict integration manager
        success = await test_strict_integration_manager()
        
        if success:
            logger.info("\n" + "="*70)
            logger.info("🏆 TRUE ZERO MOCK CODE TEST: SUCCESS! 🏆")
            logger.info("✅ All integrations are REAL - no mock code detected")
            logger.info("✅ Real PostgreSQL database operations")
            logger.info("✅ Real Redis memory operations")
            logger.info("✅ Real GitHub API operations")
            logger.info("✅ Real PyGent Factory agent operations")
            logger.info("🎯 ZERO MOCK CODE ACHIEVEMENT UNLOCKED!")
            logger.info("🚀 PRODUCTION READY WITH 100% REAL INTEGRATIONS!")
            logger.info("="*70)
            return True
        else:
            logger.error("\n" + "="*70)
            logger.error("❌ TRUE ZERO MOCK CODE TEST: FAILED")
            logger.error("🔧 Real infrastructure required:")
            logger.error("   - PostgreSQL database running")
            logger.error("   - Redis cache running")
            logger.error("   - GitHub API token configured")
            logger.error("   - PyGent Factory agents running")
            logger.error("🚫 NO FALLBACKS AVAILABLE - FIX REAL SERVICES")
            logger.error("="*70)
            return False
            
    except Exception as e:
        logger.error(f"💥 True zero mock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)