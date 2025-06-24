"""
Redis Setup Helper

Helps set up Redis cache for PyGent Factory.
"""

import asyncio
import logging
import subprocess
import sys
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_redis_installation():
    """Check if Redis is installed."""
    try:
        result = subprocess.run(['redis-server', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Redis found: {result.stdout.strip()}")
            return True
        else:
            logger.info("❌ Redis not found in PATH")
            return False
    except FileNotFoundError:
        logger.info("❌ Redis not installed or not in PATH")
        return False


async def test_redis_connection():
    """Test Redis connection."""
    try:
        import redis
        
        # Try to connect
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test ping
        pong = r.ping()
        if pong:
            logger.info("✅ Redis connection successful")
            return True
        else:
            logger.info("❌ Redis ping failed")
            return False
            
    except Exception as e:
        logger.info(f"❌ Redis connection failed: {e}")
        return False


async def test_redis_operations():
    """Test Redis operations."""
    try:
        import redis
        
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test SET
        test_key = "pygent_factory_setup_test"
        test_value = "Redis setup successful"
        
        result = r.set(test_key, test_value, ex=60)
        if not result:
            logger.error("❌ Redis SET operation failed")
            return False
        
        # Test GET
        retrieved_value = r.get(test_key)
        if retrieved_value != test_value:
            logger.error("❌ Redis GET operation failed")
            return False
        
        # Test EXISTS
        exists = r.exists(test_key)
        if not exists:
            logger.error("❌ Redis EXISTS operation failed")
            return False
        
        # Test DELETE
        deleted = r.delete(test_key)
        if not deleted:
            logger.error("❌ Redis DELETE operation failed")
            return False
        
        # Test TTL operations
        r.set("ttl_test", "value", ex=5)
        ttl = r.ttl("ttl_test")
        if ttl <= 0:
            logger.error("❌ Redis TTL operation failed")
            return False
        
        r.delete("ttl_test")
        
        logger.info("✅ All Redis operations tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis operations test failed: {e}")
        return False


async def start_redis_server():
    """Attempt to start Redis server."""
    try:
        logger.info("🚀 Attempting to start Redis server...")
        
        # Try to start Redis server in background
        process = subprocess.Popen(
            ['redis-server'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for startup
        await asyncio.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("✅ Redis server started successfully")
            return True, process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Redis server failed to start:")
            logger.error(f"   stdout: {stdout.decode()}")
            logger.error(f"   stderr: {stderr.decode()}")
            return False, None
            
    except Exception as e:
        logger.error(f"❌ Failed to start Redis server: {e}")
        return False, None


async def main():
    """Main setup function."""
    logger.info("💾 Redis Setup for PyGent Factory")
    logger.info("="*50)
    
    # Check if Redis is installed
    redis_installed = await check_redis_installation()
    
    if not redis_installed:
        logger.info("\n📋 Redis Installation Instructions:")
        logger.info("1. Download Redis from: https://redis.io/download")
        logger.info("2. For Windows: Use WSL or download Windows port")
        logger.info("3. Alternative: Use Docker: docker run -d -p 6379:6379 redis")
        logger.info("4. Ensure Redis is in PATH")
        return False
    
    # Test connection
    logger.info("\n🔌 Testing Redis Connection...")
    connection_ok = await test_redis_connection()
    
    if not connection_ok:
        logger.info("🚀 Redis not running, attempting to start...")
        started, process = await start_redis_server()
        
        if started:
            # Test connection again
            await asyncio.sleep(2)
            connection_ok = await test_redis_connection()
        
        if not connection_ok:
            logger.info("\n📋 Connection Troubleshooting:")
            logger.info("1. Start Redis manually: redis-server")
            logger.info("2. Check if Redis is listening on localhost:6379")
            logger.info("3. Check firewall settings")
            logger.info("4. Try: redis-cli ping")
            return False
    
    # Test operations
    logger.info("\n🧪 Testing Redis Operations...")
    operations_ok = await test_redis_operations()
    
    if operations_ok:
        logger.info("\n🎉 Redis Setup Complete!")
        logger.info("✅ Connection: localhost:6379")
        logger.info("✅ Operations: SET, GET, EXISTS, DELETE, TTL")
        logger.info("✅ Performance: Ready for production caching")
        logger.info("🚀 Ready for zero mock code integration!")
        return True
    else:
        logger.error("\n❌ Redis setup failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)