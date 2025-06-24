#!/usr/bin/env python3
"""
Initialize PyGent Factory Services

Properly initialize all core services for testing and production use.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def initialize_database():
    """Initialize database service"""
    try:
        logger.info("Initializing database service...")
        
        # Set environment variables for testing
        os.environ.setdefault("DB_HOST", "localhost")
        os.environ.setdefault("DB_PORT", "5432")
        os.environ.setdefault("DB_NAME", "pygent_factory")
        os.environ.setdefault("DB_USER", "postgres")
        os.environ.setdefault("DB_PASSWORD", "postgres")
        
        from database.production_manager import db_manager, initialize_database
        
        success = await initialize_database()
        if success:
            logger.info("✅ Database service initialized successfully")
            return True
        else:
            logger.warning("⚠️ Database service initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        return False

async def initialize_redis():
    """Initialize Redis service"""
    try:
        logger.info("Initializing Redis service...")
        
        # Set environment variables for testing
        os.environ.setdefault("REDIS_HOST", "localhost")
        os.environ.setdefault("REDIS_PORT", "6379")
        os.environ.setdefault("REDIS_DB", "0")
        
        from cache.redis_manager import redis_manager, initialize_redis
        
        success = await initialize_redis()
        if success:
            logger.info("✅ Redis service initialized successfully")
            return True
        else:
            logger.warning("⚠️ Redis service initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Redis initialization error: {e}")
        return False

async def initialize_communication():
    """Initialize communication system"""
    try:
        logger.info("Initializing communication system...")
        
        from agents.communication_system import communication_system
        
        success = await communication_system.initialize()
        if success:
            logger.info("✅ Communication system initialized successfully")
            return True
        else:
            logger.warning("⚠️ Communication system initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Communication system initialization error: {e}")
        return False

async def initialize_coordination():
    """Initialize coordination system"""
    try:
        logger.info("Initializing coordination system...")
        
        from agents.coordination_system import coordination_system
        
        success = await coordination_system.initialize()
        if success:
            logger.info("✅ Coordination system initialized successfully")
            return True
        else:
            logger.warning("⚠️ Coordination system initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Coordination system initialization error: {e}")
        return False

async def test_services():
    """Test all initialized services"""
    logger.info("\n🧪 Testing initialized services...")
    
    # Test database
    try:
        from database.production_manager import db_manager
        if db_manager.is_initialized:
            health = await db_manager.health_check()
            if health.get('status') == 'healthy':
                logger.info("✅ Database service is healthy")
            else:
                logger.warning(f"⚠️ Database health check: {health}")
        else:
            logger.warning("⚠️ Database not initialized")
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
    
    # Test Redis
    try:
        from cache.redis_manager import redis_manager
        if redis_manager.is_initialized:
            health = await redis_manager.health_check()
            if health.get('status') == 'healthy':
                logger.info("✅ Redis service is healthy")
            else:
                logger.warning(f"⚠️ Redis health check: {health}")
        else:
            logger.warning("⚠️ Redis not initialized")
    except Exception as e:
        logger.error(f"❌ Redis test failed: {e}")
    
    # Test communication
    try:
        from agents.communication_system import communication_system
        if communication_system.is_initialized:
            logger.info("✅ Communication system is running")
        else:
            logger.warning("⚠️ Communication system not initialized")
    except Exception as e:
        logger.error(f"❌ Communication test failed: {e}")
    
    # Test coordination
    try:
        from agents.coordination_system import coordination_system
        if coordination_system.is_initialized:
            logger.info("✅ Coordination system is running")
        else:
            logger.warning("⚠️ Coordination system not initialized")
    except Exception as e:
        logger.error(f"❌ Coordination test failed: {e}")

async def main():
    """Initialize all PyGent Factory services"""
    logger.info("🚀 INITIALIZING PYGENT FACTORY SERVICES")
    logger.info("=" * 50)
    
    services = [
        ("Database", initialize_database),
        ("Redis", initialize_redis),
        ("Communication", initialize_communication),
        ("Coordination", initialize_coordination)
    ]
    
    initialized_count = 0
    for service_name, init_func in services:
        try:
            success = await init_func()
            if success:
                initialized_count += 1
        except Exception as e:
            logger.error(f"❌ {service_name} initialization failed: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 SERVICE INITIALIZATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Services initialized: {initialized_count}/{len(services)}")
    
    if initialized_count == len(services):
        logger.info("🎉 ALL SERVICES INITIALIZED SUCCESSFULLY!")
    else:
        logger.warning(f"⚠️ {len(services) - initialized_count} services failed to initialize")
    
    # Test services
    await test_services()
    
    logger.info("\n🎯 Services are ready for testing and production use!")
    return initialized_count == len(services)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
