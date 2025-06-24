"""
Dependency Verification Script

Verifies that all required dependencies for real integrations are installed and working.
"""

import sys
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_dependency(module_name, description=""):
    """Verify a dependency is installed and importable."""
    try:
        importlib.import_module(module_name)
        logger.info(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        logger.error(f"❌ {module_name} - {description} - ERROR: {e}")
        return False


def main():
    """Verify all dependencies."""
    logger.info("🔍 Verifying Real Integration Dependencies...")
    
    dependencies = [
        # Database
        ("asyncpg", "PostgreSQL async driver"),
        ("psycopg2", "PostgreSQL sync driver"),
        
        # Memory/Cache
        ("aioredis", "Redis async client"),
        ("redis", "Redis sync client"),
        
        # HTTP/API
        ("aiohttp", "Async HTTP client"),
        ("httpx", "Alternative HTTP client"),
        ("requests", "Sync HTTP client"),
        
        # Core Python async
        ("asyncio", "Async I/O support"),
        ("json", "JSON processing"),
        ("datetime", "Date/time handling"),
        ("typing", "Type hints"),
        
        # Configuration
        ("dotenv", "Environment variables"),
        ("pydantic", "Data validation"),
        
        # Testing
        ("pytest", "Testing framework"),
        
        # Standard library essentials
        ("os", "Operating system interface"),
        ("sys", "System-specific parameters"),
        ("logging", "Logging facility"),
        ("collections", "Specialized container datatypes"),
        ("dataclasses", "Data classes"),
        ("enum", "Enumerations")
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for module_name, description in dependencies:
        if verify_dependency(module_name, description):
            success_count += 1
    
    logger.info(f"\n📊 DEPENDENCY VERIFICATION RESULTS:")
    logger.info(f"✅ Successful: {success_count}/{total_count}")
    logger.info(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        logger.info("🎉 ALL DEPENDENCIES VERIFIED SUCCESSFULLY!")
        logger.info("🚀 Ready for real integration implementation!")
        return True
    else:
        logger.error("⚠️  Some dependencies are missing or failed to import")
        logger.error("Please install missing dependencies before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)