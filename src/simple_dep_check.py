"""
Simple Dependency Check

Quick verification of core dependencies.
"""

def check_deps():
    """Check core dependencies."""
    print("🔍 Checking Core Dependencies...")
    
    try:
        import asyncpg
        print("✅ asyncpg - PostgreSQL async driver")
    except ImportError as e:
        print(f"❌ asyncpg failed: {e}")
    
    try:
        import psycopg2
        print("✅ psycopg2 - PostgreSQL sync driver")
    except ImportError as e:
        print(f"❌ psycopg2 failed: {e}")
    
    try:
        import redis
        print("✅ redis - Redis client")
    except ImportError as e:
        print(f"❌ redis failed: {e}")
    
    try:
        import aiohttp
        print("✅ aiohttp - Async HTTP client")
    except ImportError as e:
        print(f"❌ aiohttp failed: {e}")
    
    try:
        import httpx
        print("✅ httpx - Alternative HTTP client")
    except ImportError as e:
        print(f"❌ httpx failed: {e}")
    
    try:
        import pydantic
        print("✅ pydantic - Data validation")
    except ImportError as e:
        print(f"❌ pydantic failed: {e}")
    
    print("\n🎉 Core dependencies check complete!")
    print("🚀 Ready to implement real integrations!")


if __name__ == "__main__":
    check_deps()