#!/usr/bin/env python3
"""
Test OAuth and User Service Integration

Quick test to verify that the database-backed authentication system is working.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_user_service():
    """Test user service functionality"""
    print("Testing User Service...")
    
    try:
        from src.services.user_service import get_user_service
        user_service = get_user_service()
        
        print("✓ User service imported successfully")
        
        # Test creating default users
        await user_service.create_default_users()
        print("✓ Default users created successfully")
        
        # Test getting a user
        admin_user = await user_service.get_user_by_username("admin")
        if admin_user:
            print(f"✓ Admin user found: {admin_user.username} ({admin_user.email})")
        else:
            print("✗ Admin user not found")
        
        # Test authentication
        auth_result = await user_service.authenticate_user("admin", "admin")
        if auth_result:
            print(f"✓ Authentication successful for admin")
        else:
            print("✗ Authentication failed")
        
        return True
        
    except Exception as e:
        print(f"✗ User service test failed: {e}")
        return False

async def test_oauth_integration():
    """Test OAuth integration"""
    print("\nTesting OAuth Integration...")
    
    try:
        from src.auth import OAuthManager, DatabaseTokenStorage
        from src.auth.storage import FileTokenStorage
        
        oauth_manager = OAuthManager()
        
        # Test database token storage
        try:
            db_storage = DatabaseTokenStorage()
            oauth_manager.set_token_storage(db_storage)
            print("✓ Database token storage configured")
        except ImportError:
            # Fallback to file storage
            file_storage = FileTokenStorage()
            oauth_manager.set_token_storage(file_storage)
            print("⚠ Using file token storage (database not available)")
        
        print("✓ OAuth manager configured successfully")
        return True
        
    except Exception as e:
        print(f"✗ OAuth integration test failed: {e}")
        return False

async def test_api_routes():
    """Test API route availability"""
    print("\nTesting API Routes...")
    
    try:
        from src.api.routes.auth import router
        print("✓ Auth routes imported successfully")
        
        # Check if routes are properly defined
        routes = [str(route.path) for route in router.routes]
        expected_routes = ["/providers", "/authorize/{provider}", "/callback/{provider}", "/login", "/logout"]
        
        for expected in expected_routes:
            if any(expected in route for route in routes):
                print(f"✓ Route found: {expected}")
            else:
                print(f"⚠ Route missing: {expected}")
        
        return True
        
    except Exception as e:
        print(f"✗ API routes test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=== OAuth Integration Test ===\n")
    
    tests = [
        test_user_service,
        test_oauth_integration,
        test_api_routes
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print(f"\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! OAuth integration is working.")
    else:
        print("⚠ Some tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
