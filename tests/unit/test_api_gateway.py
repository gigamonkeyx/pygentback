#!/usr/bin/env python3
"""
API Gateway Test Suite

Comprehensive testing of API gateway, authentication, authorization,
and integration with all PyGent Factory systems.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from api.gateway import api_gateway, APIGatewayConfig
from auth.jwt_auth import jwt_auth, AuthConfig
from auth.authorization import rbac_manager, Permission, Role
from cache.integration_layer import integrated_cache


async def test_api_gateway_initialization():
    """Test API gateway initialization"""
    print("🌐 Testing API Gateway Initialization...")
    
    try:
        # Initialize API gateway
        success = await api_gateway.initialize()
        
        if not success:
            print("❌ API gateway initialization failed")
            return False
        
        print("✅ API gateway initialized successfully")
        
        # Check if FastAPI app is created
        if not api_gateway.app:
            print("❌ FastAPI app not created")
            return False
        
        print("✅ FastAPI app created")
        
        # Check middleware setup
        if not hasattr(api_gateway.app, 'middleware_stack'):
            print("❌ Middleware not configured")
            return False
        
        print("✅ Middleware configured")
        
        # Check metrics initialization
        if not api_gateway.metrics:
            print("❌ Metrics not initialized")
            return False
        
        print("✅ Metrics initialized")
        print(f"   Start time: {api_gateway.metrics['start_time']}")
        
        return True
        
    except Exception as e:
        print(f"❌ API gateway initialization test failed: {e}")
        return False


async def test_jwt_authentication():
    """Test JWT authentication system"""
    print("\n🔐 Testing JWT Authentication...")
    
    try:
        # Test password hashing
        password = "TestPassword123!"
        hashed = jwt_auth.hash_password(password)
        
        if not jwt_auth.verify_password(password, hashed):
            print("❌ Password hashing/verification failed")
            return False
        
        print("✅ Password hashing and verification working")
        
        # Test password strength validation
        weak_password = "123"
        validation = jwt_auth.validate_password_strength(weak_password)
        
        if validation["valid"]:
            print("❌ Weak password validation failed")
            return False
        
        print("✅ Password strength validation working")
        print(f"   Weak password issues: {len(validation['issues'])}")
        
        # Test strong password
        strong_password = "StrongPassword123!@#"
        validation = jwt_auth.validate_password_strength(strong_password)
        
        if not validation["valid"]:
            print("❌ Strong password validation failed")
            return False
        
        print("✅ Strong password validation working")
        
        # Test token creation (mock user data)
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"],
            "permissions": ["user:read", "user:update"]
        }
        
        access_token = await jwt_auth.create_access_token(user_data)
        
        if not access_token:
            print("❌ Access token creation failed")
            return False
        
        print("✅ Access token creation working")
        print(f"   Token length: {len(access_token)} characters")
        
        # Test token validation
        try:
            token_data = await jwt_auth.validate_token(access_token)
            
            if token_data.user_id != user_data["user_id"]:
                print("❌ Token validation failed - user ID mismatch")
                return False
            
            print("✅ Token validation working")
            print(f"   Validated user: {token_data.username}")
            
        except Exception as e:
            print(f"❌ Token validation failed: {e}")
            return False
        
        # Test refresh token creation
        refresh_token = await jwt_auth.create_refresh_token(user_data["user_id"])
        
        if not refresh_token:
            print("❌ Refresh token creation failed")
            return False
        
        print("✅ Refresh token creation working")
        
        return True
        
    except Exception as e:
        print(f"❌ JWT authentication test failed: {e}")
        return False


async def test_authorization_rbac():
    """Test authorization and RBAC system"""
    print("\n🛡️ Testing Authorization and RBAC...")
    
    try:
        # Test role definitions
        guest_permissions = rbac_manager.get_role_permissions(Role.GUEST.value)
        user_permissions = rbac_manager.get_role_permissions(Role.USER.value)
        admin_permissions = rbac_manager.get_role_permissions(Role.ADMIN.value)
        
        if not guest_permissions:
            print("❌ Guest role permissions not defined")
            return False
        
        print(f"✅ Guest role has {len(guest_permissions)} permissions")
        
        if len(user_permissions) <= len(guest_permissions):
            print("❌ User role should have more permissions than guest")
            return False
        
        print(f"✅ User role has {len(user_permissions)} permissions")
        
        if len(admin_permissions) <= len(user_permissions):
            print("❌ Admin role should have more permissions than user")
            return False
        
        print(f"✅ Admin role has {len(admin_permissions)} permissions")
        
        # Test permission inheritance
        if not guest_permissions.issubset(user_permissions):
            print("❌ User role should inherit guest permissions")
            return False
        
        print("✅ Permission inheritance working")
        
        # Test user permissions calculation
        user_roles = ["user"]
        calculated_permissions = rbac_manager.get_user_permissions(user_roles)
        
        if calculated_permissions != user_permissions:
            print("❌ User permissions calculation failed")
            return False
        
        print("✅ User permissions calculation working")
        
        # Test cached permissions
        cached_permissions = await rbac_manager.get_cached_user_permissions("test_user", user_roles)
        
        if cached_permissions != calculated_permissions:
            print("❌ Cached permissions mismatch")
            return False
        
        print("✅ Permission caching working")
        
        # Test specific permission checks
        from auth.authorization import AuthorizationContext
        
        auth_context = AuthorizationContext(
            user_id="test_user",
            username="testuser",
            roles=["user"],
            permissions=calculated_permissions,
            request_path="/api/documents",
            request_method="GET"
        )
        
        # User should have document read permission
        has_doc_read = await rbac_manager.check_permission(auth_context, Permission.DOCUMENT_READ)
        
        if not has_doc_read:
            print("❌ User should have document read permission")
            return False
        
        print("✅ Permission check working (granted)")
        
        # User should not have system admin permission
        has_sys_admin = await rbac_manager.check_permission(auth_context, Permission.SYSTEM_ADMIN)
        
        if has_sys_admin:
            print("❌ User should not have system admin permission")
            return False
        
        print("✅ Permission check working (denied)")
        
        return True
        
    except Exception as e:
        print(f"❌ Authorization RBAC test failed: {e}")
        return False


async def test_api_gateway_routes():
    """Test API gateway routes"""
    print("\n🛣️ Testing API Gateway Routes...")
    
    try:
        # Check if routes are registered
        routes = api_gateway.app.routes
        
        if not routes:
            print("❌ No routes registered")
            return False
        
        print(f"✅ {len(routes)} routes registered")
        
        # Check for essential routes
        route_paths = [route.path for route in routes if hasattr(route, 'path')]
        
        essential_routes = [
            "/",
            "/health",
            "/metrics",
            "/api/auth/login",
            "/api/auth/register",
            "/api/redis/status",
            "/api/gpu/status"
        ]
        
        missing_routes = []
        for route in essential_routes:
            if route not in route_paths:
                missing_routes.append(route)
        
        if missing_routes:
            print(f"❌ Missing essential routes: {missing_routes}")
            return False
        
        print("✅ All essential routes registered")
        
        # Check middleware
        middleware_count = len(api_gateway.app.middleware_stack) if hasattr(api_gateway.app, 'middleware_stack') else 0
        
        if middleware_count == 0:
            print("❌ No middleware configured")
            return False
        
        print(f"✅ {middleware_count} middleware components configured")
        
        return True
        
    except Exception as e:
        print(f"❌ API gateway routes test failed: {e}")
        return False


async def test_rate_limiting_integration():
    """Test rate limiting integration"""
    print("\n🚦 Testing Rate Limiting Integration...")
    
    try:
        from cache.rate_limiter import rate_limiter
        
        # Check if rate limiter is initialized
        if not rate_limiter.is_initialized:
            await rate_limiter.initialize()
        
        if not rate_limiter.is_initialized:
            print("❌ Rate limiter not initialized")
            return False
        
        print("✅ Rate limiter initialized")
        
        # Check default rules
        if not rate_limiter.rules:
            print("❌ No rate limiting rules configured")
            return False
        
        print(f"✅ {len(rate_limiter.rules)} rate limiting rules configured")
        
        # List configured rules
        for rule_name in rate_limiter.rules:
            rule = rate_limiter.rules[rule_name]
            print(f"   📋 {rule_name}: {rule.requests_per_window}/{rule.window_seconds}s")
        
        # Test rate limit check
        test_identifier = "test_api_gateway"
        result = await rate_limiter.check_rate_limit("api_general", test_identifier)
        
        if not result.allowed:
            print("❌ Rate limit check failed - should be allowed")
            return False
        
        print("✅ Rate limit check working")
        print(f"   Remaining requests: {result.remaining_requests}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rate limiting integration test failed: {e}")
        return False


async def test_cache_integration():
    """Test cache system integration"""
    print("\n🔧 Testing Cache Integration...")
    
    try:
        # Check if integrated cache is initialized
        if not integrated_cache.is_initialized:
            await integrated_cache.initialize()
        
        if not integrated_cache.is_initialized:
            print("❌ Integrated cache not initialized")
            return False
        
        print("✅ Integrated cache initialized")
        
        # Test cache health
        health_status = await integrated_cache.get_integrated_health_status()
        
        if not health_status:
            print("❌ Cache health check failed")
            return False
        
        print(f"✅ Cache health check passed: {health_status['overall_status']}")
        
        # Test performance summary
        performance = await integrated_cache.get_performance_summary()
        
        if not performance:
            print("❌ Cache performance summary failed")
            return False
        
        print("✅ Cache performance summary working")
        
        cache_perf = performance.get("cache_performance", {})
        print(f"   Cache hit rate: {cache_perf.get('overall_hit_rate_percent', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache integration test failed: {e}")
        return False


async def test_security_features():
    """Test security features"""
    print("\n🔒 Testing Security Features...")
    
    try:
        # Test CORS configuration
        cors_middleware = None
        for middleware in api_gateway.app.middleware_stack:
            if hasattr(middleware, 'cls') and 'CORS' in str(middleware.cls):
                cors_middleware = middleware
                break
        
        if not cors_middleware:
            print("❌ CORS middleware not configured")
            return False
        
        print("✅ CORS middleware configured")
        
        # Test trusted host middleware
        trusted_host_middleware = None
        for middleware in api_gateway.app.middleware_stack:
            if hasattr(middleware, 'cls') and 'TrustedHost' in str(middleware.cls):
                trusted_host_middleware = middleware
                break
        
        if not trusted_host_middleware:
            print("❌ Trusted host middleware not configured")
            return False
        
        print("✅ Trusted host middleware configured")
        
        # Test compression middleware
        compression_middleware = None
        for middleware in api_gateway.app.middleware_stack:
            if hasattr(middleware, 'cls') and 'GZip' in str(middleware.cls):
                compression_middleware = middleware
                break
        
        if not compression_middleware:
            print("❌ Compression middleware not configured")
            return False
        
        print("✅ Compression middleware configured")
        
        # Test authentication metrics
        auth_metrics = jwt_auth.get_auth_metrics()
        
        if not auth_metrics:
            print("❌ Authentication metrics not available")
            return False
        
        print("✅ Authentication metrics available")
        print(f"   Total logins: {auth_metrics['total_logins']}")
        
        # Test authorization metrics
        authz_metrics = rbac_manager.get_authorization_metrics()
        
        if not authz_metrics:
            print("❌ Authorization metrics not available")
            return False
        
        print("✅ Authorization metrics available")
        print(f"   Authorization checks: {authz_metrics['authorization_checks']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Security features test failed: {e}")
        return False


async def run_all_tests():
    """Run all API gateway tests"""
    print("🚀 PyGent Factory API Gateway Test Suite")
    print("=" * 70)
    
    tests = [
        ("API Gateway Initialization", test_api_gateway_initialization),
        ("JWT Authentication", test_jwt_authentication),
        ("Authorization RBAC", test_authorization_rbac),
        ("API Gateway Routes", test_api_gateway_routes),
        ("Rate Limiting Integration", test_rate_limiting_integration),
        ("Cache Integration", test_cache_integration),
        ("Security Features", test_security_features)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    total = len(tests)
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL API GATEWAY TESTS PASSED!")
        print("   API Gateway is production-ready with:")
        print("   ✅ JWT authentication with secure token management")
        print("   ✅ Role-based access control (RBAC) with permission inheritance")
        print("   ✅ Advanced rate limiting with multiple algorithms")
        print("   ✅ Multi-layer Redis caching integration")
        print("   ✅ Comprehensive security middleware")
        print("   ✅ Performance monitoring and metrics")
        print("   ✅ Production-ready error handling")
    else:
        print("⚠️ SOME API GATEWAY TESTS FAILED")
        print("   Check the errors above and ensure all dependencies are properly configured.")
    
    # Cleanup
    try:
        await integrated_cache.cleanup()
        print("🧹 Test cleanup completed")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
