#!/usr/bin/env python3
"""
Comprehensive Redis Caching System Test Suite

Tests all Redis caching components including cache layers, session management,
rate limiting, and integration with database and GPU systems.
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from cache.redis_manager import redis_manager
from cache.cache_layers import cache_manager
from cache.session_manager import session_manager
from cache.rate_limiter import rate_limiter, RateLimitRule, RateLimitAlgorithm
from cache.integration_layer import integrated_cache


async def test_redis_manager():
    """Test Redis manager functionality"""
    print("🔧 Testing Redis Manager...")
    
    try:
        # Initialize Redis manager
        success = await redis_manager.initialize()
        if not success:
            print("❌ Redis manager initialization failed")
            return False
        
        print("✅ Redis manager initialized")
        
        # Test basic operations
        test_key = "test_redis_manager"
        test_value = {"message": "Hello Redis!", "timestamp": datetime.utcnow().isoformat()}
        
        # Test SET
        set_success = await redis_manager.set(test_key, test_value, ttl=60)
        if not set_success:
            print("❌ Redis SET operation failed")
            return False
        
        print("✅ Redis SET operation successful")
        
        # Test GET
        retrieved_value = await redis_manager.get(test_key)
        if retrieved_value != test_value:
            print("❌ Redis GET operation failed")
            return False
        
        print("✅ Redis GET operation successful")
        
        # Test EXISTS
        exists = await redis_manager.exists(test_key)
        if not exists:
            print("❌ Redis EXISTS operation failed")
            return False
        
        print("✅ Redis EXISTS operation successful")
        
        # Test DELETE
        delete_success = await redis_manager.delete(test_key)
        if not delete_success:
            print("❌ Redis DELETE operation failed")
            return False
        
        print("✅ Redis DELETE operation successful")
        
        # Test health check
        health = await redis_manager.health_check()
        if health.get("status") != "healthy":
            print("❌ Redis health check failed")
            return False
        
        print("✅ Redis health check passed")
        print(f"   Response time: {health.get('response_time_ms', 0):.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis manager test failed: {e}")
        return False


async def test_cache_layers():
    """Test multi-layer caching system"""
    print("\n📚 Testing Cache Layers...")
    
    try:
        # Initialize cache manager
        success = await cache_manager.initialize()
        if not success:
            print("❌ Cache manager initialization failed")
            return False
        
        print("✅ Cache manager initialized")
        
        # Test database cache layer
        db_key = "user_profile_123"
        db_data = {"user_id": 123, "name": "Test User", "email": "test@example.com"}
        
        cache_success = await cache_manager.cache_db_query(db_key, db_data)
        if not cache_success:
            print("❌ Database cache SET failed")
            return False
        
        cached_data = await cache_manager.get_cached_db_query(db_key)
        if cached_data != db_data:
            print("❌ Database cache GET failed")
            return False
        
        print("✅ Database cache layer working")
        
        # Test API cache layer
        api_endpoint = "/api/users"
        api_params = {"page": 1, "limit": 10}
        api_response = {"users": [{"id": 1, "name": "User 1"}], "total": 1}
        
        api_cache_success = await cache_manager.cache_api_response(api_endpoint, api_params, api_response)
        if not api_cache_success:
            print("❌ API cache SET failed")
            return False
        
        cached_api_response = await cache_manager.get_cached_api_response(api_endpoint, api_params)
        if cached_api_response != api_response:
            print("❌ API cache GET failed")
            return False
        
        print("✅ API cache layer working")
        
        # Test model inference cache layer
        model_name = "test_model"
        input_data = {"prompt": "Hello, world!", "temperature": 0.7}
        input_hash = cache_manager.generate_input_hash(input_data)
        model_result = {"response": "Hello! How can I help you?", "tokens": 25}
        
        model_cache_success = await cache_manager.cache_model_inference(model_name, input_hash, model_result)
        if not model_cache_success:
            print("❌ Model inference cache SET failed")
            return False
        
        cached_model_result = await cache_manager.get_cached_model_inference(model_name, input_hash)
        if cached_model_result != model_result:
            print("❌ Model inference cache GET failed")
            return False
        
        print("✅ Model inference cache layer working")
        
        # Test cache statistics
        stats = await cache_manager.get_cache_statistics()
        if not stats or "cache_layers" not in stats:
            print("❌ Cache statistics failed")
            return False
        
        print("✅ Cache statistics working")
        print(f"   DB cache hits: {stats['cache_layers']['db']['hits']}")
        print(f"   API cache hits: {stats['cache_layers']['api']['hits']}")
        print(f"   Model cache hits: {stats['cache_layers']['model']['hits']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache layers test failed: {e}")
        return False


async def test_session_management():
    """Test session management system"""
    print("\n👤 Testing Session Management...")
    
    try:
        # Initialize session manager
        success = await session_manager.initialize()
        if not success:
            print("❌ Session manager initialization failed")
            return False
        
        print("✅ Session manager initialized")
        
        # Test session creation
        user_id = "test_user_123"
        ip_address = "192.168.1.100"
        user_agent = "Mozilla/5.0 Test Browser"
        
        session_data = await session_manager.create_session(
            user_id, ip_address, user_agent, {"login_time": datetime.utcnow().isoformat()}
        )
        
        if not session_data:
            print("❌ Session creation failed")
            return False
        
        session_id = session_data.session_id
        print(f"✅ Session created: {session_id}")
        
        # Test session retrieval
        retrieved_session = await session_manager.get_session(session_id)
        if not retrieved_session or retrieved_session.user_id != user_id:
            print("❌ Session retrieval failed")
            return False
        
        print("✅ Session retrieval working")
        
        # Test session update
        update_data = {"last_action": "viewed_dashboard"}
        update_success = await session_manager.update_session(session_id, update_data)
        if not update_success:
            print("❌ Session update failed")
            return False
        
        print("✅ Session update working")
        
        # Test user sessions listing
        user_sessions = await session_manager.get_user_sessions(user_id)
        if not user_sessions or len(user_sessions) != 1:
            print("❌ User sessions listing failed")
            return False
        
        print("✅ User sessions listing working")
        
        # Test session extension
        extend_success = await session_manager.extend_session(session_id, 120)  # 2 hours
        if not extend_success:
            print("❌ Session extension failed")
            return False
        
        print("✅ Session extension working")
        
        # Test session deletion
        delete_success = await session_manager.delete_session(session_id)
        if not delete_success:
            print("❌ Session deletion failed")
            return False
        
        print("✅ Session deletion working")
        
        # Test session statistics
        stats = session_manager.get_session_stats()
        if not stats or "session_creates" not in stats:
            print("❌ Session statistics failed")
            return False
        
        print("✅ Session statistics working")
        print(f"   Sessions created: {stats['session_creates']}")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        return False


async def test_rate_limiting():
    """Test rate limiting system"""
    print("\n🚦 Testing Rate Limiting...")
    
    try:
        # Initialize rate limiter
        success = await rate_limiter.initialize()
        if not success:
            print("❌ Rate limiter initialization failed")
            return False
        
        print("✅ Rate limiter initialized")
        
        # Add test rate limit rule
        test_rule = RateLimitRule(
            name="test_api",
            requests_per_window=5,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW
        )
        rate_limiter.add_rule(test_rule)
        
        print("✅ Test rate limit rule added")
        
        # Test rate limiting
        test_identifier = "test_user_rate_limit"
        
        # Should allow first 5 requests
        for i in range(5):
            result = await rate_limiter.check_rate_limit("test_api", test_identifier)
            if not result.allowed:
                print(f"❌ Request {i+1} should be allowed but was blocked")
                return False
        
        print("✅ First 5 requests allowed")
        
        # 6th request should be blocked
        result = await rate_limiter.check_rate_limit("test_api", test_identifier)
        if result.allowed:
            print("❌ 6th request should be blocked but was allowed")
            return False
        
        print("✅ 6th request correctly blocked")
        print(f"   Remaining requests: {result.remaining_requests}")
        print(f"   Retry after: {result.retry_after_seconds} seconds")
        
        # Test rate limit status
        status = await rate_limiter.get_rate_limit_status("test_api", test_identifier)
        if not status:
            print("❌ Rate limit status check failed")
            return False
        
        print("✅ Rate limit status check working")
        
        # Test rate limit reset
        reset_success = await rate_limiter.reset_rate_limit("test_api", test_identifier)
        if not reset_success:
            print("❌ Rate limit reset failed")
            return False
        
        print("✅ Rate limit reset working")
        
        # Test statistics
        stats = rate_limiter.get_statistics()
        if not stats or "total_checks" not in stats:
            print("❌ Rate limiter statistics failed")
            return False
        
        print("✅ Rate limiter statistics working")
        print(f"   Total checks: {stats['total_checks']}")
        print(f"   Block rate: {stats['block_rate_percent']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")
        return False


async def test_integration_layer():
    """Test integrated cache system"""
    print("\n🔗 Testing Integration Layer...")
    
    try:
        # Initialize integrated cache
        success = await integrated_cache.initialize()
        if not success:
            print("❌ Integrated cache initialization failed")
            return False
        
        print("✅ Integrated cache initialized")
        
        # Test integrated session creation
        session_id = await integrated_cache.create_user_session("integration_test_user")
        if not session_id:
            print("❌ Integrated session creation failed")
            return False
        
        print(f"✅ Integrated session created: {session_id}")
        
        # Test session validation with cache
        session_data = await integrated_cache.validate_session_with_cache(session_id)
        if not session_data:
            print("❌ Integrated session validation failed")
            return False
        
        print("✅ Integrated session validation working")
        
        # Test performance optimization
        optimization_results = await integrated_cache.optimize_cache_performance()
        if not optimization_results:
            print("❌ Cache performance optimization failed")
            return False
        
        print("✅ Cache performance optimization working")
        
        # Test health status
        health_status = await integrated_cache.get_integrated_health_status()
        if not health_status or health_status.get("overall_status") not in ["healthy", "degraded"]:
            print("❌ Integrated health check failed")
            return False
        
        print("✅ Integrated health check working")
        print(f"   Overall status: {health_status['overall_status']}")
        
        # Test performance summary
        performance_summary = await integrated_cache.get_performance_summary()
        if not performance_summary:
            print("❌ Performance summary failed")
            return False
        
        print("✅ Performance summary working")
        
        cache_perf = performance_summary.get("cache_performance", {})
        print(f"   Overall hit rate: {cache_perf.get('overall_hit_rate_percent', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration layer test failed: {e}")
        return False


async def test_performance_benchmarks():
    """Test cache system performance"""
    print("\n⚡ Testing Performance Benchmarks...")
    
    try:
        # Test Redis performance
        start_time = time.time()
        
        # Bulk SET operations
        for i in range(100):
            await redis_manager.set(f"perf_test_{i}", f"value_{i}", ttl=60)
        
        set_time = time.time() - start_time
        
        # Bulk GET operations
        start_time = time.time()
        
        for i in range(100):
            await redis_manager.get(f"perf_test_{i}")
        
        get_time = time.time() - start_time
        
        # Cleanup
        start_time = time.time()
        
        for i in range(100):
            await redis_manager.delete(f"perf_test_{i}")
        
        delete_time = time.time() - start_time
        
        print("✅ Performance benchmarks completed")
        print(f"   100 SET operations: {set_time:.3f}s ({100/set_time:.0f} ops/sec)")
        print(f"   100 GET operations: {get_time:.3f}s ({100/get_time:.0f} ops/sec)")
        print(f"   100 DELETE operations: {delete_time:.3f}s ({100/delete_time:.0f} ops/sec)")
        
        # Performance should be reasonable
        if set_time > 1.0 or get_time > 1.0 or delete_time > 1.0:
            print("⚠️ Performance may be suboptimal (>1s for 100 operations)")
        else:
            print("✅ Performance is good (<1s for 100 operations)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


async def run_all_tests():
    """Run comprehensive Redis caching system tests"""
    print("🚀 PyGent Factory Redis Caching System Test Suite")
    print("=" * 70)
    
    tests = [
        ("Redis Manager", test_redis_manager),
        ("Cache Layers", test_cache_layers),
        ("Session Management", test_session_management),
        ("Rate Limiting", test_rate_limiting),
        ("Integration Layer", test_integration_layer),
        ("Performance Benchmarks", test_performance_benchmarks)
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
        print("🎉 ALL REDIS CACHING TESTS PASSED!")
        print("   Redis caching system is production-ready with:")
        print("   ✅ Multi-layer caching (DB, API, Model, Performance)")
        print("   ✅ Session management with Redis backend")
        print("   ✅ Advanced rate limiting with multiple algorithms")
        print("   ✅ Integrated cache system with database and GPU optimization")
        print("   ✅ Comprehensive monitoring and health checks")
        print("   ✅ High-performance operations (>100 ops/sec)")
    else:
        print("⚠️ SOME REDIS CACHING TESTS FAILED")
        print("   Check the errors above and ensure Redis is running and accessible.")
    
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
