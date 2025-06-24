#!/usr/bin/env python3
"""
Test script for the Intelligent Documentation Build System
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_mermaid_cache():
    """Test Mermaid cache manager"""
    print("🎨 Testing Mermaid Cache Manager...")
    
    try:
        from orchestration.mermaid_cache_manager import MermaidCacheManager
        
        manager = MermaidCacheManager()
        print("✅ Created MermaidCacheManager")
        
        # Get cache status
        status = manager.get_cache_status()
        print("✅ Got cache status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Find diagram sources
        sources = manager.find_all_mermaid_sources()
        print(f"✅ Found {len(sources)} diagram sources")
        
        return True
        
    except Exception as e:
        print(f"❌ Mermaid cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_trigger_detector():
    """Test build trigger detector"""
    print("\n🔍 Testing Build Trigger Detector...")
    
    try:
        from orchestration.build_trigger_detector import BuildTriggerDetector
        
        detector = BuildTriggerDetector()
        print("✅ Created BuildTriggerDetector")
        
        # Check triggers
        status = await detector.check_all_triggers()
        print("✅ Checked triggers:")
        print(f"  Should build: {status['should_build']}")
        print(f"  Trigger reasons: {status['summary']['trigger_reasons']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trigger detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_intelligent_builder():
    """Test intelligent documentation builder"""
    print("\n🏗️  Testing Intelligent Documentation Builder...")
    
    try:
        from orchestration.intelligent_docs_builder import IntelligentDocsBuilder
        
        builder = IntelligentDocsBuilder()
        print("✅ Created IntelligentDocsBuilder")
        
        # Get build status
        status = await builder.get_build_status()
        print("✅ Got build status:")
        print(f"  System status: {status['system_status']}")
        print(f"  Should build: {status['triggers']['should_build']}")
        print(f"  Mermaid cache: {status['mermaid_cache']['cached_diagrams']}/{status['mermaid_cache']['total_diagrams']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Intelligent builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing Intelligent Documentation Build System")
    print("=" * 60)
    
    tests = [
        test_mermaid_cache,
        test_trigger_detector,
        test_intelligent_builder
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"  Passed: {passed}/{total}")
    print(f"  Success Rate: {passed/total:.1%}")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
