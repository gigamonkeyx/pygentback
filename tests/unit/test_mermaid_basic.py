#!/usr/bin/env python3
"""
Minimal test for MermaidCacheManager - just test diagram discovery
"""

import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

print(f"Current directory: {current_dir}")
print(f"Source directory: {src_dir}")
print(f"Python path: {sys.path[:3]}")  # Show first 3 entries

def test_basic_import():
    """Test if we can import the MermaidCacheManager"""
    print("\n🔍 Testing basic import...")
    
    try:
        from orchestration.mermaid_cache_manager import MermaidCacheManager
        print("✅ Successfully imported MermaidCacheManager")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_create_instance():
    """Test if we can create a MermaidCacheManager instance"""
    print("\n🏗️  Testing instance creation...")
    
    try:
        from orchestration.mermaid_cache_manager import MermaidCacheManager
        
        # Use the actual docs path
        docs_path = current_dir / "src" / "docs"
        cache_path = docs_path / "public" / "diagrams"
        
        print(f"Docs path: {docs_path}")
        print(f"Cache path: {cache_path}")
        print(f"Docs exists: {docs_path.exists()}")
        
        manager = MermaidCacheManager(
            docs_path=docs_path,
            cache_path=cache_path
        )
        print("✅ Successfully created MermaidCacheManager instance")
        return manager
    except Exception as e:
        print(f"❌ Instance creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_find_diagrams(manager):
    """Test if we can find Mermaid diagrams"""
    print("\n🔍 Testing diagram discovery...")
    
    try:
        sources = manager.find_all_mermaid_sources()
        print(f"✅ Found {len(sources)} diagram sources")
        
        if sources:
            print("📋 Diagram sources found:")
            for diagram_id, info in sources.items():
                print(f"  • {diagram_id}: {info['type']} from {info['source_file'].name}")
        else:
            print("ℹ️  No diagram sources found (this might be expected)")
        
        return sources
    except Exception as e:
        print(f"❌ Diagram discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_cache_status(manager):
    """Test if we can get cache status"""
    print("\n📊 Testing cache status...")
    
    try:
        status = manager.get_cache_status()
        print("✅ Successfully got cache status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        return status
    except Exception as e:
        print(f"❌ Cache status failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run minimal tests"""
    print("🧪 Minimal MermaidCacheManager Test")
    print("=" * 50)
    
    # Test 1: Basic import
    if not test_basic_import():
        print("\n❌ Basic import failed - stopping tests")
        return 1
    
    # Test 2: Create instance
    manager = test_create_instance()
    if not manager:
        print("\n❌ Instance creation failed - stopping tests")
        return 1
    
    # Test 3: Find diagrams
    sources = test_find_diagrams(manager)
    if sources is None:
        print("\n❌ Diagram discovery failed - stopping tests")
        return 1
    
    # Test 4: Cache status
    status = test_cache_status(manager)
    if status is None:
        print("\n❌ Cache status failed - stopping tests")
        return 1
    
    print("\n" + "=" * 50)
    print("✅ All basic tests passed!")
    print(f"📊 Summary:")
    print(f"  • Diagrams found: {len(sources)}")
    print(f"  • Cache status: {status.get('cached_diagrams', 0)}/{status.get('total_diagrams', 0)} cached")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
