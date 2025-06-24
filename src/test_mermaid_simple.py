#!/usr/bin/env python3
"""
Simple test for MermaidCacheManager using direct execution
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print(f"🔍 Testing MermaidCacheManager")
print(f"Current directory: {current_dir}")
print(f"Working from: {os.getcwd()}")

# Test 1: Basic import
print("\n1️⃣ Testing import...")
try:
    from orchestration.mermaid_cache_manager import MermaidCacheManager
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create instance
print("\n2️⃣ Testing instance creation...")
try:
    docs_path = current_dir / "docs"
    cache_path = docs_path / "public" / "diagrams"
    
    print(f"Docs path: {docs_path}")
    print(f"Docs exists: {docs_path.exists()}")
    
    manager = MermaidCacheManager(
        docs_path=docs_path,
        cache_path=cache_path
    )
    print("✅ Instance creation successful")
except Exception as e:
    print(f"❌ Instance creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Find diagrams
print("\n3️⃣ Testing diagram discovery...")
try:
    sources = manager.find_all_mermaid_sources()
    print(f"✅ Found {len(sources)} diagram sources")
    
    if sources:
        print("📋 Diagram sources:")
        for diagram_id, info in list(sources.items())[:3]:  # Show first 3
            print(f"  • {diagram_id}: {info['type']} from {info['source_file'].name}")
        if len(sources) > 3:
            print(f"  ... and {len(sources) - 3} more")
    else:
        print("ℹ️  No diagram sources found")
        
except Exception as e:
    print(f"❌ Diagram discovery failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Cache status
print("\n4️⃣ Testing cache status...")
try:
    status = manager.get_cache_status()
    print("✅ Cache status retrieved:")
    for key, value in status.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"❌ Cache status failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All tests passed!")
print(f"📊 Summary: {len(sources)} diagrams found, {status.get('cached_diagrams', 0)} cached")