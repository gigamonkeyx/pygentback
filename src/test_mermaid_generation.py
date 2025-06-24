#!/usr/bin/env python3
"""
Test Mermaid diagram generation functionality
"""

import sys
import os
import asyncio
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print(f"🎨 Testing Mermaid Diagram Generation")
print(f"Current directory: {current_dir}")

async def test_diagram_generation():
    """Test if we can generate a single diagram"""
    print("\n1️⃣ Setting up MermaidCacheManager...")
    
    try:
        from orchestration.mermaid_cache_manager import MermaidCacheManager
        
        docs_path = current_dir / "docs"
        cache_path = docs_path / "public" / "diagrams"
        
        manager = MermaidCacheManager(
            docs_path=docs_path,
            cache_path=cache_path
        )
        print("✅ MermaidCacheManager created")
        
        # Find diagram sources
        sources = manager.find_all_mermaid_sources()
        print(f"✅ Found {len(sources)} diagram sources")
        
        if not sources:
            print("❌ No diagram sources found - cannot test generation")
            return False
        
        # Pick the first diagram to test
        first_diagram_id = list(sources.keys())[0]
        first_diagram = sources[first_diagram_id]
        
        print(f"\n2️⃣ Testing generation of: {first_diagram_id}")
        print(f"   Source: {first_diagram['source_file'].name}")
        print(f"   Type: {first_diagram['type']}")
        print(f"   Content preview: {first_diagram['content'][:100]}...")
        
        # Check if Mermaid CLI is available
        print("\n3️⃣ Checking Mermaid CLI availability...")
        import subprocess

        # Check local mmdc first
        import platform
        if platform.system() == "Windows":
            local_mmdc = docs_path / "node_modules" / ".bin" / "mmdc.cmd"
        else:
            local_mmdc = docs_path / "node_modules" / ".bin" / "mmdc"

        if local_mmdc.exists():
            try:
                result = subprocess.run(
                    [str(local_mmdc), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(docs_path)
                )
                if result.returncode == 0:
                    print(f"✅ Local Mermaid CLI available: {result.stdout.strip()}")
                else:
                    print(f"⚠️  Local Mermaid CLI check failed: {result.stderr}")
            except Exception as e:
                print(f"⚠️  Error checking local Mermaid CLI: {e}")
        else:
            print("⚠️  Local Mermaid CLI not found, checking npx...")
            try:
                result = subprocess.run(
                    ["npx", "@mermaid-js/mermaid-cli", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(docs_path)
                )
                if result.returncode == 0:
                    print(f"✅ Mermaid CLI available via npx: {result.stdout.strip()}")
                else:
                    print(f"⚠️  Mermaid CLI check failed: {result.stderr}")
                    print("   Attempting to install locally...")

                    # Try to install Mermaid CLI locally
                    install_result = subprocess.run(
                        ["npm", "install", "@mermaid-js/mermaid-cli"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=str(docs_path)
                    )
                    if install_result.returncode == 0:
                        print("✅ Mermaid CLI installed locally successfully")
                    else:
                        print(f"❌ Failed to install Mermaid CLI: {install_result.stderr}")
                        return False
            except Exception as e:
                print(f"❌ Error checking Mermaid CLI: {e}")
                return False
        
        # Test diagram generation
        print(f"\n4️⃣ Generating diagram: {first_diagram_id}")
        
        success = await manager.generate_diagram(
            first_diagram_id,
            first_diagram['content']
        )
        
        if success:
            print("✅ Diagram generation successful!")
            
            # Check if SVG file was created
            svg_file = cache_path / f"{first_diagram_id}.svg"
            if svg_file.exists():
                file_size = svg_file.stat().st_size
                print(f"✅ SVG file created: {svg_file}")
                print(f"   File size: {file_size} bytes")
                
                # Read first few lines to verify it's valid SVG
                with open(svg_file, 'r') as f:
                    first_lines = f.read(200)
                    if '<svg' in first_lines:
                        print("✅ Valid SVG content detected")
                        return True
                    else:
                        print("❌ Invalid SVG content")
                        return False
            else:
                print("❌ SVG file not created")
                return False
        else:
            print("❌ Diagram generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cache_regeneration():
    """Test cache regeneration functionality"""
    print("\n5️⃣ Testing cache regeneration...")
    
    try:
        from orchestration.mermaid_cache_manager import MermaidCacheManager
        
        docs_path = current_dir / "docs"
        cache_path = docs_path / "public" / "diagrams"
        
        manager = MermaidCacheManager(
            docs_path=docs_path,
            cache_path=cache_path
        )
        
        # Test regeneration (this will only regenerate missing/outdated diagrams)
        results = await manager.regenerate_diagrams(force=False)
        
        if results:
            successful = sum(1 for success in results.values() if success)
            print(f"✅ Regeneration complete: {successful}/{len(results)} successful")
            
            for diagram_id, success in results.items():
                status = "✅" if success else "❌"
                print(f"   {status} {diagram_id}")
            
            return successful > 0
        else:
            print("ℹ️  No diagrams needed regeneration")
            return True
            
    except Exception as e:
        print(f"❌ Cache regeneration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run generation tests"""
    print("🧪 Mermaid Diagram Generation Test")
    print("=" * 50)
    
    # Test single diagram generation
    generation_success = await test_diagram_generation()
    
    if generation_success:
        # Test cache regeneration
        regeneration_success = await test_cache_regeneration()
        
        if regeneration_success:
            print("\n" + "=" * 50)
            print("🎉 ALL GENERATION TESTS PASSED!")
            print("✅ Mermaid diagram generation is working")
            print("✅ Cache regeneration is working")
            return 0
        else:
            print("\n❌ Cache regeneration failed")
            return 1
    else:
        print("\n❌ Diagram generation failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)