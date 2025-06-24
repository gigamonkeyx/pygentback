#!/usr/bin/env python3
"""
Debug script to test Mermaid content extraction
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_extraction():
    """Test the Mermaid extraction logic"""
    
    from orchestration.mermaid_cache_manager import MermaidCacheManager
    
    docs_path = current_dir / "docs"
    cache_path = docs_path / "public" / "diagrams"
    
    manager = MermaidCacheManager(
        docs_path=docs_path,
        cache_path=cache_path
    )
    
    # Test extraction from index.md
    index_file = docs_path / "index.md"
    if index_file.exists():
        print(f"Testing extraction from: {index_file}")
        
        diagrams = manager._extract_mermaid_from_markdown(index_file)
        
        print(f"Found {len(diagrams)} diagrams:")
        for i, (diagram_id, content) in enumerate(diagrams):
            print(f"\n--- Diagram {i+1}: {diagram_id} ---")
            print(f"Content length: {len(content)} characters")
            print("First 200 characters:")
            print(repr(content[:200]))
            print("Last 200 characters:")
            print(repr(content[-200:]))
            
            # Write to temp file to inspect
            temp_file = cache_path / f"debug_{diagram_id}.mmd"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Written to: {temp_file}")
    else:
        print(f"Index file not found: {index_file}")

if __name__ == "__main__":
    test_extraction()
