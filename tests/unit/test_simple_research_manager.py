#!/usr/bin/env python3
"""
Quick test of the research manager with better test queries
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.research.simple_enhanced_research_manager import simple_research_manager

async def test_research_queries():
    """Test research manager with realistic queries"""
    
    print("Testing Enhanced Research Manager")
    print("=" * 50)
    
    # Test with realistic academic queries
    test_queries = [
        "machine learning",
        "artificial intelligence",
        "quantum computing",
        "neural networks"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print("-" * 40)
        
        try:
            results = await simple_research_manager.search_all_portals(query, 5)
            
            print(f"📊 Results: {len(results)} papers found")
            
            if results:
                # Group by source
                sources = {}
                for paper in results:
                    source = paper.get('source', 'unknown')
                    if source not in sources:
                        sources[source] = 0
                    sources[source] += 1
                
                print(f"📚 Sources: {dict(sources)}")
                
                # Show first paper as example
                first_paper = results[0]
                print(f"📄 Example paper:")
                print(f"   Title: {first_paper.get('title', 'N/A')[:80]}...")
                print(f"   Authors: {first_paper.get('authors', ['N/A'])[0] if isinstance(first_paper.get('authors'), list) else str(first_paper.get('authors', 'N/A'))[:50]}...")
                print(f"   Year: {first_paper.get('year', 'N/A')}")
                print(f"   Source: {first_paper.get('source', 'N/A')}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("✅ Research manager test completed")

if __name__ == "__main__":
    asyncio.run(test_research_queries())
