#!/usr/bin/env python3
"""
Test the updated research manager with MCP server integration
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research.simple_enhanced_research_manager import simple_research_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_research_manager():
    """Test the research manager with all available sources"""
    print("=" * 60)
    print("Testing Enhanced Research Manager with MCP Integration")
    print("=" * 60)
    
    # Test query
    query = "machine learning algorithms"
    
    print(f"Searching for: '{query}'")
    print("-" * 40)
    
    try:
        results = await simple_research_manager.search_all_portals(query, max_results_per_portal=3)
        
        print(f"\nTotal results found: {len(results)}")
        
        # Group results by source
        source_counts = {}
        for result in results:
            source = result.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"Results by source: {source_counts}")
        
        # Show sample results
        if results:
            print(f"\nFirst {min(5, len(results))} results:")
            for i, result in enumerate(results[:5], 1):
                print(f"\n{i}. {result.get('title', 'No title')}")
                print(f"   Source: {result.get('source', 'Unknown')}")
                print(f"   Authors: {result.get('authors', 'Unknown')}")
                if isinstance(result.get('authors'), list):
                    print(f"   Authors: {', '.join(result['authors'][:3])}")
                else:
                    authors_str = str(result.get('authors', 'Unknown'))
                    print(f"   Authors: {authors_str[:100]}...")
                print(f"   Year: {result.get('year', 'Unknown')}")
                if result.get('url'):
                    print(f"   URL: {result['url']}")
                if result.get('abstract'):
                    print(f"   Abstract: {result['abstract'][:150]}...")
        else:
            print("\nNo results found")
    
    except Exception as e:
        print(f"Error during search: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    print("Research Manager MCP Integration Test")
    print("====================================")
    
    # Run the async test
    asyncio.run(test_research_manager())
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
