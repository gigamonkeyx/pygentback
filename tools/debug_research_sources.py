#!/usr/bin/env python3
"""
Quick debug test to see why research sources return 0 papers
"""

import asyncio
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.research.simple_enhanced_research_manager import simple_research_manager

# Set up logging to see debug info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def debug_research_sources():
    """Debug why research sources return 0 papers"""
    
    print("🔍 Debugging Research Sources")
    print("=" * 50)
    
    # Test with very basic, common queries that should definitely return results
    test_queries = [
        "machine learning",  # Very common topic
        "deep learning",     # Very common topic
        "AI",               # Extremely broad
        "neural network",   # Common ML topic
        "computer science"  # Very broad academic field
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        print("-" * 40)
        
        try:
            # Test each source individually first
            print("🔬 Testing individual sources:")
            
            # Test ArXiv
            print("  📚 ArXiv...")
            arxiv_results = await simple_research_manager._search_arxiv_fixed(query, 3)
            print(f"     Result: {len(arxiv_results.get('papers', []))} papers, success: {arxiv_results.get('success')}")
            if not arxiv_results.get('success'):
                print(f"     Error: {arxiv_results.get('error', 'Unknown')}")
            
            # Test Semantic Scholar
            print("  🧠 Semantic Scholar...")
            try:
                semantic_results = await simple_research_manager._search_semantic_scholar_fixed(query, 3)
                print(f"     Result: {len(semantic_results.get('papers', []))} papers, success: {semantic_results.get('success')}")
                if not semantic_results.get('success'):
                    print(f"     Error: {semantic_results.get('error', 'Unknown')}")
            except Exception as e:
                print(f"     Exception: {e}")
            
            # Test CrossRef
            print("  🔗 CrossRef...")
            try:
                crossref_results = await simple_research_manager._search_crossref_fixed(query, 3)
                print(f"     Result: {len(crossref_results.get('papers', []))} papers, success: {crossref_results.get('success')}")
                if not crossref_results.get('success'):
                    print(f"     Error: {crossref_results.get('error', 'Unknown')}")
            except Exception as e:
                print(f"     Exception: {e}")
            
            # Test combined search
            print("  🔄 Combined search...")
            all_results = await simple_research_manager.search_all_portals(query, 5)
            print(f"     Total results: {len(all_results)} papers")
            
            if all_results:
                # Show sources breakdown
                sources = {}
                for paper in all_results:
                    source = paper.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                print(f"     Sources: {sources}")
                
                # Show first paper as example
                first_paper = all_results[0]
                print(f"     Example: '{first_paper.get('title', 'No title')[:60]}...'")
                print(f"     From: {first_paper.get('source', 'unknown')}")
            else:
                print("     ❌ No papers found from any source")
            
        except Exception as e:
            print(f"❌ Error testing '{query}': {e}")
            import traceback
            traceback.print_exc()
        
        # Add delay between queries
        if i < len(test_queries):
            print("   ⏳ Waiting before next query...")
            await asyncio.sleep(2)
    
    print("\n" + "=" * 50)
    print("🏁 Debug test completed")
    
    # Show failure counts
    print(f"\nFailure counts: {simple_research_manager.failure_counts}")

if __name__ == "__main__":
    asyncio.run(debug_research_sources())
