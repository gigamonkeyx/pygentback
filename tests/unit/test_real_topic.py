#!/usr/bin/env python3
"""
Quick test with a topic that will return real papers
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.research.simple_enhanced_research_manager import simple_research_manager

async def test_real_topic():
    """Test with a topic that should return real papers"""
    
    print("🔍 Testing with a real academic topic")
    print("=" * 50)
    
    # Use a very common topic that definitely has papers
    query = "machine learning"
    
    print(f"Testing query: '{query}'")
    
    results = await simple_research_manager.search_all_portals(query, 10)
    
    print(f"Found {len(results)} papers!")
    
    if results:
        print("\nFirst few papers:")
        for i, paper in enumerate(results[:3], 1):
            print(f"\n{i}. {paper.get('title', 'No title')}")
            print(f"   Authors: {paper.get('authors', 'Unknown')}")
            print(f"   Year: {paper.get('year', 'Unknown')}")
            print(f"   Source: {paper.get('source', 'Unknown')}")
            if paper.get('abstract'):
                print(f"   Abstract: {paper.get('abstract', '')[:100]}...")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_real_topic())
