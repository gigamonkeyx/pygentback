#!/usr/bin/env python3
"""
Test HathiTrust bibliographic search functionality
"""

import asyncio
import logging
from src.orchestration.research_models import ResearchQuery
from src.orchestration.hathitrust_integration import HathiTrustBibliographicAPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_hathitrust_search():
    """Test HathiTrust bibliographic search"""
    print("🔍 Testing HathiTrust Bibliographic Search...")
    
    # Create a simple research query
    query = ResearchQuery(
        query_id="test-hathi-123",
        topic="American Civil War",
        domain="historical",
        keywords=["civil", "war", "american", "1861", "1865"],
        metadata={"test": True}
    )
    
    # Test the HathiTrust search
    async with HathiTrustBibliographicAPI() as hathi:
        try:
            print(f"📚 Searching HathiTrust catalog for: {query.topic}")
            sources = await hathi.search_bibliographic_catalog(query)
            
            print(f"✅ Found {len(sources)} sources from HathiTrust!")
            
            # Show first few results
            for i, source in enumerate(sources[:5]):
                print(f"  {i+1}. {source.title}")
                print(f"     Author: {', '.join(source.authors)}")
                print(f"     URL: {source.url}")
                print(f"     Credibility: {source.credibility_score}")
                print()
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_hathitrust_search())
