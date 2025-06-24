#!/usr/bin/env python3
"""
Debug Internet Archive API calls
"""

import asyncio
import aiohttp
import logging
from src.orchestration.research_models import ResearchQuery
from src.orchestration.internet_archive_integration import InternetArchiveAPI

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_internet_archive():
    """Test Internet Archive API calls"""
    print("🔍 Testing Internet Archive API...")
      # Create a simple research query
    query = ResearchQuery(
        query_id="test-123",
        topic="American Civil War",
        domain="historical",
        keywords=["civil", "war", "american", "1861", "1865"],
        metadata={"geographic_scope": ["United States"]}
    )
    
    # Test the API
    async with InternetArchiveAPI() as ia:
        # Build search params to see what we're sending
        params = ia._build_search_params(query)
        print(f"📋 Search parameters: {params}")
        
        # Try a manual API call to debug
        print(f"🌐 API URL: {ia.search_api_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(ia.search_api_url, params=params) as response:
                    print(f"📊 Response status: {response.status}")
                    print(f"📝 Response headers: {dict(response.headers)}")
                    
                    if response.status != 200:
                        text = await response.text()
                        print(f"❌ Error response: {text[:500]}...")
                    else:
                        data = await response.json()
                        print(f"✅ Success! Found {len(data.get('response', {}).get('docs', []))} documents")
                        
                        # Show first result if any
                        docs = data.get('response', {}).get('docs', [])
                        if docs:
                            first_doc = docs[0]
                            print(f"📚 First result: {first_doc.get('title', 'No title')}")
                            
        except Exception as e:
            print(f"💥 Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_internet_archive())
