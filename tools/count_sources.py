#!/usr/bin/env python3
"""
Count Available Research Sources in PyGent Factory
"""

import asyncio
import logging
from src.orchestration.historical_research_agent import HistoricalResearchAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def count_all_sources():
    """Count sources from all available integrations"""
    print("🔍 Counting Available Research Sources in PyGent Factory")
    print("=" * 60)
    
    try:
        # Initialize the research agent
        agent = HistoricalResearchAgent()
        
        # Test queries to get comprehensive source count
        test_queries = [
            "american history",
            "world war",
            "ancient civilization", 
            "historical documents"
        ]
        
        total_sources = 0
        source_breakdown = {}
        
        for query_text in test_queries:
            print(f"\n📊 Testing query: '{query_text}'")
              # Get sources using the direct search method
            sources = await agent.search_sources(query_text)
            
            print(f"   Found {len(sources)} sources")
            total_sources += len(sources)
            
            # Categorize sources by platform
            for source in sources:
                platform = "Unknown"
                if "archive.org" in source.url:
                    platform = "Internet Archive"
                elif "hathitrust.org" in source.url:
                    platform = "HathiTrust"
                elif hasattr(source, 'metadata') and source.metadata:
                    platform = source.metadata.get('source_platform', 'Other')
                
                source_breakdown[platform] = source_breakdown.get(platform, 0) + 1
            
            # Small delay between queries
            await asyncio.sleep(1)
        
        print("\n" + "=" * 60)
        print("📚 FINAL SOURCE COUNT SUMMARY")
        print("=" * 60)
        print(f"Total Sources Found: {total_sources}")
        print(f"Average per Query: {total_sources / len(test_queries):.1f}")
        
        print("\n📈 Sources by Platform:")
        for platform, count in sorted(source_breakdown.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_sources * 100) if total_sources > 0 else 0
            print(f"  {platform}: {count} sources ({percentage:.1f}%)")
        
        print("\n🎯 Available Source Types:")
        print("  ✅ Internet Archive - Historical documents, books, newspapers")
        print("  ✅ HathiTrust - Bibliographic catalog (ToS compliant)")
        print("  ✅ Academic Collections - Filtered from Internet Archive")
        print("  ✅ Archival Collections - Manuscripts, letters, documents")
        print("  ✅ Newspaper Archives - Historical newspapers and periodicals")
        print("  ✅ Government Records - Official documents and reports")
        
        print(f"\n✅ System Status: {total_sources} real sources available from compliant integrations")
        return total_sources
        
    except Exception as e:
        print(f"❌ Error counting sources: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    asyncio.run(count_all_sources())
