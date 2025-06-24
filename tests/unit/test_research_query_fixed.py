#!/usr/bin/env python3
"""
Test Research Query Runner
Executes a research query against the existing working sources to validate functionality
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestration.historical_research_agent import HistoricalResearchAgent
from src.orchestration.research_models import ResearchQuery
from src.orchestration.coordination_models import OrchestrationConfig
from src.orchestration.hathitrust_integration import HathiTrustBibliographicAPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_hathitrust_integration():
    """Test HathiTrust integration with known volume ID"""
    print("\n🔍 Testing HathiTrust Bibliographic API Integration...")
    
    # Test with a known HathiTrust volume ID
    test_id = "mdp.39015078707601"  # A known historical document
    
    async with HathiTrustBibliographicAPI() as hathi_api:
        print(f"📚 Testing volume retrieval for ID: {test_id}")
        
        # Test volume existence
        exists = await hathi_api.verify_volume_exists(test_id)
        print(f"   Volume exists: {exists}")
        
        if exists:
            # Retrieve volume information
            source = await hathi_api.get_volume_by_id(test_id)
            if source:
                print("   ✅ Successfully retrieved:")
                print(f"      Title: {source.title}")
                print(f"      Authors: {', '.join(source.authors)}")
                print(f"      Publisher: {source.publisher}")
                print(f"      URL: {source.url}")
                print(f"      Credibility Score: {source.credibility_score}")
                return source
            else:
                print("   ❌ Failed to retrieve volume data")
        else:
            print("   ❌ Volume not found")
    
    return None


async def test_historical_research_agent():
    """Test the Historical Research Agent with a simple query"""
    print("\n🏛️ Testing Historical Research Agent...")
    
    # Create configuration
    config = OrchestrationConfig()
    
    # Create research agent
    research_agent = HistoricalResearchAgent(config)
    
    # Create a simple research query
    query = ResearchQuery(
        topic="American Civil War",
        domain="historical",
        metadata={
            "time_period": "1861-1865",
            "geographic_scope": ["United States"],
            "focus": "primary_sources",
            "research_type": "military"
        }
    )
    
    print(f"📖 Conducting research on: {query.topic}")
    print(f"   Time Period: {query.metadata.get('time_period')}")
    print(f"   Geographic Scope: {query.metadata.get('geographic_scope')}")
    
    try:
        # Conduct research
        analysis = await research_agent.conduct_historical_research(query)
        
        print("   ✅ Research completed successfully!")
        print(f"      Key Themes: {len(analysis.key_themes)} identified")
        print(f"      Events: {len(analysis.events)} documented")
        print(f"      Confidence: {analysis.confidence_metrics}")
        
        # Display some results
        if analysis.key_themes:
            print(f"      Top Theme: {analysis.key_themes[0]}")
        
        if analysis.events:
            print(f"      First Event: {analysis.events[0].title}")
        
        return analysis
        
    except Exception as e:
        print(f"   ❌ Research failed: {e}")
        logger.exception("Research agent test failed")
        return None


async def test_simplified_research_query():
    """Test a simplified research query to validate basic functionality"""
    print("\n🔬 Testing Simplified Research Query...")
    
    # Simple query that should work with available sources
    query = ResearchQuery(
        topic="World War II",
        domain="historical", 
        metadata={
            "time_period": "1939-1945",
            "focus": "overview"
        }
    )
    
    print(f"📚 Simple query: {query.topic}")
    print(f"   Domain: {query.domain}")
    print(f"   Period: {query.metadata.get('time_period')}")
      # Just log the query creation - this tests the basic models work
    print("   ✅ Query created successfully")
    print(f"   Query ID: {query.query_id}")
    print(f"   Created At: {query.created_at}")
    
    return query


async def main():
    """Main test execution"""
    print("🧪 PyGent Factory Research Query Test")
    print("=" * 50)
    print(f"🕐 Test started at: {datetime.now()}")
    
    results = {}
    
    # Test 1: Basic model functionality
    print("\n1️⃣ Testing Basic Query Models...")
    query = await test_simplified_research_query()
    results['basic_query'] = query is not None
    
    # Test 2: HathiTrust integration (if network available)
    print("\n2️⃣ Testing HathiTrust Integration...")
    try:
        hathi_result = await test_hathitrust_integration()
        results['hathitrust'] = hathi_result is not None
    except Exception as e:
        print(f"   ❌ HathiTrust test failed: {e}")
        results['hathitrust'] = False
    
    # Test 3: Full research agent (may have limitations)
    print("\n3️⃣ Testing Historical Research Agent...")
    try:
        research_result = await test_historical_research_agent()
        results['research_agent'] = research_result is not None
    except Exception as e:
        print(f"   ❌ Research agent test failed: {e}")
        results['research_agent'] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   ✅ Basic Query Models: {'PASS' if results['basic_query'] else 'FAIL'}")
    print(f"   ✅ HathiTrust Integration: {'PASS' if results['hathitrust'] else 'FAIL'}")
    print(f"   ✅ Research Agent: {'PASS' if results['research_agent'] else 'FAIL'}")
    
    success_count = sum(results.values())
    total_tests = len(results)
    print(f"\n🎯 Overall: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Research system is working.")
    elif success_count > 0:
        print("⚠️  Some tests passed. Partial functionality available.")
    else:
        print("❌ All tests failed. System needs debugging.")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
