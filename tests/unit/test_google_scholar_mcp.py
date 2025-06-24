#!/usr/bin/env python3
"""
Test script for Google Scholar MCP Server
Tests the direct search functions to debug why we're getting 0 results
"""

import sys
import os
import asyncio
import logging

# Add the google-scholar server to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mcp_servers', 'google-scholar'))

from google_scholar_web_search import google_scholar_search, advanced_google_scholar_search

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_direct_search():
    """Test the direct Google Scholar search function"""
    print("=" * 60)
    print("Testing Direct Google Scholar Search")
    print("=" * 60)
    
    # Test with a well-known query
    queries = [
        "artificial intelligence",
        "machine learning algorithms", 
        "deep learning neural networks",
        "transformer architecture attention"
    ]
    
    for query in queries:
        print(f"\nTesting query: '{query}'")
        print("-" * 40)
        
        try:
            results = google_scholar_search(query, num_results=3)
            print(f"Number of results: {len(results)}")
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"  Title: {result.get('Title', 'N/A')}")
                    print(f"  Authors: {result.get('Authors', 'N/A')[:100]}...")
                    print(f"  URL: {result.get('URL', 'N/A')}")
                    if result.get('Abstract'):
                        print(f"  Abstract: {result.get('Abstract', 'N/A')[:150]}...")
            else:
                print("  No results returned")
                
        except Exception as e:
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()

def test_advanced_search():
    """Test the advanced Google Scholar search function"""
    print("\n\n" + "=" * 60)
    print("Testing Advanced Google Scholar Search")
    print("=" * 60)
    
    # Test advanced search with different parameters
    test_cases = [
        {
            "query": "machine learning",
            "author": None,
            "year_range": (2020, 2024),
            "description": "Recent ML papers (2020-2024)"
        },
        {
            "query": "neural networks",
            "author": "Hinton",
            "year_range": None,
            "description": "Neural networks by Hinton"
        },
        {
            "query": "attention mechanism",
            "author": None,
            "year_range": (2017, 2023),
            "description": "Attention mechanism papers (2017-2023)"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        if test_case['author']:
            print(f"Author: {test_case['author']}")
        if test_case['year_range']:
            print(f"Year range: {test_case['year_range']}")
        print("-" * 40)
        
        try:
            results = advanced_google_scholar_search(
                query=test_case['query'],
                author=test_case['author'],
                year_range=test_case['year_range'],
                num_results=2
            )
            print(f"Number of results: {len(results)}")
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"  Title: {result.get('Title', 'N/A')}")
                    print(f"  Authors: {result.get('Authors', 'N/A')[:100]}...")
                    print(f"  URL: {result.get('URL', 'N/A')}")
            else:
                print("  No results returned")
                
        except Exception as e:
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()

async def test_mcp_server_tools():
    """Test the actual MCP server tools"""
    print("\n\n" + "=" * 60)
    print("Testing MCP Server Tools")
    print("=" * 60)
    
    # Import the MCP server functions
    from google_scholar_server import search_google_scholar_key_words, search_google_scholar_advanced
    
    # Test keyword search
    print("\nTesting MCP keyword search tool:")
    print("-" * 40)
    
    try:
        results = await search_google_scholar_key_words("artificial intelligence", 3)
        print(f"MCP Keyword search results: {len(results) if isinstance(results, list) else 'Error'}")
        if isinstance(results, list) and results:
            for i, result in enumerate(results[:2], 1):
                print(f"Result {i}: {result.get('Title', result)[:100]}...")
        else:
            print(f"Results: {results}")
    except Exception as e:
        print(f"Error in MCP keyword search: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test advanced search
    print("\nTesting MCP advanced search tool:")
    print("-" * 40)
    
    try:
        results = await search_google_scholar_advanced(
            query="machine learning", 
            year_range=(2020, 2024), 
            num_results=2
        )
        print(f"MCP Advanced search results: {len(results) if isinstance(results, list) else 'Error'}")
        if isinstance(results, list) and results:
            for i, result in enumerate(results, 1):
                print(f"Result {i}: {result.get('Title', result)[:100]}...")
        else:
            print(f"Results: {results}")
    except Exception as e:
        print(f"Error in MCP advanced search: {str(e)}")
        import traceback
        traceback.print_exc()

def inspect_response():
    """Inspect the raw response from Google Scholar to debug"""
    print("\n\n" + "=" * 60)
    print("Inspecting Raw Google Scholar Response")
    print("=" * 60)
    
    import requests
    from bs4 import BeautifulSoup
    
    query = "artificial intelligence"
    search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    print(f"URL: {search_url}")
    print(f"Headers: {headers}")
    
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response Length: {len(response.text)}")
        print(f"Content Type: {response.headers.get('content-type', 'Unknown')}")
        
        # Check if we're being blocked
        if "blocked" in response.text.lower() or "captcha" in response.text.lower():
            print("WARNING: Appears to be blocked or requires CAPTCHA")
        
        # Parse and look for result elements
        soup = BeautifulSoup(response.text, 'html.parser')
        gs_ri_elements = soup.find_all('div', class_='gs_ri')
        print(f"Found {len(gs_ri_elements)} 'gs_ri' elements (expected result containers)")
        
        # Look for other potential result containers
        other_elements = soup.find_all('div', class_=lambda x: x and 'gs_' in x)
        unique_classes = set()
        for elem in other_elements:
            unique_classes.update(elem.get('class', []))
        
        gs_classes = [cls for cls in unique_classes if cls.startswith('gs_')]
        print(f"Other Google Scholar classes found: {sorted(gs_classes)}")
        
        # Save a sample of the response for manual inspection
        sample_file = "d:/mcp/pygent-factory/google_scholar_response_sample.html"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(response.text[:5000])  # First 5000 chars
        print(f"Response sample saved to: {sample_file}")
        
    except Exception as e:
        print(f"Error inspecting response: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("Google Scholar MCP Server Test Suite")
    print("====================================")
    
    # Test direct search functions
    test_direct_search()
    test_advanced_search()
    
    # Inspect raw response
    inspect_response()
    
    # Test MCP server tools
    print("\nRunning async MCP server tests...")
    asyncio.run(test_mcp_server_tools())
    
    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
