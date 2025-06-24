#!/usr/bin/env python3
"""
Enhanced Google Scholar search with improved anti-detection techniques
"""

import requests
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urlencode, quote_plus
import json
import logging

logger = logging.getLogger(__name__)

class EnhancedGoogleScholarSearch:
    """Enhanced Google Scholar search with better anti-detection"""
    
    def __init__(self):
        self.session = requests.Session()
        self.last_request_time = 0
        self.request_count = 0
        
        # Rotate between different user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15'
        ]
    
    def _get_headers(self):
        """Get headers with rotating user agent"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting with jitter"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Minimum delay between requests (5-15 seconds with jitter)
        min_delay = random.uniform(5.0, 15.0)
        
        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def search(self, query, num_results=5, start_year=None, end_year=None, author=None):
        """
        Enhanced search with better parameters and error handling
        """
        try:
            self._enforce_rate_limit()
            
            # Build search URL with proper encoding
            base_url = "https://scholar.google.com/scholar"
            params = {
                'q': query,
                'hl': 'en',
                'as_sdt': '0,5',  # Include patents and citations
                'num': min(num_results, 10)  # Limit to 10 max
            }
            
            # Add year range if specified
            if start_year:
                params['as_ylo'] = start_year
            if end_year:
                params['as_yhi'] = end_year
            
            # Add author if specified
            if author:
                params['as_sauth'] = author
            
            url = f"{base_url}?{urlencode(params)}"
            logger.info(f"Searching: {url}")
            
            # Make request with enhanced headers
            headers = self._get_headers()
            response = self.session.get(url, headers=headers, timeout=30)
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response length: {len(response.text)}")
            
            # Check for common blocking indicators
            if response.status_code == 429:
                logger.warning("Rate limited (429)")
                return []
            
            if response.status_code != 200:
                logger.error(f"Non-200 status code: {response.status_code}")
                return []
            
            # Check response content for blocking/CAPTCHA
            response_text = response.text.lower()
            blocking_indicators = [
                'blocked', 'captcha', 'unusual traffic', 'automated queries',
                'verify you are human', 'please complete the security check'
            ]
            
            for indicator in blocking_indicators:
                if indicator in response_text:
                    logger.warning(f"Blocking detected: '{indicator}' found in response")
                    return []
            
            # Parse results
            results = self._parse_results(response.text)
            logger.info(f"Parsed {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def _parse_results(self, html_content):
        """Parse search results from HTML"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Debug: Check what elements we can find
            all_divs = soup.find_all('div')
            gs_classes = set()
            for div in all_divs:
                classes = div.get('class', [])
                for cls in classes:
                    if cls.startswith('gs_'):
                        gs_classes.add(cls)
            
            logger.info(f"Found GS classes: {sorted(gs_classes)}")
            
            # Try multiple selectors for result containers
            result_selectors = [
                'div.gs_ri',           # Standard result item
                'div.gs_r',            # Alternative result item
                'div[data-lid]',       # Data-driven result item
                '.gs_ri',              # Class-based selector
                '[id^="gs_res_ccl_"]'  # ID-based selector
            ]
            
            for selector in result_selectors:
                items = soup.select(selector)
                if items:
                    logger.info(f"Found {len(items)} items with selector: {selector}")
                    
                    for item in items[:10]:  # Limit to first 10 items
                        try:
                            result = self._parse_single_result(item)
                            if result and result.get('title'):
                                results.append(result)
                        except Exception as e:
                            logger.warning(f"Failed to parse result item: {e}")
                            continue
                    
                    break  # Use first successful selector
            
            # If no results found, try alternative parsing
            if not results:
                logger.info("No results with standard selectors, trying alternative parsing")
                results = self._parse_alternative(soup)
            
        except Exception as e:
            logger.error(f"Failed to parse HTML: {str(e)}")
        
        return results
    
    def _parse_single_result(self, item):
        """Parse a single result item"""
        result = {}
        
        try:
            # Extract title
            title_selectors = ['h3.gs_rt', 'h3 a', '.gs_rt a', 'h3']
            title_elem = None
            for selector in title_selectors:
                title_elem = item.select_one(selector)
                if title_elem:
                    break
            
            if title_elem:
                result['title'] = title_elem.get_text().strip()
                
                # Extract URL
                link = title_elem.find('a') if title_elem.name != 'a' else title_elem
                if link and link.get('href'):
                    result['url'] = link['href']
            
            # Extract authors and publication info
            author_selectors = ['.gs_a', 'div.gs_a']
            for selector in author_selectors:
                authors_elem = item.select_one(selector)
                if authors_elem:
                    result['authors'] = authors_elem.get_text().strip()
                    break
            
            # Extract abstract/snippet
            abstract_selectors = ['.gs_rs', 'div.gs_rs', '.gs_fl']
            for selector in abstract_selectors:
                abstract_elem = item.select_one(selector)
                if abstract_elem:
                    result['abstract'] = abstract_elem.get_text().strip()
                    break
            
            # Extract citation count
            citation_selectors = ['.gs_fl a', 'a[href*="cites"]']
            for selector in citation_selectors:
                citation_elem = item.select_one(selector)
                if citation_elem and 'cited by' in citation_elem.get_text().lower():
                    citation_text = citation_elem.get_text()
                    # Extract number from "Cited by X"
                    import re
                    match = re.search(r'cited by (\d+)', citation_text.lower())
                    if match:
                        result['citations'] = int(match.group(1))
                    break
            
        except Exception as e:
            logger.warning(f"Error parsing result item: {e}")
        
        return result
    
    def _parse_alternative(self, soup):
        """Alternative parsing method if standard selectors fail"""
        results = []
        
        try:
            # Look for any links that might be paper titles
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link.get('href', '')
                text = link.get_text().strip()
                
                # Filter for what looks like academic paper links
                if (len(text) > 20 and 
                    not href.startswith('javascript:') and
                    not href.startswith('#') and
                    'scholar.google' not in href and
                    any(word in text.lower() for word in ['learning', 'algorithm', 'method', 'analysis', 'study', 'research'])):
                    
                    result = {
                        'title': text,
                        'url': href,
                        'authors': 'Unknown',
                        'abstract': '',
                        'source': 'alternative_parse'
                    }
                    results.append(result)
                    
                    if len(results) >= 5:  # Limit fallback results
                        break
        
        except Exception as e:
            logger.warning(f"Alternative parsing failed: {e}")
        
        return results

def test_enhanced_search():
    """Test the enhanced search functionality"""
    print("Testing Enhanced Google Scholar Search")
    print("=" * 50)
    
    searcher = EnhancedGoogleScholarSearch()
    
    # Test queries
    test_queries = [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural networks"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        print("-" * 30)
        
        results = searcher.search(query, num_results=3)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.get('title', 'No title')}")
                if result.get('authors'):
                    print(f"   Authors: {result['authors'][:100]}...")
                if result.get('url'):
                    print(f"   URL: {result['url']}")
                if result.get('abstract'):
                    print(f"   Abstract: {result['abstract'][:150]}...")
                if result.get('citations'):
                    print(f"   Citations: {result['citations']}")
        else:
            print("No results found")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_enhanced_search()
