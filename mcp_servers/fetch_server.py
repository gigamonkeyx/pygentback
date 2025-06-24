#!/usr/bin/env python3
"""
MCP Fetch Server - HTTP request capabilities
Provides web fetching and API calling functionality
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
import aiohttp
import argparse
from urllib.parse import urlparse, urljoin


class MCPFetchServer:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize the HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'MCP-Fetch-Server/1.0'}
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def fetch_url(self, url: str, method: str = 'GET', headers: Dict[str, str] = None, data: Any = None) -> Dict[str, Any]:
        """Fetch data from a URL"""
        if not self.session:
            await self.initialize()
            
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                url = f"https://{url}"
                
            kwargs = {
                'method': method.upper(),
                'url': url,
                'headers': headers or {}
            }
            
            if data:
                if isinstance(data, (dict, list)):
                    kwargs['json'] = data
                else:
                    kwargs['data'] = data
                    
            async with self.session.request(**kwargs) as response:
                content_type = response.headers.get('content-type', '').lower()
                
                if 'application/json' in content_type:
                    content = await response.json()
                else:
                    content = await response.text()
                    
                return {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'content': content,
                    'url': str(response.url),
                    'content_type': content_type
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'status': 0,
                'content': None
            }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""
        method = request.get('method')
        
        if method == 'tools/list':
            return {
                'tools': [
                    {
                        'name': 'fetch',
                        'description': 'Fetch data from a URL with HTTP methods',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'url': {'type': 'string', 'description': 'URL to fetch'},
                                'method': {'type': 'string', 'default': 'GET', 'description': 'HTTP method'},
                                'headers': {'type': 'object', 'description': 'HTTP headers'},
                                'data': {'description': 'Request body data'}
                            },
                            'required': ['url']
                        }
                    }
                ]
            }
        elif method == 'tools/call':
            tool_name = request.get('params', {}).get('name')
            arguments = request.get('params', {}).get('arguments', {})
            
            if tool_name == 'fetch':
                result = await self.fetch_url(
                    url=arguments.get('url'),
                    method=arguments.get('method', 'GET'),
                    headers=arguments.get('headers'),
                    data=arguments.get('data')
                )
                return {'content': [{'type': 'text', 'text': json.dumps(result, indent=2)}]}
        
        return {'error': f'Unknown method: {method}'}


async def main():
    server = MCPFetchServer()
    
    try:
        # Read from stdin and write to stdout for MCP protocol
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                    
                request = json.loads(line.strip())
                response = await server.handle_request(request)
                
                print(json.dumps(response))
                sys.stdout.flush()
                
            except json.JSONDecodeError:
                continue
            except KeyboardInterrupt:
                break
                
    finally:
        await server.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
