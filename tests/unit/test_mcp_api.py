#!/usr/bin/env python3
"""
Test MCP servers via API
"""

import requests
import json


def test_mcp_api():
    """Test MCP server endpoints via API"""
    try:
        # Test MCP server endpoints
        response = requests.get('https://timpayne.net/api/v1/mcp/servers', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f'Found {len(data)} MCP servers:')
            for server in data:
                print(f'- {server["name"]}: {server.get("status", "unknown")}')
                print(f'  ID: {server.get("id", "N/A")}')
                print(f'  Type: {server.get("type", "N/A")}')
                if "tools" in server:
                    print(f'  Tools: {len(server["tools"])}')
                print()
        else:
            print(f'MCP servers endpoint returned: {response.status_code}')
            print(response.text[:500])
    except Exception as e:
        print(f'Error accessing MCP servers: {e}')


if __name__ == "__main__":
    test_mcp_api()
