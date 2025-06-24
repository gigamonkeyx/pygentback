#!/usr/bin/env python3
"""
Proper MCP client using the official MCP Python SDK to interact with Context7
"""

import asyncio
import json
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_context7_with_mcp_sdk():
    """Test Context7 using the official MCP Python SDK"""
    print("🚀 Testing Context7 with Official MCP Python SDK")
    print("=" * 55)
    
    # Configure Context7 server parameters
    server_params = StdioServerParameters(
        command="D:\\nodejs\\npx.cmd",
        args=["-y", "@upstash/context7-mcp"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print("✅ Connected to Context7 MCP server")
            
            # Initialize the session
            await session.initialize()
            print("✅ Session initialized")
            
            # List available tools
            print("\n📋 Listing available tools...")
            tools_result = await session.list_tools()
            
            print(f"Found {len(tools_result.tools)} tools:")
            for tool in tools_result.tools:
                print(f"  🔧 {tool.name}: {tool.description}")
            
            # Test 1: Resolve library ID for FastAPI
            print(f"\n🔍 Testing: Resolve library ID for 'fastapi'")
            try:
                result = await session.call_tool("resolve-library-id", {
                    "libraryName": "fastapi"
                })
                
                print("📝 Result:")
                for content in result.content:
                    if hasattr(content, 'text'):
                        print(f"   {content.text}")
                    
            except Exception as e:
                print(f"❌ Error resolving 'fastapi': {e}")
            
            # Test 2: Get FastAPI documentation  
            print(f"\n📚 Testing: Get FastAPI documentation")
            try:
                result = await session.call_tool("get-library-docs", {
                    "context7CompatibleLibraryID": "tiangolo/fastapi",
                    "topic": "getting started",
                    "tokens": 3000
                })
                
                print("📖 FastAPI Documentation:")
                print("-" * 50)
                for content in result.content:
                    if hasattr(content, 'text'):
                        text = content.text
                        # Show first 500 characters
                        print(text[:500] + "..." if len(text) > 500 else text)
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Error getting FastAPI docs: {e}")
            
            # Test 3: Resolve and get docs for pandas
            print(f"\n🐼 Testing: Pandas library")
            try:
                # First resolve pandas
                resolve_result = await session.call_tool("resolve-library-id", {
                    "libraryName": "pandas"
                })
                
                pandas_id = None
                for content in resolve_result.content:
                    if hasattr(content, 'text') and 'pandas' in content.text.lower():
                        # Extract the ID from the response
                        pandas_id = content.text.strip()
                        break
                
                if pandas_id:
                    print(f"📝 Pandas ID resolved: {pandas_id}")
                    
                    # Get pandas documentation
                    docs_result = await session.call_tool("get-library-docs", {
                        "context7CompatibleLibraryID": "pandas-dev/pandas",  # Common pandas ID
                        "topic": "dataframe operations",
                        "tokens": 2000
                    })
                    
                    print("📊 Pandas Documentation:")
                    print("-" * 30)
                    for content in docs_result.content:
                        if hasattr(content, 'text'):
                            text = content.text
                            print(text[:300] + "..." if len(text) > 300 else text)
                    print("-" * 30)
                    
            except Exception as e:
                print(f"❌ Error with pandas: {e}")

async def main():
    """Main function"""
    try:
        await test_context7_with_mcp_sdk()
        
        print("\n🎉 Context7 SDK Test Complete!")
        print("\n💡 Key Features Demonstrated:")
        print("  ✅ resolve-library-id: Find correct library identifiers")
        print("  ✅ get-library-docs: Get up-to-date documentation") 
        print("  ✅ Live documentation: Always current, never stale")
        print("  ✅ Code examples: Real working code snippets")
        
        print("\n🔥 In VS Code, just add 'use context7' to your prompts!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
