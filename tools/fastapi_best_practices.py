#!/usr/bin/env python3
"""
Context7 FastAPI Best Practices Client
Get up-to-date FastAPI application structure and best practices
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def get_fastapi_best_practices():
    """Get FastAPI best practices from Context7"""
    
    # Start the Context7 MCP server
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@upstash/context7-mcp@latest"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("🔍 Getting FastAPI Best Architecture documentation...")
              # Get FastAPI best architecture docs
            result = await session.call_tool(
                "get-library-docs", 
                {
                    "context7CompatibleLibraryID": "/fastapi-practices/fastapi_best_architecture",
                    "query": "application structure startup lifespan middleware"
                }
            )
            
            print("📚 FastAPI Best Architecture Patterns:")
            print("=" * 60)
            print(result.content[0].text if result.content else "No content received")
            print("\n" + "=" * 60)
            
            print("\n🔍 Getting FastAPI Full Stack Template documentation...")
              # Get FastAPI full stack template docs
            result2 = await session.call_tool(
                "get-library-docs", 
                {
                    "context7CompatibleLibraryID": "/fastapi/full-stack-fastapi-template", 
                    "query": "main.py application setup lifespan middleware structure"
                }
            )
            
            print("📚 FastAPI Full Stack Template:")
            print("=" * 60)
            print(result2.content[0].text if result2.content else "No content received")
            print("\n" + "=" * 60)
            
            print("\n🔍 Getting general FastAPI application structure...")
              # Get general FastAPI application structure
            result3 = await session.call_tool(
                "get-library-docs", 
                {
                    "context7CompatibleLibraryID": "/tiangolo/fastapi",
                    "query": "lifespan startup shutdown middleware cors application structure"
                }
            )
            
            print("📚 FastAPI Application Structure:")
            print("=" * 60)
            print(result3.content[0].text if result3.content else "No content received")

if __name__ == "__main__":
    asyncio.run(get_fastapi_best_practices())
