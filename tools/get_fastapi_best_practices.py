#!/usr/bin/env python3
"""
Get FastAPI best practices and structure improvements from Context7
"""

import asyncio
import json
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client


async def get_fastapi_guidance():
    """Get comprehensive FastAPI best practices from Context7"""
    
    server_params = StdioServerParameters(
        command="npx",
        args=["@upstash/context7-mcp", "context7"],
        env={"UPSTASH_REDIS_REST_URL": "https://usw1-polite-beetle-43014.upstash.io", 
             "UPSTASH_REDIS_REST_TOKEN": "AcaAACQjMGZmOTQxNWQtOGM3Zi00NWEzLWJkNTMtZDVmZjEzODJiYzQ0YzBlNzVjOGIzNTBmNDZkOWE4ZDc4MGQ0MWEyZQ=="}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
              # First, resolve FastAPI library ID
            print("🔍 Resolving FastAPI library ID...")
            result = await session.call_tool("resolve-library-id", {"libraryName": "fastapi best practices"})
            print("📋 FastAPI Libraries Found:")
            print(result.content[0].text[:500] + "...\n")
            
            # Get FastAPI main library documentation
            print("📚 Getting FastAPI best practices documentation...")
            topics = [
                "fastapi application structure lifespan startup shutdown",
                "fastapi middleware error handling cors",
                "fastapi dependency injection dependency provider",
                "fastapi router organization modular architecture",
                "fastapi exception handlers validation errors",
                "fastapi configuration settings environment variables",
                "fastapi logging structured logging",
                "fastapi database connection pooling sqlalchemy",
                "fastapi authentication authorization security",
                "fastapi testing pytest fixtures"
            ]
            
            all_docs = []
            for topic in topics:
                print(f"  📖 Getting: {topic}")
                try:
                    docs = await session.call_tool("get-library-docs", {
                        "library_id": "/tiangolo/fastapi",
                        "query": topic
                    })
                    all_docs.append({
                        "topic": topic,
                        "content": docs.content[0].text
                    })
                    print(f"  ✅ Retrieved {len(docs.content[0].text)} characters")
                except Exception as e:
                    print(f"  ❌ Error for {topic}: {e}")
            
            # Get Full Stack Template docs
            print("\n📚 Getting Full Stack FastAPI Template documentation...")
            try:
                template_docs = await session.call_tool("get-library-docs", {
                    "library_id": "/fastapi/full-stack-fastapi-template",
                    "query": "application structure backend architecture main.py setup"
                })
                all_docs.append({
                    "topic": "full-stack-template",
                    "content": template_docs.content[0].text
                })
                print(f"  ✅ Retrieved template docs: {len(template_docs.content[0].text)} characters")
            except Exception as e:
                print(f"  ❌ Error getting template docs: {e}")
            
            # Get Best Architecture practices
            print("\n📚 Getting FastAPI Best Architecture documentation...")
            try:
                arch_docs = await session.call_tool("get-library-docs", {
                    "library_id": "/fastapi-practices/fastapi_best_architecture",
                    "query": "enterprise architecture backend structure main.py application setup"
                })
                all_docs.append({
                    "topic": "best-architecture",
                    "content": arch_docs.content[0].text
                })
                print(f"  ✅ Retrieved architecture docs: {len(arch_docs.content[0].text)} characters")
            except Exception as e:
                print(f"  ❌ Error getting architecture docs: {e}")
            
            return all_docs


async def main():
    """Main function to get and save FastAPI guidance"""
    print("🚀 Getting FastAPI Best Practices from Context7")
    print("=" * 60)
    
    try:
        docs = await get_fastapi_guidance()
        
        # Save documentation to file
        output_file = "fastapi_best_practices_context7.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved {len(docs)} documentation sections to {output_file}")
        
        # Print summary
        print("\n📊 Documentation Summary:")
        for doc in docs:
            print(f"  📁 {doc['topic']}: {len(doc['content'])} characters")
        
        print("\n🎉 FastAPI best practices retrieved successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
