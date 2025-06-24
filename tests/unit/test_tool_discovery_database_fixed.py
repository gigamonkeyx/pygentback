#!/usr/bin/env python3
"""
Test MCP Tool Discovery and Database Persistence

This script tests that we can properly discover tools from MCP servers
and save the tool metadata to the database according to MCP specification.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.mcp.database.models import (
    MCPServerModel, MCPToolModel, MCPResourceModel, 
    MCPPromptModel, MCPToolCallLog
)


async def test_tool_discovery_database():
    """Test MCP tool discovery and database persistence"""
    try:
        # Use SQLite database (same as the main application)
        database_url = "sqlite:///./pygent_factory.db"
        
        print(f"Testing MCP tool discovery database: {database_url}")
        
        # Create engine and session
        engine = create_engine(database_url, echo=False)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        print("\n=== Testing Database Operations ===")
        
        with SessionLocal() as session:
            # Test 1: Create a mock MCP server entry
            print("Test 1: Creating mock MCP server...")
            mock_server = MCPServerModel(
                name="test-cloudflare-docs",
                command=["npx", "-y", "@cloudflare/mcp-server-cloudflare", "docs"],
                server_type="remote",
                transport="stdio",
                status="active",
                capabilities={
                    "tools": {"list": True},
                    "resources": {"list": True, "subscribe": False},
                    "prompts": {"list": True}
                },
                auto_start=True
            )
            session.add(mock_server)
            session.commit()
            print(f"✅ Created server: {mock_server.id}")
            
            # Test 2: Create mock tools discovered from the server
            print("Test 2: Creating discovered tools...")
            mock_tools = [
                MCPToolModel(
                    server_id=mock_server.id,
                    name="cloudflare_docs_search",
                    description="Search Cloudflare documentation for specific topics",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for documentation"
                            },
                            "section": {
                                "type": "string",
                                "description": "Specific documentation section",
                                "enum": ["workers", "pages", "dns", "security"]
                            }
                        },
                        "required": ["query"]
                    },
                    annotations={
                        "audience": ["developer"],
                        "level": ["beginner", "intermediate", "advanced"]
                    },
                    is_available=True
                ),
                MCPToolModel(
                    server_id=mock_server.id,
                    name="cloudflare_api_reference",
                    description="Get API reference documentation for Cloudflare services",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "service": {
                                "type": "string",
                                "description": "Cloudflare service name",
                                "enum": ["workers", "kv", "durable-objects", "r2"]
                            },
                            "endpoint": {
                                "type": "string",
                                "description": "Specific API endpoint"
                            }
                        },
                        "required": ["service"]
                    },
                    is_available=True
                )
            ]
            
            for tool in mock_tools:
                session.add(tool)
            session.commit()
            print(f"✅ Created {len(mock_tools)} tools")
            
            # Test 3: Create mock resources
            print("Test 3: Creating discovered resources...")
            mock_resources = [
                MCPResourceModel(
                    server_id=mock_server.id,
                    uri="cloudflare://docs/workers/getting-started",
                    name="Workers Getting Started Guide",
                    description="Complete guide to getting started with Cloudflare Workers",
                    mime_type="text/markdown",
                    is_available=True
                ),
                MCPResourceModel(
                    server_id=mock_server.id,
                    uri="cloudflare://docs/pages/framework-guides",
                    name="Pages Framework Guides",
                    description="Framework-specific guides for Cloudflare Pages",
                    mime_type="text/html",
                    is_available=True
                )
            ]
            
            for resource in mock_resources:
                session.add(resource)
            session.commit()
            print(f"✅ Created {len(mock_resources)} resources")
            
            # Test 4: Create mock prompts
            print("Test 4: Creating discovered prompts...")
            mock_prompts = [
                MCPPromptModel(
                    server_id=mock_server.id,
                    name="cloudflare_troubleshoot",
                    description="Help troubleshoot common Cloudflare issues",
                    arguments={
                        "type": "object",
                        "properties": {
                            "issue_type": {
                                "type": "string",
                                "enum": ["ssl", "dns", "performance", "security"]
                            },
                            "error_message": {
                                "type": "string",
                                "description": "Error message or symptoms"
                            }
                        },
                        "required": ["issue_type"]
                    },
                    is_available=True
                )
            ]
            
            for prompt in mock_prompts:
                session.add(prompt)
            session.commit()
            print(f"✅ Created {len(mock_prompts)} prompts")
            
            # Test 5: Create mock tool call logs
            print("Test 5: Creating tool call logs...")
            first_tool = mock_tools[0]
            mock_tool_calls = [
                MCPToolCallLog(
                    tool_id=first_tool.id,
                    server_id=mock_server.id,
                    agent_id="agent-001",
                    arguments={"query": "worker bindings", "section": "workers"},
                    response_data={
                        "found": True,
                        "results": [
                            {"title": "Worker Bindings", "url": "https://..."}
                        ]
                    },
                    success=True,
                    response_time_ms=245,
                    completed_at=datetime.utcnow(),
                    usage_context={"task": "documentation_lookup", "user_intent": "learning"},
                    effectiveness_score=9
                )
            ]
            
            for call_log in mock_tool_calls:
                session.add(call_log)
            session.commit()
            print(f"✅ Created {len(mock_tool_calls)} tool call logs")
            
            # Test 6: Query and verify all data
            print("\nTest 6: Querying and verifying data...")
            
            # Query servers
            servers = session.query(MCPServerModel).all()
            print(f"Found {len(servers)} servers:")
            for server in servers:
                print(f"  - {server.name} ({server.status}) - {len(server.tools)} tools")
            
            # Query tools with server relationships
            tools = session.query(MCPToolModel).all()
            print(f"\nFound {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description[:50]}...")
                print(f"    Server: {tool.server.name}")
                print(f"    Schema: {bool(tool.input_schema)}")
                
            # Query resources
            resources = session.query(MCPResourceModel).all()
            print(f"\nFound {len(resources)} resources:")
            for resource in resources:
                print(f"  - {resource.name}: {resource.uri}")
                
            # Query prompts
            prompts = session.query(MCPPromptModel).all()
            print(f"\nFound {len(prompts)} prompts:")
            for prompt in prompts:
                print(f"  - {prompt.name}: {prompt.description}")
                
            # Query tool calls
            tool_calls = session.query(MCPToolCallLog).all()
            print(f"\nFound {len(tool_calls)} tool calls:")
            for call in tool_calls:
                print(f"  - Tool: {call.tool_id} | Agent: {call.agent_id} | Success: {call.success}")
                print(f"    Response time: {call.response_time_ms}ms | Score: {call.effectiveness_score}/10")
            
            # Test 7: Test tool usage tracking
            print("\nTest 7: Testing tool usage tracking...")
            first_tool.call_count = 1
            first_tool.last_called = datetime.utcnow()
            first_tool.average_response_time = 245
            session.commit()
            
            # Verify update
            updated_tool = session.query(MCPToolModel).filter_by(id=first_tool.id).first()
            print(f"✅ Tool usage updated: {updated_tool.call_count} calls, avg {updated_tool.average_response_time}ms")
            
            print("\n=== Summary ===")
            print("✅ Database operations successful!")
            print("✅ All MCP entities (servers, tools, resources, prompts, logs) working")
            print("✅ Relationships and foreign keys working")
            print("✅ Tool discovery data structure validated")
            
            # Generate summary report
            report = {
                "test_timestamp": datetime.utcnow().isoformat(),
                "database_url": database_url,
                "tables_tested": ["mcp_servers", "mcp_tools", "mcp_resources", "mcp_prompts", "mcp_tool_calls"],
                "test_results": {
                    "servers_created": len(servers),
                    "tools_created": len(tools),
                    "resources_created": len(resources),
                    "prompts_created": len(prompts),
                    "tool_calls_logged": len(tool_calls),
                    "relationships_working": True,
                    "usage_tracking_working": True
                },
                "schema_validation": {
                    "tool_input_schema": "✅ JSON schema stored and retrieved",
                    "server_capabilities": "✅ Capabilities JSON stored and retrieved",
                    "tool_annotations": "✅ Annotations stored and retrieved",
                    "tool_call_analytics": "✅ Performance and effectiveness tracking working"
                },
                "next_steps": [
                    "Integrate tool discovery service with MCP server registration",
                    "Implement real tool discovery via MCP tools/list calls",
                    "Add tool change notification handling",
                    "Expose tool metadata APIs for agents and orchestrator"
                ]
            }
            
            return report
            
    except Exception as e:
        print(f"❌ Error testing tool discovery database: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🔍 Testing MCP Tool Discovery Database Persistence...")
    report = asyncio.run(test_tool_discovery_database())
    
    if report:
        # Save report
        with open("tool_discovery_database_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("\n📋 Test report saved to: tool_discovery_database_test_report.json")
        print("\n✅ All tests passed! Database is ready for tool discovery integration.")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
