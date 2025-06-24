#!/usr/bin/env python3
"""
Live MCP Server Tool Investigation
Connects to running MCP servers and analyzes their tool capabilities
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def investigate_live_tools():
    """Investigate tools from live running MCP servers"""
    
    print("=" * 80)
    print("LIVE MCP SERVER TOOL INVESTIGATION")
    print("=" * 80)
    
    try:
        # Start backend first to get live servers
        print("\n1. CONNECTING TO LIVE MCP SERVERS:")
        print("-" * 50)
        
        # Import and wait for backend
        import time
        time.sleep(2)  # Give backend time to start
        
        from src.mcp.server.manager import MCPServerManager
        from src.config.settings import Settings
        
        settings = Settings()
        manager = MCPServerManager(settings)
        await manager.initialize()
        print("✓ Connected to MCP Manager")
        
        # Get all active servers
        servers = await manager.list_servers()
        print(f"Found {len(servers)} registered servers")
        
        if not servers:
            print("❌ No servers found - trying to register test servers")
              # Load real servers from our config
            from update_mcp_servers import update_mcp_servers
            await update_mcp_servers()
            
            # Check again
            servers = await manager.list_servers()
            print(f"After registration: {len(servers)} servers")
        
        # 2. Deep dive into each server's tool capabilities
        print("\n2. ANALYZING TOOL CAPABILITIES PER SERVER:")
        print("-" * 50)
        
        total_tools = 0
        server_tool_details = {}
        
        for server in servers:
            server_name = server.get('name', 'unknown')
            server_id = server.get('id', 'unknown')
            server_status = server.get('status', 'unknown')
            
            print(f"\n--- SERVER: {server_name} ({server_id}) ---")
            print(f"Status: {server_status}")
            
            # Check server capabilities
            capabilities = server.get('capabilities', {})
            print(f"Capabilities: {list(capabilities.keys())}")
            
            # Look for tools specifically
            if 'tools' in capabilities:
                tools_cap = capabilities['tools']
                print(f"Tools capability config: {tools_cap}")
            
            # Get tools from the server
            tools = server.get('tools', [])
            print(f"Registered tools: {len(tools)}")
            total_tools += len(tools)
            
            server_tool_details[server_name] = {
                'id': server_id,
                'status': server_status,
                'capabilities': capabilities,
                'tool_count': len(tools),
                'tools': []
            }
            
            # Analyze each tool in detail
            for i, tool in enumerate(tools):
                tool_name = tool.get('name', f'tool_{i}')
                tool_desc = tool.get('description', 'No description')
                tool_schema = tool.get('inputSchema', {})
                tool_annotations = tool.get('annotations', {})
                
                print(f"  Tool {i+1}: {tool_name}")
                print(f"    Description: {tool_desc[:100]}...")
                print(f"    Input Schema Type: {tool_schema.get('type', 'unknown')}")
                
                # Check schema properties
                if 'properties' in tool_schema:
                    props = tool_schema['properties']
                    print(f"    Parameters: {list(props.keys())}")
                
                # Check annotations for agent guidance
                if tool_annotations:
                    print(f"    Annotations: {tool_annotations}")
                    
                    # Check important annotations for agent behavior
                    if tool_annotations.get('readOnlyHint'):
                        print(f"      🔒 Read-only tool")
                    if tool_annotations.get('destructiveHint'):
                        print(f"      ⚠️  Potentially destructive")
                    if tool_annotations.get('openWorldHint'):
                        print(f"      🌐 Interacts with external world")
                
                # Store tool details
                server_tool_details[server_name]['tools'].append({
                    'name': tool_name,
                    'description': tool_desc,
                    'schema': tool_schema,
                    'annotations': tool_annotations
                })
        
        # 3. Test tool discovery through manager
        print("\n3. TESTING TOOL DISCOVERY MECHANISMS:")
        print("-" * 50)
        
        # Test finding servers by capability
        try:
            tool_servers = await manager.get_servers_by_capability("tools")
            print(f"Servers advertising tools capability: {len(tool_servers)}")
        except Exception as e:
            print(f"❌ get_servers_by_capability failed: {e}")
        
        # Test finding specific tools
        test_tools = ["read_file", "write_file", "search", "resolve-library-id", "get-library-docs"]
        for tool_name in test_tools:
            try:
                tool_server = await manager.find_tool_server(tool_name)
                if tool_server:
                    print(f"✓ Found '{tool_name}' in server: {tool_server}")
                else:
                    print(f"❌ Tool '{tool_name}' not found")
            except Exception as e:
                print(f"❌ Error finding '{tool_name}': {e}")
        
        # 4. Analyze capability advertising for agents
        print("\n4. CAPABILITY ADVERTISING ANALYSIS:")
        print("-" * 50)
        
        print(f"Total tools across all servers: {total_tools}")
        
        # Check if tools are properly categorized
        tool_categories = {}
        for server_name, details in server_tool_details.items():
            for tool in details['tools']:
                annotations = tool.get('annotations', {})
                
                # Categorize by behavior
                if annotations.get('readOnlyHint'):
                    category = 'read-only'
                elif annotations.get('destructiveHint'):
                    category = 'destructive'
                else:
                    category = 'standard'
                
                if category not in tool_categories:
                    tool_categories[category] = []
                tool_categories[category].append(f"{server_name}:{tool['name']}")
        
        print("Tool categorization:")
        for category, tools in tool_categories.items():
            print(f"  {category}: {len(tools)} tools")
            for tool in tools[:3]:  # Show first 3
                print(f"    - {tool}")
            if len(tools) > 3:
                print(f"    ... and {len(tools) - 3} more")
        
        # 5. Check storage and persistence issues
        print("\n5. STORAGE AND PERSISTENCE ANALYSIS:")
        print("-" * 50)
        
        # Check if tool metadata is properly stored
        storage_issues = []
        
        for server_name, details in server_tool_details.items():
            if details['tool_count'] == 0:
                storage_issues.append(f"Server {server_name} has no tools registered")
            
            for tool in details['tools']:
                if not tool.get('description'):
                    storage_issues.append(f"Tool {tool['name']} lacks description")
                
                if not tool.get('schema'):
                    storage_issues.append(f"Tool {tool['name']} lacks input schema")
                
                # Check if schema is complete
                schema = tool.get('schema', {})
                if schema.get('type') == 'object' and 'properties' not in schema:
                    storage_issues.append(f"Tool {tool['name']} has incomplete schema")
        
        if storage_issues:
            print("Storage/metadata issues found:")
            for issue in storage_issues[:10]:  # Show first 10
                print(f"  ❌ {issue}")
            if len(storage_issues) > 10:
                print(f"  ... and {len(storage_issues) - 10} more issues")
        else:
            print("✓ No major storage issues detected")
        
        # 6. Generate recommendations
        print("\n6. RECOMMENDATIONS FOR MAXIMUM AGENT USABILITY:")
        print("-" * 50)
        
        recommendations = []
        
        if total_tools == 0:
            recommendations.append("Register and start MCP servers with working tools")
        
        if not any(s.get('capabilities', {}).get('tools') for s in servers):
            recommendations.append("Ensure servers properly advertise tools capability")
        
        # Check for missing annotations
        tools_without_annotations = sum(
            1 for details in server_tool_details.values()
            for tool in details['tools']
            if not tool.get('annotations')
        )
        
        if tools_without_annotations > 0:
            recommendations.append(f"Add annotations to {tools_without_annotations} tools for better agent guidance")
        
        # Check for proper schema definitions
        tools_without_schemas = sum(
            1 for details in server_tool_details.values()
            for tool in details['tools']
            if not tool.get('schema') or not tool['schema'].get('properties')
        )
        
        if tools_without_schemas > 0:
            recommendations.append(f"Improve input schemas for {tools_without_schemas} tools")
        
        if not recommendations:
            recommendations.append("Tool capabilities appear well-configured for agent use")
        
        print("Priority recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # 7. Save detailed analysis
        print("\n7. SAVING DETAILED ANALYSIS:")
        print("-" * 50)
        
        analysis_data = {
            'timestamp': str(asyncio.get_event_loop().time()),
            'total_servers': len(servers),
            'total_tools': total_tools,
            'server_details': server_tool_details,
            'tool_categories': tool_categories,
            'storage_issues': storage_issues,
            'recommendations': recommendations
        }
        
        with open('mcp_tool_analysis.json', 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print("✓ Analysis saved to mcp_tool_analysis.json")
        
    except Exception as e:
        print(f"ERROR: Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(investigate_live_tools())
