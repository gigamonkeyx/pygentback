"""
Setup Cloudflare MCP Server

Sets up the official Cloudflare MCP server for PyGent Factory integration.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from config.settings import Settings
from mcp.server.manager import MCPServerManager
from mcp.cloudflare_mcp_integration import setup_cloudflare_mcp_server, get_cloudflare_mcp_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Set up Cloudflare MCP server for PyGent Factory."""
    logger.info("🚀 Setting up Cloudflare MCP Server for PyGent Factory")
    logger.info("="*60)
    
    try:
        # Initialize settings and MCP manager
        settings = Settings()
        mcp_manager = MCPServerManager(settings)
        
        # Initialize MCP manager
        await mcp_manager.initialize()
        
        # Check if Cloudflare API token is available
        cloudflare_token = os.getenv('CLOUDFLARE_API_TOKEN')
        if cloudflare_token:
            logger.info("✅ Cloudflare API token found in environment")
        else:
            logger.warning("⚠️ No Cloudflare API token found in environment")
            logger.info("   Set CLOUDFLARE_API_TOKEN environment variable for full functionality")
            logger.info("   Server will still be registered but may have limited functionality")
        
        # Set up Cloudflare MCP server
        logger.info("\n🔧 Setting up Cloudflare MCP server...")
        setup_results = await setup_cloudflare_mcp_server(mcp_manager, cloudflare_token)
        
        # Display results
        logger.info("\n📊 Setup Results:")
        logger.info(f"   Installation Check: {'✅' if setup_results.get('installation_check') else '❌'}")
        logger.info(f"   Registration: {'✅' if setup_results.get('registration') else '❌'}")
        logger.info(f"   Startup: {'✅' if setup_results.get('startup') else '❌'}")
        logger.info(f"   Functionality Test: {'✅' if setup_results.get('functionality_test') else '❌'}")
        logger.info(f"   Ready for Integration: {'✅' if setup_results.get('ready_for_integration') else '❌'}")
        
        if setup_results.get('error'):
            logger.error(f"   Error: {setup_results['error']}")
        
        # Get server info
        logger.info("\n📋 Server Information:")
        server_info = await get_cloudflare_mcp_info(mcp_manager)
        
        logger.info(f"   Server ID: {server_info.get('server_id', 'unknown')}")
        logger.info(f"   Server Name: {server_info.get('server_name', 'unknown')}")
        logger.info(f"   Status: {server_info.get('status', 'unknown')}")
        logger.info(f"   Installed: {'✅' if server_info.get('installed') else '❌'}")
        logger.info(f"   Requires Auth: {'✅' if server_info.get('requires_auth') else '❌'}")
        
        if server_info.get('capabilities'):
            logger.info("   Capabilities:")
            for capability in server_info['capabilities']:
                logger.info(f"     • {capability}")
        
        if server_info.get('last_error'):
            logger.error(f"   Last Error: {server_info['last_error']}")
        
        # Test MCP manager functionality
        logger.info("\n🧪 Testing MCP Manager Integration...")
        
        # List all servers
        servers = await mcp_manager.list_servers()
        logger.info(f"   Total MCP servers registered: {len(servers)}")
        
        # Find Cloudflare server
        cloudflare_servers = [s for s in servers if 'cloudflare' in s.get('config', {}).get('name', '').lower()]
        if cloudflare_servers:
            logger.info("   ✅ Cloudflare MCP server found in registry")
            cloudflare_server = cloudflare_servers[0]
            logger.info(f"   Server Status: {cloudflare_server.get('status', 'unknown')}")
        else:
            logger.error("   ❌ Cloudflare MCP server not found in registry")
        
        # Final assessment
        logger.info("\n" + "="*60)
        if setup_results.get('ready_for_integration'):
            logger.info("🎉 CLOUDFLARE MCP SERVER SETUP: SUCCESS!")
            logger.info("✅ Server installed and registered")
            logger.info("✅ Ready for PyGent Factory integration")
            logger.info("✅ Available for tunnel and deployment operations")
            
            if not cloudflare_token:
                logger.info("\n📋 Next Steps:")
                logger.info("1. Set CLOUDFLARE_API_TOKEN environment variable")
                logger.info("2. Restart the server for full functionality")
                logger.info("3. Test tunnel creation and DNS operations")
            else:
                logger.info("\n🚀 Ready for Phase 2: UI Integration Planning!")
                
        else:
            logger.error("❌ CLOUDFLARE MCP SERVER SETUP: INCOMPLETE")
            logger.error("🔧 Check the errors above and retry setup")
        
        logger.info("="*60)
        
        # Cleanup
        await mcp_manager.shutdown()
        
        return setup_results.get('ready_for_integration', False)
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)