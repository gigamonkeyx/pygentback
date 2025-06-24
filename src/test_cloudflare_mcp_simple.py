"""
Simple Cloudflare MCP Server Test

Tests the Cloudflare MCP server installation and basic functionality.
"""

import asyncio
import logging
import subprocess
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_node_and_npm():
    """Check if Node.js and npm are available."""
    try:
        # Check Node.js
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Node.js: {result.stdout.strip()}")
        else:
            logger.error("❌ Node.js not found")
            return False
        
        # Check npm
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ npm: {result.stdout.strip()}")
        else:
            logger.error("❌ npm not found")
            return False
        
        # Check npx
        result = subprocess.run(['npx', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ npx: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ npx not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking Node.js/npm: {e}")
        return False


async def test_cloudflare_mcp_server():
    """Test the Cloudflare MCP server availability."""
    try:
        logger.info("🔍 Testing Cloudflare MCP server availability...")
        
        # Try to run the Cloudflare MCP server with help flag
        result = subprocess.run([
            'npx', '@cloudflare/mcp-server-cloudflare', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ Cloudflare MCP server is available via npx")
            logger.info("📋 Server help output:")
            logger.info(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            logger.error(f"❌ Cloudflare MCP server failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout testing Cloudflare MCP server")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing Cloudflare MCP server: {e}")
        return False


async def install_cloudflare_mcp_server():
    """Install the Cloudflare MCP server globally."""
    try:
        logger.info("📦 Installing Cloudflare MCP server...")
        
        result = subprocess.run([
            'npm', 'install', '-g', '@cloudflare/mcp-server-cloudflare'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            logger.info("✅ Cloudflare MCP server installed successfully")
            return True
        else:
            logger.error(f"❌ Installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Installation timeout")
        return False
    except Exception as e:
        logger.error(f"❌ Installation error: {e}")
        return False


async def create_mcp_config():
    """Create MCP configuration for Cloudflare server."""
    try:
        logger.info("📝 Creating MCP configuration...")
        
        # Create MCP config directory if it doesn't exist
        config_dir = os.path.expanduser("~/.config/mcp")
        os.makedirs(config_dir, exist_ok=True)
        
        # Create configuration for Cloudflare MCP server
        config = {
            "mcpServers": {
                "cloudflare": {
                    "command": "npx",
                    "args": ["@cloudflare/mcp-server-cloudflare"],
                    "env": {
                        "CLOUDFLARE_API_TOKEN": os.getenv("CLOUDFLARE_API_TOKEN", "")
                    }
                }
            }
        }
        
        config_file = os.path.join(config_dir, "cloudflare-server.json")
        
        import json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ MCP configuration created: {config_file}")
        
        # Check if API token is set
        if os.getenv("CLOUDFLARE_API_TOKEN"):
            logger.info("✅ Cloudflare API token found in environment")
        else:
            logger.warning("⚠️ No CLOUDFLARE_API_TOKEN environment variable set")
            logger.info("   Set this for full functionality:")
            logger.info("   $env:CLOUDFLARE_API_TOKEN = 'your_token_here'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating MCP config: {e}")
        return False


async def test_mcp_server_startup():
    """Test starting the Cloudflare MCP server."""
    try:
        logger.info("🚀 Testing MCP server startup...")
        
        # Start the server and test basic functionality
        process = subprocess.Popen([
            'npx', '@cloudflare/mcp-server-cloudflare'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for startup
        await asyncio.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("✅ Cloudflare MCP server started successfully")
            
            # Terminate the test process
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            return True
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Server failed to start: {stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing server startup: {e}")
        return False


async def main():
    """Run Cloudflare MCP server setup and testing."""
    logger.info("🚀 Cloudflare MCP Server Setup and Testing")
    logger.info("="*50)
    
    try:
        # Step 1: Check prerequisites
        logger.info("\n1️⃣ Checking Prerequisites...")
        if not await check_node_and_npm():
            logger.error("❌ Node.js/npm prerequisites not met")
            logger.info("📋 Install Node.js from: https://nodejs.org/")
            return False
        
        # Step 2: Test server availability
        logger.info("\n2️⃣ Testing Server Availability...")
        server_available = await test_cloudflare_mcp_server()
        
        if not server_available:
            logger.info("📦 Server not available, attempting installation...")
            if not await install_cloudflare_mcp_server():
                logger.error("❌ Failed to install Cloudflare MCP server")
                return False
            
            # Test again after installation
            server_available = await test_cloudflare_mcp_server()
        
        if not server_available:
            logger.error("❌ Cloudflare MCP server still not available")
            return False
        
        # Step 3: Create MCP configuration
        logger.info("\n3️⃣ Creating MCP Configuration...")
        if not await create_mcp_config():
            logger.error("❌ Failed to create MCP configuration")
            return False
        
        # Step 4: Test server startup
        logger.info("\n4️⃣ Testing Server Startup...")
        if not await test_mcp_server_startup():
            logger.error("❌ Failed to start MCP server")
            return False
        
        # Success summary
        logger.info("\n" + "="*50)
        logger.info("🎉 CLOUDFLARE MCP SERVER SETUP: SUCCESS!")
        logger.info("✅ Prerequisites met")
        logger.info("✅ Server installed and available")
        logger.info("✅ Configuration created")
        logger.info("✅ Server startup tested")
        logger.info("")
        logger.info("📋 Next Steps:")
        logger.info("1. Set CLOUDFLARE_API_TOKEN environment variable")
        logger.info("2. Integrate with PyGent Factory MCP registry")
        logger.info("3. Test tunnel creation and management")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)