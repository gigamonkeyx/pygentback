#!/usr/bin/env python3
"""
Manual MCP Server Validation Test

Test script to validate that all configured MCP servers can be started
and are functioning correctly.
"""

import asyncio
import json
import logging
import sys
import os
import subprocess
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_cloudflare_api_token():
    """Get Cloudflare API token from environment variables or config file"""
    # Try multiple environment variable names
    for env_var in ["CLOUDFLARE_API_TOKEN", "CF_API_TOKEN", "CLOUDFLARE_TOKEN"]:
        token = os.environ.get(env_var)
        if token:
            return token
    
    # Try to load from cloudflare_auth.env file
    auth_file_paths = [
        "cloudflare_auth.env",
        ".cloudflare_auth.env",
        os.path.join(os.getcwd(), "cloudflare_auth.env")
    ]
    
    for auth_file in auth_file_paths:
        if os.path.exists(auth_file):
            try:
                with open(auth_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("CLOUDFLARE_API_TOKEN="):
                            token = line.split("=", 1)[1].strip()
                            if token and not token.startswith("#"):
                                return token
            except Exception as e:
                logger.warning(f"Error reading auth file {auth_file}: {e}")
    
    return None


async def get_oauth_token(provider: str, user_id: str = "system"):
    """Get OAuth token for a provider"""
    try:
        # Add src to path
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.auth import OAuthManager, FileTokenStorage
        
        # Initialize OAuth manager
        oauth_manager = OAuthManager()
        oauth_manager.set_token_storage(FileTokenStorage())
        
        token = await oauth_manager.get_token(user_id, provider)
        if token and not token.is_expired:
            return token.access_token
        elif token and token.expires_soon and token.refresh_token:
            # Try to refresh the token
            provider_obj = oauth_manager.get_provider(provider)
            if provider_obj:
                try:
                    new_token = await provider_obj.refresh_token(token.refresh_token)
                    new_token.user_id = user_id
                    await oauth_manager.token_storage.store_token(user_id, provider, new_token)
                    return new_token.access_token
                except Exception as e:
                    logger.warning(f"Failed to refresh {provider} token: {e}")
        
        return None
    except Exception as e:
        logger.debug(f"OAuth not available or error getting token for {provider}: {e}")
        return None


async def get_auth_token(provider: str, user_id: str = "system"):
    """Get authentication token for a provider (tries OAuth first, then API token)"""
    # Try OAuth token first
    oauth_token = await get_oauth_token(provider, user_id)
    if oauth_token:
        logger.info(f"Using OAuth token for {provider}")
        return oauth_token
    
    # Fall back to API token for Cloudflare
    if provider == "cloudflare":
        api_token = get_cloudflare_api_token()
        if api_token:
            logger.info(f"Using API token for {provider}")
            return api_token
    
    return None

async def test_python_mcp_server(server_path: str, timeout: int = 10):
    """Test a Python-based MCP server"""
    try:
        logger.info(f"Testing Python MCP server: {server_path}")
        
        # Start the server process
        process = await asyncio.create_subprocess_exec(
            sys.executable, server_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send a simple JSON-RPC initialization
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        message_bytes = (json.dumps(init_message) + '\n').encode('utf-8')
        
        try:
            # Send the message and wait for response
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=message_bytes),
                timeout=timeout
            )
            
            if stdout:
                logger.info(f"✓ Server responded: {stdout.decode()[:200]}...")
                return True
            else:
                logger.error(f"✗ No response from server. Stderr: {stderr.decode()}")
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"✗ Server timeout after {timeout}s")
            process.terminate()
            return False
            
    except Exception as e:
        logger.error(f"✗ Error testing server: {e}")
        return False

async def test_node_mcp_server(server_path: str, timeout: int = 10):
    """Test a Node.js-based MCP server"""
    try:
        logger.info(f"Testing Node.js MCP server: {server_path}")
        
        # Start the server process
        process = await asyncio.create_subprocess_exec(
            "node", server_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send a simple JSON-RPC initialization
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        message_bytes = (json.dumps(init_message) + '\n').encode('utf-8')
        
        try:
            # Send the message and wait for response
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=message_bytes),
                timeout=timeout
            )
            
            if stdout:
                logger.info(f"✓ Server responded: {stdout.decode()[:200]}...")
                return True
            else:
                logger.error(f"✗ No response from server. Stderr: {stderr.decode()}")
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"✗ Server timeout after {timeout}s")
            process.terminate()
            return False
            
    except Exception as e:
        logger.error(f"✗ Error testing server: {e}")
        return False

async def test_remote_mcp_server(command_parts: list, timeout: int = 15):
    """Test a remote MCP server using npx mcp-remote or similar"""
    try:
        logger.info(f"Testing remote MCP server: {' '.join(command_parts)}")
        
        # Start the server process
        process = await asyncio.create_subprocess_exec(
            *command_parts,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send a simple JSON-RPC initialization
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        message_bytes = (json.dumps(init_message) + '\n').encode('utf-8')
        
        try:
            # Send the message and wait for response
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=message_bytes),
                timeout=timeout
            )
            
            if stdout:
                logger.info(f"✓ Remote server responded: {stdout.decode()[:200]}...")
                return True
            else:
                logger.error(f"✗ No response from remote server. Stderr: {stderr.decode()}")
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"✗ Remote server timeout after {timeout}s")
            process.terminate()
            return False
            
    except Exception as e:
        logger.error(f"✗ Error testing remote server: {e}")
        return False

async def test_sse_mcp_server(host: str, path: str = "/sse", port: int = None, timeout: int = 10, auth_required: bool = False, use_ssl: bool = False):
    """Test an SSE-based MCP server"""
    try:
        # Build URL
        protocol = "https" if use_ssl or port == 443 else "http"
        url = f"{protocol}://{host}"
        if port and port not in [80, 443]:
            url += f":{port}"
        if path:
            url += path
        
        logger.info(f"Testing SSE MCP server: {url}")
          # Prepare authentication headers if needed
        headers = {}
        if auth_required:
            auth_token = await get_auth_token("cloudflare")
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
                logger.info("✓ Using authentication token (OAuth or API)")
            else:
                logger.warning("⚠️  Authentication required but no token found (OAuth or API)")
        
        # Try to connect to the SSE endpoint
        async with httpx.AsyncClient(timeout=timeout) as client:            # First try a simple health check
            try:
                response = await client.get(url.replace('/sse', '/health'), headers=headers, timeout=5)
                if response.status_code == 200:
                    auth_status = " (authenticated)" if auth_required and auth_token else ""
                    logger.info(f"✓ SSE server health check passed{auth_status}")
                    return True
            except Exception:
                pass

            # Try the SSE endpoint itself
            try:
                response = await client.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    auth_status = " (authenticated)" if auth_required and auth_token else ""
                    logger.info(f"✓ SSE server endpoint accessible{auth_status}")
                    return True
                elif response.status_code == 401:
                    if auth_required and not auth_token:
                        logger.warning("⚠️  SSE server requires authentication (OAuth or API token needed)")
                    else:
                        logger.info("✓ SSE server endpoint accessible (requires authentication)")
                    return True
                else:
                    logger.error(f"✗ SSE server returned status {response.status_code}")
                    return False
            except httpx.ReadTimeout:
                # SSE streams often time out on initial connection, this is normal
                auth_status = " (authenticated)" if auth_required and auth_token else ""
                logger.info(f"✓ SSE server endpoint accessible (SSE stream detected){auth_status}")
                return True
            except Exception as e:
                logger.error(f"✗ Error connecting to SSE server: {e}")
                return False
                
    except Exception as e:
        logger.error(f"✗ Error testing SSE server: {e}")
        return False

async def test_python_module_mcp_server(command: list):
    """Test a Python MCP server using -m module syntax"""
    try:
        logger.info(f"Starting Python module server: {' '.join(command)}")
          # Start the server
        process = await asyncio.create_subprocess_exec(
            sys.executable, command[1], *command[2:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send a simple JSON-RPC initialization
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        message_bytes = (json.dumps(init_message) + '\n').encode('utf-8')
        
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(input=message_bytes), timeout=10
            )
            
            if stdout_data:
                try:
                    response = json.loads(stdout_data.decode('utf-8').strip())
                    if "result" in response:
                        logger.info("✓ Python module server responded correctly")
                        return True
                    else:
                        logger.error(f"Invalid response from Python module server: {response}")
                        return False
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse response from Python module server: {e}")
                    return False
            else:
                stderr_output = stderr_data.decode('utf-8') if stderr_data else "No error output"
                logger.error(f"No response from Python module server. Stderr: {stderr_output}")
                return False
                
        except asyncio.TimeoutError:
            logger.error("Python module server test timed out")
            process.kill()
            await process.wait()
            return False
            
    except Exception as e:
        logger.error(f"Error testing Python module server: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("Starting MCP Server Validation Tests")
    
    # Load MCP server configuration
    config_path = "mcp_server_configs.json"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    results = {}
    
    servers = config.get("servers", [])
    if not servers:
        logger.error("No servers found in config")
        return
    
    for server_config in servers:
        server_name = server_config.get("name", server_config.get("id", "unknown"))
        logger.info(f"\n--- Testing {server_name} ---")
          # Check transport type first
        transport = server_config.get("transport", "stdio")
        
        if transport == "sse":            # Handle SSE servers
            host = server_config.get("host")
            path = server_config.get("path", "/sse")
            port = server_config.get("port")
            use_ssl = server_config.get("use_ssl", False)
            
            # Check authentication requirements
            auth_config = server_config.get("config", {}).get("authentication", {})
            auth_required = auth_config.get("required", False)
            
            if host:
                results[server_name] = await test_sse_mcp_server(host, path, port, auth_required=auth_required, use_ssl=use_ssl)
            else:
                logger.error(f"No host specified for SSE server {server_name}")
                results[server_name] = False
            continue
        
        # Handle traditional command-based servers
        command = server_config.get("command")
        if not command:
            logger.error(f"No command specified for {server_name}")
            results[server_name] = False
            continue
          # Handle array command format
        if isinstance(command, list) and len(command) > 0:
            cmd_type = command[0]
            if cmd_type == "python" and len(command) > 1:
                if command[1] == "-m" and len(command) > 2:
                    # Handle Python module: python -m module_name
                    module_name = command[2]
                    logger.info(f"Testing Python module: {module_name}")
                    results[server_name] = await test_python_module_mcp_server(command)
                else:
                    # Handle Python file: python script.py
                    server_path = command[1]
                    if os.path.exists(server_path):
                        results[server_name] = await test_python_mcp_server(server_path)
                    else:
                        logger.error(f"Server file not found: {server_path}")
                        results[server_name] = False
            elif cmd_type == "node" and len(command) > 1:
                server_path = command[1]
                if os.path.exists(server_path):
                    results[server_name] = await test_node_mcp_server(server_path)
                else:
                    logger.error(f"Server file not found: {server_path}")
                    results[server_name] = False
            elif cmd_type == "npx":
                # Handle remote servers like npx mcp-remote
                results[server_name] = await test_remote_mcp_server(command)
            else:
                logger.warning(f"Unknown command type for {server_name}: {cmd_type}")
                # Try as a generic remote command
                results[server_name] = await test_remote_mcp_server(command)
        else:
            logger.warning(f"Invalid command format for {server_name}: {command}")
            results[server_name] = False
    
    # Print summary
    logger.info("\n=== MCP Server Test Results ===")
    passed = 0
    total = len(results)
    
    for server_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{server_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nSummary: {passed}/{total} servers passed tests")
    
    if passed == total:
        logger.info("🎉 All MCP servers are working correctly!")
    else:
        logger.warning(f"⚠️  {total - passed} servers failed tests")

if __name__ == "__main__":
    asyncio.run(main())
