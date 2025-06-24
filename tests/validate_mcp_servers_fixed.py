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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_python_module(module_name: str, timeout: int = 10):
    """Test a Python module-based MCP server"""
    try:
        logger.info(f"Testing Python module: {module_name}")
        
        # Start the server process
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", module_name,
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
                response = stdout.decode()
                logger.info(f"✓ Module responded: {response[:200]}...")
                return True
            else:
                logger.error(f"✗ No response from module. Stderr: {stderr.decode()}")
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"✗ Module timeout after {timeout}s")
            process.terminate()
            return False
            
    except Exception as e:
        logger.error(f"✗ Error testing module: {e}")
        return False

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
                response = stdout.decode()
                logger.info(f"✓ Server responded: {response[:200]}...")
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
        
        command = server_config.get("command")
        if not command:
            logger.error(f"No command specified for {server_name}")
            results[server_name] = False
            continue
          # Handle array command format
        if isinstance(command, list) and len(command) > 0:
            cmd_type = command[0]
            if cmd_type == "python" and len(command) >= 3 and command[1] == "-m":
                # Python module format: ["python", "-m", "module_name"]
                module_name = command[2]
                logger.info(f"Testing Python module: {module_name}")
                results[server_name] = await test_python_module(module_name)
            elif cmd_type == "python" and len(command) > 1:
                server_path = command[1]
                if os.path.exists(server_path):
                    results[server_name] = await test_python_mcp_server(server_path)
                else:
                    logger.error(f"Server file not found: {server_path}")
                    results[server_name] = False            elif cmd_type == "node" and len(command) > 1:
                # Node.js format: ["node", "server.js", ...args]
                server_path = command[1]
                args = command[2:] if len(command) > 2 else []
                if os.path.exists(server_path):
                    results[server_name] = await test_node_mcp_server_with_args(server_path, args)
                else:
                    logger.error(f"Server file not found: {server_path}")
                    results[server_name] = False
            else:
                logger.warning(f"Unknown command format for {server_name}: {command}")
                results[server_name] = False
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
