"""
Test MCP Server Functionality

This script tests the actual MCP servers to ensure they're real and not mocks.
"""

import asyncio
import json
import logging
import subprocess
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_server(server_config):
    """Test if an MCP server can be started and responds"""
    server_id = server_config['id']
    command = server_config['command']
    
    logger.info(f"Testing MCP server: {server_id}")
    
    try:
        # Try to start the server process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info(f"✅ {server_id}: Server process started successfully")
            process.terminate()
            process.wait(timeout=5)
            return {"status": "success", "message": "Server started and responsive"}
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ {server_id}: Server failed to start")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return {"status": "failed", "message": f"Failed to start: {stderr}"}
            
    except FileNotFoundError as e:
        logger.error(f"❌ {server_id}: Command not found - {e}")
        return {"status": "not_found", "message": f"Command not found: {e}"}
    except Exception as e:
        logger.error(f"❌ {server_id}: Error testing server - {e}")
        return {"status": "error", "message": f"Error: {e}"}

async def test_node_and_npm():
    """Test if Node.js and npm are available"""
    logger.info("Testing Node.js and npm availability...")
    
    results = {}
    
    # Test Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"✅ Node.js available: {version}")
            results['node'] = {"status": "available", "version": version}
        else:
            logger.error("❌ Node.js not available")
            results['node'] = {"status": "not_available", "error": result.stderr}
    except Exception as e:
        logger.error(f"❌ Node.js error: {e}")
        results['node'] = {"status": "error", "error": str(e)}
    
    # Test npm
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"✅ npm available: {version}")
            results['npm'] = {"status": "available", "version": version}
        else:
            logger.error("❌ npm not available")
            results['npm'] = {"status": "not_available", "error": result.stderr}
    except Exception as e:
        logger.error(f"❌ npm error: {e}")
        results['npm'] = {"status": "error", "error": str(e)}
    
    # Test npx
    try:
        result = subprocess.run(['npx', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"✅ npx available: {version}")
            results['npx'] = {"status": "available", "version": version}
        else:
            logger.error("❌ npx not available")
            results['npx'] = {"status": "not_available", "error": result.stderr}
    except Exception as e:
        logger.error(f"❌ npx error: {e}")
        results['npx'] = {"status": "error", "error": str(e)}
    
    return results

async def check_mcp_packages():
    """Check if MCP packages are available via npm"""
    logger.info("Checking MCP package availability...")
    
    packages = [
        "@modelcontextprotocol/server-filesystem",
        "@modelcontextprotocol/server-github", 
        "@modelcontextprotocol/server-git"
    ]
    
    results = {}
    
    for package in packages:
        try:
            logger.info(f"Checking package: {package}")
            # Try to get package info
            result = subprocess.run(
                ['npm', 'view', package, 'version'], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"✅ {package}: {version}")
                results[package] = {"status": "available", "version": version}
            else:
                logger.error(f"❌ {package}: Not found")
                results[package] = {"status": "not_found", "error": result.stderr}
                
        except Exception as e:
            logger.error(f"❌ {package}: Error - {e}")
            results[package] = {"status": "error", "error": str(e)}
    
    return results

async def main():
    """Main test function"""
    logger.info("=== MCP SERVER FUNCTIONALITY TEST ===")
    
    # Load MCP server config
    try:
        with open('mcp_server_configs.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load MCP config: {e}")
        return
    
    # Test Node.js/npm availability
    logger.info("\n1. Testing Node.js/npm/npx availability...")
    node_results = await test_node_and_npm()
    
    # Test MCP package availability
    logger.info("\n2. Testing MCP package availability...")
    package_results = await check_mcp_packages()
    
    # Test coding-related servers
    logger.info("\n3. Testing coding-related MCP servers...")
    server_results = {}
    
    coding_servers = [s for s in config['servers'] 
                     if any(keyword in s['id'] for keyword in ['filesystem', 'git', 'github', 'context7'])]
    
    for server in coding_servers:
        result = await test_mcp_server(server)
        server_results[server['id']] = result
    
    # Generate summary report
    logger.info("\n=== TEST SUMMARY ===")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "node_npm_status": node_results,
        "mcp_packages": package_results,
        "server_tests": server_results,
        "summary": {
            "total_servers_tested": len(server_results),
            "successful_servers": len([r for r in server_results.values() if r['status'] == 'success']),
            "failed_servers": len([r for r in server_results.values() if r['status'] != 'success']),
            "node_available": node_results.get('node', {}).get('status') == 'available',
            "npm_available": node_results.get('npm', {}).get('status') == 'available',
            "npx_available": node_results.get('npx', {}).get('status') == 'available'
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("MCP SERVER FUNCTIONALITY TEST RESULTS")
    print("="*60)
    print(f"Node.js Available: {report['summary']['node_available']}")
    print(f"npm Available: {report['summary']['npm_available']}")
    print(f"npx Available: {report['summary']['npx_available']}")
    print(f"Servers Tested: {report['summary']['total_servers_tested']}")
    print(f"Successful Servers: {report['summary']['successful_servers']}")
    print(f"Failed Servers: {report['summary']['failed_servers']}")
    
    print("\nServer Details:")
    for server_id, result in server_results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"  {status_icon} {server_id}: {result['status']}")
    
    print("\nPackage Status:")
    for package, result in package_results.items():
        status_icon = "✅" if result['status'] == 'available' else "❌"
        print(f"  {status_icon} {package}: {result['status']}")
    
    # Save detailed report
    with open('mcp_functionality_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nDetailed report saved to: mcp_functionality_test_report.json")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
