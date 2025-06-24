#!/usr/bin/env python3
"""
Startup script for PyGent Factory Embedding MCP Server

This script starts the embedding server with proper configuration,
health checks, and monitoring.
"""

import asyncio
import logging
import sys
import time
import subprocess
import signal
import os
from pathlib import Path
from typing import Optional
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingServerManager:
    """Manages the embedding MCP server lifecycle"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"
        
    def start_server(self) -> bool:
        """Start the embedding server"""
        try:
            # Check if server is already running
            if self.is_server_running():
                logger.info(f"Server already running at {self.base_url}")
                return True
            
            # Start server process
            logger.info(f"Starting Embedding MCP Server at {self.base_url}")
            
            cmd = [
                sys.executable,
                "src/servers/embedding_mcp_server.py",
                self.host,
                str(self.port)
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for server to start
            if self.wait_for_server_ready(timeout=30):
                logger.info("✅ Embedding MCP Server started successfully")
                return True
            else:
                logger.error("❌ Server failed to start within timeout")
                self.stop_server()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """Stop the embedding server"""
        try:
            if self.process:
                logger.info("Stopping Embedding MCP Server...")
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Server didn't stop gracefully, forcing termination")
                    self.process.kill()
                    self.process.wait()
                
                self.process = None
                logger.info("✅ Server stopped")
                return True
            else:
                logger.info("Server not running")
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            return False
    
    def is_server_running(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_server_ready(self, timeout: int = 30) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_server_running():
                return True
            
            # Check if process died
            if self.process and self.process.poll() is not None:
                logger.error("Server process died during startup")
                return False
            
            time.sleep(1)
        
        return False
    
    def get_server_status(self) -> dict:
        """Get detailed server status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
    
    def test_embedding_generation(self) -> bool:
        """Test basic embedding generation"""
        try:
            payload = {
                "input": "Test embedding generation",
                "model": "text-embedding-ada-002"
            }
            
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return (
                    'data' in data and 
                    len(data['data']) > 0 and 
                    'embedding' in data['data'][0] and
                    len(data['data'][0]['embedding']) > 0
                )
            else:
                logger.error(f"Embedding test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Embedding test failed: {e}")
            return False


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, 'server_manager'):
        signal_handler.server_manager.stop_server()
    sys.exit(0)


def main():
    """Main startup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PyGent Factory Embedding MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--test", action="store_true", help="Run tests after startup")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (don't block)")
    
    args = parser.parse_args()
    
    # Create server manager
    server_manager = EmbeddingServerManager(args.host, args.port)
    signal_handler.server_manager = server_manager
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 PyGent Factory Embedding MCP Server Manager")
    print("=" * 50)
    
    # Start server
    if not server_manager.start_server():
        print("❌ Failed to start server")
        return 1
    
    # Get initial status
    status = server_manager.get_server_status()
    print(f"📊 Server Status: {status.get('status', 'unknown')}")
    
    if status.get('status') == 'healthy':
        providers = status.get('providers', {})
        print(f"🔧 Providers: {providers.get('provider_count', 0)} available")
        print(f"🎯 Current Provider: {providers.get('current_provider', 'unknown')}")
    
    # Test embedding generation
    print("\n🧪 Testing embedding generation...")
    if server_manager.test_embedding_generation():
        print("✅ Embedding generation test passed")
    else:
        print("❌ Embedding generation test failed")
    
    # Run comprehensive tests if requested
    if args.test:
        print("\n🔬 Running comprehensive tests...")
        try:
            from test_embedding_mcp_server import EmbeddingMCPServerTester
            tester = EmbeddingMCPServerTester(server_manager.base_url)
            results = tester.run_all_tests()
            
            if results['overall_status'] == 'PASS':
                print("✅ All tests passed!")
            else:
                print(f"❌ {results['failed_tests']} tests failed")
        except ImportError:
            print("⚠️  Test module not available")
    
    print(f"\n🌐 Server running at: {server_manager.base_url}")
    print("📚 API Documentation:")
    print(f"  - Health: {server_manager.base_url}/health")
    print(f"  - Embeddings: {server_manager.base_url}/v1/embeddings")
    print(f"  - Root: {server_manager.base_url}/")
    
    if args.daemon:
        print("\n🔄 Running in daemon mode")
        return 0
    else:
        print("\n⌨️  Press Ctrl+C to stop the server")
        try:
            # Keep the script running
            while True:
                time.sleep(1)
                
                # Check if server process is still alive
                if server_manager.process and server_manager.process.poll() is not None:
                    logger.error("Server process died unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested")
        finally:
            server_manager.stop_server()
    
    return 0


if __name__ == "__main__":
    exit(main())
