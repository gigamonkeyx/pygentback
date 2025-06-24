#!/usr/bin/env python3
"""
HTTP wrapper for Context7 MCP server to make it accessible via URL
"""

import asyncio
import json
import subprocess
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import queue
import time

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)

class Context7MCPWrapper:
    def __init__(self):
        self.process = None
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.initialized = False
        
    def start_mcp_server(self):
        """Start the Context7 MCP server process"""
        try:
            cmd = ["D:\\nodejs\\npx.cmd", "-y", "@upstash/context7-mcp"]
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            
            # Initialize the MCP connection
            init_request = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "http-wrapper", "version": "1.0"}
                }
            }
            
            self.process.stdin.write(json.dumps(init_request) + "\n")
            self.process.stdin.flush()
            
            # Wait for initialization response
            response_line = self.process.stdout.readline()
            if response_line:
                self.initialized = True
                logger.info("Context7 MCP server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Context7 MCP server: {e}")
    
    def send_request(self, method, params=None):
        """Send a request to the MCP server"""
        if not self.initialized:
            return {"error": "MCP server not initialized"}
            
        try:
            request_data = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),  # Use timestamp as ID
                "method": method,
                "params": params or {}
            }
            
            self.process.stdin.write(json.dumps(request_data) + "\n")
            self.process.stdin.flush()
            
            # Read response (with timeout)
            response_line = self.process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            else:
                return {"error": "No response from MCP server"}
                
        except Exception as e:
            return {"error": f"Failed to send request: {e}"}

# Global wrapper instance
context7_wrapper = Context7MCPWrapper()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Context7 HTTP Wrapper",
        "mcp_initialized": context7_wrapper.initialized
    })

@app.route('/resolve-library-id', methods=['POST'])
def resolve_library_id():
    """Resolve a library name to Context7 compatible ID"""
    data = request.get_json()
    library_name = data.get('libraryName', '')
    
    if not library_name:
        return jsonify({"error": "libraryName is required"}), 400
    
    response = context7_wrapper.send_request("tools/call", {
        "name": "resolve-library-id",
        "arguments": {"libraryName": library_name}
    })
    
    return jsonify(response)

@app.route('/get-library-docs', methods=['POST'])
def get_library_docs():
    """Get documentation for a library"""
    data = request.get_json()
    library_id = data.get('context7CompatibleLibraryID', '')
    topic = data.get('topic', '')
    tokens = data.get('tokens', 10000)
    
    if not library_id:
        return jsonify({"error": "context7CompatibleLibraryID is required"}), 400
    
    params = {
        "name": "get-library-docs",
        "arguments": {
            "context7CompatibleLibraryID": library_id,
            "tokens": tokens
        }
    }
    
    if topic:
        params["arguments"]["topic"] = topic
    
    response = context7_wrapper.send_request("tools/call", params)
    return jsonify(response)

@app.route('/mcp', methods=['POST'])
def mcp_proxy():
    """Generic MCP proxy endpoint"""
    data = request.get_json()
    method = data.get('method', '')
    params = data.get('params', {})
    
    response = context7_wrapper.send_request(method, params)
    return jsonify(response)

@app.route('/', methods=['GET'])
def index():
    """Index page with API documentation"""
    return jsonify({
        "service": "Context7 HTTP Wrapper",
        "description": "HTTP wrapper for Context7 MCP server",
        "endpoints": {
            "/health": "GET - Health check",
            "/resolve-library-id": "POST - Resolve library name to ID",
            "/get-library-docs": "POST - Get library documentation", 
            "/mcp": "POST - Generic MCP proxy"
        },
        "example_usage": {
            "resolve": {
                "url": "/resolve-library-id",
                "method": "POST",
                "body": {"libraryName": "fastapi"}
            },
            "docs": {
                "url": "/get-library-docs", 
                "method": "POST",
                "body": {
                    "context7CompatibleLibraryID": "tiangolo/fastapi",
                    "topic": "getting started",
                    "tokens": 5000
                }
            }
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Context7 HTTP Wrapper...")
    
    # Start the MCP server
    context7_wrapper.start_mcp_server()
    
    if context7_wrapper.initialized:
        print("✅ Context7 MCP server initialized")
        print("🌐 Starting HTTP server on http://localhost:8080")
        print("📖 API documentation available at http://localhost:8080")
        
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        print("❌ Failed to initialize Context7 MCP server")
