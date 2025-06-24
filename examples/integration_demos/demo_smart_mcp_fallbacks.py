#!/usr/bin/env python3
"""
Smart MCP Fallback Strategy Demo

This demonstrates the CORRECT approach to MCP hyper-availability:
1. ALWAYS try MCP first (don't bypass)
2. Smart fallback hierarchy when MCP fails
3. Clear warnings when native fallbacks are used
4. Helpful suggestions when no fallback exists
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartMCPRegistry:
    """Demonstrates the correct MCP fallback approach."""
    
    def __init__(self):
        self.mcp_tool_registry = {}
        self.native_tool_registry = {}
        self.tool_fallback_strategies = {}
        self.mcp_available = False  # Simulate MCP server being down
    
    async def execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """SMART execution: Always try MCP first, then intelligent fallbacks."""
        
        logger.info(f"🎯 Agent requested tool: {tool_name}")
        
        # STEP 1: Always try MCP first (this ensures MCP gets priority)
        try:
            result = await self._try_primary_mcp(tool_name, parameters)
            logger.info(f"✅ MCP succeeded for {tool_name}")
            return result
        except Exception as e:
            logger.warning(f"❌ Primary MCP failed: {e}")
        
        # STEP 2: Try alternative MCP servers
        try:
            result = await self._try_alternative_mcp(tool_name, parameters)
            logger.info(f"✅ Alternative MCP succeeded for {tool_name}")
            return result
        except Exception as e:
            logger.warning(f"❌ Alternative MCP failed: {e}")
        
        # STEP 3: Try degraded MCP (simpler parameters)
        try:
            result = await self._try_degraded_mcp(tool_name, parameters)
            logger.info(f"✅ Degraded MCP succeeded for {tool_name}")
            return result
        except Exception as e:
            logger.warning(f"❌ Degraded MCP failed: {e}")
        
        # STEP 4: Native fallback (LAST RESORT with clear warning)
        if tool_name in self.native_tool_registry:
            logger.warning(f"⚠️ Using native fallback for {tool_name} - MCP unavailable!")
            result = await self._try_native_fallback(tool_name, parameters)
            result['warning'] = 'MCP unavailable - using local fallback'
            result['mcp_preferred'] = True
            return result
        
        # STEP 5: No fallback available - helpful error
        return {
            'success': False,
            'error': f"No MCP server or native fallback for {tool_name}",
            'suggestions': self._get_suggestions(tool_name),
            'recommendation': 'Install MCP server or implement native fallback'
        }
    
    async def _try_primary_mcp(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Try primary MCP server."""
        if not self.mcp_available:
            raise Exception("Primary MCP server unavailable")
        
        # This would be real MCP call:
        # return await mcp_client.call_tool(tool_name, parameters)
        return {
            'success': True,
            'result': f"MCP executed {tool_name}",
            'source': 'primary_mcp'
        }
    
    async def _try_alternative_mcp(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Try alternative MCP servers."""
        alternatives = self.tool_fallback_strategies.get(tool_name, {}).get('alternatives', [])
        if not alternatives:
            raise Exception("No alternative MCP servers configured")
        
        # Try each alternative
        for alt in alternatives:
            try:
                # This would be real alternative MCP call
                return {
                    'success': True, 
                    'result': f"Alternative MCP {alt} executed {tool_name}",
                    'source': f'alternative_mcp_{alt}'
                }
            except:
                continue
        
        raise Exception("All alternative MCP servers failed")
    
    async def _try_degraded_mcp(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Try MCP with simplified parameters."""
        if not self.tool_fallback_strategies.get(tool_name, {}).get('allow_degraded', False):
            raise Exception("Degraded mode not allowed for this tool")
        
        # Simplify parameters and try again
        simple_params = self._simplify_params(tool_name, parameters)
        # This would be real MCP call with simple params
        return {
            'success': True,
            'result': f"Degraded MCP executed {tool_name} with simplified params",
            'source': 'degraded_mcp',
            'original_params': parameters,
            'simplified_params': simple_params
        }
    
    async def _try_native_fallback(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Last resort: native Python implementation."""
        if tool_name not in self.native_tool_registry:
            raise Exception(f"No native fallback for {tool_name}")
        
        native_func = self.native_tool_registry[tool_name]
        result = await native_func(parameters)
        
        return {
            'success': True,
            'result': result,
            'source': 'native_fallback',
            'warning': 'This used local Python code, not MCP server'
        }
    
    def _simplify_params(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify parameters for degraded mode."""
        if tool_name == "create_file":
            return {
                "path": parameters.get("path"),
                "content": parameters.get("content", "")[:1000]  # Limit content
            }
        return parameters
    
    def _get_suggestions(self, tool_name: str) -> List[str]:
        """Get helpful suggestions when tool completely fails."""
        return [
            f"Install MCP server that supports '{tool_name}'",
            f"Implement native fallback for '{tool_name}'",
            f"Use alternative tool from: {list(self.native_tool_registry.keys())}"
        ]
    
    # Register tools and fallbacks
    
    def register_tool_with_smart_fallbacks(self, tool_name: str):
        """Register a tool with comprehensive fallback strategy."""
        
        # Register the MCP tool
        self.mcp_tool_registry[tool_name] = {
            'primary_server': 'main_mcp_server',
            'registered': datetime.utcnow()
        }
        
        # Configure smart fallback strategy
        self.tool_fallback_strategies[tool_name] = {
            'alternatives': ['backup_mcp_1', 'backup_mcp_2'],
            'allow_degraded': True,
            'native_available': tool_name in self.native_tool_registry
        }
        
        logger.info(f"Registered {tool_name} with smart fallback strategy")
    
    # Example native fallbacks (only as safety net)
    
    async def _native_create_file(self, params: Dict[str, Any]) -> str:
        """Native fallback - clearly marked as backup."""
        try:
            with open(params["path"], "w") as f:
                f.write(params["content"])
            return f"⚠️ NATIVE FALLBACK: Created {params['path']} (MCP preferred)"
        except Exception as e:
            return f"❌ Native fallback failed: {e}"
    
    async def _native_read_file(self, params: Dict[str, Any]) -> str:
        """Native fallback - clearly marked as backup.""" 
        try:
            with open(params["path"], "r") as f:
                content = f.read()
            return f"⚠️ NATIVE FALLBACK: Read {params['path']} ({len(content)} chars)"
        except Exception as e:
            return f"❌ Native fallback failed: {e}"

async def demo_smart_fallbacks():
    """Demonstrate the smart fallback approach."""
    
    print("🎯 SMART MCP FALLBACK DEMO")
    print("=" * 50)
    
    registry = SmartMCPRegistry()
    
    # Register native fallbacks (safety net only)
    registry.native_tool_registry["create_file"] = registry._native_create_file
    registry.native_tool_registry["read_file"] = registry._native_read_file
    
    # Register tools with smart fallback strategies
    registry.register_tool_with_smart_fallbacks("create_file")
    registry.register_tool_with_smart_fallbacks("read_file")
    registry.register_tool_with_smart_fallbacks("analyze_data")  # No native fallback
    
    print(f"✅ Registered tools with smart fallbacks")
    
    # Demo 1: Tool with native fallback (but MCP tried first)
    print("\n📝 Demo 1: create_file (has native fallback)")
    result = await registry.execute_mcp_tool("create_file", {
        "path": "test.txt",
        "content": "Hello World"
    })
    print(f"Result: {result}")
    
    # Demo 2: Tool with native fallback
    print("\n📖 Demo 2: read_file (has native fallback)")
    result = await registry.execute_mcp_tool("read_file", {
        "path": "test.txt"
    })
    print(f"Result: {result}")
    
    # Demo 3: Tool WITHOUT native fallback (shows proper error)
    print("\n🧠 Demo 3: analyze_data (NO native fallback)")
    result = await registry.execute_mcp_tool("analyze_data", {
        "data": [1, 2, 3, 4, 5]
    })
    print(f"Result: {result}")
    
    print("\n💡 KEY INSIGHTS:")
    print("✅ MCP is ALWAYS tried first")
    print("⚠️ Native fallbacks clearly marked as backups")
    print("❌ Tools without fallbacks give helpful errors")
    print("🎯 Agents learn to prefer MCP, use fallbacks as safety net")

if __name__ == "__main__":
    asyncio.run(demo_smart_fallbacks())
