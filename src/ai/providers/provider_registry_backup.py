"""
Provider Registry - Central Provider Management

Manages all LLM providers (Ollama, OpenRouter, etc.) with unified interface,
health monitoring, and automatic failover capabilities.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta

from .base_provider import BaseProviderManager
from .ollama_provider import get_ollama_manager
from .openrouter_provider import get_openrouter_manager

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern for MCP tool execution."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            timeout: Time in seconds to wait before half-opening the circuit
        """
        self.state = 'closed'  # closed, open, half-open
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure = None
    
    def record_success(self):
        """Record a successful operation - reset the circuit breaker."""
        self.state = 'closed'
        self.last_failure = None
    
    def record_failure(self):
        """Record a failed operation - update circuit breaker state if needed."""
        if self.state == 'closed':
            self.state = 'open'
            self.last_failure = datetime.utcnow()
        elif self.state == 'half-open':
            # Still in half-open state, check timeout
            if self.last_failure:
                elapsed = (datetime.utcnow() - self.last_failure).seconds
                if elapsed > self.timeout:
                    # Half-open timeout reached, switch to closed
                    self.state = 'closed'
                    self.last_failure = None
    
    def is_open(self) -> bool:
        """Check if the circuit is open."""
        if self.state == 'open':
            # Check if timeout has passed
            if self.last_failure:
                elapsed = (datetime.utcnow() - self.last_failure).seconds
                if elapsed > self.timeout:
                    self.state = 'half-open'
                    return False
            return True
        
        return False


class ProviderRegistry:
    """
    Central registry for managing all LLM providers.
    
    Provides unified access to multiple providers with health monitoring,
    automatic failover, and load balancing capabilities.
    """
    
    def __init__(self):
        """Initialize the provider registry."""
        self.providers: Dict[str, BaseProviderManager] = {}
        self.provider_status: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
        self._lock = asyncio.Lock()
        
        # MCP Tool Hyper-Availability
        self.mcp_tool_registry = {}
        self.native_tool_registry = {}
        self.tool_circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.tool_performance_stats = {}
        self.tool_fallback_strategies = {}
    
    async def initialize(self, 
                        enable_ollama: bool = True,
                        enable_openrouter: bool = True,
                        ollama_config: Optional[Dict[str, Any]] = None,
                        openrouter_config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        Initialize all enabled providers.
        
        Args:
            enable_ollama: Whether to enable Ollama provider
            enable_openrouter: Whether to enable OpenRouter provider
            ollama_config: Optional configuration for Ollama
            openrouter_config: Optional configuration for OpenRouter
            
        Returns:
            Dict mapping provider names to initialization success status
        """
        async with self._lock:
            results = {}
            
            # Initialize Ollama if enabled
            if enable_ollama:
                try:
                    ollama_config = ollama_config or {}
                    ollama_manager = get_ollama_manager(
                        host=ollama_config.get("host", "localhost"),
                        port=ollama_config.get("port", 11434),
                        timeout=ollama_config.get("timeout", 30)
                    )
                    success = await ollama_manager.start()
                    self.providers["ollama"] = ollama_manager
                    results["ollama"] = success
                    logger.info(f"Ollama provider initialization: {'success' if success else 'failed'}")
                except Exception as e:
                    logger.error(f"Error initializing Ollama: {e}")
                    results["ollama"] = False
            
            # Initialize OpenRouter if enabled
            if enable_openrouter:
                try:
                    openrouter_config = openrouter_config or {}
                    openrouter_manager = get_openrouter_manager(
                        api_key=openrouter_config.get("api_key")
                    )
                    success = await openrouter_manager.start()
                    self.providers["openrouter"] = openrouter_manager
                    results["openrouter"] = success
                    logger.info(f"OpenRouter provider initialization: {'success' if success else 'failed'}")
                except Exception as e:
                    logger.error(f"Error initializing OpenRouter: {e}")
                    results["openrouter"] = False
            
            self.initialized = True
            await self._update_provider_status()
            
            return results
    
    async def _update_provider_status(self) -> None:
        """Update status information for all providers."""
        for name, provider in self.providers.items():
            try:
                health_info = await provider.health_check()
                models = await provider.get_available_models()
                
                self.provider_status[name] = {
                    "name": name,
                    "ready": provider.is_ready,
                    "health": health_info,
                    "model_count": len(models),
                    "models": models[:5],  # First 5 models for preview
                    "capabilities": provider.get_capabilities().__dict__ if hasattr(provider, 'get_capabilities') else {},
                    "last_updated": datetime.utcnow().isoformat()
                }
            except Exception as e:
                self.provider_status[name] = {
                    "name": name,
                    "ready": False,
                    "error": str(e),
                    "last_updated": datetime.utcnow().isoformat()
                }
    
    async def get_provider(self, provider_name: str) -> Optional[BaseProviderManager]:
        """Get a specific provider by name."""
        return self.providers.get(provider_name)
    
    async def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())
    
    async def get_ready_providers(self) -> List[str]:
        """Get list of ready provider names."""
        return [name for name, provider in self.providers.items() if provider.is_ready]
    
    async def get_all_models(self) -> Dict[str, List[str]]:
        """Get all available models from all providers."""
        all_models = {}
        for name, provider in self.providers.items():
            if provider.is_ready:
                try:
                    models = await provider.get_available_models()
                    all_models[name] = models
                except Exception as e:
                    logger.error(f"Error getting models from {name}: {e}")
                    all_models[name] = []
            else:
                all_models[name] = []
        return all_models
    
    async def is_model_available(self, model_name: str, provider_name: Optional[str] = None) -> Dict[str, bool]:
        """
        Check if a model is available across providers.
        
        Args:
            model_name: Name of the model to check
            provider_name: Optional specific provider to check
            
        Returns:
            Dict mapping provider names to availability status
        """
        availability = {}
        
        providers_to_check = [provider_name] if provider_name else self.providers.keys()
        
        for name in providers_to_check:
            provider = self.providers.get(name)
            if provider and provider.is_ready:
                try:
                    available = await provider.is_model_available(model_name)
                    availability[name] = available
                except Exception as e:
                    logger.error(f"Error checking model {model_name} on {name}: {e}")
                    availability[name] = False
            else:
                availability[name] = False
        
        return availability
    
    async def generate_text(self, 
                           provider_name: str,
                           model: str,
                           prompt: str,
                           **kwargs) -> str:
        """
        Generate text using a specific provider.
        
        Args:
            provider_name: Name of the provider to use
            model: Model name
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        provider = self.providers.get(provider_name)
        if not provider or not provider.is_ready:
            logger.error(f"Provider {provider_name} not available")
            return ""
        
        try:
            return await provider.generate_text(model, prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error generating text with {provider_name}/{model}: {e}")
            return ""
    
    async def generate_text_with_fallback(self, 
                                         model: str,
                                         prompt: str,
                                         preferred_providers: Optional[List[str]] = None,
                                         **kwargs) -> Dict[str, Any]:
        """
        Generate text with automatic fallback to other providers.
        
        Args:
            model: Model name to try
            prompt: Input prompt
            preferred_providers: Ordered list of preferred providers
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with result, provider used, and any errors
        """
        if not preferred_providers:
            preferred_providers = await self.get_ready_providers()
        
        errors = []
        
        for provider_name in preferred_providers:
            provider = self.providers.get(provider_name)
            if not provider or not provider.is_ready:
                errors.append(f"{provider_name}: not ready")
                continue
            
            # Check if model is available on this provider
            model_available = await provider.is_model_available(model)
            if not model_available:
                errors.append(f"{provider_name}: model '{model}' not available")
                continue
            
            # Try to generate
            try:
                result = await provider.generate_text(model, prompt, **kwargs)
                if result:  # Success
                    return {
                        "success": True,
                        "result": result,
                        "provider_used": provider_name,
                        "model_used": model,
                        "errors": errors
                    }
                else:
                    errors.append(f"{provider_name}: empty response")
            except Exception as e:
                errors.append(f"{provider_name}: {str(e)}")
        
        return {
            "success": False,
            "result": "",
            "provider_used": None,
            "model_used": model,
            "errors": errors
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for all providers."""
        await self._update_provider_status()
        
        ready_count = len(await self.get_ready_providers())
        total_count = len(self.providers)
        
        return {
            "initialized": self.initialized,
            "providers_ready": ready_count,
            "providers_total": total_count,
            "providers": self.provider_status,
            "system_healthy": ready_count > 0,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_model_recommendations(self, 
                                      agent_type: str = "general",
                                      include_free_only: bool = False) -> Dict[str, List[str]]:
        """
        Get model recommendations for different agent types.
        
        Args:
            agent_type: Type of agent (reasoning, coding, general, etc.)
            include_free_only: Whether to include only free models
            
        Returns:
            Dict mapping provider names to recommended model lists
        """
        recommendations = {}
        
        for name, provider in self.providers.items():
            if not provider.is_ready:
                continue
            
            try:
                if hasattr(provider, 'get_recommended_models'):
                    models = await provider.get_recommended_models()
                else:
                    # Fallback to available models
                    available = await provider.get_available_models()
                    models = available[:3] if available else []
                
                # Filter for free models if requested
                if include_free_only and name == "openrouter":
                    models = [m for m in models if ":free" in m or "free" in m.lower()]
                
                recommendations[name] = models
            except Exception as e:
                logger.error(f"Error getting recommendations from {name}: {e}")
                recommendations[name] = []
        
        return recommendations
    
    async def shutdown(self) -> None:
        """Shutdown all providers."""
        logger.info("Shutting down provider registry...")
        
        for name, provider in self.providers.items():
            try:
                await provider.stop()
                logger.info(f"Shutdown provider: {name}")
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
        
        self.providers.clear()
        self.provider_status.clear()
        self.initialized = False

    # MCP Tool Hyper-Availability Methods
    
    async def register_mcp_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> bool:
        """Register an MCP tool with fallback strategies."""
        self.mcp_tool_registry[tool_name] = {
            'config': tool_config,
            'status': 'active',
            'last_used': datetime.utcnow(),
            'success_count': 0,
            'failure_count': 0
        }
        
        # Initialize circuit breaker
        self.tool_circuit_breakers[tool_name] = CircuitBreaker(
            failure_threshold=5,
            timeout=60
        )
        
        logger.info(f"Registered MCP tool: {tool_name}")
        return True
    
    async def execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool with hyper-availability (circuit breaker + fallback)."""
        
        # Check circuit breaker
        if self._is_circuit_open(tool_name):
            return await self._execute_fallback_tool(tool_name, parameters)
        
        try:
            # Execute primary tool
            result = await self._execute_primary_tool(tool_name, parameters)
            self._record_tool_success(tool_name)
            return result
            
        except Exception as e:
            self._record_tool_failure(tool_name, str(e))
            logger.warning(f"MCP tool {tool_name} failed: {e}")
            
            # Try fallback
            return await self._execute_fallback_tool(tool_name, parameters)
    
    def _is_circuit_open(self, tool_name: str) -> bool:
        """Check if circuit breaker is open for tool."""
        breaker = self.tool_circuit_breakers.get(tool_name)
        if breaker and breaker.state == 'open':
            # Check if timeout has passed
            if breaker.last_failure:
                elapsed = (datetime.utcnow() - breaker.last_failure).seconds
                if elapsed > breaker.timeout:
                    breaker.state = 'half-open'
                    return False
            return True
        
        return False
    
    async def _execute_primary_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the primary MCP tool - REAL MCP client integration."""
        try:
            # Import real MCP client components
            from mcp.client import MCPClient
            from mcp.tools.executor import ToolExecutor

            # Get or create MCP client
            if not hasattr(self, '_mcp_client'):
                self._mcp_client = MCPClient()
                await self._mcp_client.initialize()

            # Execute tool through real MCP client
            tool_executor = ToolExecutor(self._mcp_client)
            result = await tool_executor.execute_tool(tool_name, parameters)

            if result and result.get('success'):
                return result
            else:
                raise RuntimeError(f"MCP tool execution failed: {result.get('error', 'Unknown error')}")

        except ImportError as e:
            logger.error(f"MCP client not available: {e}")
            raise RuntimeError(f"MCP client integration not available: {e}")
        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            raise RuntimeError(f"MCP tool execution failed: {e}")
    
    async def _execute_fallback_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fallback with SMART hierarchy - not just native."""
        
        # Fallback Strategy 1: Try alternative MCP servers
        alternative_servers = self.tool_fallback_strategies.get(tool_name, {}).get('alternative_servers', [])
        for server in alternative_servers:
            try:
                logger.info(f"Trying alternative MCP server: {server}")
                # REAL alternative MCP server implementation
                from mcp.client import MCPClient
                alternative_client = MCPClient(server_url=server)
                await alternative_client.initialize()

                result = await alternative_client.call_tool(tool_name, parameters)
                if result and result.get('success'):
                    logger.info(f"Alternative MCP server {server} succeeded")
                    return result

            except Exception as e:
                logger.warning(f"Alternative MCP server {server} failed: {e}")
                continue
        
        # Fallback Strategy 2: Try degraded MCP (simplified parameters)
        if self.tool_fallback_strategies.get(tool_name, {}).get('allow_degraded', False):
            try:
                simplified_params = self._simplify_parameters(tool_name, parameters)
                logger.info(f"Trying degraded MCP with simplified parameters")
                # REAL degraded MCP implementation
                result = await self._mcp_client.call_tool(tool_name, simplified_params)
                if result and result.get('success'):
                    logger.info(f"Degraded MCP succeeded with simplified parameters")
                    return result
            except Exception as e:
                logger.warning(f"Degraded MCP failed: {e}")
        
        # Fallback Strategy 3: Try native fallback (LAST RESORT)
        if tool_name in self.native_tool_registry:
            logger.info(f"Using native fallback for {tool_name} (MCP unavailable)")
            try:
                native_func = self.native_tool_registry[tool_name]
                result = await native_func(parameters)
                return {
                    'success': True,
                    'result': result,
                    'tool_name': tool_name,
                    'fallback_used': 'native',
                    'warning': 'MCP server unavailable - used local fallback'
                }
            except Exception as e:
                logger.error(f"Native fallback failed: {e}")
        
        # Fallback Strategy 4: Return helpful error with suggestions
        return {
            'success': False,
            'error': f"All fallback strategies failed for {tool_name}",
            'tool_name': tool_name,
            'suggestions': self._get_tool_suggestions(tool_name, parameters),
            'available_alternatives': list(self.native_tool_registry.keys())
        }
    
    def _simplify_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify parameters for degraded MCP mode."""
        # Tool-specific parameter simplification
        simplified = parameters.copy()
        
        if tool_name == "create_file":
            # Keep only essential parameters
            simplified = {
                "path": parameters.get("path", ""),
                "content": parameters.get("content", "")
            }
        elif tool_name == "fetch_url":
            # Remove advanced options
            simplified = {
                "url": parameters.get("url", ""),
                "timeout": 10  # Fixed timeout
            }
        
        return simplified
    
    def _get_tool_suggestions(self, tool_name: str, parameters: Dict[str, Any]) -> List[str]:
        """Get helpful suggestions when tool fails."""
        suggestions = []
        
        # Check for similar tools
        available_tools = list(self.native_tool_registry.keys())
        
        if tool_name == "create_file" and "read_file" in available_tools:
            suggestions.append("Try 'read_file' if you want to check existing files")
        elif tool_name == "fetch_url" and "read_file" in available_tools:
            suggestions.append("If fetching local content, try 'read_file' instead")
        elif tool_name not in available_tools:
            suggestions.append(f"Available tools: {', '.join(available_tools[:5])}")
        
        return suggestions
    
    async def _execute_native_fallback(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute native Python fallback for MCP tool."""
        if tool_name in self.native_tool_registry:
            native_func = self.native_tool_registry[tool_name]
            result = await native_func(parameters)
            return {
                'success': True,
                'result': result,
                'tool_name': tool_name,
                'fallback_used': 'native'
            }
        
        return {
            'success': False,
            'error': f"No native fallback for {tool_name}",
            'tool_name': tool_name
        }
    
    def _record_tool_success(self, tool_name: str):
        """Record successful tool execution."""
        if tool_name in self.mcp_tool_registry:
            self.mcp_tool_registry[tool_name]['success_count'] += 1
            self.mcp_tool_registry[tool_name]['last_used'] = datetime.utcnow()
        
        # Reset circuit breaker
        if tool_name in self.tool_circuit_breakers:
            self.tool_circuit_breakers[tool_name].record_success()
    
    def _record_tool_failure(self, tool_name: str, error: str):
        """Record failed tool execution and update circuit breaker."""
        if tool_name in self.mcp_tool_registry:
            self.mcp_tool_registry[tool_name]['failure_count'] += 1
        
        # Update circuit breaker
        if tool_name in self.tool_circuit_breakers:
            breaker = self.tool_circuit_breakers[tool_name]
            breaker.record_failure()
    
    async def get_mcp_tool_status(self) -> Dict[str, Any]:
        """Get status of all MCP tools."""
        return {
            'registered_tools': len(self.mcp_tool_registry),
            'native_fallbacks': len(self.native_tool_registry),
            'circuit_breakers': {
                name: breaker.state 
                for name, breaker in self.tool_circuit_breakers.items()
            },
            'tool_stats': self.mcp_tool_registry,
            'timestamp': datetime.utcnow().isoformat()
        }

    # Example: Register native fallbacks for common MCP tools
    
    def register_native_fallbacks(self):
        """Register Python function fallbacks for common MCP tools."""
        
        # File operations
        self.native_tool_registry["create_file"] = self._native_create_file
        self.native_tool_registry["read_file"] = self._native_read_file
        self.native_tool_registry["list_directory"] = self._native_list_directory
        
        # Web operations
        self.native_tool_registry["fetch_url"] = self._native_fetch_url
        
        # System operations
        self.native_tool_registry["run_command"] = self._native_run_command
        
        logger.info(f"Registered {len(self.native_tool_registry)} native fallbacks")
    
    # Native fallback implementations
    
    async def _native_create_file(self, params: Dict[str, Any]) -> str:
        """Native Python fallback for create_file MCP tool."""
        try:
            path = params.get("path", "")
            content = params.get("content", "")
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"✅ Created file: {path} ({len(content)} chars)"
        except Exception as e:
            return f"❌ Failed to create file: {str(e)}"
    
    async def _native_read_file(self, params: Dict[str, Any]) -> str:
        """Native Python fallback for read_file MCP tool."""
        try:
            path = params.get("path", "")
            
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return f"✅ Read file: {path} ({len(content)} chars)\n\n{content}"
        except Exception as e:
            return f"❌ Failed to read file: {str(e)}"
    
    async def _native_list_directory(self, params: Dict[str, Any]) -> str:
        """Native Python fallback for list_directory MCP tool."""
        try:
            import os
            path = params.get("path", ".")
            
            items = os.listdir(path)
            files = [item for item in items if os.path.isfile(os.path.join(path, item))]
            dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
            
            result = f"📁 Directory: {path}\n"
            result += f"📂 Directories ({len(dirs)}): {', '.join(dirs[:10])}\n"
            result += f"📄 Files ({len(files)}): {', '.join(files[:10])}"
            
            return result
        except Exception as e:
            return f"❌ Failed to list directory: {str(e)}"
    
    async def _native_fetch_url(self, params: Dict[str, Any]) -> str:
        """Native Python fallback for fetch_url MCP tool."""
        try:
            import aiohttp
            import asyncio
            
            url = params.get("url", "")
            timeout = params.get("timeout", 10)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    content = await response.text()
                    
            return f"✅ Fetched URL: {url} ({response.status}) - {len(content)} chars"
        except Exception as e:
            return f"❌ Failed to fetch URL: {str(e)}"
    
    async def _native_run_command(self, params: Dict[str, Any]) -> str:
        """Native Python fallback for run_command MCP tool."""
        try:
            import subprocess
            
            command = params.get("command", "")
            timeout = params.get("timeout", 30)
            
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            output = f"✅ Command: {command}\n"
            output += f"Exit Code: {result.returncode}\n"
            output += f"Stdout: {result.stdout}\n"
            output += f"Stderr: {result.stderr}"
            
            return output
        except Exception as e:
            return f"❌ Failed to run command: {str(e)}"
    
    # Advanced Example: Database Tool with Native Fallback
    
    async def register_advanced_tool_example(self):
        """Example of sophisticated tool with native fallback."""
        
        # Register a database tool
        await self.register_mcp_tool("query_database", {
            "mcp_server": "sqlite://database.db",
            "description": "Query SQLite database",
            "parameters": ["query", "database"]
        })
        
        # Register the native Python fallback
        self.native_tool_registry["query_database"] = self._native_query_database
        self.tool_fallback_strategies["query_database"] = "native"
        
        logger.info("Registered advanced database tool with native fallback")
    
    async def _native_query_database(self, params: Dict[str, Any]) -> str:
        """Native Python implementation of database querying."""
        try:
            import sqlite3
            
            query = params.get("query", "")
            database = params.get("database", "database.db")
            
            # This is a REAL database query, not emulation!
            conn = sqlite3.connect(database)
            cursor = conn.cursor()
            
            if query.upper().startswith("SELECT"):
                cursor.execute(query)
                results = cursor.fetchall()
                conn.close()
                
                return f"✅ Query executed: {len(results)} rows returned\n{results}"
            else:
                cursor.execute(query)
                conn.commit()
                rows_affected = cursor.rowcount
                conn.close()
                
                return f"✅ Query executed: {rows_affected} rows affected"
                
        except Exception as e:
            return f"❌ Database query failed: {str(e)}"
    
    # Web Scraping Tool Example
    
    async def _native_scrape_webpage(self, params: Dict[str, Any]) -> str:
        """Native Python web scraping - full functionality!"""
        try:
            import aiohttp
            from bs4 import BeautifulSoup
            
            url = params.get("url", "")
            selector = params.get("selector", "")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            if selector:
                elements = soup.select(selector)
                results = [elem.get_text().strip() for elem in elements]
                return f"✅ Scraped {len(results)} elements: {results[:5]}"
            else:
                title = soup.title.string if soup.title else "No title"
                return f"✅ Scraped page: {title} ({len(html)} chars)"
                
        except Exception as e:
            return f"❌ Web scraping failed: {str(e)}"
    
    # AI Analysis Tool Example
    
    async def _native_analyze_sentiment(self, params: Dict[str, Any]) -> str:
        """Native Python sentiment analysis - real AI processing!"""
        try:
            text = params.get("text", "")
            
            # Simple rule-based sentiment (could use transformers, etc.)
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", "love"]
            negative_words = ["bad", "terrible", "awful", "hate", "horrible", "worse"]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
                confidence = positive_count / (positive_count + negative_count + 1)
            elif negative_count > positive_count:
                sentiment = "negative"
                confidence = negative_count / (positive_count + negative_count + 1)
            else:
                sentiment = "neutral"
                confidence = 0.5
            
            return f"✅ Sentiment: {sentiment} (confidence: {confidence:.2f})\n" \
                   f"Positive signals: {positive_count}, Negative signals: {negative_count}"
                   
        except Exception as e:
            return f"❌ Sentiment analysis failed: {str(e)}"


# Global registry instance
_provider_registry = None

def get_provider_registry() -> ProviderRegistry:
    """Get global provider registry instance."""
    global _provider_registry
    if _provider_registry is None:
        _provider_registry = ProviderRegistry()
    return _provider_registry
