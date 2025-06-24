"""
PyGent Factory Integration

Integration layer connecting the orchestration system with existing
PyGent Factory components including ToT, RAG, web UI, and research workflows.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

from .coordination_models import (
    OrchestrationConfig, TaskRequest, TaskPriority, MCPServerType
)

logger = logging.getLogger(__name__)


@dataclass
class PyGentComponent:
    """PyGent Factory component definition."""
    component_id: str
    component_type: str
    name: str
    description: str
    capabilities: List[str]
    endpoints: Dict[str, str]
    status: str = "inactive"
    last_health_check: Optional[datetime] = None


@dataclass
class IntegrationMapping:
    """Mapping between orchestration and PyGent components."""
    orchestration_capability: str
    pygent_component: str
    endpoint: str
    transformation_function: Optional[Callable] = None


class PyGentIntegration:
    """
    Integration layer for PyGent Factory components.
    
    Features:
    - ToT reasoning system integration
    - RAG retrieval/generation integration
    - Web UI orchestration interface
    - Research workflow automation
    - Real-time component health monitoring
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        
        # Component registry
        self.components: Dict[str, PyGentComponent] = {}
        self.integration_mappings: List[IntegrationMapping] = []
        
        # Integration state
        self.is_connected = False
        self.connection_status: Dict[str, str] = {}
        
        # Health monitoring
        self.health_check_interval = timedelta(minutes=2)
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Initialize component definitions
        self._initialize_component_definitions()
        self._initialize_integration_mappings()
        
        logger.info("PyGent Factory Integration initialized")
    
    async def start(self):
        """Start the integration system."""
        try:
            # Connect to PyGent components
            await self._connect_to_components()
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.is_connected = True
            logger.info("PyGent Factory Integration started")
            
        except Exception as e:
            logger.error(f"Integration startup failed: {e}")
            raise
    
    async def stop(self):
        """Stop the integration system."""
        try:
            # Stop health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
            
            # Disconnect from components
            await self._disconnect_from_components()
            
            self.is_connected = False
            logger.info("PyGent Factory Integration stopped")
            
        except Exception as e:
            logger.error(f"Integration shutdown failed: {e}")
    
    async def execute_tot_reasoning(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute Tree of Thought reasoning using PyGent ToT system."""
        try:
            tot_component = self.components.get("tot_reasoning")
            if not tot_component or tot_component.status != "active":
                raise ValueError("ToT reasoning component not available")
            
            # Prepare request
            request_data = {
                "problem": problem,
                "context": context or {},
                "reasoning_depth": 3,
                "exploration_breadth": 4,
                "evaluation_criteria": ["feasibility", "creativity", "effectiveness"]
            }
            
            # Execute reasoning
            result = await self._call_component_endpoint(
                tot_component, "reason", request_data
            )
            
            # Transform result for orchestration system
            transformed_result = {
                "reasoning_path": result.get("reasoning_steps", []),
                "solution": result.get("best_solution", ""),
                "confidence": result.get("confidence_score", 0.0),
                "alternatives": result.get("alternative_solutions", []),
                "execution_time": result.get("processing_time", 0.0)
            }
            
            logger.info(f"ToT reasoning completed: {problem[:50]}...")
            return transformed_result
            
        except Exception as e:
            logger.error(f"ToT reasoning failed: {e}")
            raise
    
    async def execute_rag_retrieval(self, query: str, domain: str = None) -> Dict[str, Any]:
        """Execute RAG retrieval using PyGent RAG system."""
        try:
            rag_component = self.components.get("rag_retrieval")
            if not rag_component or rag_component.status != "active":
                raise ValueError("RAG retrieval component not available")
            
            # Prepare request
            request_data = {
                "query": query,
                "domain": domain,
                "max_results": 10,
                "similarity_threshold": 0.7,
                "include_metadata": True
            }
            
            # Execute retrieval
            result = await self._call_component_endpoint(
                rag_component, "retrieve", request_data
            )
            
            # Transform result
            transformed_result = {
                "documents": result.get("retrieved_documents", []),
                "total_results": result.get("total_count", 0),
                "query_embedding": result.get("query_vector", []),
                "retrieval_time": result.get("processing_time", 0.0)
            }
            
            logger.info(f"RAG retrieval completed: {len(transformed_result['documents'])} documents")
            return transformed_result
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            raise
    
    async def execute_rag_generation(self, context: List[str], query: str) -> Dict[str, Any]:
        """Execute RAG generation using PyGent RAG system."""
        try:
            rag_component = self.components.get("rag_generation")
            if not rag_component or rag_component.status != "active":
                raise ValueError("RAG generation component not available")
            
            # Prepare request
            request_data = {
                "context_documents": context,
                "query": query,
                "max_length": 500,
                "temperature": 0.7,
                "include_citations": True
            }
            
            # Execute generation
            result = await self._call_component_endpoint(
                rag_component, "generate", request_data
            )
            
            # Transform result
            transformed_result = {
                "generated_text": result.get("generated_response", ""),
                "citations": result.get("source_citations", []),
                "confidence": result.get("generation_confidence", 0.0),
                "generation_time": result.get("processing_time", 0.0)
            }
            
            logger.info(f"RAG generation completed: {len(transformed_result['generated_text'])} characters")
            return transformed_result
            
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            raise
    
    async def execute_research_workflow(self, research_topic: str, workflow_type: str = "comprehensive") -> Dict[str, Any]:
        """Execute research workflow using PyGent research system."""
        try:
            research_component = self.components.get("research_workflow")
            if not research_component or research_component.status != "active":
                raise ValueError("Research workflow component not available")
            
            # Prepare request
            request_data = {
                "topic": research_topic,
                "workflow_type": workflow_type,
                "depth": "detailed",
                "sources": ["academic", "primary", "secondary"],
                "output_format": "structured"
            }
            
            # Execute workflow
            result = await self._call_component_endpoint(
                research_component, "execute_workflow", request_data
            )
            
            # Transform result
            transformed_result = {
                "research_summary": result.get("summary", ""),
                "sources_found": result.get("source_count", 0),
                "key_findings": result.get("findings", []),
                "research_quality": result.get("quality_score", 0.0),
                "execution_time": result.get("total_time", 0.0)
            }
            
            logger.info(f"Research workflow completed: {research_topic}")
            return transformed_result
            
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            raise
    
    async def update_web_ui_status(self, orchestration_status: Dict[str, Any]) -> bool:
        """Update web UI with orchestration status."""
        try:
            ui_component = self.components.get("web_ui")
            if not ui_component or ui_component.status != "active":
                return False
            
            # Prepare status update
            ui_update = {
                "orchestration_status": orchestration_status,
                "timestamp": datetime.utcnow().isoformat(),
                "component_health": self.connection_status
            }
            
            # Send update to UI
            result = await self._call_component_endpoint(
                ui_component, "update_status", ui_update
            )
            
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Web UI status update failed: {e}")
            return False
    
    async def get_component_health(self) -> Dict[str, Any]:
        """Get health status of all PyGent components."""
        health_status = {}
        
        for component_id, component in self.components.items():
            try:
                # Perform health check
                health_result = await self._call_component_endpoint(
                    component, "health", {}
                )
                
                health_status[component_id] = {
                    "status": "healthy" if health_result.get("healthy", False) else "unhealthy",
                    "last_check": datetime.utcnow().isoformat(),
                    "response_time": health_result.get("response_time", 0.0),
                    "details": health_result.get("details", {})
                }
                
            except Exception as e:
                health_status[component_id] = {
                    "status": "error",
                    "last_check": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
        
        return health_status
    
    def _initialize_component_definitions(self):
        """Initialize PyGent Factory component definitions."""
        components = [
            PyGentComponent(
                component_id="tot_reasoning",
                component_type="reasoning",
                name="Tree of Thought Reasoning",
                description="Advanced reasoning system using tree-of-thought methodology",
                capabilities=["reasoning", "problem_solving", "analysis"],
                endpoints={
                    "reason": "/api/tot/reason",
                    "health": "/api/tot/health"
                }
            ),
            PyGentComponent(
                component_id="rag_retrieval",
                component_type="retrieval",
                name="RAG Retrieval System",
                description="Retrieval-Augmented Generation retrieval component",
                capabilities=["document_retrieval", "semantic_search"],
                endpoints={
                    "retrieve": "/api/rag/retrieve",
                    "health": "/api/rag/health"
                }
            ),
            PyGentComponent(
                component_id="rag_generation",
                component_type="generation",
                name="RAG Generation System",
                description="Retrieval-Augmented Generation text generation component",
                capabilities=["text_generation", "context_synthesis"],
                endpoints={
                    "generate": "/api/rag/generate",
                    "health": "/api/rag/health"
                }
            ),
            PyGentComponent(
                component_id="research_workflow",
                component_type="workflow",
                name="Research Workflow System",
                description="Automated research workflow execution",
                capabilities=["research_automation", "workflow_execution"],
                endpoints={
                    "execute_workflow": "/api/research/execute",
                    "health": "/api/research/health"
                }
            ),
            PyGentComponent(
                component_id="web_ui",
                component_type="interface",
                name="Web User Interface",
                description="Web-based user interface for PyGent Factory",
                capabilities=["user_interface", "status_display"],
                endpoints={
                    "update_status": "/api/ui/status",
                    "health": "/api/ui/health"
                }
            )
        ]
        
        for component in components:
            self.components[component.component_id] = component
    
    def _initialize_integration_mappings(self):
        """Initialize integration mappings between orchestration and PyGent."""
        mappings = [
            IntegrationMapping(
                orchestration_capability="reasoning",
                pygent_component="tot_reasoning",
                endpoint="reason"
            ),
            IntegrationMapping(
                orchestration_capability="retrieval",
                pygent_component="rag_retrieval",
                endpoint="retrieve"
            ),
            IntegrationMapping(
                orchestration_capability="generation",
                pygent_component="rag_generation",
                endpoint="generate"
            ),
            IntegrationMapping(
                orchestration_capability="research",
                pygent_component="research_workflow",
                endpoint="execute_workflow"
            )
        ]
        
        self.integration_mappings = mappings
    
    async def _connect_to_components(self):
        """Connect to all PyGent Factory components."""
        for component_id, component in self.components.items():
            try:
                # Attempt to connect to component
                # In a real implementation, this would establish actual connections
                component.status = "active"
                component.last_health_check = datetime.utcnow()
                self.connection_status[component_id] = "connected"
                
                logger.info(f"Connected to component: {component.name}")
                
            except Exception as e:
                component.status = "error"
                self.connection_status[component_id] = f"error: {str(e)}"
                logger.error(f"Failed to connect to component {component.name}: {e}")
    
    async def _disconnect_from_components(self):
        """Disconnect from all PyGent Factory components."""
        for component_id, component in self.components.items():
            component.status = "inactive"
            self.connection_status[component_id] = "disconnected"
    
    async def _call_component_endpoint(self, component: PyGentComponent, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call a component endpoint."""
        try:
            # In a real implementation, this would make actual HTTP/API calls
            # For now, return simulated responses
            
            if endpoint == "reason":
                return {
                    "reasoning_steps": [
                        f"Analyzed problem: {data.get('problem', '')[:50]}...",
                        "Generated multiple solution approaches",
                        "Evaluated feasibility and effectiveness",
                        "Selected optimal solution path"
                    ],
                    "best_solution": f"Optimized solution for: {data.get('problem', '')}",
                    "confidence_score": 0.85,
                    "alternative_solutions": ["Alternative 1", "Alternative 2"],
                    "processing_time": 1.2
                }
            
            elif endpoint == "retrieve":
                return {
                    "retrieved_documents": [
                        {"title": f"Document 1 for {data.get('query', '')}", "relevance": 0.9},
                        {"title": f"Document 2 for {data.get('query', '')}", "relevance": 0.8},
                        {"title": f"Document 3 for {data.get('query', '')}", "relevance": 0.7}
                    ],
                    "total_count": 3,
                    "query_vector": [0.1, 0.2, 0.3],
                    "processing_time": 0.5
                }
            
            elif endpoint == "generate":
                return {
                    "generated_response": f"Generated response based on context for query: {data.get('query', '')}",
                    "source_citations": ["Source 1", "Source 2"],
                    "generation_confidence": 0.88,
                    "processing_time": 0.8
                }
            
            elif endpoint == "execute_workflow":
                return {
                    "summary": f"Research completed on topic: {data.get('topic', '')}",
                    "source_count": 15,
                    "findings": ["Finding 1", "Finding 2", "Finding 3"],
                    "quality_score": 0.92,
                    "total_time": 45.0
                }
            
            elif endpoint == "update_status":
                return {"success": True}
            
            elif endpoint == "health":
                return {
                    "healthy": True,
                    "response_time": 0.1,
                    "details": {"status": "operational"}
                }
            
            else:
                return {"error": f"Unknown endpoint: {endpoint}"}
                
        except Exception as e:
            logger.error(f"Component endpoint call failed: {e}")
            raise
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                # Perform health checks on all components
                health_status = await self.get_component_health()
                
                # Update component statuses
                for component_id, health in health_status.items():
                    if component_id in self.components:
                        component = self.components[component_id]
                        component.last_health_check = datetime.utcnow()
                        
                        if health["status"] == "healthy":
                            component.status = "active"
                        else:
                            component.status = "error"
                
                await asyncio.sleep(self.health_check_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration metrics."""
        return {
            "is_connected": self.is_connected,
            "total_components": len(self.components),
            "active_components": len([c for c in self.components.values() if c.status == "active"]),
            "connection_status": dict(self.connection_status),
            "component_details": {
                component_id: {
                    "name": component.name,
                    "type": component.component_type,
                    "status": component.status,
                    "capabilities": component.capabilities,
                    "last_health_check": component.last_health_check.isoformat() if component.last_health_check else None
                }
                for component_id, component in self.components.items()
            },
            "integration_mappings": len(self.integration_mappings)
        }