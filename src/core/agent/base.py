"""
Base Agent Implementation

This module provides the base agent class that all agents inherit from.
It implements the core MCP-compliant agent functionality.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from .message import AgentMessage, MessageType
from .capability import AgentCapability
from .config import AgentConfig
from .status import AgentStatus, AgentStatusManager


logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent-related errors"""
    pass


class BaseAgent(ABC):
    """
    Base agent class implementing MCP-compliant agent functionality.
    
    This abstract base class provides the core infrastructure that all
    agents inherit from, including message handling, capability management,
    status tracking, and lifecycle management.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.agent_id = config.agent_id
        self.name = config.name
        self.type = config.agent_type
        
        # Status management
        self.status_manager = AgentStatusManager()
        
        # Core components
        self.capabilities: Dict[str, AgentCapability] = {}
        self.mcp_tools: Dict[str, Any] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=config.message_queue_size)
        
        # Lifecycle tracking
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Task management
        self.active_tasks: Set[asyncio.Task] = set()
        self.task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        
        # Shutdown flag
        self._shutdown_event = asyncio.Event()

        # Enhanced augmentation support - Phase 1.2
        self.augmentation_enabled = config.custom_config.get("augmentation_enabled", False)
        self.rag_enabled = config.custom_config.get("rag_enabled", False)
        self.lora_enabled = config.custom_config.get("lora_enabled", False)
        self.riper_omega_enabled = config.custom_config.get("riper_omega_enabled", False)
        self.cooperative_enabled = config.custom_config.get("cooperative_enabled", False)

        # Augmentation components (initialized later)
        self.rag_augmenter = None
        self.lora_adapter = None
        self.riper_omega_manager = None
        self.cooperation_manager = None

        # Performance tracking for augmentation
        self.augmentation_metrics = {
            "total_requests": 0,
            "augmented_requests": 0,
            "rag_retrievals": 0,
            "lora_generations": 0,
            "lora_adaptations": 0,
            "riper_omega_chains": 0,
            "cooperative_actions": 0,
            "performance_improvement": 0.0
        }

        # Initialize logger
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self.logger.setLevel(getattr(logging, config.log_level))
    
    @property
    def status(self) -> AgentStatus:
        """Get current agent status"""
        return self.status_manager.get_status()
    
    @property
    def is_active(self) -> bool:
        """Check if agent is active"""
        return self.status_manager.is_active()
    
    @property
    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return self.status_manager.is_available()
    
    async def initialize(self) -> None:
        """
        Initialize the agent.
        
        This method should be called after creating the agent instance
        to complete the initialization process.
        """
        try:
            self.logger.info(f"Initializing agent {self.name} ({self.agent_id})")
            
            # Transition to initializing status
            self.status_manager.transition_to(AgentStatus.INITIALIZING, "Starting initialization")
            
            # Load capabilities
            await self._load_capabilities()
            
            # Initialize MCP tools
            await self._initialize_mcp_tools()

            # Initialize augmentation components - Phase 1.2
            await self._initialize_augmentations()

            # Perform agent-specific initialization
            await self._agent_initialize()

            # Transition to active status
            self.status_manager.transition_to(AgentStatus.ACTIVE, "Initialization complete")
            
            self.logger.info(f"Agent {self.name} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {str(e)}")
            self.status_manager.transition_to(AgentStatus.ERROR, f"Initialization failed: {str(e)}")
            raise AgentError(f"Agent initialization failed: {str(e)}")
    
    async def shutdown(self) -> None:
        """
        Shutdown the agent gracefully.
        """
        try:
            self.logger.info(f"Shutting down agent {self.name}")
            
            # Transition to stopping status
            self.status_manager.transition_to(AgentStatus.STOPPING, "Shutdown initiated")
            
            # Set shutdown event
            self._shutdown_event.set()
            
            # Cancel active tasks
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete or timeout
            if self.active_tasks:
                await asyncio.wait(self.active_tasks, timeout=30.0)
            
            # Perform agent-specific cleanup
            await self._agent_shutdown()
            
            # Transition to stopped status
            self.status_manager.transition_to(AgentStatus.STOPPED, "Shutdown complete")
            
            self.logger.info(f"Agent {self.name} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            self.status_manager.transition_to(AgentStatus.ERROR, f"Shutdown error: {str(e)}")
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message.
        
        Args:
            message: The message to process
            
        Returns:
            AgentMessage: Response message
        """
        try:
            self.last_activity = datetime.utcnow()
            
            # Validate message
            if message.recipient != self.agent_id:
                raise AgentError(f"Message not addressed to this agent: {message.recipient}")
            
            # Update status to busy
            if self.status_manager.can_transition_to(AgentStatus.BUSY):
                self.status_manager.transition_to(AgentStatus.BUSY, "Processing message")
            
            # Route message based on type
            if message.type == MessageType.REQUEST:
                response = await self._handle_request(message)
            elif message.type == MessageType.TOOL_CALL:
                response = await self._handle_tool_call(message)
            elif message.type == MessageType.CAPABILITY_REQUEST:
                response = await self._handle_capability_request(message)
            elif message.type == MessageType.NOTIFICATION:
                response = await self._handle_notification(message)
            else:
                response = message.create_error_response(
                    "UNSUPPORTED_MESSAGE_TYPE",
                    f"Unsupported message type: {message.type.value}"
                )
            
            # Mark message as processed
            message.mark_processed()
            
            # Return to idle status
            if self.status_manager.can_transition_to(AgentStatus.IDLE):
                self.status_manager.transition_to(AgentStatus.IDLE, "Message processed")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            self.status_manager.add_error(str(e))
            
            return message.create_error_response(
                "MESSAGE_PROCESSING_ERROR",
                f"Failed to process message: {str(e)}"
            )
    
    async def execute_capability(self, capability_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a capability.
        
        Args:
            capability_name: Name of the capability to execute
            parameters: Parameters for the capability
            
        Returns:
            Any: Capability execution result
        """
        try:
            self.last_activity = datetime.utcnow()
            
            # Check if capability exists
            if capability_name not in self.capabilities:
                raise AgentError(f"Capability not found: {capability_name}")
            
            capability = self.capabilities[capability_name]
            
            # Check if capability is enabled
            if not self.config.is_capability_enabled(capability_name):
                raise AgentError(f"Capability disabled: {capability_name}")
            
            # Validate parameters
            is_valid, errors = capability.validate_parameters(parameters)
            if not is_valid:
                raise AgentError(f"Parameter validation failed: {', '.join(errors)}")
            
            # Execute capability
            async with self.task_semaphore:
                result = await self._execute_capability_impl(capability, parameters)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing capability {capability_name}: {str(e)}")
            raise AgentError(f"Capability execution failed: {str(e)}")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get list of agent capabilities"""
        return list(self.capabilities.values())
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed agent status"""
        status_info = self.status_manager.get_status_info()
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.type,
            "status": status_info.to_dict(),
            "capabilities": [cap.name for cap in self.capabilities.values()],
            "mcp_tools": list(self.mcp_tools.keys()),
            "active_tasks": len(self.active_tasks),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "config": {
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "memory_enabled": self.config.memory_enabled,
                "memory_limit": self.config.memory_limit
            }
        }
    
    async def _initialize_augmentations(self) -> None:
        """
        Initialize augmentation components based on configuration.
        Phase 1.2: Enhanced Agent Base Class Integration
        """
        try:
            if not self.augmentation_enabled:
                self.logger.info("Augmentation disabled for this agent")
                return

            self.logger.info("Initializing agent augmentations...")

            # Initialize RAG augmentation (Phase 2)
            if self.rag_enabled:
                try:
                    # Import RAG augmentation components
                    from ...agents.augmentation.rag_augmenter import RAGAugmenter
                    from ...storage.vector import VectorStoreManager
                    from ...utils.embedding import get_embedding_service
                    from ...database.connection import get_database_manager
                    from ...config.settings import Settings

                    # Initialize components
                    settings = Settings()
                    db_manager = get_database_manager()
                    vector_store_manager = VectorStoreManager(settings, db_manager)
                    embedding_service = get_embedding_service()

                    # Create RAG augmenter
                    self.rag_augmenter = RAGAugmenter(
                        vector_store_manager=vector_store_manager,
                        embedding_service=embedding_service
                    )

                    # Initialize RAG augmenter
                    await self.rag_augmenter.initialize()

                    self.logger.info("RAG augmentation initialized successfully")

                except Exception as e:
                    self.logger.warning(f"RAG initialization failed: {e}")
                    self.rag_enabled = False
                    self.rag_augmenter = None

            # Initialize LoRA adaptation (Phase 2.3)
            if self.lora_enabled:
                try:
                    from ...ai.fine_tune import LoRAFineTuner, LoRAConfig

                    # Create LoRA fine-tuner with agent-specific config
                    lora_config = LoRAConfig(
                        max_steps=self.config.get_custom_config("lora_max_steps", 30),
                        learning_rate=self.config.get_custom_config("lora_learning_rate", 2e-4),
                        r=self.config.get_custom_config("lora_r", 16)
                    )

                    self.lora_fine_tuner = LoRAFineTuner(lora_config)

                    # Try to load existing fine-tuned model if specified
                    model_path = self.config.get_custom_config("lora_model_path")
                    if model_path:
                        await self.lora_fine_tuner.load_fine_tuned_model(model_path)
                        self.logger.info(f"LoRA model loaded: {model_path}")
                    else:
                        await self.lora_fine_tuner.initialize()
                        self.logger.info("LoRA fine-tuner initialized")

                except Exception as e:
                    self.logger.warning(f"LoRA initialization failed: {e}")
                    self.lora_enabled = False
                    self.lora_fine_tuner = None

            # Initialize RIPER-Ω protocol (Phase 2.3)
            if self.riper_omega_enabled:
                try:
                    from ...core.riper_omega_protocol import RIPERProtocol

                    # Create RIPER-Ω protocol with agent-specific config
                    hallucination_threshold = self.config.get_custom_config("riper_hallucination_threshold", 0.4)
                    self.riper_protocol = RIPERProtocol(hallucination_threshold)

                    self.logger.info(f"RIPER-Ω protocol initialized (threshold: {hallucination_threshold})")

                except Exception as e:
                    self.logger.warning(f"RIPER-Ω initialization failed: {e}")
                    self.riper_omega_enabled = False
                    self.riper_protocol = None

            # Initialize cooperative capabilities (Phase 5)
            if self.cooperative_enabled:
                try:
                    # Will be implemented in Phase 5
                    self.logger.info("Cooperative capabilities enabled (will be initialized in Phase 5)")
                except Exception as e:
                    self.logger.warning(f"Cooperative initialization failed: {e}")
                    self.cooperative_enabled = False

            self.logger.info("Augmentation initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize augmentations: {e}")
            # Disable all augmentations on failure
            self.augmentation_enabled = False
            self.rag_enabled = False
            self.lora_enabled = False
            self.riper_omega_enabled = False
            self.cooperative_enabled = False

    async def _augmented_generate(self, prompt: str, **kwargs) -> str:
        """
        Enhanced generation method with augmentation support.
        Phase 1.2: Foundation for augmented generation
        """
        try:
            self.augmentation_metrics["total_requests"] += 1

            # Track if any augmentation is used
            augmentation_used = False

            # Phase 2: RAG augmentation (implemented)
            if self.rag_enabled and self.rag_augmenter:
                try:
                    # Use RAG augmenter to enhance the prompt
                    rag_result = await self.rag_augmenter.augment_prompt(prompt, **kwargs)
                    if rag_result.success:
                        prompt = rag_result.augmented_prompt
                        self.augmentation_metrics["rag_retrievals"] += 1
                        augmentation_used = True

                        # Add RAG metadata to kwargs for potential use by subclasses
                        kwargs['rag_metadata'] = {
                            'retrieved_documents': len(rag_result.retrieved_documents),
                            'retrieval_time_ms': rag_result.retrieval_time_ms,
                            'average_relevance': sum(rag_result.relevance_scores) / len(rag_result.relevance_scores) if rag_result.relevance_scores else 0.0
                        }
                    else:
                        self.logger.warning(f"RAG augmentation failed: {rag_result.error_message}")
                except Exception as e:
                    self.logger.error(f"RAG augmentation error: {e}")
            elif self.rag_enabled:
                # RAG enabled but augmenter not initialized - simulate for backward compatibility
                self.augmentation_metrics["rag_retrievals"] += 1
                augmentation_used = True

            # Phase 2.3: LoRA fine-tuning enhancement
            if self.lora_enabled and self.lora_fine_tuner:
                try:
                    # Use LoRA fine-tuned model for generation
                    lora_generated = await self.lora_fine_tuner.generate(prompt, max_length=1024)
                    if lora_generated and len(lora_generated.strip()) > 10:
                        prompt = lora_generated  # Use LoRA output as enhanced prompt
                        self.augmentation_metrics["lora_generations"] += 1
                        augmentation_used = True

                        # Add LoRA metadata to kwargs
                        kwargs['lora_metadata'] = {
                            'model_used': True,
                            'generation_length': len(lora_generated),
                            'model_path': getattr(self.lora_fine_tuner, 'current_model_path', 'unknown')
                        }
                    else:
                        self.logger.warning("LoRA generation produced empty or short result")
                except Exception as e:
                    self.logger.error(f"LoRA generation error: {e}")
            elif self.lora_enabled:
                # LoRA enabled but fine-tuner not initialized
                self.augmentation_metrics["lora_generations"] += 1
                augmentation_used = True

            # Phase 2.3: RIPER-Ω protocol integration
            if self.riper_omega_enabled and self.riper_protocol:
                try:
                    # Use RIPER-Ω protocol for structured generation
                    task_type = kwargs.get('task_type', 'general')

                    # Run RIPER protocol chain for complex tasks
                    if task_type in ['code_generation', 'research', 'analysis']:
                        riper_result = await self.riper_protocol.run_full_protocol(prompt)

                        if riper_result.success and riper_result.hallucination_score < 0.4:
                            prompt = riper_result.final_output
                            self.augmentation_metrics["riper_omega_chains"] += 1
                            augmentation_used = True

                            # Add RIPER metadata to kwargs
                            kwargs['riper_metadata'] = {
                                'protocol_used': True,
                                'confidence_score': riper_result.confidence_score,
                                'hallucination_score': riper_result.hallucination_score,
                                'mode_chain': [mode.value for mode in riper_result.mode_chain]
                            }
                        else:
                            self.logger.warning(f"RIPER-Ω protocol failed or high hallucination score: {riper_result.hallucination_score:.2f}")

                except Exception as e:
                    self.logger.error(f"RIPER-Ω protocol error: {e}")
            elif self.riper_omega_enabled:
                # RIPER-Ω enabled but protocol not initialized
                self.augmentation_metrics["riper_omega_chains"] += 1
                augmentation_used = True

            # Phase 3: LoRA adaptation (placeholder - simulate for Phase 1.2 testing)
            if self.lora_enabled:
                # Will be implemented in Phase 3, for now simulate the augmentation
                self.augmentation_metrics["lora_adaptations"] += 1
                augmentation_used = True

            # Phase 4: RIPER-Ω chaining (placeholder - simulate for Phase 1.2 testing)
            if self.riper_omega_enabled:
                # Will be implemented in Phase 4, for now simulate the augmentation
                self.augmentation_metrics["riper_omega_chains"] += 1
                augmentation_used = True

            # Phase 5: Cooperative generation (placeholder - simulate for Phase 1.2 testing)
            if self.cooperative_enabled:
                # Will be implemented in Phase 5, for now simulate the augmentation
                self.augmentation_metrics["cooperative_actions"] += 1
                augmentation_used = True

            if augmentation_used:
                self.augmentation_metrics["augmented_requests"] += 1

            # For now, return the original prompt (will be enhanced in later phases)
            if augmentation_used:
                return f"[Augmented] {prompt}"
            else:
                return prompt

        except Exception as e:
            self.logger.error(f"Augmented generation failed: {e}")
            # Fallback to basic generation
            return prompt

    def get_augmentation_metrics(self) -> Dict[str, Any]:
        """Get current augmentation performance metrics"""
        total_requests = self.augmentation_metrics["total_requests"]
        augmented_requests = self.augmentation_metrics["augmented_requests"]

        if total_requests > 0:
            augmentation_rate = augmented_requests / total_requests
        else:
            augmentation_rate = 0.0

        return {
            **self.augmentation_metrics,
            "augmentation_rate": augmentation_rate,
            "augmentation_enabled": self.augmentation_enabled,
            "rag_enabled": self.rag_enabled,
            "lora_enabled": self.lora_enabled,
            "riper_omega_enabled": self.riper_omega_enabled,
            "cooperative_enabled": self.cooperative_enabled
        }

    # Abstract methods that subclasses must implement
    @abstractmethod
    async def _agent_initialize(self) -> None:
        """Agent-specific initialization logic"""
        pass
    
    @abstractmethod
    async def _agent_shutdown(self) -> None:
        """Agent-specific shutdown logic"""
        pass
    
    @abstractmethod
    async def _handle_request(self, message: AgentMessage) -> AgentMessage:
        """Handle a request message"""
        pass
    
    # Protected methods for internal use
    async def _load_capabilities(self) -> None:
        """Load agent capabilities"""
        # This will be implemented by capability manager integration
        pass
    
    async def _initialize_mcp_tools(self) -> None:
        """Initialize MCP tools"""
        # This will be implemented by MCP manager integration
        pass
    
    async def _handle_tool_call(self, message: AgentMessage) -> AgentMessage:
        """Handle a tool call message"""
        try:
            content = message.content
            tool_name = content.get("tool_name")
            arguments = content.get("arguments", {})
            
            if not tool_name:
                return message.create_error_response(
                    "MISSING_TOOL_NAME",
                    "Tool name is required for tool call"
                )
            
            # Execute tool (this will be implemented by MCP integration)
            result = await self._execute_tool(tool_name, arguments)
            
            return message.create_response({
                "tool_result": result
            }, MessageType.TOOL_RESULT)
            
        except Exception as e:
            return message.create_error_response(
                "TOOL_EXECUTION_ERROR",
                f"Tool execution failed: {str(e)}"
            )
    
    async def _handle_capability_request(self, message: AgentMessage) -> AgentMessage:
        """Handle a capability request message"""
        try:
            content = message.content
            capability_name = content.get("capability")
            parameters = content.get("parameters", {})
            
            if not capability_name:
                return message.create_error_response(
                    "MISSING_CAPABILITY_NAME",
                    "Capability name is required"
                )
            
            result = await self.execute_capability(capability_name, parameters)
            
            return message.create_response({
                "capability_result": result
            }, MessageType.CAPABILITY_RESPONSE)
            
        except Exception as e:
            return message.create_error_response(
                "CAPABILITY_EXECUTION_ERROR",
                f"Capability execution failed: {str(e)}"
            )
    
    async def _handle_notification(self, message: AgentMessage) -> AgentMessage:
        """Handle a notification message"""
        # Default implementation - just acknowledge
        return message.create_response({
            "acknowledged": True,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _execute_capability_impl(self, capability: AgentCapability,
                                     parameters: Dict[str, Any]) -> Any:
        """Execute capability implementation"""
        try:
            # Get capability manager from global state
            from ...core.capability_system import get_capability_manager
            capability_manager = await get_capability_manager()

            if capability_manager:
                return await capability_manager.execute_capability(
                    self.agent_id, capability, parameters
                )
            else:
                # Fallback: basic capability execution
                logger.warning(f"No capability manager available, using basic execution for {capability.name}")
                return {
                    "capability": capability.name,
                    "status": "executed",
                    "result": f"Basic execution of {capability.name} with parameters: {parameters}",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"Capability execution failed: {e}")
            return {
                "capability": capability.name,
                "status": "failed",
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute MCP tool"""
        try:
            # Get MCP manager from global state
            from ...mcp.server_registry import get_mcp_manager
            mcp_manager = await get_mcp_manager()

            if mcp_manager:
                return await mcp_manager.execute_tool(tool_name, arguments)
            else:
                # Fallback: basic tool execution simulation
                logger.warning(f"No MCP manager available, using basic execution for tool {tool_name}")
                return {
                    "tool": tool_name,
                    "status": "executed",
                    "result": f"Basic execution of tool {tool_name} with arguments: {arguments}",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "tool": tool_name,
                "status": "failed",
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def __str__(self) -> str:
        """String representation"""
        return f"BaseAgent(id={self.agent_id}, name={self.name}, type={self.type})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"BaseAgent(id={self.agent_id}, name={self.name}, type={self.type}, "
                f"status={self.status.value})")
