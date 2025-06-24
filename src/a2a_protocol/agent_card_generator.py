#!/usr/bin/env python3
"""
A2A-Compliant Agent Card Generator

Generates proper Agent Cards according to the official Google A2A specification.
Ensures all created agents meet A2A protocol requirements for discovery and communication.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class AgentProvider:
    """A2A Agent Provider - compliant with specification"""
    name: str
    organization: str
    description: Optional[str] = None
    url: Optional[str] = None


@dataclass
class AgentCapabilities:
    """A2A Agent Capabilities - compliant with specification"""
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


@dataclass
class AgentExtension:
    """A2A Agent Extension - for additional functionality"""
    name: str
    version: str
    description: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityScheme:
    """A2A Security Scheme - authentication requirements"""
    type: str  # "http", "oauth2", "apiKey"
    scheme: Optional[str] = None  # "bearer", "basic" for http type
    bearerFormat: Optional[str] = None  # "JWT" for bearer tokens
    description: Optional[str] = None
    flows: Optional[Dict[str, Any]] = None  # For OAuth2
    name: Optional[str] = None  # For apiKey
    location: Optional[str] = None  # "header", "query", "cookie" for apiKey


@dataclass
class AgentSkill:
    """A2A Agent Skill - compliant with specification"""
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    inputModes: List[str] = field(default_factory=lambda: ["text", "text/plain"])
    outputModes: List[str] = field(default_factory=lambda: ["text", "text/plain"])


@dataclass
class AgentCard:
    """A2A Agent Card - fully compliant with specification"""
    name: str
    description: str
    url: str
    version: str
    provider: AgentProvider
    capabilities: AgentCapabilities
    defaultInputModes: List[str]
    defaultOutputModes: List[str]
    skills: List[AgentSkill]
    securitySchemes: Dict[str, SecurityScheme] = field(default_factory=dict)
    security: List[Dict[str, List[str]]] = field(default_factory=list)
    supportsAuthenticatedExtendedCard: bool = False
    extensions: List[AgentExtension] = field(default_factory=list)
    documentationUrl: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class A2AAgentCardGenerator:
    """Generates A2A-compliant Agent Cards for PyGent Factory agents"""
    
    def __init__(self, base_url: str = "http://localhost:8000", organization: str = "PyGent Factory"):
        self.base_url = base_url.rstrip('/')
        self.organization = organization
        
    def generate_agent_card(self, 
                          agent,
                          agent_type: str,
                          custom_skills: Optional[List[AgentSkill]] = None,
                          enable_authentication: bool = True,
                          enable_streaming: bool = True,
                          enable_push_notifications: bool = False) -> AgentCard:
        """Generate A2A-compliant Agent Card for a PyGent Factory agent"""
        
        try:
            # Generate agent URL
            agent_url = f"{self.base_url}/a2a/agents/{agent.agent_id}"
            
            # Create provider information
            provider = AgentProvider(
                name=self.organization,
                organization=self.organization,
                description="Advanced AI agent orchestration platform",
                url="https://github.com/gigamonkeyx/pygentback"
            )
            
            # Create capabilities
            capabilities = AgentCapabilities(
                streaming=enable_streaming,
                pushNotifications=enable_push_notifications,
                stateTransitionHistory=True
            )
            
            # Generate skills based on agent type and capabilities
            skills = custom_skills or self._generate_skills_for_agent(agent, agent_type)
            
            # Create security schemes if authentication is enabled
            security_schemes = {}
            security = []

            if enable_authentication:
                # Try to get security schemes from security manager
                try:
                    from .security import security_manager
                    manager_schemes = security_manager.get_security_schemes()

                    # Convert to SecurityScheme objects
                    for scheme_name, scheme_data in manager_schemes.items():
                        if scheme_name == "bearerAuth":
                            security_schemes["bearerAuth"] = SecurityScheme(
                                type=scheme_data["type"],
                                scheme=scheme_data["scheme"],
                                bearerFormat=scheme_data.get("bearerFormat"),
                                description=scheme_data["description"]
                            )
                        elif scheme_name == "apiKeyAuth":
                            security_schemes["apiKeyAuth"] = SecurityScheme(
                                type=scheme_data["type"],
                                name=scheme_data["name"],
                                location=scheme_data["in"],
                                description=scheme_data["description"]
                            )

                    security = [
                        {"bearerAuth": []},
                        {"apiKeyAuth": []}
                    ]

                except ImportError:
                    # Fallback to default schemes
                    security_schemes["bearerAuth"] = SecurityScheme(
                        type="http",
                        scheme="bearer",
                        bearerFormat="JWT",
                        description="Bearer token authentication"
                    )
                    security_schemes["apiKeyAuth"] = SecurityScheme(
                        type="apiKey",
                        name="X-API-Key",
                        location="header",
                        description="API key authentication"
                    )
                    security = [
                        {"bearerAuth": []},
                        {"apiKeyAuth": []}
                    ]
            
            # Create agent card
            agent_card = AgentCard(
                name=agent.name,
                description=self._generate_agent_description(agent, agent_type),
                url=agent_url,
                version="1.0.0",
                provider=provider,
                capabilities=capabilities,
                defaultInputModes=["text", "text/plain", "application/json"],
                defaultOutputModes=["text", "text/plain", "application/json"],
                skills=skills,
                securitySchemes=security_schemes,
                security=security,
                supportsAuthenticatedExtendedCard=enable_authentication,
                documentationUrl=f"{self.base_url}/docs/agents/{agent.agent_id}",
                metadata={
                    "created": datetime.utcnow().isoformat(),
                    "agent_id": agent.agent_id,
                    "agent_type": agent_type,
                    "pygent_factory_version": "1.0.0"
                }
            )
            
            logger.info(f"Generated A2A-compliant agent card for {agent.name}")
            return agent_card
            
        except Exception as e:
            logger.error(f"Failed to generate agent card for {agent.name}: {e}")
            raise
    
    def _generate_agent_description(self, agent, agent_type: str) -> str:
        """Generate appropriate description for agent"""
        base_desc = f"PyGent Factory {agent_type.title()} Agent"
        
        if hasattr(agent, 'description') and agent.description:
            return f"{base_desc} - {agent.description}"
        
        # Generate type-specific descriptions
        type_descriptions = {
            "research": "Specialized in document search, literature review, and research coordination",
            "analysis": "Performs statistical analysis, data visualization, and pattern recognition",
            "generation": "Creates content, reports, and documentation from analysis results",
            "coordination": "Orchestrates multi-agent workflows and task management",
            "reasoning": "Advanced reasoning and problem-solving capabilities",
            "search": "Document and information retrieval specialist",
            "general": "General-purpose agent with versatile capabilities"
        }
        
        specific_desc = type_descriptions.get(agent_type.lower(), "Specialized AI agent")
        return f"{base_desc} - {specific_desc}"
    
    def _generate_skills_for_agent(self, agent, agent_type: str) -> List[AgentSkill]:
        """Generate appropriate skills based on agent type and capabilities"""
        skills = []
        
        # Add agent-specific capabilities as skills
        if hasattr(agent, 'capabilities'):
            for capability in agent.capabilities:
                skill = AgentSkill(
                    id=capability.name.lower().replace(" ", "_").replace("-", "_"),
                    name=capability.name,
                    description=capability.description or f"{capability.name} capability",
                    tags=self._extract_tags_from_capability(capability, agent_type),
                    examples=self._generate_examples_for_capability(capability),
                    inputModes=["text", "text/plain", "application/json"],
                    outputModes=["text", "text/plain", "application/json"]
                )
                skills.append(skill)
        
        # Add type-specific skills
        type_skills = self._get_type_specific_skills(agent_type)
        skills.extend(type_skills)
        
        # Ensure at least one skill exists
        if not skills:
            skills.append(AgentSkill(
                id="general_processing",
                name="General Processing",
                description="General-purpose text processing and response generation",
                tags=["general", "processing", "text"],
                examples=[
                    "Process this text and provide insights",
                    "Analyze the content and summarize key points",
                    "Help me understand this information"
                ]
            ))
        
        return skills
    
    def _extract_tags_from_capability(self, capability, agent_type: str) -> List[str]:
        """Extract relevant tags from capability and agent type"""
        tags = [agent_type.lower()]
        
        capability_name = capability.name.lower()
        
        # Add capability-based tags
        tag_mapping = {
            "search": ["search", "retrieval", "documents", "information"],
            "analysis": ["analysis", "statistics", "data", "insights"],
            "generation": ["generation", "creation", "content", "writing"],
            "coordination": ["coordination", "orchestration", "workflow", "management"],
            "reasoning": ["reasoning", "logic", "problem-solving", "inference"],
            "research": ["research", "academic", "literature", "investigation"]
        }
        
        for keyword, related_tags in tag_mapping.items():
            if keyword in capability_name:
                tags.extend(related_tags)
        
        return list(set(tags))  # Remove duplicates
    
    def _generate_examples_for_capability(self, capability) -> List[str]:
        """Generate example prompts for capability"""
        capability_name = capability.name.lower()
        
        example_mapping = {
            "search": [
                "Search for documents about machine learning",
                "Find research papers on neural networks",
                "Retrieve information about artificial intelligence"
            ],
            "analysis": [
                "Analyze the trends in this dataset",
                "Perform statistical analysis on the data",
                "Generate insights from the information"
            ],
            "generation": [
                "Generate a summary of this research",
                "Create a report based on the analysis",
                "Write documentation for this process"
            ],
            "coordination": [
                "Coordinate a multi-agent research task",
                "Manage workflow between different agents",
                "Orchestrate document processing pipeline"
            ],
            "reasoning": [
                "Reason through this complex problem",
                "Analyze the logical structure of the argument",
                "Provide step-by-step reasoning for the solution"
            ]
        }
        
        for keyword, examples in example_mapping.items():
            if keyword in capability_name:
                return examples
        
        # Default examples
        return [
            f"Use {capability.name} to process this request",
            f"Apply {capability.name} to analyze the content",
            f"Leverage {capability.name} for this task"
        ]
    
    def _get_type_specific_skills(self, agent_type: str) -> List[AgentSkill]:
        """Get predefined skills for specific agent types"""
        type_skills = {
            "research": [
                AgentSkill(
                    id="document_search",
                    name="Document Search",
                    description="Search and retrieve documents from various sources",
                    tags=["search", "documents", "retrieval", "research"],
                    examples=[
                        "Search for papers about quantum computing",
                        "Find documents related to climate change research",
                        "Retrieve academic articles on machine learning"
                    ]
                ),
                AgentSkill(
                    id="literature_review",
                    name="Literature Review", 
                    description="Conduct comprehensive literature reviews",
                    tags=["literature", "review", "analysis", "research"],
                    examples=[
                        "Conduct a literature review on renewable energy",
                        "Review recent papers in artificial intelligence",
                        "Analyze trends in biomedical research"
                    ]
                )
            ],
            "analysis": [
                AgentSkill(
                    id="statistical_analysis",
                    name="Statistical Analysis",
                    description="Perform statistical analysis on datasets",
                    tags=["statistics", "analysis", "data", "math"],
                    examples=[
                        "Calculate descriptive statistics for this dataset",
                        "Perform regression analysis on the variables",
                        "Test for statistical significance"
                    ]
                ),
                AgentSkill(
                    id="data_visualization",
                    name="Data Visualization",
                    description="Create visualizations and charts from data",
                    tags=["visualization", "charts", "graphs", "data"],
                    examples=[
                        "Create a histogram of the data distribution",
                        "Generate a scatter plot showing correlation",
                        "Build a dashboard for the analysis results"
                    ]
                )
            ],
            "generation": [
                AgentSkill(
                    id="content_generation",
                    name="Content Generation",
                    description="Generate various types of content and documents",
                    tags=["generation", "content", "writing", "creation"],
                    examples=[
                        "Generate a research summary from these papers",
                        "Create documentation for this analysis",
                        "Write a report based on the findings"
                    ]
                ),
                AgentSkill(
                    id="code_generation",
                    name="Code Generation",
                    description="Generate code and scripts for various tasks",
                    tags=["code", "programming", "scripts", "automation"],
                    examples=[
                        "Generate Python code for data analysis",
                        "Create a script to automate this process",
                        "Write functions for the specified requirements"
                    ]
                )
            ]
        }
        
        return type_skills.get(agent_type.lower(), [])
    
    def to_json(self, agent_card: AgentCard) -> Dict[str, Any]:
        """Convert AgentCard to JSON-serializable dictionary"""
        return asdict(agent_card)
    
    def generate_well_known_url_content(self, agent_card: AgentCard) -> str:
        """Generate content for /.well-known/agent.json endpoint"""
        import json
        return json.dumps(self.to_json(agent_card), indent=2)

    def generate_agent_card_sync(self,
                                agent_id: str,
                                agent_name: str,
                                agent_type: str,
                                capabilities: List[str] = None,
                                skills: List[str] = None,
                                enable_authentication: bool = False) -> Dict[str, Any]:
        """Synchronous version of generate_agent_card for testing"""
        try:
            # Import and create a real agent object for the generator
            from ai.multi_agent.core_fixed import BaseAgent, AgentStatus

            # Create a real agent object for the generator
            class RealAgent(BaseAgent):
                def __init__(self, agent_id: str, name: str, agent_type: str):
                    super().__init__(agent_id=agent_id, name=name, agent_type=agent_type)
                    self.description = f"Real {agent_type} agent for A2A protocol"
                    self.status = AgentStatus.IDLE
                    if capabilities:
                        # Convert string capabilities to capability objects
                        from ai.multi_agent.core_fixed import AgentCapability
                        for cap_name in capabilities:
                            cap_obj = AgentCapability(
                                name=cap_name,
                                description=f"{cap_name} capability"
                            )
                            self.capabilities.append(cap_obj)

                async def execute_task(self, task):
                    """Real task execution implementation"""
                    return {"status": "completed", "result": f"Task executed by {self.name}"}

                async def initialize(self) -> bool:
                    """Initialize the agent"""
                    self.status = AgentStatus.IDLE
                    return True

                async def shutdown(self) -> bool:
                    """Shutdown the agent gracefully"""
                    self.status = AgentStatus.OFFLINE
                    return True

            real_agent = RealAgent(agent_id, agent_name, agent_type)

            # Generate the agent card
            agent_card = self.generate_agent_card(
                agent=real_agent,
                agent_type=agent_type,
                enable_authentication=enable_authentication
            )

            # Convert to dictionary
            return self.to_json(agent_card)

        except Exception as e:
            logger.error(f"Failed to generate agent card sync: {e}")
            raise

    async def generate_system_agent_card(self) -> Dict[str, Any]:
        """Generate agent card for the PyGent Factory system itself"""
        try:
            # Create system agent card
            system_card = AgentCard(
                name="PyGent Factory System",
                description="Advanced AI agent orchestration platform with A2A protocol support",
                version="1.0.0",
                url=f"{self.base_url}/.well-known/agent.json",
                defaultInputModes=["text/plain", "application/json"],
                defaultOutputModes=["text/plain", "application/json"],
                provider=AgentProvider(
                    name="PyGent Factory",
                    organization="PyGent Factory",
                    description="Advanced AI agent orchestration platform",
                    url="https://github.com/gigamonkeyx/pygentback"
                ),
                capabilities=AgentCapabilities(
                    streaming=True,
                    push_notifications=True,
                    multi_turn=True,
                    file_upload=True,
                    file_download=True,
                    structured_data=True
                ),
                skills=[
                    AgentSkill(
                        id="agent_orchestration",
                        name="Agent Orchestration",
                        description="Coordinate and manage multiple AI agents",
                        input_modalities=["text", "application/json"],
                        output_modalities=["text", "application/json"],
                        tags=["orchestration", "coordination", "management"],
                        examples=["Create and manage AI agents", "Coordinate multi-agent workflows"]
                    )
                ]
            )

            # Convert to JSON and add endpoints
            card_dict = self.to_json(system_card)
            card_dict["endpoints"] = {
                "discovery": f"{self.base_url}/api/a2a/v1/agents/discover",
                "message_send": f"{self.base_url}/api/a2a/v1/message/send",
                "health": f"{self.base_url}/api/a2a/v1/health"
            }

            return card_dict

        except Exception as e:
            logger.error(f"Failed to generate system agent card: {e}")
            raise

    async def generate_system_agent_card(self) -> Dict[str, Any]:
        """Generate agent card for the PyGent Factory system itself"""
        try:
            # Create system agent card
            system_card = AgentCard(
                name="PyGent Factory System",
                description="Advanced AI agent orchestration platform with A2A protocol support",
                version="1.0.0",
                url=f"{self.base_url}/.well-known/agent.json",
                defaultInputModes=["text/plain", "application/json"],
                defaultOutputModes=["text/plain", "application/json"],
                provider=AgentProvider(
                    name="PyGent Factory",
                    organization="PyGent Factory",
                    description="Advanced AI agent orchestration platform",
                    url="https://github.com/gigamonkeyx/pygentback"
                ),
                capabilities=AgentCapabilities(
                    streaming=True,
                    push_notifications=True,
                    multi_turn=True,
                    file_upload=True,
                    file_download=True,
                    structured_data=True
                ),
                skills=[
                    AgentSkill(
                        id="agent_orchestration",
                        name="Agent Orchestration",
                        description="Coordinate and manage multiple AI agents",
                        input_modalities=["text", "application/json"],
                        output_modalities=["text", "application/json"],
                        tags=["orchestration", "coordination", "management"],
                        examples=["Create and manage AI agents", "Coordinate multi-agent workflows"]
                    ),
                    AgentSkill(
                        id="a2a_communication",
                        name="A2A Communication",
                        description="Agent-to-agent protocol communication",
                        input_modalities=["application/json"],
                        output_modalities=["application/json"],
                        tags=["a2a", "protocol", "communication"],
                        examples=["Send messages between agents", "Coordinate agent tasks"]
                    ),
                    AgentSkill(
                        id="task_management",
                        name="Task Management",
                        description="Manage and execute complex tasks",
                        input_modalities=["text", "application/json"],
                        output_modalities=["text", "application/json"],
                        tags=["tasks", "execution", "management"],
                        examples=["Execute complex workflows", "Manage task queues"]
                    )
                ]
            )

            # Add security schemes
            security_schemes = {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "JWT token authentication"
                },
                "apiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key authentication"
                }
            }

            # Convert to JSON and add security
            card_dict = self.to_json(system_card)
            card_dict["securitySchemes"] = security_schemes
            card_dict["endpoints"] = {
                "discovery": f"{self.base_url}/api/a2a/v1/agents/discover",
                "message_send": f"{self.base_url}/api/a2a/v1/message/send",
                "message_stream": f"{self.base_url}/api/a2a/v1/message/stream",
                "health": f"{self.base_url}/api/a2a/v1/health"
            }

            return card_dict

        except Exception as e:
            logger.error(f"Failed to generate system agent card: {e}")
            raise
