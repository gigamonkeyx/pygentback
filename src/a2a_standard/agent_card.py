"""
Agent Card Implementation

Implements the Agent Card specification from Google A2A Protocol.
Agent Cards are JSON metadata documents that describe an A2A Server's
identity, capabilities, skills, and authentication requirements.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class SecuritySchemeType(str, Enum):
    """Security scheme types supported by A2A"""
    BEARER = "bearer"
    API_KEY = "apiKey"
    OAUTH2 = "oauth2"
    BASIC = "basic"


@dataclass
class SecurityScheme:
    """Security scheme definition for agent authentication"""
    type: SecuritySchemeType
    description: Optional[str] = None
    name: Optional[str] = None  # For apiKey schemes
    in_: Optional[str] = None  # For apiKey schemes (header, query, cookie)
    scheme: Optional[str] = None  # For bearer schemes
    bearer_format: Optional[str] = None  # For bearer schemes
    flows: Optional[Dict[str, Any]] = None  # For oauth2 schemes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {"type": self.type.value}
        if self.description:
            result["description"] = self.description
        if self.name:
            result["name"] = self.name
        if self.in_:
            result["in"] = self.in_
        if self.scheme:
            result["scheme"] = self.scheme
        if self.bearer_format:
            result["bearerFormat"] = self.bearer_format
        if self.flows:
            result["flows"] = self.flows
        return result


@dataclass
class AgentProvider:
    """Information about the agent provider/vendor"""
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    email: Optional[str] = None
    organization: Optional[str] = None  # Added for compatibility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.url:
            result["url"] = self.url
        if self.email:
            result["email"] = self.email
        return result


@dataclass
class AgentExtension:
    """Agent extension definition for custom capabilities"""
    name: str
    version: str
    description: Optional[str] = None
    schema_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "name": self.name,
            "version": self.version
        }
        if self.description:
            result["description"] = self.description
        if self.schema_url:
            result["schemaUrl"] = self.schema_url
        return result


@dataclass
class AgentCapabilities:
    """Agent capabilities and supported features"""
    streaming: bool = False
    push_notifications: bool = False
    multi_turn: bool = True
    file_upload: bool = True
    file_download: bool = True
    structured_data: bool = True
    extensions: Optional[List[AgentExtension]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "streaming": self.streaming,
            "pushNotifications": self.push_notifications,
            "multiTurn": self.multi_turn,
            "fileUpload": self.file_upload,
            "fileDownload": self.file_download,
            "structuredData": self.structured_data
        }
        if self.extensions:
            result["extensions"] = [ext.to_dict() for ext in self.extensions]
        return result


@dataclass
class AgentSkill:
    """Individual skill/capability offered by the agent"""
    name: str
    description: str
    input_modalities: List[str]
    output_modalities: List[str]
    id: Optional[str] = None  # Optional ID for A2A protocol compatibility
    examples: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None  # Optional tags for A2A protocol compatibility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "name": self.name,
            "description": self.description,
            "inputModalities": self.input_modalities,
            "outputModalities": self.output_modalities
        }
        if self.id:
            result["id"] = self.id
        if self.examples:
            result["examples"] = self.examples
        if self.parameters:
            result["parameters"] = self.parameters
        if self.tags:
            result["tags"] = self.tags
        return result


@dataclass
class AgentCard:
    """
    Agent Card - JSON metadata document describing an A2A Server
    
    This is the primary discovery mechanism for A2A agents.
    Typically served at /.well-known/agent.json
    """
    
    # Required fields
    name: str
    description: str
    version: str
    url: str  # Base URL for A2A endpoints
    
    # Optional identification
    provider: Optional[AgentProvider] = None
    
    # Capabilities and features
    capabilities: Optional[AgentCapabilities] = None
    
    # Skills offered by this agent
    skills: Optional[List[AgentSkill]] = None

    # Input/Output modes
    defaultInputModes: Optional[List[str]] = None
    defaultOutputModes: Optional[List[str]] = None

    # Security configuration
    security_schemes: Optional[Dict[str, SecurityScheme]] = None
    security: Optional[List[Dict[str, List[str]]]] = None

    # Extended card support
    supports_authenticated_extended_card: bool = False

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    documentationUrl: Optional[str] = None  # Documentation URL for A2A protocol compatibility
    
    def __post_init__(self):
        """Initialize default values"""
        if self.capabilities is None:
            self.capabilities = AgentCapabilities()
        if self.skills is None:
            self.skills = []
        if self.defaultInputModes is None:
            self.defaultInputModes = ["text/plain", "application/json"]
        if self.defaultOutputModes is None:
            self.defaultOutputModes = ["text/plain", "application/json"]
        if self.security_schemes is None:
            self.security_schemes = {}
        if self.security is None:
            self.security = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url
        }
        
        if self.provider:
            result["provider"] = self.provider.to_dict()
        
        if self.capabilities:
            result["capabilities"] = self.capabilities.to_dict()
        
        if self.skills:
            result["skills"] = [skill.to_dict() for skill in self.skills]

        if self.defaultInputModes:
            result["defaultInputModes"] = self.defaultInputModes

        if self.defaultOutputModes:
            result["defaultOutputModes"] = self.defaultOutputModes

        if self.documentationUrl:
            result["documentationUrl"] = self.documentationUrl

        if self.security_schemes:
            result["securitySchemes"] = {
                name: scheme.to_dict() 
                for name, scheme in self.security_schemes.items()
            }
        
        if self.security:
            result["security"] = self.security
        
        if self.supports_authenticated_extended_card:
            result["supportsAuthenticatedExtendedCard"] = True
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Create AgentCard from dictionary"""
        # Parse provider
        provider = None
        if "provider" in data:
            provider_data = data["provider"]
            provider = AgentProvider(
                name=provider_data["name"],
                description=provider_data.get("description"),
                url=provider_data.get("url"),
                email=provider_data.get("email")
            )
        
        # Parse capabilities
        capabilities = None
        if "capabilities" in data:
            cap_data = data["capabilities"]
            extensions = None
            if "extensions" in cap_data:
                extensions = [
                    AgentExtension(
                        name=ext["name"],
                        version=ext["version"],
                        description=ext.get("description"),
                        schema_url=ext.get("schemaUrl")
                    )
                    for ext in cap_data["extensions"]
                ]
            
            capabilities = AgentCapabilities(
                streaming=cap_data.get("streaming", False),
                push_notifications=cap_data.get("pushNotifications", False),
                multi_turn=cap_data.get("multiTurn", True),
                file_upload=cap_data.get("fileUpload", True),
                file_download=cap_data.get("fileDownload", True),
                structured_data=cap_data.get("structuredData", True),
                extensions=extensions
            )
        
        # Parse skills
        skills = []
        if "skills" in data:
            skills = [
                AgentSkill(
                    name=skill["name"],
                    description=skill["description"],
                    input_modalities=skill["inputModalities"],
                    output_modalities=skill["outputModalities"],
                    id=skill.get("id"),
                    examples=skill.get("examples"),
                    parameters=skill.get("parameters"),
                    tags=skill.get("tags")
                )
                for skill in data["skills"]
            ]
        
        # Parse security schemes
        security_schemes = {}
        if "securitySchemes" in data:
            for name, scheme_data in data["securitySchemes"].items():
                security_schemes[name] = SecurityScheme(
                    type=SecuritySchemeType(scheme_data["type"]),
                    description=scheme_data.get("description"),
                    name=scheme_data.get("name"),
                    in_=scheme_data.get("in"),
                    scheme=scheme_data.get("scheme"),
                    bearer_format=scheme_data.get("bearerFormat"),
                    flows=scheme_data.get("flows")
                )
        
        return cls(
            name=data["name"],
            description=data["description"],
            version=data["version"],
            url=data["url"],
            provider=provider,
            capabilities=capabilities,
            skills=skills,
            defaultInputModes=data.get("defaultInputModes"),
            defaultOutputModes=data.get("defaultOutputModes"),
            security_schemes=security_schemes,
            security=data.get("security", []),
            supports_authenticated_extended_card=data.get("supportsAuthenticatedExtendedCard", False),
            metadata=data.get("metadata", {}),
            documentationUrl=data.get("documentationUrl")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "AgentCard":
        """Create AgentCard from JSON string"""
        return cls.from_dict(json.loads(json_str))

    def add_skill(self, skill: AgentSkill) -> None:
        """Add a skill to the agent card"""
        if self.skills is None:
            self.skills = []
        self.skills.append(skill)

    def add_security_scheme(self, name: str, scheme: SecurityScheme) -> None:
        """Add a security scheme to the agent card"""
        if self.security_schemes is None:
            self.security_schemes = {}
        self.security_schemes[name] = scheme

    def enable_capability(self, capability: str, enabled: bool = True) -> None:
        """Enable or disable a specific capability"""
        if self.capabilities is None:
            self.capabilities = AgentCapabilities()

        if hasattr(self.capabilities, capability):
            setattr(self.capabilities, capability, enabled)
