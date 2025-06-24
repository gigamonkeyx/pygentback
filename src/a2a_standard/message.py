"""
Message and Part Implementation

Implements the Message and Part specifications from Google A2A Protocol.
Messages represent communication turns between clients and agents,
containing various types of content parts.
"""

import uuid
import base64
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class PartKind(str, Enum):
    """Types of content parts supported by A2A"""
    TEXT = "text"
    FILE = "file"
    DATA = "data"


@dataclass
class TextPart:
    """Text content part"""
    kind: str = "text"
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "kind": self.kind,
            "text": self.text
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextPart":
        """Create TextPart from dictionary"""
        return cls(
            kind=data.get("kind", "text"),
            text=data.get("text", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class FileWithBytes:
    """File content with inline bytes"""
    name: Optional[str] = None
    mime_type: Optional[str] = None
    bytes: Optional[str] = None  # Base64 encoded
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {}
        if self.name:
            result["name"] = self.name
        if self.mime_type:
            result["mimeType"] = self.mime_type
        if self.bytes:
            result["bytes"] = self.bytes
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileWithBytes":
        """Create FileWithBytes from dictionary"""
        return cls(
            name=data.get("name"),
            mime_type=data.get("mimeType"),
            bytes=data.get("bytes")
        )
    
    def set_content(self, content: bytes, mime_type: str, name: Optional[str] = None) -> None:
        """Set file content from bytes"""
        self.bytes = base64.b64encode(content).decode('utf-8')
        self.mime_type = mime_type
        if name:
            self.name = name
    
    def get_content(self) -> Optional[bytes]:
        """Get file content as bytes"""
        if self.bytes:
            return base64.b64decode(self.bytes)
        return None


@dataclass
class FileWithUri:
    """File content with URI reference"""
    name: Optional[str] = None
    mime_type: Optional[str] = None
    uri: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {}
        if self.name:
            result["name"] = self.name
        if self.mime_type:
            result["mimeType"] = self.mime_type
        if self.uri:
            result["uri"] = self.uri
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileWithUri":
        """Create FileWithUri from dictionary"""
        return cls(
            name=data.get("name"),
            mime_type=data.get("mimeType"),
            uri=data.get("uri")
        )


@dataclass
class FilePart:
    """File content part"""
    kind: str = "file"
    file: Optional[Union[FileWithBytes, FileWithUri]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "kind": self.kind
        }
        if self.file:
            result["file"] = self.file.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilePart":
        """Create FilePart from dictionary"""
        file_obj = None
        if "file" in data:
            file_data = data["file"]
            if "bytes" in file_data:
                file_obj = FileWithBytes.from_dict(file_data)
            elif "uri" in file_data:
                file_obj = FileWithUri.from_dict(file_data)
        
        return cls(
            kind=data.get("kind", "file"),
            file=file_obj,
            metadata=data.get("metadata", {})
        )
    
    def set_file_content(self, content: bytes, mime_type: str, name: Optional[str] = None) -> None:
        """Set file content from bytes"""
        self.file = FileWithBytes()
        self.file.set_content(content, mime_type, name)
    
    def set_file_uri(self, uri: str, mime_type: str, name: Optional[str] = None) -> None:
        """Set file content from URI"""
        self.file = FileWithUri(name=name, mime_type=mime_type, uri=uri)


@dataclass
class DataPart:
    """Structured data content part"""
    kind: str = "data"
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "kind": self.kind
        }
        if self.data is not None:
            result["data"] = self.data
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPart":
        """Create DataPart from dictionary"""
        return cls(
            kind=data.get("kind", "data"),
            data=data.get("data"),
            metadata=data.get("metadata", {})
        )


# Union type for all part types
Part = Union[TextPart, FilePart, DataPart]


def create_part(kind: str, **kwargs) -> Part:
    """Factory function to create parts"""
    if kind == "text":
        return TextPart(**kwargs)
    elif kind == "file":
        return FilePart(**kwargs)
    elif kind == "data":
        return DataPart(**kwargs)
    else:
        raise ValueError(f"Unknown part kind: {kind}")


def part_from_dict(data: Dict[str, Any]) -> Part:
    """Create part from dictionary based on kind"""
    kind = data.get("kind", "text")
    
    if kind == "text":
        return TextPart.from_dict(data)
    elif kind == "file":
        return FilePart.from_dict(data)
    elif kind == "data":
        return DataPart.from_dict(data)
    else:
        raise ValueError(f"Unknown part kind: {kind}")


@dataclass
class Message:
    """
    Message - single turn of communication between client and agent
    
    Messages have a role (user or agent) and contain one or more parts
    with different types of content.
    """
    
    # Required fields
    role: str  # "user" or "agent"
    parts: List[Part] = field(default_factory=list)
    
    # Optional identification
    message_id: Optional[str] = None
    task_id: Optional[str] = None
    context_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Initialize message with unique ID if not provided"""
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a text part to the message"""
        part = TextPart(text=text, metadata=metadata or {})
        self.parts.append(part)
    
    def add_file_bytes(self, content: bytes, mime_type: str, name: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a file part with inline bytes"""
        part = FilePart(metadata=metadata or {})
        part.set_file_content(content, mime_type, name)
        self.parts.append(part)
    
    def add_file_uri(self, uri: str, mime_type: str, name: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a file part with URI reference"""
        part = FilePart(metadata=metadata or {})
        part.set_file_uri(uri, mime_type, name)
        self.parts.append(part)
    
    def add_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a structured data part"""
        part = DataPart(data=data, metadata=metadata or {})
        self.parts.append(part)
    
    def get_text_content(self) -> str:
        """Get all text content concatenated"""
        text_parts = [part.text for part in self.parts if isinstance(part, TextPart)]
        return "\n".join(text_parts)
    
    def get_file_parts(self) -> List[FilePart]:
        """Get all file parts"""
        return [part for part in self.parts if isinstance(part, FilePart)]
    
    def get_data_parts(self) -> List[DataPart]:
        """Get all data parts"""
        return [part for part in self.parts if isinstance(part, DataPart)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "role": self.role,
            "parts": [part.to_dict() for part in self.parts]
        }
        
        if self.message_id:
            result["messageId"] = self.message_id
        if self.task_id:
            result["taskId"] = self.task_id
        if self.context_id:
            result["contextId"] = self.context_id
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create Message from dictionary"""
        # Parse parts
        parts = []
        if "parts" in data:
            parts = [part_from_dict(part_data) for part_data in data["parts"]]
        
        return cls(
            role=data["role"],
            parts=parts,
            message_id=data.get("messageId"),
            task_id=data.get("taskId"),
            context_id=data.get("contextId"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def create_user_message(cls, text: str, task_id: Optional[str] = None,
                           context_id: Optional[str] = None) -> "Message":
        """Create a user message with text content"""
        message = cls(role="user", task_id=task_id, context_id=context_id)
        message.add_text(text)
        return message
    
    @classmethod
    def create_agent_message(cls, text: str, task_id: Optional[str] = None,
                            context_id: Optional[str] = None) -> "Message":
        """Create an agent message with text content"""
        message = cls(role="agent", task_id=task_id, context_id=context_id)
        message.add_text(text)
        return message
