#!/usr/bin/env python3
"""
A2A Message and Part Structure Support

Implements proper message structure with roles and Parts according to Google A2A specification.
Supports TextPart, FilePart, and DataPart for rich communication between agents.
"""

import base64
import json
import logging
import mimetypes
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FileWithBytes:
    """A2A File with embedded bytes"""
    name: str
    mimeType: str
    bytes: str  # Base64 encoded
    size: Optional[int] = None
    checksum: Optional[str] = None


@dataclass
class FileWithUri:
    """A2A File with URI reference"""
    name: str
    mimeType: str
    uri: str
    size: Optional[int] = None
    checksum: Optional[str] = None


@dataclass
class TextPart:
    """A2A Text Part"""
    kind: str = "text"
    text: str = ""


@dataclass
class FilePart:
    """A2A File Part"""
    kind: str = "file"
    file: Union[FileWithBytes, FileWithUri] = None


@dataclass
class DataPart:
    """A2A Data Part for structured data"""
    kind: str = "data"
    data: Dict[str, Any] = field(default_factory=dict)
    mimeType: str = "application/json"


class A2AMessagePartFactory:
    """Factory for creating A2A message parts"""
    
    @staticmethod
    def create_text_part(text: str) -> TextPart:
        """Create a text part"""
        return TextPart(text=text)
    
    @staticmethod
    def create_file_part_from_bytes(file_data: bytes, filename: str, 
                                  mime_type: Optional[str] = None) -> FilePart:
        """Create a file part from bytes"""
        if not mime_type:
            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        
        # Encode bytes to base64
        encoded_bytes = base64.b64encode(file_data).decode('utf-8')
        
        # Calculate checksum
        checksum = hashlib.sha256(file_data).hexdigest()
        
        file_with_bytes = FileWithBytes(
            name=filename,
            mimeType=mime_type,
            bytes=encoded_bytes,
            size=len(file_data),
            checksum=checksum
        )
        
        return FilePart(file=file_with_bytes)
    
    @staticmethod
    def create_file_part_from_path(file_path: Union[str, Path], 
                                 filename: Optional[str] = None) -> FilePart:
        """Create a file part from file path"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not filename:
            filename = path.name
        
        # Read file data
        with open(path, 'rb') as f:
            file_data = f.read()
        
        return A2AMessagePartFactory.create_file_part_from_bytes(
            file_data, filename
        )
    
    @staticmethod
    def create_file_part_from_uri(uri: str, filename: str, 
                                mime_type: Optional[str] = None,
                                size: Optional[int] = None,
                                checksum: Optional[str] = None) -> FilePart:
        """Create a file part from URI"""
        if not mime_type:
            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        
        file_with_uri = FileWithUri(
            name=filename,
            mimeType=mime_type,
            uri=uri,
            size=size,
            checksum=checksum
        )
        
        return FilePart(file=file_with_uri)
    
    @staticmethod
    def create_data_part(data: Dict[str, Any], mime_type: str = "application/json") -> DataPart:
        """Create a data part"""
        return DataPart(data=data, mimeType=mime_type)
    
    @staticmethod
    def create_json_data_part(obj: Any) -> DataPart:
        """Create a data part from JSON-serializable object"""
        if hasattr(obj, '__dict__'):
            data = asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__
        else:
            data = obj
        
        return DataPart(data=data, mimeType="application/json")


@dataclass
class A2AMessage:
    """A2A Message with proper structure"""
    messageId: str
    role: str  # "user" or "agent"
    parts: List[Union[TextPart, FilePart, DataPart]]
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class A2AMessageBuilder:
    """Builder for constructing A2A messages"""
    
    def __init__(self, role: str, message_id: Optional[str] = None):
        self.role = role
        self.message_id = message_id or str(uuid.uuid4())
        self.parts: List[Union[TextPart, FilePart, DataPart]] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_text(self, text: str) -> 'A2AMessageBuilder':
        """Add text part to message"""
        self.parts.append(A2AMessagePartFactory.create_text_part(text))
        return self
    
    def add_file_from_bytes(self, file_data: bytes, filename: str, 
                          mime_type: Optional[str] = None) -> 'A2AMessageBuilder':
        """Add file part from bytes"""
        self.parts.append(A2AMessagePartFactory.create_file_part_from_bytes(
            file_data, filename, mime_type
        ))
        return self
    
    def add_file_from_path(self, file_path: Union[str, Path], 
                         filename: Optional[str] = None) -> 'A2AMessageBuilder':
        """Add file part from file path"""
        self.parts.append(A2AMessagePartFactory.create_file_part_from_path(
            file_path, filename
        ))
        return self
    
    def add_file_from_uri(self, uri: str, filename: str, 
                        mime_type: Optional[str] = None,
                        size: Optional[int] = None,
                        checksum: Optional[str] = None) -> 'A2AMessageBuilder':
        """Add file part from URI"""
        self.parts.append(A2AMessagePartFactory.create_file_part_from_uri(
            uri, filename, mime_type, size, checksum
        ))
        return self
    
    def add_data(self, data: Dict[str, Any], 
               mime_type: str = "application/json") -> 'A2AMessageBuilder':
        """Add data part"""
        self.parts.append(A2AMessagePartFactory.create_data_part(data, mime_type))
        return self
    
    def add_json_data(self, obj: Any) -> 'A2AMessageBuilder':
        """Add JSON data part"""
        self.parts.append(A2AMessagePartFactory.create_json_data_part(obj))
        return self
    
    def set_metadata(self, key: str, value: Any) -> 'A2AMessageBuilder':
        """Set metadata field"""
        self.metadata[key] = value
        return self
    
    def build(self) -> A2AMessage:
        """Build the A2A message"""
        return A2AMessage(
            messageId=self.message_id,
            role=self.role,
            parts=self.parts,
            metadata=self.metadata
        )


class A2AMessageValidator:
    """Validates A2A messages according to specification"""
    
    @staticmethod
    def validate_message(message: A2AMessage) -> List[str]:
        """Validate A2A message structure"""
        errors = []
        
        # Check required fields
        if not message.messageId:
            errors.append("messageId is required")
        
        if not message.role:
            errors.append("role is required")
        elif message.role not in ["user", "agent"]:
            errors.append("role must be 'user' or 'agent'")
        
        if not message.parts:
            errors.append("parts array cannot be empty")
        
        # Validate parts
        for i, part in enumerate(message.parts):
            part_errors = A2AMessageValidator.validate_part(part)
            for error in part_errors:
                errors.append(f"parts[{i}]: {error}")
        
        return errors
    
    @staticmethod
    def validate_part(part: Union[TextPart, FilePart, DataPart]) -> List[str]:
        """Validate individual message part"""
        errors = []
        
        if not hasattr(part, 'kind'):
            errors.append("part must have 'kind' field")
            return errors
        
        if part.kind == "text":
            if not isinstance(part, TextPart):
                errors.append("text part must be TextPart instance")
            elif not hasattr(part, 'text'):
                errors.append("text part must have 'text' field")
        
        elif part.kind == "file":
            if not isinstance(part, FilePart):
                errors.append("file part must be FilePart instance")
            elif not hasattr(part, 'file') or not part.file:
                errors.append("file part must have 'file' field")
            else:
                file_errors = A2AMessageValidator.validate_file(part.file)
                errors.extend(file_errors)
        
        elif part.kind == "data":
            if not isinstance(part, DataPart):
                errors.append("data part must be DataPart instance")
            elif not hasattr(part, 'data'):
                errors.append("data part must have 'data' field")
        
        else:
            errors.append(f"unknown part kind: {part.kind}")
        
        return errors
    
    @staticmethod
    def validate_file(file_obj: Union[FileWithBytes, FileWithUri]) -> List[str]:
        """Validate file object"""
        errors = []
        
        if not hasattr(file_obj, 'name') or not file_obj.name:
            errors.append("file must have 'name' field")
        
        if not hasattr(file_obj, 'mimeType') or not file_obj.mimeType:
            errors.append("file must have 'mimeType' field")
        
        if isinstance(file_obj, FileWithBytes):
            if not hasattr(file_obj, 'bytes') or not file_obj.bytes:
                errors.append("FileWithBytes must have 'bytes' field")
            else:
                # Validate base64 encoding
                try:
                    base64.b64decode(file_obj.bytes)
                except Exception:
                    errors.append("FileWithBytes 'bytes' must be valid base64")
        
        elif isinstance(file_obj, FileWithUri):
            if not hasattr(file_obj, 'uri') or not file_obj.uri:
                errors.append("FileWithUri must have 'uri' field")
        
        else:
            errors.append("file must be FileWithBytes or FileWithUri")
        
        return errors


class A2AMessageSerializer:
    """Serializes A2A messages to/from dictionaries"""
    
    @staticmethod
    def to_dict(message: A2AMessage) -> Dict[str, Any]:
        """Convert A2A message to dictionary"""
        return {
            "messageId": message.messageId,
            "role": message.role,
            "parts": [A2AMessageSerializer._part_to_dict(part) for part in message.parts],
            "timestamp": message.timestamp,
            "metadata": message.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> A2AMessage:
        """Create A2A message from dictionary"""
        parts = []
        for part_data in data.get("parts", []):
            part = A2AMessageSerializer._part_from_dict(part_data)
            parts.append(part)
        
        return A2AMessage(
            messageId=data["messageId"],
            role=data["role"],
            parts=parts,
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )
    
    @staticmethod
    def _part_to_dict(part: Union[TextPart, FilePart, DataPart]) -> Dict[str, Any]:
        """Convert part to dictionary"""
        if isinstance(part, TextPart):
            return {"kind": "text", "text": part.text}
        elif isinstance(part, FilePart):
            file_dict = asdict(part.file)
            return {"kind": "file", "file": file_dict}
        elif isinstance(part, DataPart):
            return {"kind": "data", "data": part.data, "mimeType": part.mimeType}
        else:
            raise ValueError(f"Unknown part type: {type(part)}")
    
    @staticmethod
    def _part_from_dict(data: Dict[str, Any]) -> Union[TextPart, FilePart, DataPart]:
        """Create part from dictionary"""
        kind = data.get("kind")
        
        if kind == "text":
            return TextPart(text=data.get("text", ""))
        elif kind == "file":
            file_data = data.get("file", {})
            if "bytes" in file_data:
                file_obj = FileWithBytes(**file_data)
            else:
                file_obj = FileWithUri(**file_data)
            return FilePart(file=file_obj)
        elif kind == "data":
            return DataPart(
                data=data.get("data", {}),
                mimeType=data.get("mimeType", "application/json")
            )
        else:
            raise ValueError(f"Unknown part kind: {kind}")


# Convenience functions
def create_user_message(text: str, message_id: Optional[str] = None) -> A2AMessage:
    """Create a simple user text message"""
    return A2AMessageBuilder("user", message_id).add_text(text).build()


def create_agent_message(text: str, message_id: Optional[str] = None) -> A2AMessage:
    """Create a simple agent text message"""
    return A2AMessageBuilder("agent", message_id).add_text(text).build()
