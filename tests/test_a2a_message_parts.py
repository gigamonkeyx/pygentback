#!/usr/bin/env python3
"""
Test A2A Message and Part Structure Support

Tests the A2A-compliant message structure implementation according to Google A2A specification.
"""

import pytest
import base64
import json
import tempfile
from pathlib import Path

# Import the A2A message components
try:
    from src.a2a_protocol.message_parts import (
        TextPart, FilePart, DataPart, FileWithBytes, FileWithUri,
        A2AMessage, A2AMessageBuilder, A2AMessageValidator, A2AMessageSerializer,
        A2AMessagePartFactory, create_user_message, create_agent_message
    )
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AMessageParts:
    """Test A2A Message Parts"""
    
    def test_text_part_creation(self):
        """Test text part creation"""
        text_part = A2AMessagePartFactory.create_text_part("Hello, world!")
        
        assert text_part.kind == "text"
        assert text_part.text == "Hello, world!"
    
    def test_file_part_from_bytes(self):
        """Test file part creation from bytes"""
        file_data = b"Hello, file content!"
        filename = "test.txt"
        
        file_part = A2AMessagePartFactory.create_file_part_from_bytes(
            file_data, filename, "text/plain"
        )
        
        assert file_part.kind == "file"
        assert isinstance(file_part.file, FileWithBytes)
        assert file_part.file.name == filename
        assert file_part.file.mimeType == "text/plain"
        assert file_part.file.size == len(file_data)
        assert file_part.file.checksum is not None
        
        # Verify base64 encoding
        decoded_data = base64.b64decode(file_part.file.bytes)
        assert decoded_data == file_data
    
    def test_file_part_from_path(self):
        """Test file part creation from file path"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            temp_path = f.name
        
        try:
            file_part = A2AMessagePartFactory.create_file_part_from_path(temp_path)
            
            assert file_part.kind == "file"
            assert isinstance(file_part.file, FileWithBytes)
            assert file_part.file.name == Path(temp_path).name
            assert file_part.file.mimeType == "text/plain"
            assert file_part.file.size > 0
            
            # Verify content
            decoded_data = base64.b64decode(file_part.file.bytes)
            assert decoded_data == b"Test file content"
            
        finally:
            Path(temp_path).unlink()
    
    def test_file_part_from_uri(self):
        """Test file part creation from URI"""
        uri = "https://example.com/test.pdf"
        filename = "test.pdf"
        
        file_part = A2AMessagePartFactory.create_file_part_from_uri(
            uri, filename, "application/pdf", size=1024, checksum="abc123"
        )
        
        assert file_part.kind == "file"
        assert isinstance(file_part.file, FileWithUri)
        assert file_part.file.name == filename
        assert file_part.file.mimeType == "application/pdf"
        assert file_part.file.uri == uri
        assert file_part.file.size == 1024
        assert file_part.file.checksum == "abc123"
    
    def test_data_part_creation(self):
        """Test data part creation"""
        data = {"key": "value", "number": 42}
        
        data_part = A2AMessagePartFactory.create_data_part(data)
        
        assert data_part.kind == "data"
        assert data_part.data == data
        assert data_part.mimeType == "application/json"
    
    def test_json_data_part_creation(self):
        """Test JSON data part creation from object"""
        class TestObject:
            def __init__(self):
                self.name = "test"
                self.value = 123
        
        obj = TestObject()
        data_part = A2AMessagePartFactory.create_json_data_part(obj)
        
        assert data_part.kind == "data"
        assert data_part.data["name"] == "test"
        assert data_part.data["value"] == 123
        assert data_part.mimeType == "application/json"


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AMessageBuilder:
    """Test A2A Message Builder"""
    
    def test_build_text_message(self):
        """Test building text message"""
        message = (A2AMessageBuilder("user")
                  .add_text("Hello, world!")
                  .build())
        
        assert message.role == "user"
        assert len(message.parts) == 1
        assert message.parts[0].kind == "text"
        assert message.parts[0].text == "Hello, world!"
        assert message.messageId is not None
        assert message.timestamp is not None
    
    def test_build_multi_part_message(self):
        """Test building message with multiple parts"""
        data = {"analysis": "complete", "score": 0.95}
        
        message = (A2AMessageBuilder("agent", "custom-msg-123")
                  .add_text("Analysis complete:")
                  .add_data(data)
                  .set_metadata("priority", "high")
                  .build())
        
        assert message.messageId == "custom-msg-123"
        assert message.role == "agent"
        assert len(message.parts) == 2
        
        # Check text part
        assert message.parts[0].kind == "text"
        assert message.parts[0].text == "Analysis complete:"
        
        # Check data part
        assert message.parts[1].kind == "data"
        assert message.parts[1].data == data
        
        # Check metadata
        assert message.metadata["priority"] == "high"
    
    def test_build_message_with_file(self):
        """Test building message with file"""
        file_data = b"Binary file content"
        
        message = (A2AMessageBuilder("user")
                  .add_text("Please analyze this file:")
                  .add_file_from_bytes(file_data, "data.bin", "application/octet-stream")
                  .build())
        
        assert len(message.parts) == 2
        assert message.parts[0].kind == "text"
        assert message.parts[1].kind == "file"
        assert message.parts[1].file.name == "data.bin"
        assert message.parts[1].file.mimeType == "application/octet-stream"


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AMessageValidator:
    """Test A2A Message Validator"""
    
    def test_validate_valid_message(self):
        """Test validation of valid message"""
        message = create_user_message("Hello, world!")
        errors = A2AMessageValidator.validate_message(message)
        
        assert len(errors) == 0
    
    def test_validate_message_missing_fields(self):
        """Test validation of message with missing fields"""
        # Create message with missing messageId
        message = A2AMessage(
            messageId="",
            role="user",
            parts=[TextPart(text="Hello")]
        )
        
        errors = A2AMessageValidator.validate_message(message)
        assert any("messageId is required" in error for error in errors)
    
    def test_validate_message_invalid_role(self):
        """Test validation of message with invalid role"""
        message = A2AMessage(
            messageId="test-123",
            role="invalid",
            parts=[TextPart(text="Hello")]
        )
        
        errors = A2AMessageValidator.validate_message(message)
        assert any("role must be 'user' or 'agent'" in error for error in errors)
    
    def test_validate_message_empty_parts(self):
        """Test validation of message with empty parts"""
        message = A2AMessage(
            messageId="test-123",
            role="user",
            parts=[]
        )
        
        errors = A2AMessageValidator.validate_message(message)
        assert any("parts array cannot be empty" in error for error in errors)
    
    def test_validate_file_part(self):
        """Test validation of file parts"""
        # Valid file with bytes
        file_bytes = FileWithBytes(
            name="test.txt",
            mimeType="text/plain",
            bytes=base64.b64encode(b"content").decode()
        )
        file_part = FilePart(file=file_bytes)
        
        errors = A2AMessageValidator.validate_part(file_part)
        assert len(errors) == 0
        
        # Invalid file with invalid base64
        invalid_file = FileWithBytes(
            name="test.txt",
            mimeType="text/plain",
            bytes="invalid-base64!"
        )
        invalid_part = FilePart(file=invalid_file)
        
        errors = A2AMessageValidator.validate_part(invalid_part)
        assert any("valid base64" in error for error in errors)


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2AMessageSerializer:
    """Test A2A Message Serializer"""
    
    def test_serialize_text_message(self):
        """Test serialization of text message"""
        message = create_user_message("Hello, world!", "msg-123")
        
        # Serialize to dict
        message_dict = A2AMessageSerializer.to_dict(message)
        
        # Verify structure
        assert message_dict["messageId"] == "msg-123"
        assert message_dict["role"] == "user"
        assert len(message_dict["parts"]) == 1
        assert message_dict["parts"][0]["kind"] == "text"
        assert message_dict["parts"][0]["text"] == "Hello, world!"
        
        # Verify JSON serializable
        json_str = json.dumps(message_dict)
        assert isinstance(json_str, str)
    
    def test_deserialize_text_message(self):
        """Test deserialization of text message"""
        message_dict = {
            "messageId": "msg-123",
            "role": "user",
            "parts": [
                {"kind": "text", "text": "Hello, world!"}
            ],
            "timestamp": "2024-01-01T00:00:00Z",
            "metadata": {"priority": "normal"}
        }
        
        # Deserialize from dict
        message = A2AMessageSerializer.from_dict(message_dict)
        
        # Verify structure
        assert message.messageId == "msg-123"
        assert message.role == "user"
        assert len(message.parts) == 1
        assert message.parts[0].kind == "text"
        assert message.parts[0].text == "Hello, world!"
        assert message.timestamp == "2024-01-01T00:00:00Z"
        assert message.metadata["priority"] == "normal"
    
    def test_serialize_deserialize_roundtrip(self):
        """Test serialize/deserialize roundtrip"""
        # Create complex message
        original_message = (A2AMessageBuilder("agent", "complex-msg")
                          .add_text("Analysis results:")
                          .add_data({"score": 0.95, "confidence": "high"})
                          .set_metadata("type", "analysis")
                          .build())
        
        # Serialize and deserialize
        message_dict = A2AMessageSerializer.to_dict(original_message)
        restored_message = A2AMessageSerializer.from_dict(message_dict)
        
        # Verify equality
        assert restored_message.messageId == original_message.messageId
        assert restored_message.role == original_message.role
        assert len(restored_message.parts) == len(original_message.parts)
        assert restored_message.metadata == original_message.metadata
        
        # Verify parts
        for i, (orig_part, rest_part) in enumerate(zip(original_message.parts, restored_message.parts)):
            assert orig_part.kind == rest_part.kind
            if orig_part.kind == "text":
                assert orig_part.text == rest_part.text
            elif orig_part.kind == "data":
                assert orig_part.data == rest_part.data


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_user_message(self):
        """Test create_user_message convenience function"""
        message = create_user_message("Hello!")
        
        assert message.role == "user"
        assert len(message.parts) == 1
        assert message.parts[0].text == "Hello!"
        assert message.messageId is not None
    
    def test_create_agent_message(self):
        """Test create_agent_message convenience function"""
        message = create_agent_message("Hello back!", "agent-msg-123")
        
        assert message.role == "agent"
        assert message.messageId == "agent-msg-123"
        assert len(message.parts) == 1
        assert message.parts[0].text == "Hello back!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
