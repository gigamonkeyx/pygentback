#!/usr/bin/env python3
"""
Test A2A Security and Authentication

Tests the A2A-compliant security implementation according to Google A2A specification.
"""

import pytest
import jwt
import time
from datetime import datetime, timedelta

# Import the A2A security components
try:
    from src.a2a_protocol.security import (
        A2ASecurityManager, A2AAuthenticationHandler, AuthenticationResult,
        APIKey, JWTConfig, AuthSchemeType
    )
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2ASecurityManager:
    """Test A2A Security Manager"""
    
    def setup_method(self):
        """Setup test environment"""
        jwt_config = JWTConfig(
            secret_key="test-secret-key",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7
        )
        self.security_manager = A2ASecurityManager(jwt_config)
    
    def test_generate_api_key(self):
        """Test API key generation"""
        raw_key, api_key = self.security_manager.generate_api_key(
            name="test-key",
            scopes=["read", "write"],
            expires_in_days=30
        )
        
        # Verify raw key format
        assert isinstance(raw_key, str)
        assert len(raw_key) > 20  # Should be reasonably long
        
        # Verify API key object
        assert api_key.name == "test-key"
        assert api_key.scopes == ["read", "write"]
        assert api_key.active == True
        assert api_key.expires_at is not None
        assert api_key.key_id in self.security_manager.api_keys
        
        # Verify key is stored
        stored_key = self.security_manager.api_keys[api_key.key_id]
        assert stored_key == api_key
    
    def test_validate_api_key_success(self):
        """Test successful API key validation"""
        raw_key, api_key = self.security_manager.generate_api_key("test-key", ["read"])
        
        # Validate the key
        result = self.security_manager.validate_api_key(raw_key)
        
        # Verify successful validation
        assert result.success == True
        assert result.user_id == "test-key"
        assert result.scopes == ["read"]
        assert result.metadata["key_id"] == api_key.key_id
        assert result.error is None
    
    def test_validate_api_key_invalid(self):
        """Test invalid API key validation"""
        result = self.security_manager.validate_api_key("invalid-key")
        
        assert result.success == False
        assert result.error == "Invalid API key"
    
    def test_validate_api_key_expired(self):
        """Test expired API key validation"""
        # Generate key that expires immediately
        raw_key, api_key = self.security_manager.generate_api_key(
            "expired-key", 
            expires_in_days=-1  # Already expired
        )
        
        result = self.security_manager.validate_api_key(raw_key)
        
        assert result.success == False
        assert result.error == "API key expired"
    
    def test_revoke_api_key(self):
        """Test API key revocation"""
        raw_key, api_key = self.security_manager.generate_api_key("test-key")
        
        # Verify key works before revocation
        result = self.security_manager.validate_api_key(raw_key)
        assert result.success == True
        
        # Revoke key
        success = self.security_manager.revoke_api_key(api_key.key_id)
        assert success == True
        
        # Verify key no longer works
        result = self.security_manager.validate_api_key(raw_key)
        assert result.success == False
        assert result.error == "Invalid API key"
    
    def test_create_access_token(self):
        """Test JWT access token creation"""
        token = self.security_manager.create_access_token(
            user_id="test-user",
            agent_id="test-agent",
            scopes=["read", "write"],
            extra_claims={"custom": "value"}
        )
        
        # Verify token format
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT has 3 parts
        
        # Decode token to verify contents
        payload = jwt.decode(
            token,
            self.security_manager.jwt_config.secret_key,
            algorithms=[self.security_manager.jwt_config.algorithm],
            audience=self.security_manager.jwt_config.audience,
            issuer=self.security_manager.jwt_config.issuer
        )
        
        assert payload["sub"] == "test-user"
        assert payload["agent_id"] == "test-agent"
        assert payload["scopes"] == ["read", "write"]
        assert payload["custom"] == "value"
        assert payload["type"] == "access"
    
    def test_validate_jwt_token_success(self):
        """Test successful JWT token validation"""
        token = self.security_manager.create_access_token(
            user_id="test-user",
            scopes=["read"]
        )
        
        result = self.security_manager.validate_jwt_token(token)
        
        assert result.success == True
        assert result.user_id == "test-user"
        assert result.scopes == ["read"]
        assert result.error is None
        assert "jti" in result.metadata
    
    def test_validate_jwt_token_invalid(self):
        """Test invalid JWT token validation"""
        result = self.security_manager.validate_jwt_token("invalid.token.here")
        
        assert result.success == False
        assert "Invalid token" in result.error
    
    def test_validate_jwt_token_expired(self):
        """Test expired JWT token validation"""
        # Create token with very short expiration
        old_expire_minutes = self.security_manager.jwt_config.access_token_expire_minutes
        self.security_manager.jwt_config.access_token_expire_minutes = 0  # Immediate expiration
        
        token = self.security_manager.create_access_token("test-user")
        
        # Restore original expiration
        self.security_manager.jwt_config.access_token_expire_minutes = old_expire_minutes
        
        # Wait a moment to ensure expiration
        time.sleep(1)
        
        result = self.security_manager.validate_jwt_token(token)
        
        assert result.success == False
        assert result.error == "Token expired"
    
    def test_revoke_jwt_token(self):
        """Test JWT token revocation"""
        token = self.security_manager.create_access_token("test-user")
        
        # Verify token works before revocation
        result = self.security_manager.validate_jwt_token(token)
        assert result.success == True
        
        # Revoke token
        success = self.security_manager.revoke_token(token)
        assert success == True
        
        # Verify token no longer works
        result = self.security_manager.validate_jwt_token(token)
        assert result.success == False
        assert result.error == "Token revoked"
    
    def test_check_scope_permission(self):
        """Test scope permission checking"""
        user_scopes = ["read", "write", "admin"]
        
        # Test with no required scopes
        assert self.security_manager.check_scope_permission([], user_scopes) == True
        
        # Test with subset of user scopes
        assert self.security_manager.check_scope_permission(["read"], user_scopes) == True
        assert self.security_manager.check_scope_permission(["read", "write"], user_scopes) == True
        
        # Test with scope user doesn't have
        assert self.security_manager.check_scope_permission(["delete"], user_scopes) == False
        assert self.security_manager.check_scope_permission(["read", "delete"], user_scopes) == False
    
    def test_get_security_schemes(self):
        """Test getting security schemes"""
        schemes = self.security_manager.get_security_schemes()
        
        assert "bearerAuth" in schemes
        assert "apiKeyAuth" in schemes
        
        # Verify bearer auth scheme
        bearer_scheme = schemes["bearerAuth"]
        assert bearer_scheme["type"] == "http"
        assert bearer_scheme["scheme"] == "bearer"
        assert bearer_scheme["bearerFormat"] == "JWT"
        
        # Verify API key scheme
        api_key_scheme = schemes["apiKeyAuth"]
        assert api_key_scheme["type"] == "apiKey"
        assert api_key_scheme["name"] == "X-API-Key"
        assert api_key_scheme["in"] == "header"


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestJWTConfig:
    """Test JWT Configuration"""
    
    def test_jwt_config_creation(self):
        """Test JWT config creation"""
        config = JWTConfig(
            secret_key="test-secret",
            algorithm="HS256",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
            issuer="test-issuer",
            audience="test-audience"
        )
        
        assert config.secret_key == "test-secret"
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 60
        assert config.refresh_token_expire_days == 30
        assert config.issuer == "test-issuer"
        assert config.audience == "test-audience"
    
    def test_jwt_config_defaults(self):
        """Test JWT config default values"""
        config = JWTConfig(secret_key="test-secret")
        
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 7
        assert config.issuer == "pygent-factory"
        assert config.audience == "a2a-agents"


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestAPIKey:
    """Test API Key data structure"""
    
    def test_api_key_creation(self):
        """Test API key creation"""
        api_key = APIKey(
            key_id="test-id",
            key_hash="test-hash",
            name="test-key",
            scopes=["read", "write"],
            expires_at="2024-12-31T23:59:59Z",
            metadata={"purpose": "testing"}
        )
        
        assert api_key.key_id == "test-id"
        assert api_key.key_hash == "test-hash"
        assert api_key.name == "test-key"
        assert api_key.scopes == ["read", "write"]
        assert api_key.expires_at == "2024-12-31T23:59:59Z"
        assert api_key.active == True  # Default value
        assert api_key.metadata["purpose"] == "testing"
        assert api_key.created_at is not None  # Should have default timestamp


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestAuthenticationResult:
    """Test Authentication Result data structure"""
    
    def test_authentication_result_success(self):
        """Test successful authentication result"""
        result = AuthenticationResult(
            success=True,
            user_id="test-user",
            agent_id="test-agent",
            scopes=["read", "write"],
            metadata={"source": "jwt"}
        )
        
        assert result.success == True
        assert result.user_id == "test-user"
        assert result.agent_id == "test-agent"
        assert result.scopes == ["read", "write"]
        assert result.metadata["source"] == "jwt"
        assert result.error is None
    
    def test_authentication_result_failure(self):
        """Test failed authentication result"""
        result = AuthenticationResult(
            success=False,
            error="Invalid credentials"
        )
        
        assert result.success == False
        assert result.error == "Invalid credentials"
        assert result.user_id is None
        assert result.agent_id is None
        assert result.scopes == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
