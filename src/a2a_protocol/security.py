#!/usr/bin/env python3
"""
A2A Security and Authentication

Implements proper authentication schemes and authorization mechanisms 
according to Google A2A specification for enterprise-ready agent communication.
"""

import hashlib
import hmac
import jwt
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

logger = logging.getLogger(__name__)


class AuthSchemeType(Enum):
    """A2A Authentication Scheme Types"""
    BEARER = "bearer"
    API_KEY = "apiKey"
    BASIC = "basic"
    OAUTH2 = "oauth2"


@dataclass
class AuthenticationResult:
    """Result of authentication attempt"""
    success: bool
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class APIKey:
    """API Key configuration"""
    key_id: str
    key_hash: str  # SHA256 hash of the actual key
    name: str
    scopes: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JWTConfig:
    """JWT Configuration"""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    issuer: str = "pygent-factory"
    audience: str = "a2a-agents"


class A2ASecurityManager:
    """Manages A2A security and authentication"""
    
    def __init__(self, jwt_config: Optional[JWTConfig] = None):
        self.jwt_config = jwt_config or self._create_default_jwt_config()
        self.api_keys: Dict[str, APIKey] = {}
        self.revoked_tokens: set = set()
        
        # Security schemes for agent cards
        self.security_schemes = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Bearer token authentication using JWT"
            },
            "apiKeyAuth": {
                "type": "apiKey",
                "name": "X-API-Key",
                "in": "header",
                "description": "API key authentication"
            }
        }
    
    def _create_default_jwt_config(self) -> JWTConfig:
        """Create default JWT configuration"""
        return JWTConfig(
            secret_key=secrets.token_urlsafe(32),
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7
        )
    
    def generate_api_key(self, name: str, scopes: Optional[List[str]] = None,
                        expires_in_days: Optional[int] = None) -> tuple[str, APIKey]:
        """Generate a new API key"""
        # Generate random key
        raw_key = secrets.token_urlsafe(32)
        key_id = secrets.token_urlsafe(16)
        
        # Hash the key for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = (datetime.utcnow() + timedelta(days=expires_in_days)).isoformat()
        
        # Create API key record
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes or [],
            expires_at=expires_at
        )
        
        # Store API key
        self.api_keys[key_id] = api_key
        
        logger.info(f"Generated API key '{name}' with ID {key_id}")
        return raw_key, api_key
    
    def validate_api_key(self, api_key: str) -> AuthenticationResult:
        """Validate API key"""
        try:
            # Hash the provided key
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Find matching API key
            for key_id, stored_key in self.api_keys.items():
                if stored_key.key_hash == key_hash and stored_key.active:
                    # Check expiration
                    if stored_key.expires_at:
                        expires_at = datetime.fromisoformat(stored_key.expires_at.replace('Z', '+00:00'))
                        if datetime.utcnow() > expires_at.replace(tzinfo=None):
                            return AuthenticationResult(
                                success=False,
                                error="API key expired"
                            )
                    
                    return AuthenticationResult(
                        success=True,
                        user_id=stored_key.name,
                        scopes=stored_key.scopes,
                        metadata={"key_id": key_id, "key_name": stored_key.name}
                    )
            
            return AuthenticationResult(
                success=False,
                error="Invalid API key"
            )
            
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return AuthenticationResult(
                success=False,
                error="Authentication error"
            )
    
    def create_access_token(self, user_id: str, agent_id: Optional[str] = None,
                          scopes: Optional[List[str]] = None,
                          extra_claims: Optional[Dict[str, Any]] = None) -> str:
        """Create JWT access token"""
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=self.jwt_config.access_token_expire_minutes)
        
        payload = {
            "sub": user_id,
            "iss": self.jwt_config.issuer,
            "aud": self.jwt_config.audience,
            "iat": int(now.timestamp()) - 5,  # Subtract 5 seconds to avoid timing issues
            "exp": int(expires_at.timestamp()),
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
            "scopes": scopes or [],
            "type": "access"
        }
        
        if agent_id:
            payload["agent_id"] = agent_id
        
        if extra_claims:
            payload.update(extra_claims)
        
        token = jwt.encode(payload, self.jwt_config.secret_key, algorithm=self.jwt_config.algorithm)
        logger.debug(f"Created access token for user {user_id}")
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        expires_at = now + timedelta(days=self.jwt_config.refresh_token_expire_days)
        
        payload = {
            "sub": user_id,
            "iss": self.jwt_config.issuer,
            "aud": self.jwt_config.audience,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": secrets.token_urlsafe(16),
            "type": "refresh"
        }
        
        token = jwt.encode(payload, self.jwt_config.secret_key, algorithm=self.jwt_config.algorithm)
        logger.debug(f"Created refresh token for user {user_id}")
        return token
    
    def validate_jwt_token(self, token: str) -> AuthenticationResult:
        """Validate JWT token"""
        try:
            # Decode token with leeway for clock skew and disabled iat verification
            payload = jwt.decode(
                token,
                self.jwt_config.secret_key,
                algorithms=[self.jwt_config.algorithm],
                audience=self.jwt_config.audience,
                issuer=self.jwt_config.issuer,
                leeway=10,  # Allow 10 seconds of clock skew
                options={"verify_iat": False}  # Disable iat verification for now
            )
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti in self.revoked_tokens:
                return AuthenticationResult(
                    success=False,
                    error="Token revoked"
                )
            
            # Check token type
            token_type = payload.get("type", "access")
            if token_type != "access":
                return AuthenticationResult(
                    success=False,
                    error="Invalid token type"
                )
            
            return AuthenticationResult(
                success=True,
                user_id=payload.get("sub"),
                agent_id=payload.get("agent_id"),
                scopes=payload.get("scopes", []),
                metadata={
                    "jti": jti,
                    "iat": payload.get("iat"),
                    "exp": payload.get("exp")
                }
            )
            
        except jwt.ExpiredSignatureError:
            return AuthenticationResult(
                success=False,
                error="Token expired"
            )
        except jwt.InvalidTokenError as e:
            return AuthenticationResult(
                success=False,
                error=f"Invalid token: {str(e)}"
            )
        except Exception as e:
            logger.error(f"JWT validation error: {e}")
            return AuthenticationResult(
                success=False,
                error="Authentication error"
            )
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token"""
        try:
            # Decode without verification to get JTI
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti")
            
            if jti:
                self.revoked_tokens.add(jti)
                logger.info(f"Revoked token with JTI: {jti}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Token revocation error: {e}")
            return False
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id in self.api_keys:
            self.api_keys[key_id].active = False
            logger.info(f"Revoked API key: {key_id}")
            return True
        return False
    
    def check_scope_permission(self, required_scopes: List[str], 
                             user_scopes: List[str]) -> bool:
        """Check if user has required scopes"""
        if not required_scopes:
            return True
        
        # Check if user has all required scopes
        return all(scope in user_scopes for scope in required_scopes)
    
    def get_security_schemes(self) -> Dict[str, Any]:
        """Get security schemes for agent cards"""
        return self.security_schemes.copy()

    def generate_jwt_token(self, payload: Dict[str, Any]) -> str:
        """Generate JWT token (alias for create_access_token)"""
        user_id = payload.get("user_id", "unknown")
        scopes = payload.get("scope", "").split() if payload.get("scope") else []
        agent_id = payload.get("agent_id")

        return self.create_access_token(
            user_id=user_id,
            agent_id=agent_id,
            scopes=scopes,
            extra_claims={k: v for k, v in payload.items() if k not in ["user_id", "scope", "agent_id"]}
        )




class A2AAuthenticationHandler:
    """Handles A2A authentication for FastAPI"""
    
    def __init__(self, security_manager: A2ASecurityManager):
        self.security_manager = security_manager
        self.bearer_security = HTTPBearer(auto_error=False)
        self.api_key_security = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    async def authenticate_request(self, 
                                 bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
                                 api_key: Optional[str] = Depends(APIKeyHeader(name="X-API-Key", auto_error=False))) -> AuthenticationResult:
        """Authenticate request using bearer token or API key"""
        
        # Try bearer token first
        if bearer_token and bearer_token.credentials:
            result = self.security_manager.validate_jwt_token(bearer_token.credentials)
            if result.success:
                return result
        
        # Try API key
        if api_key:
            result = self.security_manager.validate_api_key(api_key)
            if result.success:
                return result
        
        # No valid authentication
        return AuthenticationResult(
            success=False,
            error="No valid authentication provided"
        )
    
    async def require_authentication(self, 
                                   auth_result: AuthenticationResult = Depends(lambda self=None: self.authenticate_request() if self else None)) -> AuthenticationResult:
        """Require valid authentication"""
        if not auth_result.success:
            raise HTTPException(
                status_code=401,
                detail=auth_result.error or "Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return auth_result
    
    async def require_scopes(self, required_scopes: List[str]):
        """Create dependency that requires specific scopes"""
        async def check_scopes(auth_result: AuthenticationResult = Depends(self.require_authentication)):
            if not self.security_manager.check_scope_permission(required_scopes, auth_result.scopes):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required scopes: {required_scopes}"
                )
            return auth_result
        return check_scopes


# Global security manager instance
security_manager = A2ASecurityManager()
auth_handler = A2AAuthenticationHandler(security_manager)
