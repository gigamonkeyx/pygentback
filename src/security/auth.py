"""
Authentication and Authorization

This module provides authentication and authorization capabilities for PyGent Factory,
including JWT token management, API key authentication, role-based access control,
and security middleware for protecting endpoints.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import secrets
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel

from ..config.settings import get_settings
from ..database.connection import get_database_manager

# Import database models conditionally
try:
    from ..database.models import User as UserModel, OAuthToken as OAuthTokenModel
    from sqlalchemy import select, and_
    from sqlalchemy.ext.asyncio import AsyncSession
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for access control"""
    ADMIN = "admin"
    USER = "user"
    AGENT = "agent"
    READONLY = "readonly"
    SERVICE = "service"


class Permission(Enum):
    """System permissions"""
    # Agent permissions
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_EXECUTE = "agent:execute"
    
    # Memory permissions
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    MEMORY_DELETE = "memory:delete"
    
    # MCP permissions
    MCP_READ = "mcp:read"
    MCP_MANAGE = "mcp:manage"
    MCP_EXECUTE = "mcp:execute"
    
    # RAG permissions
    RAG_READ = "rag:read"
    RAG_WRITE = "rag:write"
    RAG_DELETE = "rag:delete"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"


# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [p for p in Permission],  # All permissions
    UserRole.USER: [
        Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE, Permission.AGENT_EXECUTE,
        Permission.MEMORY_READ, Permission.MEMORY_WRITE,
        Permission.MCP_READ, Permission.MCP_EXECUTE,
        Permission.RAG_READ, Permission.RAG_WRITE,
        Permission.SYSTEM_MONITOR
    ],
    UserRole.AGENT: [
        Permission.AGENT_READ, Permission.AGENT_EXECUTE,
        Permission.MEMORY_READ, Permission.MEMORY_WRITE,
        Permission.MCP_EXECUTE,
        Permission.RAG_READ
    ],
    UserRole.READONLY: [
        Permission.AGENT_READ,
        Permission.MEMORY_READ,
        Permission.MCP_READ,
        Permission.RAG_READ,
        Permission.SYSTEM_MONITOR
    ],
    UserRole.SERVICE: [
        Permission.AGENT_READ, Permission.AGENT_EXECUTE,
        Permission.MEMORY_READ, Permission.MEMORY_WRITE,
        Permission.MCP_READ, Permission.MCP_EXECUTE,
        Permission.RAG_READ, Permission.RAG_WRITE
    ]
}


class User(BaseModel):
    """User model"""
    id: str
    username: str
    email: Optional[str] = None
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    permissions: List[Permission] = []
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions or permission in ROLE_PERMISSIONS.get(self.role, [])
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        return any(self.has_permission(p) for p in permissions)


class TokenData(BaseModel):
    """JWT token data"""
    user_id: str
    username: str
    role: str
    permissions: List[str] = []
    exp: datetime
    iat: datetime


class AuthenticationError(Exception):
    """Authentication error"""
    pass


class AuthorizationError(Exception):
    """Authorization error"""
    pass


class PasswordManager:
    """Password hashing and verification"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def generate_password(self, length: int = 16) -> str:
        """Generate a secure random password"""
        return secrets.token_urlsafe(length)


class TokenManager:
    """JWT token management"""
    
    def __init__(self, settings):
        self.settings = settings
        self.secret_key = settings.security.SECRET_KEY
        self.algorithm = settings.security.ALGORITHM
        self.access_token_expire_minutes = settings.security.ACCESS_TOKEN_EXPIRE_MINUTES
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        # Get user permissions
        permissions = ROLE_PERMISSIONS.get(user.role, [])
        permissions.extend(user.permissions)
        
        token_data = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in permissions],
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        encoded_jwt = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            exp = datetime.fromtimestamp(payload.get("exp", 0))
            if datetime.utcnow() > exp:
                raise AuthenticationError("Token has expired")
            
            return TokenData(
                user_id=payload.get("user_id"),
                username=payload.get("username"),
                role=payload.get("role"),
                permissions=payload.get("permissions", []),
                exp=exp,
                iat=datetime.fromtimestamp(payload.get("iat", 0))
            )
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    def refresh_token(self, token: str) -> str:
        """Refresh an access token"""
        try:
            # Verify current token (allow expired for refresh)
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            
            # Create new token with same data but new expiration
            new_expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            payload["exp"] = new_expire
            payload["iat"] = datetime.utcnow()
            
            return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
        except jwt.JWTError as e:
            raise AuthenticationError(f"Cannot refresh token: {str(e)}")


class APIKeyManager:
    """API key management"""
    
    def __init__(self):
        self.api_keys: Dict[str, User] = {}  # In production, store in database
    
    def generate_api_key(self, user: User) -> str:
        """Generate API key for user"""
        # Create API key with user info encoded
        key_data = f"{user.id}:{user.username}:{datetime.utcnow().timestamp()}"
        api_key = hashlib.sha256(key_data.encode()).hexdigest()
        
        # Store mapping
        self.api_keys[api_key] = user
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[User]:
        """Verify API key and return associated user"""
        return self.api_keys.get(api_key)
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        return False


class AuthenticationService:
    """Main authentication service"""
    
    def __init__(self, settings):
        self.settings = settings
        self.password_manager = PasswordManager()
        self.token_manager = TokenManager(settings)
        self.api_key_manager = APIKeyManager()
        self.users: Dict[str, User] = {}  # In production, use database
        
        # Create default admin user
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default system users"""
        # Default admin user
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@pygent.factory",
            role=UserRole.ADMIN,
            created_at=datetime.utcnow()
        )
        self.users["admin"] = admin_user
        
        # Default service user for agents
        service_user = User(
            id="service",
            username="service",
            role=UserRole.SERVICE,
            created_at=datetime.utcnow()
        )
        self.users["service"] = service_user
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        # In production, verify against database
        user = self.users.get(username)
        if user and user.is_active:
            # For demo purposes, accept any password for default users
            # In production, verify against hashed password
            user.last_login = datetime.utcnow()
            return user
        return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        for user in self.users.values():
            if user.id == user_id:
                return user
        return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.users.get(username)
    
    def create_access_token(self, user: User) -> str:
        """Create access token for user"""
        return self.token_manager.create_access_token(user)
    
    def verify_token(self, token: str) -> TokenData:
        """Verify access token"""
        return self.token_manager.verify_token(token)
    
    def generate_api_key(self, user: User) -> str:
        """Generate API key for user"""
        return self.api_key_manager.generate_api_key(user)
    
    def verify_api_key(self, api_key: str) -> Optional[User]:
        """Verify API key"""
        return self.api_key_manager.verify_api_key(api_key)


# Global authentication service
_auth_service: Optional[AuthenticationService] = None


def get_auth_service() -> AuthenticationService:
    """Get global authentication service"""
    global _auth_service
    if _auth_service is None:
        settings = get_settings()
        _auth_service = AuthenticationService(settings)
    return _auth_service


# FastAPI security schemes
security_bearer = HTTPBearer()
security_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Security(security_bearer)
) -> User:
    """Get current user from JWT token"""
    try:
        auth_service = get_auth_service()
        token_data = auth_service.verify_token(credentials.credentials)
        
        user = await auth_service.get_user_by_id(token_data.user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        if not user.is_active:
            raise HTTPException(status_code=401, detail="User is inactive")
        
        return user
        
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))


async def get_current_user_from_api_key(
    api_key: Optional[str] = Security(security_api_key)
) -> Optional[User]:
    """Get current user from API key"""
    if not api_key:
        return None
    
    try:
        auth_service = get_auth_service()
        user = auth_service.verify_api_key(api_key)
        
        if user and user.is_active:
            return user
        
        return None
        
    except Exception as e:
        logger.error(f"API key authentication error: {str(e)}")
        return None


async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> User:
    """Get current user from either token or API key"""
    user = token_user or api_key_user
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


def require_permissions(*permissions: Permission):
    """Decorator to require specific permissions"""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_any_permission(list(permissions)):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {[p.value for p in permissions]}"
            )
        return current_user
    
    return permission_checker


def require_role(*roles: UserRole):
    """Decorator to require specific roles"""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient role. Required: {[r.value for r in roles]}"
            )
        return current_user
    
    return role_checker


# Common permission dependencies
require_admin = require_role(UserRole.ADMIN)
require_user_or_admin = require_role(UserRole.USER, UserRole.ADMIN)
require_agent_permissions = require_permissions(Permission.AGENT_READ, Permission.AGENT_EXECUTE)
require_memory_read = require_permissions(Permission.MEMORY_READ)
require_memory_write = require_permissions(Permission.MEMORY_WRITE)
require_mcp_execute = require_permissions(Permission.MCP_EXECUTE)
require_rag_read = require_permissions(Permission.RAG_READ)
require_rag_write = require_permissions(Permission.RAG_WRITE)
require_system_monitor = require_permissions(Permission.SYSTEM_MONITOR)


# Security middleware
async def security_middleware(request: Request, call_next):
    """Security middleware for additional protection"""
    # Add security headers
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# Rate limiting (basic implementation)
class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True


# Global rate limiter
rate_limiter = RateLimiter()


def rate_limit(identifier_func=lambda request: request.client.host):
    """Rate limiting dependency"""
    def rate_limit_checker(request: Request):
        identifier = identifier_func(request)
        
        if not rate_limiter.is_allowed(identifier):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        
        return True
    
    return rate_limit_checker
