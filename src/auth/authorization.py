#!/usr/bin/env python3
"""
Authorization and Role-Based Access Control (RBAC)

Comprehensive authorization system with role-based access control,
permission management, and resource-level security.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps

from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials

from .jwt_auth import jwt_auth, TokenData
from ..cache.cache_layers import cache_manager

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions"""
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_LIST = "user:list"
    
    # Agent management
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_LIST = "agent:list"
    AGENT_EXECUTE = "agent:execute"
    
    # Task management
    TASK_CREATE = "task:create"
    TASK_READ = "task:read"
    TASK_UPDATE = "task:update"
    TASK_DELETE = "task:delete"
    TASK_LIST = "task:list"
    TASK_EXECUTE = "task:execute"
    
    # Document management
    DOCUMENT_CREATE = "document:create"
    DOCUMENT_READ = "document:read"
    DOCUMENT_UPDATE = "document:update"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_LIST = "document:list"
    DOCUMENT_UPLOAD = "document:upload"
    
    # Model inference
    MODEL_INFERENCE = "model:inference"
    MODEL_MANAGE = "model:manage"
    MODEL_CONFIGURE = "model:configure"
    
    # System administration
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_LOGS = "system:logs"
    
    # API access
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"


class Role(Enum):
    """System roles with hierarchical permissions"""
    GUEST = "guest"
    USER = "user"
    PREMIUM_USER = "premium_user"
    DEVELOPER = "developer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class RoleDefinition:
    """Role definition with permissions"""
    name: str
    permissions: Set[Permission]
    description: str
    inherits_from: Optional[List[str]] = None


@dataclass
class AuthorizationContext:
    """Authorization context for requests"""
    user_id: str
    username: str
    roles: List[str]
    permissions: Set[Permission]
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None


class RBACManager:
    """Role-Based Access Control manager"""
    
    def __init__(self):
        self.role_definitions = self._initialize_roles()
        self.permission_cache_ttl = 300  # 5 minutes
        
        # Authorization metrics
        self.auth_metrics = {
            "authorization_checks": 0,
            "authorization_grants": 0,
            "authorization_denials": 0,
            "permission_cache_hits": 0,
            "permission_cache_misses": 0
        }
    
    def _initialize_roles(self) -> Dict[str, RoleDefinition]:
        """Initialize role definitions"""
        roles = {}
        
        # Guest role - minimal permissions
        roles[Role.GUEST.value] = RoleDefinition(
            name=Role.GUEST.value,
            permissions={
                Permission.API_READ,
                Permission.DOCUMENT_READ,
                Permission.USER_READ
            },
            description="Guest user with read-only access"
        )
        
        # User role - basic user permissions
        roles[Role.USER.value] = RoleDefinition(
            name=Role.USER.value,
            permissions={
                Permission.API_READ,
                Permission.API_WRITE,
                Permission.USER_READ,
                Permission.USER_UPDATE,
                Permission.DOCUMENT_CREATE,
                Permission.DOCUMENT_READ,
                Permission.DOCUMENT_UPDATE,
                Permission.DOCUMENT_DELETE,
                Permission.DOCUMENT_LIST,
                Permission.DOCUMENT_UPLOAD,
                Permission.AGENT_CREATE,
                Permission.AGENT_READ,
                Permission.AGENT_UPDATE,
                Permission.AGENT_DELETE,
                Permission.AGENT_LIST,
                Permission.TASK_CREATE,
                Permission.TASK_READ,
                Permission.TASK_UPDATE,
                Permission.TASK_DELETE,
                Permission.TASK_LIST,
                Permission.MODEL_INFERENCE
            },
            description="Standard user with basic functionality",
            inherits_from=[Role.GUEST.value]
        )
        
        # Premium user role - enhanced permissions
        roles[Role.PREMIUM_USER.value] = RoleDefinition(
            name=Role.PREMIUM_USER.value,
            permissions={
                Permission.AGENT_EXECUTE,
                Permission.TASK_EXECUTE,
                Permission.MODEL_MANAGE
            },
            description="Premium user with advanced features",
            inherits_from=[Role.USER.value]
        )
        
        # Developer role - development permissions
        roles[Role.DEVELOPER.value] = RoleDefinition(
            name=Role.DEVELOPER.value,
            permissions={
                Permission.API_ADMIN,
                Permission.MODEL_CONFIGURE,
                Permission.SYSTEM_MONITOR,
                Permission.SYSTEM_LOGS
            },
            description="Developer with system access",
            inherits_from=[Role.PREMIUM_USER.value]
        )
        
        # Admin role - administrative permissions
        roles[Role.ADMIN.value] = RoleDefinition(
            name=Role.ADMIN.value,
            permissions={
                Permission.USER_CREATE,
                Permission.USER_DELETE,
                Permission.USER_LIST,
                Permission.SYSTEM_ADMIN,
                Permission.SYSTEM_CONFIG
            },
            description="Administrator with full user management",
            inherits_from=[Role.DEVELOPER.value]
        )
        
        # Super admin role - all permissions
        roles[Role.SUPER_ADMIN.value] = RoleDefinition(
            name=Role.SUPER_ADMIN.value,
            permissions=set(Permission),  # All permissions
            description="Super administrator with unrestricted access"
        )
        
        return roles
    
    def get_role_permissions(self, role_name: str) -> Set[Permission]:
        """Get all permissions for a role including inherited permissions"""
        if role_name not in self.role_definitions:
            return set()
        
        role_def = self.role_definitions[role_name]
        permissions = role_def.permissions.copy()
        
        # Add inherited permissions
        if role_def.inherits_from:
            for parent_role in role_def.inherits_from:
                permissions.update(self.get_role_permissions(parent_role))
        
        return permissions
    
    def get_user_permissions(self, roles: List[str]) -> Set[Permission]:
        """Get all permissions for a user based on their roles"""
        permissions = set()
        
        for role in roles:
            permissions.update(self.get_role_permissions(role))
        
        return permissions
    
    async def get_cached_user_permissions(self, user_id: str, roles: List[str]) -> Set[Permission]:
        """Get user permissions with caching"""
        try:
            self.auth_metrics["authorization_checks"] += 1
            
            # Try cache first
            cache_key = f"user_permissions:{user_id}:{':'.join(sorted(roles))}"
            cached_permissions = await cache_manager.get_cached_performance_metric(cache_key)
            
            if cached_permissions:
                self.auth_metrics["permission_cache_hits"] += 1
                return {Permission(p) for p in cached_permissions}
            
            # Calculate permissions
            permissions = self.get_user_permissions(roles)
            
            # Cache permissions
            permission_values = [p.value for p in permissions]
            await cache_manager.cache_performance_metric(
                cache_key, 
                permission_values, 
                ttl=self.permission_cache_ttl
            )
            
            self.auth_metrics["permission_cache_misses"] += 1
            return permissions
            
        except Exception as e:
            logger.error(f"Failed to get cached user permissions: {e}")
            # Fallback to direct calculation
            return self.get_user_permissions(roles)
    
    async def check_permission(self, context: AuthorizationContext, 
                             required_permission: Permission) -> bool:
        """Check if user has required permission"""
        try:
            user_permissions = await self.get_cached_user_permissions(
                context.user_id, 
                context.roles
            )
            
            has_permission = required_permission in user_permissions
            
            if has_permission:
                self.auth_metrics["authorization_grants"] += 1
            else:
                self.auth_metrics["authorization_denials"] += 1
                logger.warning(
                    f"Permission denied: user {context.username} "
                    f"lacks {required_permission.value} for {context.request_path}"
                )
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    async def check_resource_access(self, context: AuthorizationContext, 
                                  resource_type: str, resource_id: str,
                                  required_permission: Permission) -> bool:
        """Check if user has permission to access specific resource"""
        try:
            # First check if user has the general permission
            if not await self.check_permission(context, required_permission):
                return False
            
            # Check resource-specific access
            return await self._check_resource_ownership(
                context.user_id, 
                resource_type, 
                resource_id
            )
            
        except Exception as e:
            logger.error(f"Resource access check failed: {e}")
            return False
    
    async def _check_resource_ownership(self, user_id: str, resource_type: str, 
                                      resource_id: str) -> bool:
        """Check if user owns or has access to specific resource"""
        try:
            # Implement resource ownership checks based on resource type
            # This would query the database to check ownership
            
            # For now, return True for basic implementation
            # In production, implement proper ownership checks
            return True
            
        except Exception as e:
            logger.error(f"Resource ownership check failed: {e}")
            return False
    
    def get_authorization_metrics(self) -> Dict[str, Any]:
        """Get authorization metrics"""
        total_checks = self.auth_metrics["authorization_checks"]
        grant_rate = (self.auth_metrics["authorization_grants"] / total_checks * 100) if total_checks > 0 else 0
        cache_hit_rate = (self.auth_metrics["permission_cache_hits"] / total_checks * 100) if total_checks > 0 else 0
        
        return {
            **self.auth_metrics,
            "grant_rate_percent": round(grant_rate, 2),
            "cache_hit_rate_percent": round(cache_hit_rate, 2)
        }


class AuthorizationDependency:
    """FastAPI dependency for authorization"""
    
    def __init__(self, rbac_manager: RBACManager):
        self.rbac_manager = rbac_manager
    
    async def __call__(self, request: Request, 
                      credentials: Optional[HTTPAuthorizationCredentials] = Depends(jwt_auth.security)):
        """Validate authentication and create authorization context"""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        try:
            # Validate JWT token
            token_data = await jwt_auth.validate_token(credentials.credentials)
            
            # Get user permissions
            permissions = await self.rbac_manager.get_cached_user_permissions(
                token_data.user_id,
                token_data.roles
            )
            
            # Create authorization context
            context = AuthorizationContext(
                user_id=token_data.user_id,
                username=token_data.username,
                roles=token_data.roles,
                permissions=permissions,
                session_id=token_data.session_id,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
                request_path=request.url.path,
                request_method=request.method
            )
            
            return context
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authorization dependency failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )


def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get authorization context from kwargs
            auth_context = None
            for arg in args:
                if isinstance(arg, AuthorizationContext):
                    auth_context = arg
                    break
            
            if not auth_context:
                for value in kwargs.values():
                    if isinstance(value, AuthorizationContext):
                        auth_context = value
                        break
            
            if not auth_context:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authorization context not found"
                )
            
            # Check permission
            if not await rbac_manager.check_permission(auth_context, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission {permission.value} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(role: Role):
    """Decorator to require specific role"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get authorization context from kwargs
            auth_context = None
            for arg in args:
                if isinstance(arg, AuthorizationContext):
                    auth_context = arg
                    break
            
            if not auth_context:
                for value in kwargs.values():
                    if isinstance(value, AuthorizationContext):
                        auth_context = value
                        break
            
            if not auth_context:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authorization context not found"
                )
            
            # Check role
            if role.value not in auth_context.roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role {role.value} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global RBAC manager and dependency
rbac_manager = RBACManager()
get_auth_context = AuthorizationDependency(rbac_manager)
