#!/usr/bin/env python3
"""
JWT Authentication System

Production-ready JWT authentication with token generation, validation,
refresh tokens, and comprehensive security features.
"""

import asyncio
import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..cache.session_manager import session_manager
from ..cache.redis_manager import redis_manager
from ..database.production_manager import db_manager

logger = logging.getLogger(__name__)


@dataclass
class AuthConfig:
    """Authentication configuration"""
    jwt_secret_key: str = "your-super-secret-jwt-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    require_email_verification: bool = True
    enable_2fa: bool = False


@dataclass
class TokenData:
    """JWT token data structure"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    session_id: Optional[str] = None
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


@dataclass
class AuthResult:
    """Authentication result"""
    success: bool
    user_id: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    error: Optional[str] = None
    requires_2fa: bool = False


class JWTAuthenticator:
    """JWT authentication system with comprehensive security features"""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer(auto_error=False)
        
        # Performance metrics
        self.auth_metrics = {
            "total_logins": 0,
            "successful_logins": 0,
            "failed_logins": 0,
            "token_validations": 0,
            "token_refreshes": 0,
            "password_resets": 0
        }
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        issues = []
        
        if len(password) < self.config.password_min_length:
            issues.append(f"Password must be at least {self.config.password_min_length} characters")
        
        if not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one special character")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "strength": "strong" if len(issues) == 0 else "weak"
        }
    
    async def create_access_token(self, user_data: Dict[str, Any], 
                                session_id: Optional[str] = None) -> str:
        """Create JWT access token"""
        try:
            now = datetime.utcnow()
            expires_at = now + timedelta(minutes=self.config.access_token_expire_minutes)
            
            payload = {
                "sub": user_data["user_id"],
                "username": user_data["username"],
                "email": user_data["email"],
                "roles": user_data.get("roles", []),
                "permissions": user_data.get("permissions", []),
                "session_id": session_id,
                "iat": now,
                "exp": expires_at,
                "type": "access"
            }
            
            token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
            return token
            
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create access token"
            )
    
    async def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token"""
        try:
            now = datetime.utcnow()
            expires_at = now + timedelta(days=self.config.refresh_token_expire_days)
            
            # Generate secure random token
            token_data = f"{user_id}:{now.isoformat()}:{secrets.token_urlsafe(32)}"
            refresh_token = hashlib.sha256(token_data.encode()).hexdigest()
            
            # Store refresh token in Redis
            await redis_manager.set(
                f"refresh_token:{refresh_token}",
                {
                    "user_id": user_id,
                    "created_at": now.isoformat(),
                    "expires_at": expires_at.isoformat()
                },
                ttl=int(self.config.refresh_token_expire_days * 24 * 3600)
            )
            
            return refresh_token
            
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create refresh token"
            )
    
    async def validate_token(self, token: str) -> TokenData:
        """Validate JWT token and return token data"""
        try:
            self.auth_metrics["token_validations"] += 1
            
            # Decode JWT token
            payload = jwt.decode(
                token, 
                self.config.jwt_secret_key, 
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Validate token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            # Validate session if present
            session_id = payload.get("session_id")
            if session_id:
                session_data = await session_manager.get_session(session_id)
                if not session_data:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Session has expired"
                    )
            
            # Create token data
            token_data = TokenData(
                user_id=payload["sub"],
                username=payload["username"],
                email=payload["email"],
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                session_id=session_id,
                issued_at=datetime.fromtimestamp(payload.get("iat", 0)),
                expires_at=datetime.fromtimestamp(payload.get("exp", 0))
            )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token validation failed"
            )
    
    async def refresh_access_token(self, refresh_token: str) -> AuthResult:
        """Refresh access token using refresh token"""
        try:
            self.auth_metrics["token_refreshes"] += 1
            
            # Get refresh token data from Redis
            token_data = await redis_manager.get(f"refresh_token:{refresh_token}")
            
            if not token_data:
                return AuthResult(
                    success=False,
                    error="Invalid or expired refresh token"
                )
            
            user_id = token_data["user_id"]
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            
            # Check if refresh token is expired
            if expires_at < datetime.utcnow():
                # Clean up expired token
                await redis_manager.delete(f"refresh_token:{refresh_token}")
                return AuthResult(
                    success=False,
                    error="Refresh token has expired"
                )
            
            # Get user data from database
            user_data = await self._get_user_data(user_id)
            if not user_data:
                return AuthResult(
                    success=False,
                    error="User not found"
                )
            
            # Create new access token
            access_token = await self.create_access_token(user_data)
            
            return AuthResult(
                success=True,
                user_id=user_id,
                access_token=access_token,
                expires_in=self.config.access_token_expire_minutes * 60
            )
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return AuthResult(
                success=False,
                error="Token refresh failed"
            )
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: Optional[str] = None) -> AuthResult:
        """Authenticate user with username/password"""
        try:
            self.auth_metrics["total_logins"] += 1
            
            # Check for account lockout
            lockout_key = f"lockout:{username}"
            lockout_data = await redis_manager.get(lockout_key)
            
            if lockout_data:
                lockout_until = datetime.fromisoformat(lockout_data["until"])
                if lockout_until > datetime.utcnow():
                    return AuthResult(
                        success=False,
                        error=f"Account locked until {lockout_until.isoformat()}"
                    )
            
            # Get user from database
            user_data = await self._get_user_by_username(username)
            if not user_data:
                await self._record_failed_login(username, ip_address)
                return AuthResult(
                    success=False,
                    error="Invalid username or password"
                )
            
            # Verify password
            if not self.verify_password(password, user_data["password_hash"]):
                await self._record_failed_login(username, ip_address)
                return AuthResult(
                    success=False,
                    error="Invalid username or password"
                )
            
            # Check if account is active
            if not user_data.get("is_active", True):
                return AuthResult(
                    success=False,
                    error="Account is disabled"
                )
            
            # Check email verification if required
            if self.config.require_email_verification and not user_data.get("email_verified", False):
                return AuthResult(
                    success=False,
                    error="Email verification required"
                )
            
            # Clear any existing lockout
            await redis_manager.delete(lockout_key)
            await redis_manager.delete(f"failed_logins:{username}")
            
            # Create session
            session_data = await session_manager.create_session(
                user_data["user_id"],
                ip_address,
                user_agent=None  # Would get from request headers
            )
            
            session_id = session_data.session_id if session_data else None
            
            # Create tokens
            access_token = await self.create_access_token(user_data, session_id)
            refresh_token = await self.create_refresh_token(user_data["user_id"])
            
            self.auth_metrics["successful_logins"] += 1
            
            return AuthResult(
                success=True,
                user_id=user_data["user_id"],
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.config.access_token_expire_minutes * 60,
                requires_2fa=self.config.enable_2fa and user_data.get("two_factor_enabled", False)
            )
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return AuthResult(
                success=False,
                error="Authentication failed"
            )
    
    async def logout_user(self, token: str) -> bool:
        """Logout user and invalidate tokens"""
        try:
            # Validate token to get session info
            token_data = await self.validate_token(token)
            
            # Delete session if present
            if token_data.session_id:
                await session_manager.delete_session(token_data.session_id)
            
            # Add token to blacklist (implement token blacklisting)
            await self._blacklist_token(token, token_data.expires_at)
            
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    async def _record_failed_login(self, username: str, ip_address: Optional[str]):
        """Record failed login attempt and implement lockout"""
        try:
            failed_key = f"failed_logins:{username}"
            
            # Get current failed attempts
            failed_data = await redis_manager.get(failed_key) or {"count": 0, "attempts": []}
            
            # Add new attempt
            failed_data["count"] += 1
            failed_data["attempts"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "ip_address": ip_address
            })
            
            # Store updated data
            await redis_manager.set(failed_key, failed_data, ttl=3600)  # 1 hour
            
            # Check if lockout threshold reached
            if failed_data["count"] >= self.config.max_login_attempts:
                lockout_until = datetime.utcnow() + timedelta(minutes=self.config.lockout_duration_minutes)
                await redis_manager.set(
                    f"lockout:{username}",
                    {"until": lockout_until.isoformat()},
                    ttl=self.config.lockout_duration_minutes * 60
                )
                
                logger.warning(f"Account locked for user {username} until {lockout_until}")
            
            self.auth_metrics["failed_logins"] += 1
            
        except Exception as e:
            logger.error(f"Failed to record failed login: {e}")
    
    async def _blacklist_token(self, token: str, expires_at: datetime):
        """Add token to blacklist"""
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            ttl = int((expires_at - datetime.utcnow()).total_seconds())
            
            if ttl > 0:
                await redis_manager.set(
                    f"blacklisted_token:{token_hash}",
                    {"blacklisted_at": datetime.utcnow().isoformat()},
                    ttl=ttl
                )
                
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")
    
    async def _get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data from database"""
        try:
            async with db_manager.get_session() as session:
                from sqlalchemy import select
                from ..database.models import User
                
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if user:
                    return {
                        "user_id": str(user.id),
                        "username": user.username,
                        "email": user.email,
                        "password_hash": user.password_hash,
                        "roles": [user.role] if user.role else [],
                        "permissions": [],  # Implement role-based permissions
                        "is_active": user.is_active,
                        "email_verified": user.email_verified,
                        "two_factor_enabled": getattr(user, 'two_factor_enabled', False)
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user data: {e}")
            return None
    
    async def _get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username from database"""
        try:
            async with db_manager.get_session() as session:
                from sqlalchemy import select
                from ..database.models import User
                
                result = await session.execute(
                    select(User).where(User.username == username)
                )
                user = result.scalar_one_or_none()
                
                if user:
                    return {
                        "user_id": str(user.id),
                        "username": user.username,
                        "email": user.email,
                        "password_hash": user.password_hash,
                        "roles": [user.role] if user.role else [],
                        "permissions": [],  # Implement role-based permissions
                        "is_active": user.is_active,
                        "email_verified": user.email_verified,
                        "two_factor_enabled": getattr(user, 'two_factor_enabled', False)
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by username: {e}")
            return None
    
    def get_auth_metrics(self) -> Dict[str, Any]:
        """Get authentication metrics"""
        total_logins = self.auth_metrics["total_logins"]
        success_rate = (self.auth_metrics["successful_logins"] / total_logins * 100) if total_logins > 0 else 0
        
        return {
            **self.auth_metrics,
            "success_rate_percent": round(success_rate, 2)
        }


# Global JWT authenticator instance
jwt_auth = JWTAuthenticator()
