"""
Security utilities for PyGent Factory Startup Service
JWT authentication, password hashing, and secure credential management.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
from fastapi import HTTPException, status

from .logging_config import security_logger


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Encryption key for sensitive data
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
cipher_suite = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    try:
        hashed = pwd_context.hash(password)
        security_logger.debug("Password hashed successfully")
        return hashed
    except Exception as e:
        security_logger.error(f"Failed to hash password: {e}")
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        result = pwd_context.verify(plain_password, hashed_password)
        security_logger.debug(f"Password verification: {'success' if result else 'failed'}")
        return result
    except Exception as e:
        security_logger.error(f"Failed to verify password: {e}")
        return False


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    try:
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        security_logger.info(
            "Access token created",
            user=data.get("sub", "unknown"),
            expires_at=expire.isoformat()
        )
        
        return encoded_jwt
    except Exception as e:
        security_logger.error(f"Failed to create access token: {e}")
        raise


def verify_jwt_token(token: str) -> Dict[str, Any]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check if token is expired
        exp = payload.get("exp")
        if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
            security_logger.warning("Token verification failed: token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        security_logger.debug(
            "Token verified successfully",
            user=payload.get("sub", "unknown")
        )
        
        return payload
    except jwt.ExpiredSignatureError:
        security_logger.warning("Token verification failed: signature expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError as e:
        security_logger.warning(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def encrypt_sensitive_data(data: str) -> str:
    """Encrypt sensitive data using Fernet encryption."""
    try:
        encrypted_data = cipher_suite.encrypt(data.encode())
        security_logger.debug("Sensitive data encrypted successfully")
        return encrypted_data.decode()
    except Exception as e:
        security_logger.error(f"Failed to encrypt sensitive data: {e}")
        raise


def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive data using Fernet encryption."""
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data.encode())
        security_logger.debug("Sensitive data decrypted successfully")
        return decrypted_data.decode()
    except Exception as e:
        security_logger.error(f"Failed to decrypt sensitive data: {e}")
        raise


def generate_api_key() -> str:
    """Generate a secure API key."""
    api_key = secrets.token_urlsafe(32)
    security_logger.info("API key generated")
    return api_key


def validate_api_key(api_key: str, stored_hash: str) -> bool:
    """Validate an API key against its stored hash."""
    try:
        result = pwd_context.verify(api_key, stored_hash)
        security_logger.debug(f"API key validation: {'success' if result else 'failed'}")
        return result
    except Exception as e:
        security_logger.error(f"Failed to validate API key: {e}")
        return False


class SecurityManager:
    """Centralized security management for the startup service."""
    
    def __init__(self):
        self.logger = security_logger
    
    def create_user_token(self, user_id: str, username: str, roles: list = None) -> str:
        """Create a user authentication token."""
        token_data = {
            "sub": user_id,
            "username": username,
            "roles": roles or ["user"],
            "token_type": "access"
        }
        return create_access_token(token_data)
    
    def create_service_token(self, service_name: str, permissions: list = None) -> str:
        """Create a service-to-service authentication token."""
        token_data = {
            "sub": f"service:{service_name}",
            "service": service_name,
            "permissions": permissions or [],
            "token_type": "service"
        }
        return create_access_token(token_data, expires_delta=timedelta(hours=24))
    
    def validate_user_permissions(self, token_payload: Dict[str, Any], required_role: str) -> bool:
        """Validate user permissions based on token payload."""
        try:
            user_roles = token_payload.get("roles", [])
            has_permission = required_role in user_roles or "admin" in user_roles
            
            self.logger.debug(
                "Permission validation",
                user=token_payload.get("sub"),
                required_role=required_role,
                user_roles=user_roles,
                granted=has_permission
            )
            
            return has_permission
        except Exception as e:
            self.logger.error(f"Failed to validate permissions: {e}")
            return False
    
    def secure_config_value(self, value: str, config_key: str) -> str:
        """Securely store a configuration value."""
        try:
            encrypted_value = encrypt_sensitive_data(value)
            self.logger.info(f"Configuration value secured: {config_key}")
            return encrypted_value
        except Exception as e:
            self.logger.error(f"Failed to secure config value {config_key}: {e}")
            raise
    
    def retrieve_config_value(self, encrypted_value: str, config_key: str) -> str:
        """Retrieve and decrypt a configuration value."""
        try:
            decrypted_value = decrypt_sensitive_data(encrypted_value)
            self.logger.debug(f"Configuration value retrieved: {config_key}")
            return decrypted_value
        except Exception as e:
            self.logger.error(f"Failed to retrieve config value {config_key}: {e}")
            raise
    
    def audit_security_event(self, event_type: str, user: str, details: Dict[str, Any] = None):
        """Log security events for auditing."""
        self.logger.info(
            f"Security event: {event_type}",
            event_type=event_type,
            user=user,
            details=details or {},
            timestamp=datetime.utcnow().isoformat()
        )


# Global security manager instance
security_manager = SecurityManager()


def require_role(required_role: str):
    """Decorator to require specific role for endpoint access."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be used with FastAPI dependencies
            # Implementation depends on how the token is passed
            pass
        return wrapper
    return decorator


def get_current_active_user(token_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Get current active user from token payload."""
    try:
        user_id = token_payload.get("sub")
        username = token_payload.get("username")
        
        if not user_id or not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # In a real implementation, you might check if user is active in database
        user_data = {
            "id": user_id,
            "username": username,
            "roles": token_payload.get("roles", []),
            "is_active": True
        }
        
        security_logger.debug(f"Active user retrieved: {username}")
        return user_data
        
    except Exception as e:
        security_logger.error(f"Failed to get current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate user"
        )
