#!/usr/bin/env python3
"""
Authentication API Endpoints

Comprehensive authentication API with login, logout, registration,
password management, and user profile endpoints.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, EmailStr, validator

from ..auth.jwt_auth import jwt_auth, AuthResult
from ..auth.authorization import get_auth_context, AuthorizationContext, Permission, require_permission
from ..cache.rate_limiter import rate_limiter
from ..database.production_manager import db_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# Request/Response Models
class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str
    remember_me: bool = False


class RegisterRequest(BaseModel):
    """Registration request model"""
    username: str
    email: EmailStr
    password: str
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('username')
    def username_valid(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not v.isalnum():
            raise ValueError('Username must contain only letters and numbers')
        return v


class RefreshTokenRequest(BaseModel):
    """Refresh token request model"""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Change password request model"""
    current_password: str
    new_password: str
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class UpdateProfileRequest(BaseModel):
    """Update profile request model"""
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class AuthResponse(BaseModel):
    """Authentication response model"""
    success: bool
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    user: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class UserProfileResponse(BaseModel):
    """User profile response model"""
    user_id: str
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    roles: list
    is_active: bool
    email_verified: bool
    created_at: str
    last_login: Optional[str] = None


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest, http_request: Request):
    """User login endpoint"""
    try:
        # Check rate limiting
        client_ip = http_request.client.host
        rate_result = await rate_limiter.check_rate_limit("api_auth", f"ip:{client_ip}")
        
        if not rate_result.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please try again later.",
                headers={"Retry-After": str(rate_result.retry_after_seconds)}
            )
        
        # Authenticate user
        auth_result = await jwt_auth.authenticate_user(
            request.username,
            request.password,
            client_ip
        )
        
        if not auth_result.success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=auth_result.error or "Authentication failed"
            )
        
        # Get user profile
        user_profile = await _get_user_profile(auth_result.user_id)
        
        return AuthResponse(
            success=True,
            access_token=auth_result.access_token,
            refresh_token=auth_result.refresh_token,
            expires_in=auth_result.expires_in,
            user=user_profile,
            message="Login successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest, http_request: Request):
    """User registration endpoint"""
    try:
        # Check rate limiting
        client_ip = http_request.client.host
        rate_result = await rate_limiter.check_rate_limit("api_auth", f"ip:{client_ip}")
        
        if not rate_result.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many registration attempts. Please try again later.",
                headers={"Retry-After": str(rate_result.retry_after_seconds)}
            )
        
        # Validate password strength
        password_validation = jwt_auth.validate_password_strength(request.password)
        if not password_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Password does not meet requirements",
                    "issues": password_validation["issues"]
                }
            )
        
        # Check if username already exists
        existing_user = await _check_username_exists(request.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check if email already exists
        existing_email = await _check_email_exists(request.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user
        user_id = await _create_user(request)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        # Authenticate newly created user
        auth_result = await jwt_auth.authenticate_user(
            request.username,
            request.password,
            client_ip
        )
        
        if auth_result.success:
            user_profile = await _get_user_profile(user_id)
            
            return AuthResponse(
                success=True,
                access_token=auth_result.access_token,
                refresh_token=auth_result.refresh_token,
                expires_in=auth_result.expires_in,
                user=user_profile,
                message="Registration successful"
            )
        else:
            return AuthResponse(
                success=True,
                message="Registration successful. Please login."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token"""
    try:
        auth_result = await jwt_auth.refresh_access_token(request.refresh_token)
        
        if not auth_result.success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=auth_result.error or "Token refresh failed"
            )
        
        return AuthResponse(
            success=True,
            access_token=auth_result.access_token,
            expires_in=auth_result.expires_in,
            message="Token refreshed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout(auth_context: AuthorizationContext = Depends(get_auth_context)):
    """User logout endpoint"""
    try:
        # Get token from authorization header (would need to extract from request)
        # For now, just delete the session
        if auth_context.session_id:
            from ..cache.session_manager import session_manager
            await session_manager.delete_session(auth_context.session_id)
        
        return {"success": True, "message": "Logout successful"}
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Get user profile"""
    try:
        profile = await _get_user_profile(auth_context.user_id)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        return UserProfileResponse(**profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile"
        )


@router.put("/profile")
@require_permission(Permission.USER_UPDATE)
async def update_profile(request: UpdateProfileRequest, 
                        auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Update user profile"""
    try:
        success = await _update_user_profile(auth_context.user_id, request)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update profile"
            )
        
        return {"success": True, "message": "Profile updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update profile failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.post("/change-password")
async def change_password(request: ChangePasswordRequest,
                         auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Change user password"""
    try:
        # Validate current password
        user_data = await jwt_auth._get_user_data(auth_context.user_id)
        if not user_data or not jwt_auth.verify_password(request.current_password, user_data["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password strength
        password_validation = jwt_auth.validate_password_strength(request.new_password)
        if not password_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "New password does not meet requirements",
                    "issues": password_validation["issues"]
                }
            )
        
        # Update password
        success = await _update_user_password(auth_context.user_id, request.new_password)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to change password"
            )
        
        return {"success": True, "message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change password failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.get("/me")
async def get_current_user(auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Get current authenticated user information"""
    return {
        "user_id": auth_context.user_id,
        "username": auth_context.username,
        "roles": auth_context.roles,
        "permissions": [p.value for p in auth_context.permissions],
        "session_id": auth_context.session_id
    }


# Helper functions
async def _check_username_exists(username: str) -> bool:
    """Check if username already exists"""
    try:
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            from ..database.models import User
            
            result = await session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none() is not None
            
    except Exception as e:
        logger.error(f"Failed to check username: {e}")
        return False


async def _check_email_exists(email: str) -> bool:
    """Check if email already exists"""
    try:
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            from ..database.models import User
            
            result = await session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none() is not None
            
    except Exception as e:
        logger.error(f"Failed to check email: {e}")
        return False


async def _create_user(request: RegisterRequest) -> Optional[str]:
    """Create new user in database"""
    try:
        async with db_manager.get_session() as session:
            from ..database.models import User
            import uuid
            
            user = User(
                id=uuid.uuid4(),
                username=request.username,
                email=request.email,
                password_hash=jwt_auth.hash_password(request.password),
                role="user",
                is_active=True,
                email_verified=False,
                created_at=datetime.utcnow()
            )
            
            session.add(user)
            await session.commit()
            
            return str(user.id)
            
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        return None


async def _get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user profile from database"""
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
                    "first_name": getattr(user, 'first_name', None),
                    "last_name": getattr(user, 'last_name', None),
                    "roles": [user.role] if user.role else [],
                    "is_active": user.is_active,
                    "email_verified": user.email_verified,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_login": getattr(user, 'last_login', None)
                }
            
            return None
            
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        return None


async def _update_user_profile(user_id: str, request: UpdateProfileRequest) -> bool:
    """Update user profile in database"""
    try:
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            from ..database.models import User
            
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                if request.email:
                    user.email = request.email
                if request.first_name is not None:
                    user.first_name = request.first_name
                if request.last_name is not None:
                    user.last_name = request.last_name
                
                user.updated_at = datetime.utcnow()
                await session.commit()
                return True
            
            return False
            
    except Exception as e:
        logger.error(f"Failed to update user profile: {e}")
        return False


async def _update_user_password(user_id: str, new_password: str) -> bool:
    """Update user password in database"""
    try:
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            from ..database.models import User
            
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                user.password_hash = jwt_auth.hash_password(new_password)
                user.updated_at = datetime.utcnow()
                await session.commit()
                return True
            
            return False
            
    except Exception as e:
        logger.error(f"Failed to update user password: {e}")
        return False
