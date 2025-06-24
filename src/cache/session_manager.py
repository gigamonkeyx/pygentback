#!/usr/bin/env python3
"""
Redis Session Management System

Production-ready session management with Redis backend for PyGent Factory.
Provides secure session storage, automatic expiration, and performance optimization.
"""

import asyncio
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from .redis_manager import redis_manager

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Session configuration"""
    session_timeout_minutes: int = 60  # 1 hour default
    max_sessions_per_user: int = 5
    session_key_prefix: str = "session:"
    user_sessions_prefix: str = "user_sessions:"
    cleanup_interval_minutes: int = 15
    secure_cookies: bool = True
    session_token_length: int = 32


@dataclass
class SessionData:
    """Session data structure"""
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


class SessionManager:
    """Redis-based session manager"""
    
    def __init__(self, config: Optional[SessionConfig] = None):
        self.config = config or SessionConfig()
        self.is_initialized = False
        self._cleanup_task = None
        
        # Performance metrics
        self.session_creates = 0
        self.session_reads = 0
        self.session_updates = 0
        self.session_deletes = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def initialize(self) -> bool:
        """Initialize session manager"""
        try:
            logger.info("Initializing Redis session manager...")
            
            if not redis_manager.is_initialized:
                await redis_manager.initialize()
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
            
            self.is_initialized = True
            logger.info("Redis session manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize session manager: {e}")
            return False
    
    async def create_session(self, user_id: str, ip_address: Optional[str] = None, 
                           user_agent: Optional[str] = None, 
                           initial_data: Optional[Dict[str, Any]] = None) -> Optional[SessionData]:
        """Create a new session"""
        try:
            # Generate secure session ID
            session_id = secrets.token_urlsafe(self.config.session_token_length)
            
            # Check session limits for user
            await self._enforce_session_limits(user_id)
            
            # Create session data
            now = datetime.utcnow()
            expires_at = now + timedelta(minutes=self.config.session_timeout_minutes)
            
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_accessed=now,
                expires_at=expires_at,
                data=initial_data or {},
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Store session in Redis
            session_key = f"{self.config.session_key_prefix}{session_id}"
            session_dict = self._session_to_dict(session_data)
            
            ttl_seconds = int(self.config.session_timeout_minutes * 60)
            success = await redis_manager.set(session_key, session_dict, ttl=ttl_seconds)
            
            if success:
                # Add to user sessions set
                user_sessions_key = f"{self.config.user_sessions_prefix}{user_id}"
                await redis_manager.add_to_set(user_sessions_key, session_id)
                await redis_manager.expire(user_sessions_key, ttl_seconds)
                
                self.session_creates += 1
                logger.debug(f"Session created for user {user_id}: {session_id}")
                return session_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            return None
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID"""
        try:
            session_key = f"{self.config.session_key_prefix}{session_id}"
            session_dict = await redis_manager.get(session_key)
            
            if session_dict is None:
                self.cache_misses += 1
                return None
            
            self.cache_hits += 1
            self.session_reads += 1
            
            # Convert to SessionData
            session_data = self._dict_to_session(session_dict)
            
            # Check if session is expired
            if session_data.expires_at < datetime.utcnow():
                await self.delete_session(session_id)
                return None
            
            # Update last accessed time
            session_data.last_accessed = datetime.utcnow()
            await self._update_session_access_time(session_data)
            
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            self.cache_misses += 1
            return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data"""
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                return False
            
            # Update session data
            session_data.data.update(data)
            session_data.last_accessed = datetime.utcnow()
            
            # Store updated session
            session_key = f"{self.config.session_key_prefix}{session_id}"
            session_dict = self._session_to_dict(session_data)
            
            ttl_seconds = int((session_data.expires_at - datetime.utcnow()).total_seconds())
            if ttl_seconds <= 0:
                await self.delete_session(session_id)
                return False
            
            success = await redis_manager.set(session_key, session_dict, ttl=ttl_seconds)
            
            if success:
                self.session_updates += 1
                logger.debug(f"Session updated: {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        try:
            # Get session to find user_id
            session_key = f"{self.config.session_key_prefix}{session_id}"
            session_dict = await redis_manager.get(session_key, deserialize=False)
            
            if session_dict:
                # Parse to get user_id
                try:
                    session_data = json.loads(session_dict.decode('utf-8'))
                    user_id = session_data.get('user_id')
                    
                    if user_id:
                        # Remove from user sessions set
                        user_sessions_key = f"{self.config.user_sessions_prefix}{user_id}"
                        await redis_manager.redis_client.srem(user_sessions_key, session_id)
                        
                except Exception:
                    pass
            
            # Delete session
            success = await redis_manager.delete(session_key)
            
            if success:
                self.session_deletes += 1
                logger.debug(f"Session deleted: {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all active sessions for a user"""
        try:
            user_sessions_key = f"{self.config.user_sessions_prefix}{user_id}"
            session_ids = await redis_manager.get_set_members(user_sessions_key)
            
            sessions = []
            for session_id in session_ids:
                session_data = await self.get_session(str(session_id))
                if session_data and session_data.is_active:
                    sessions.append(session_data)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id}: {e}")
            return []
    
    async def delete_user_sessions(self, user_id: str, exclude_session_id: Optional[str] = None) -> int:
        """Delete all sessions for a user"""
        try:
            sessions = await self.get_user_sessions(user_id)
            deleted_count = 0
            
            for session in sessions:
                if exclude_session_id and session.session_id == exclude_session_id:
                    continue
                
                if await self.delete_session(session.session_id):
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete user sessions for {user_id}: {e}")
            return 0
    
    async def extend_session(self, session_id: str, minutes: Optional[int] = None) -> bool:
        """Extend session expiration"""
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                return False
            
            # Extend expiration
            extension_minutes = minutes or self.config.session_timeout_minutes
            session_data.expires_at = datetime.utcnow() + timedelta(minutes=extension_minutes)
            
            # Update in Redis
            session_key = f"{self.config.session_key_prefix}{session_id}"
            session_dict = self._session_to_dict(session_data)
            
            ttl_seconds = int(extension_minutes * 60)
            success = await redis_manager.set(session_key, session_dict, ttl=ttl_seconds)
            
            if success:
                logger.debug(f"Session extended: {session_id} for {extension_minutes} minutes")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to extend session {session_id}: {e}")
            return False
    
    async def _enforce_session_limits(self, user_id: str):
        """Enforce maximum sessions per user"""
        try:
            sessions = await self.get_user_sessions(user_id)
            
            if len(sessions) >= self.config.max_sessions_per_user:
                # Sort by last accessed and delete oldest
                sessions.sort(key=lambda s: s.last_accessed)
                sessions_to_delete = sessions[:-self.config.max_sessions_per_user + 1]
                
                for session in sessions_to_delete:
                    await self.delete_session(session.session_id)
                    
        except Exception as e:
            logger.error(f"Failed to enforce session limits for user {user_id}: {e}")
    
    async def _update_session_access_time(self, session_data: SessionData):
        """Update session last accessed time"""
        try:
            session_key = f"{self.config.session_key_prefix}{session_data.session_id}"
            
            # Quick update of last_accessed field
            await redis_manager.redis_client.hset(
                session_key, 
                "last_accessed", 
                session_data.last_accessed.isoformat()
            )
            
        except Exception as e:
            logger.debug(f"Failed to update session access time: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions"""
        while self.is_initialized:
            try:
                # This is handled by Redis TTL, but we can do additional cleanup
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
                
                # Could implement additional cleanup logic here
                logger.debug("Session cleanup cycle completed")
                
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def _session_to_dict(self, session_data: SessionData) -> Dict[str, Any]:
        """Convert SessionData to dictionary"""
        return {
            "session_id": session_data.session_id,
            "user_id": session_data.user_id,
            "created_at": session_data.created_at.isoformat(),
            "last_accessed": session_data.last_accessed.isoformat(),
            "expires_at": session_data.expires_at.isoformat(),
            "data": session_data.data,
            "ip_address": session_data.ip_address,
            "user_agent": session_data.user_agent,
            "is_active": session_data.is_active
        }
    
    def _dict_to_session(self, session_dict: Dict[str, Any]) -> SessionData:
        """Convert dictionary to SessionData"""
        return SessionData(
            session_id=session_dict["session_id"],
            user_id=session_dict["user_id"],
            created_at=datetime.fromisoformat(session_dict["created_at"]),
            last_accessed=datetime.fromisoformat(session_dict["last_accessed"]),
            expires_at=datetime.fromisoformat(session_dict["expires_at"]),
            data=session_dict.get("data", {}),
            ip_address=session_dict.get("ip_address"),
            user_agent=session_dict.get("user_agent"),
            is_active=session_dict.get("is_active", True)
        )
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session management statistics"""
        return {
            "session_creates": self.session_creates,
            "session_reads": self.session_reads,
            "session_updates": self.session_updates,
            "session_deletes": self.session_deletes,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (self.cache_hits / max(self.cache_hits + self.cache_misses, 1)) * 100,
            "config": {
                "session_timeout_minutes": self.config.session_timeout_minutes,
                "max_sessions_per_user": self.config.max_sessions_per_user,
                "cleanup_interval_minutes": self.config.cleanup_interval_minutes
            }
        }
    
    async def cleanup(self):
        """Cleanup session manager"""
        try:
            self.is_initialized = False
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Session manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during session manager cleanup: {e}")


# Global session manager instance
session_manager = SessionManager()
