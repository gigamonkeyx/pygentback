"""
Token storage backends for OAuth tokens

Provides different storage mechanisms for OAuth tokens.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict
import aiofiles
import asyncio
from datetime import datetime

from .oauth import OAuthToken

# Import database models conditionally
try:
    from ..database.models import OAuthToken as OAuthTokenModel, User
    from ..database.connection import get_database_manager
    from sqlalchemy import select, delete
    from sqlalchemy.ext.asyncio import AsyncSession
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


class TokenStorage(ABC):
    """Abstract base class for token storage"""
    
    @abstractmethod
    async def store_token(self, user_id: str, provider: str, token: OAuthToken) -> bool:
        """Store a token for user and provider"""
        pass
    
    @abstractmethod
    async def get_token(self, user_id: str, provider: str) -> Optional[OAuthToken]:
        """Get token for user and provider"""
        pass
    
    @abstractmethod
    async def delete_token(self, user_id: str, provider: str) -> bool:
        """Delete token for user and provider"""
        pass
    
    @abstractmethod
    async def list_tokens(self, user_id: str) -> Dict[str, OAuthToken]:
        """List all tokens for a user"""
        pass


class FileTokenStorage(TokenStorage):
    """File-based token storage"""
    
    def __init__(self, storage_dir: str = "data/tokens"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    def _get_token_file(self, user_id: str, provider: str) -> Path:
        """Get token file path for user and provider"""
        # Use safe filename
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in '-_')
        safe_provider = "".join(c for c in provider if c.isalnum() or c in '-_')
        return self.storage_dir / f"{safe_user_id}_{safe_provider}.json"
    
    def _get_user_dir(self, user_id: str) -> Path:
        """Get user directory path"""
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in '-_')
        return self.storage_dir / safe_user_id
    
    async def store_token(self, user_id: str, provider: str, token: OAuthToken) -> bool:
        """Store a token for user and provider"""
        async with self._lock:
            try:
                token_file = self._get_token_file(user_id, provider)
                token_file.parent.mkdir(parents=True, exist_ok=True)
                
                token_data = token.to_dict()
                
                async with aiofiles.open(token_file, 'w') as f:
                    await f.write(json.dumps(token_data, indent=2))
                
                return True
            except Exception:
                return False
    
    async def get_token(self, user_id: str, provider: str) -> Optional[OAuthToken]:
        """Get token for user and provider"""
        async with self._lock:
            try:
                token_file = self._get_token_file(user_id, provider)
                
                if not token_file.exists():
                    return None
                
                async with aiofiles.open(token_file, 'r') as f:
                    token_data = json.loads(await f.read())
                
                return OAuthToken.from_dict(token_data)
            except Exception:
                return None
    
    async def delete_token(self, user_id: str, provider: str) -> bool:
        """Delete token for user and provider"""
        async with self._lock:
            try:
                token_file = self._get_token_file(user_id, provider)
                
                if token_file.exists():
                    token_file.unlink()
                
                return True
            except Exception:
                return False
    
    async def list_tokens(self, user_id: str) -> Dict[str, OAuthToken]:
        """List all tokens for a user"""
        async with self._lock:
            tokens = {}
            try:
                safe_user_id = "".join(c for c in user_id if c.isalnum() or c in '-_')
                pattern = f"{safe_user_id}_*.json"
                
                for token_file in self.storage_dir.glob(pattern):
                    try:
                        async with aiofiles.open(token_file, 'r') as f:
                            token_data = json.loads(await f.read())
                        
                        token = OAuthToken.from_dict(token_data)
                        
                        # Extract provider name from filename
                        provider = token_file.stem.split('_', 1)[1]
                        tokens[provider] = token
                    except Exception:
                        continue
                
                return tokens
            except Exception:
                return {}


class DatabaseTokenStorage(TokenStorage):
    """Database-based token storage using SQLAlchemy"""
    
    def __init__(self, database_manager=None):
        self.database_manager = database_manager
        if not DATABASE_AVAILABLE:
            raise ImportError("Database dependencies not available")
    
    async def store_token(self, user_id: str, provider: str, token: OAuthToken) -> bool:
        """Store a token for user and provider"""
        if not self.database_manager:
            self.database_manager = get_database_manager()
        
        try:
            async with self.database_manager.async_session() as session:
                # Check if token already exists
                stmt = select(OAuthTokenModel).where(
                    OAuthTokenModel.user_id == user_id,
                    OAuthTokenModel.provider == provider
                )
                result = await session.execute(stmt)
                existing_token = result.scalar_one_or_none()
                
                if existing_token:
                    # Update existing token
                    existing_token.access_token = token.access_token
                    existing_token.refresh_token = token.refresh_token
                    existing_token.token_type = token.token_type
                    existing_token.expires_at = token.expires_at
                    existing_token.scopes = token.scopes or []
                    existing_token.provider_user_id = getattr(token, 'provider_user_id', None)
                    existing_token.provider_user_info = getattr(token, 'provider_user_info', {})
                else:
                    # Create new token
                    db_token = OAuthTokenModel(
                        user_id=user_id,
                        provider=provider,
                        access_token=token.access_token,
                        refresh_token=token.refresh_token,
                        token_type=token.token_type,
                        expires_at=token.expires_at,
                        scopes=token.scopes or [],
                        provider_user_id=getattr(token, 'provider_user_id', None),
                        provider_user_info=getattr(token, 'provider_user_info', {})
                    )
                    session.add(db_token)
                
                await session.commit()
                return True
        except Exception as e:
            print(f"Error storing token: {e}")
            return False
    
    async def get_token(self, user_id: str, provider: str) -> Optional[OAuthToken]:
        """Get token for user and provider"""
        if not self.database_manager:
            self.database_manager = get_database_manager()
        
        try:
            async with self.database_manager.async_session() as session:
                stmt = select(OAuthTokenModel).where(
                    OAuthTokenModel.user_id == user_id,
                    OAuthTokenModel.provider == provider
                )
                result = await session.execute(stmt)
                db_token = result.scalar_one_or_none()
                
                if db_token:
                    # Convert to OAuthToken
                    token = OAuthToken(
                        access_token=db_token.access_token,
                        refresh_token=db_token.refresh_token,
                        token_type=db_token.token_type,
                        expires_at=db_token.expires_at,
                        scopes=db_token.scopes or []
                    )
                    token.user_id = user_id
                    token.provider_user_id = db_token.provider_user_id
                    token.provider_user_info = db_token.provider_user_info
                    return token
                
                return None
        except Exception as e:
            print(f"Error getting token: {e}")
            return None
    
    async def delete_token(self, user_id: str, provider: str) -> bool:
        """Delete token for user and provider"""
        if not self.database_manager:
            self.database_manager = get_database_manager()
        
        try:
            async with self.database_manager.async_session() as session:
                stmt = delete(OAuthTokenModel).where(
                    OAuthTokenModel.user_id == user_id,
                    OAuthTokenModel.provider == provider
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"Error deleting token: {e}")
            return False
    
    async def list_tokens(self, user_id: str) -> Dict[str, OAuthToken]:
        """List all tokens for a user"""
        if not self.database_manager:
            self.database_manager = get_database_manager()
        
        try:
            async with self.database_manager.async_session() as session:
                stmt = select(OAuthTokenModel).where(OAuthTokenModel.user_id == user_id)
                result = await session.execute(stmt)
                db_tokens = result.scalars().all()
                
                tokens = {}
                for db_token in db_tokens:
                    token = OAuthToken(
                        access_token=db_token.access_token,
                        refresh_token=db_token.refresh_token,
                        token_type=db_token.token_type,
                        expires_at=db_token.expires_at,
                        scopes=db_token.scopes or []
                    )
                    token.user_id = user_id
                    token.provider_user_id = db_token.provider_user_id
                    token.provider_user_info = db_token.provider_user_info
                    tokens[db_token.provider] = token
                
                return tokens
        except Exception as e:
            print(f"Error listing tokens: {e}")
            return {}


class MemoryTokenStorage(TokenStorage):
    """In-memory token storage (for testing/development)"""
    
    def __init__(self):
        self._tokens: Dict[str, Dict[str, OAuthToken]] = {}
        self._lock = asyncio.Lock()
    
    def _get_key(self, user_id: str, provider: str) -> str:
        """Get storage key for user and provider"""
        return f"{user_id}:{provider}"
    
    async def store_token(self, user_id: str, provider: str, token: OAuthToken) -> bool:
        """Store a token for user and provider"""
        async with self._lock:
            if user_id not in self._tokens:
                self._tokens[user_id] = {}
            self._tokens[user_id][provider] = token
            return True
    
    async def get_token(self, user_id: str, provider: str) -> Optional[OAuthToken]:
        """Get token for user and provider"""
        async with self._lock:
            return self._tokens.get(user_id, {}).get(provider)
    
    async def delete_token(self, user_id: str, provider: str) -> bool:
        """Delete token for user and provider"""
        async with self._lock:
            if user_id in self._tokens and provider in self._tokens[user_id]:
                del self._tokens[user_id][provider]
                if not self._tokens[user_id]:  # Remove empty user dict
                    del self._tokens[user_id]
                return True
            return False
    
    async def list_tokens(self, user_id: str) -> Dict[str, OAuthToken]:
        """List all tokens for a user"""
        async with self._lock:
            return self._tokens.get(user_id, {}).copy()
