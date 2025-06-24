"""
Core OAuth 2.0 implementation for PyGent Factory

Provides base classes and interfaces for OAuth authentication.
"""

import base64
import hashlib
import secrets
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import httpx
from fastapi import HTTPException


@dataclass
class OAuthToken:
    """OAuth 2.0 token representation"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    created_at: Optional[datetime] = None
    provider: Optional[str] = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    @property
    def expires_at(self) -> Optional[datetime]:
        """Calculate when the token expires"""
        if self.expires_in and self.created_at:
            return self.created_at + timedelta(seconds=self.expires_in)
        return None
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired"""
        if self.expires_at:
            return datetime.utcnow() >= self.expires_at
        return False
    
    @property
    def expires_soon(self, threshold_minutes: int = 5) -> bool:
        """Check if token expires within threshold minutes"""
        if self.expires_at:
            threshold = datetime.utcnow() + timedelta(minutes=threshold_minutes)
            return threshold >= self.expires_at
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OAuthToken':
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class OAuthProvider(ABC):
    """Abstract base class for OAuth providers"""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the OAuth provider"""
        pass
    
    @property
    @abstractmethod
    def authorization_url(self) -> str:
        """OAuth authorization endpoint URL"""
        pass
    
    @property
    @abstractmethod
    def token_url(self) -> str:
        """OAuth token endpoint URL"""
        pass
    
    @property
    @abstractmethod
    def default_scopes(self) -> List[str]:
        """Default OAuth scopes for this provider"""
        pass
    
    def generate_authorization_url(self, state: str, scopes: Optional[List[str]] = None) -> str:
        """Generate OAuth authorization URL"""
        if scopes is None:
            scopes = self.default_scopes
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': ' '.join(scopes),
            'state': state
        }
        
        # Add PKCE for security
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        params.update({
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        })
        
        # Store code_verifier for token exchange (would need session storage in practice)
        self._code_verifier = code_verifier
        
        return f"{self.authorization_url}?{urllib.parse.urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str) -> OAuthToken:
        """Exchange authorization code for access token"""
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code',
            'code': code
        }
        
        # Add PKCE verifier if available
        if hasattr(self, '_code_verifier'):
            data['code_verifier'] = self._code_verifier
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data=data,
                headers={'Accept': 'application/json'}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Token exchange failed: {response.text}"
                )
            
            token_data = response.json()
            return OAuthToken(
                access_token=token_data['access_token'],
                token_type=token_data.get('token_type', 'Bearer'),
                expires_in=token_data.get('expires_in'),
                refresh_token=token_data.get('refresh_token'),
                scope=token_data.get('scope'),
                provider=self.provider_name
            )
    
    async def refresh_token(self, refresh_token: str) -> OAuthToken:
        """Refresh an expired access token"""
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data=data,
                headers={'Accept': 'application/json'}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Token refresh failed: {response.text}"
                )
            
            token_data = response.json()
            return OAuthToken(
                access_token=token_data['access_token'],
                token_type=token_data.get('token_type', 'Bearer'),
                expires_in=token_data.get('expires_in'),
                refresh_token=token_data.get('refresh_token', refresh_token),
                scope=token_data.get('scope'),
                provider=self.provider_name
            )
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke an access token"""
        # Default implementation - override in subclasses if provider supports revocation
        return True


class OAuthManager:
    """Central OAuth manager for handling multiple providers"""
    
    def __init__(self):
        self.providers: Dict[str, OAuthProvider] = {}
        self.token_storage = None
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def register_provider(self, provider: OAuthProvider):
        """Register an OAuth provider"""
        self.providers[provider.provider_name] = provider
    
    def set_token_storage(self, storage):
        """Set token storage backend"""
        self.token_storage = storage
    
    def generate_state(self) -> str:
        """Generate secure state parameter"""
        return secrets.token_urlsafe(32)
    
    async def get_authorization_url(self, provider_name: str, user_id: str, scopes: Optional[List[str]] = None) -> str:
        """Get authorization URL for a provider"""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider = self.providers[provider_name]
        state = self.generate_state()
        
        # Store session info
        self.active_sessions[state] = {
            'provider': provider_name,
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'scopes': scopes
        }
        
        return provider.generate_authorization_url(state, scopes)
    
    async def handle_callback(self, code: str, state: str) -> OAuthToken:
        """Handle OAuth callback"""
        if state not in self.active_sessions:
            raise HTTPException(status_code=400, detail="Invalid or expired state")
        
        session = self.active_sessions[state]
        provider_name = session['provider']
        user_id = session['user_id']
        
        if provider_name not in self.providers:
            raise HTTPException(status_code=400, detail="Unknown provider")
        
        provider = self.providers[provider_name]
        token = await provider.exchange_code_for_token(code)
        token.user_id = user_id
        
        # Store token
        if self.token_storage:
            await self.token_storage.store_token(user_id, provider_name, token)
        
        # Clean up session
        del self.active_sessions[state]
        
        return token
    
    async def get_token(self, user_id: str, provider_name: str) -> Optional[OAuthToken]:
        """Get stored token for user and provider"""
        if not self.token_storage:
            return None
        
        token = await self.token_storage.get_token(user_id, provider_name)
        
        # Auto-refresh if needed
        if token and token.expires_soon and token.refresh_token:
            provider = self.providers.get(provider_name)
            if provider:
                try:
                    new_token = await provider.refresh_token(token.refresh_token)
                    new_token.user_id = user_id
                    await self.token_storage.store_token(user_id, provider_name, new_token)
                    return new_token
                except Exception:
                    # Refresh failed, return original token
                    pass
        
        return token
    
    async def revoke_token(self, user_id: str, provider_name: str) -> bool:
        """Revoke token for user and provider"""
        if not self.token_storage:
            return False
        
        token = await self.token_storage.get_token(user_id, provider_name)
        if not token:
            return False
        
        provider = self.providers.get(provider_name)
        if provider:
            await provider.revoke_token(token.access_token)
        
        await self.token_storage.delete_token(user_id, provider_name)
        return True
    
    def get_provider(self, provider_name: str) -> Optional[OAuthProvider]:
        """Get provider by name"""
        return self.providers.get(provider_name)
    
    def list_providers(self) -> List[str]:
        """List available provider names"""
        return list(self.providers.keys())
