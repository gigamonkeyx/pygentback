"""
Authentication module for PyGent Factory

Provides OAuth 2.0 and other authentication mechanisms for MCP servers
and external services.
"""

from .oauth import OAuthManager, OAuthProvider, OAuthToken
from .providers import (
    CloudflareOAuthProvider, 
    GitHubOAuthProvider, 
    GoogleOAuthProvider,
    MicrosoftOAuthProvider,
    SlackOAuthProvider,
    DiscordOAuthProvider,
    NotionOAuthProvider,
    LinearOAuthProvider
)
from .storage import TokenStorage, FileTokenStorage, DatabaseTokenStorage, MemoryTokenStorage

__all__ = [
    'OAuthManager',
    'OAuthProvider', 
    'OAuthToken',
    'CloudflareOAuthProvider',
    'GitHubOAuthProvider',
    'GoogleOAuthProvider',
    'MicrosoftOAuthProvider',
    'SlackOAuthProvider',
    'DiscordOAuthProvider',
    'NotionOAuthProvider',
    'LinearOAuthProvider',
    'TokenStorage',
    'FileTokenStorage',
    'DatabaseTokenStorage',
    'MemoryTokenStorage'
]
