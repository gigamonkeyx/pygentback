"""
OAuth provider implementations for various services

Implements specific OAuth flows for different providers.
"""

from typing import List
from .oauth import OAuthProvider


class CloudflareOAuthProvider(OAuthProvider):
    """Cloudflare OAuth provider implementation"""
    
    @property
    def provider_name(self) -> str:
        return "cloudflare"
    
    @property
    def authorization_url(self) -> str:
        return "https://dash.cloudflare.com/oauth2/auth"
    
    @property
    def token_url(self) -> str:
        return "https://dash.cloudflare.com/oauth2/token"
    
    @property
    def default_scopes(self) -> List[str]:
        # Cloudflare MCP server scopes
        return [
            "account:read",
            "zone:read", 
            "dns_records:read",
            "analytics:read",
            "workers:read",
            "workers_kv:read",
            "workers_r2:read",
            "browser_rendering:read"
        ]


class GitHubOAuthProvider(OAuthProvider):
    """GitHub OAuth provider implementation"""
    
    @property
    def provider_name(self) -> str:
        return "github"
    
    @property
    def authorization_url(self) -> str:
        return "https://github.com/login/oauth/authorize"
    
    @property
    def token_url(self) -> str:
        return "https://github.com/login/oauth/access_token"
    
    @property
    def default_scopes(self) -> List[str]:
        return ["repo", "user"]


class GoogleOAuthProvider(OAuthProvider):
    """Google OAuth provider implementation"""
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    @property
    def authorization_url(self) -> str:
        return "https://accounts.google.com/o/oauth2/v2/auth"
    
    @property
    def token_url(self) -> str:
        return "https://oauth2.googleapis.com/token"
    
    @property
    def default_scopes(self) -> List[str]:
        return ["openid", "profile", "email"]


class MicrosoftOAuthProvider(OAuthProvider):
    """Microsoft/Azure OAuth provider implementation"""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, tenant_id: str = "common"):
        super().__init__(client_id, client_secret, redirect_uri)
        self.tenant_id = tenant_id
    
    @property
    def provider_name(self) -> str:
        return "microsoft"
    
    @property
    def authorization_url(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/authorize"
    
    @property
    def token_url(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
    
    @property
    def default_scopes(self) -> List[str]:
        return ["openid", "profile", "email", "User.Read"]


class SlackOAuthProvider(OAuthProvider):
    """Slack OAuth provider implementation"""
    
    @property
    def provider_name(self) -> str:
        return "slack"
    
    @property
    def authorization_url(self) -> str:
        return "https://slack.com/oauth/v2/authorize"
    
    @property
    def token_url(self) -> str:
        return "https://slack.com/api/oauth.v2.access"
    
    @property
    def default_scopes(self) -> List[str]:
        return ["channels:read", "chat:write", "users:read"]


class DiscordOAuthProvider(OAuthProvider):
    """Discord OAuth provider implementation"""
    
    @property
    def provider_name(self) -> str:
        return "discord"
    
    @property
    def authorization_url(self) -> str:
        return "https://discord.com/api/oauth2/authorize"
    
    @property
    def token_url(self) -> str:
        return "https://discord.com/api/oauth2/token"
    
    @property
    def default_scopes(self) -> List[str]:
        return ["identify", "email"]


class NotionOAuthProvider(OAuthProvider):
    """Notion OAuth provider implementation"""
    
    @property
    def provider_name(self) -> str:
        return "notion"
    
    @property
    def authorization_url(self) -> str:
        return "https://api.notion.com/v1/oauth/authorize"
    
    @property
    def token_url(self) -> str:
        return "https://api.notion.com/v1/oauth/token"
    
    @property
    def default_scopes(self) -> List[str]:
        return []  # Notion doesn't use traditional scopes


class LinearOAuthProvider(OAuthProvider):
    """Linear OAuth provider implementation"""
    
    @property
    def provider_name(self) -> str:
        return "linear"
    
    @property
    def authorization_url(self) -> str:
        return "https://linear.app/oauth/authorize"
    
    @property
    def token_url(self) -> str:
        return "https://api.linear.app/oauth/token"
    
    @property
    def default_scopes(self) -> List[str]:
        return ["read", "write"]
