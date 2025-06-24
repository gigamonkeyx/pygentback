#!/usr/bin/env python3
"""
OAuth Setup and Management Tool for PyGent Factory

Provides commands to setup, test, and manage OAuth authentication for MCP servers.
"""

import asyncio
import os
import sys
import click

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.auth import (
    OAuthManager, 
    CloudflareOAuthProvider, 
    GitHubOAuthProvider, 
    FileTokenStorage
)


@click.group()
def cli():
    """OAuth management commands for PyGent Factory"""
    pass


@cli.command()
@click.option('--provider', required=True, help='OAuth provider (cloudflare, github)')
@click.option('--user-id', default='system', help='User ID for token storage')
def authorize(provider: str, user_id: str):
    """Start OAuth authorization flow for a provider"""
    asyncio.run(_authorize(provider, user_id))


async def _authorize(provider: str, user_id: str):
    """Internal authorize function"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    load_dotenv('oauth.env.example')
    
    # Setup OAuth manager
    oauth_manager = OAuthManager()
    oauth_manager.set_token_storage(FileTokenStorage())
    
    # Setup providers
    await _setup_providers(oauth_manager)
    
    if provider not in oauth_manager.list_providers():
        click.echo(f"❌ Provider '{provider}' not configured or not supported")
        click.echo(f"Available providers: {', '.join(oauth_manager.list_providers())}")
        return
    
    try:
        # Get authorization URL
        auth_url = await oauth_manager.get_authorization_url(provider, user_id)
        
        click.echo(f"🚀 Starting OAuth flow for {provider}")
        click.echo("📋 Please visit this URL to authorize the application:")
        click.echo(f"   {auth_url}")
        click.echo()
        click.echo("📝 After authorization, you will be redirected to a callback URL.")
        click.echo("   Copy the 'code' parameter from the callback URL and paste it below.")
        click.echo()
        
        # Get authorization code from user
        code = click.prompt("Enter the authorization code")
        state = auth_url.split('state=')[1].split('&')[0] if 'state=' in auth_url else ""
        
        # Exchange code for token
        token = await oauth_manager.handle_callback(code, state)
        
        click.echo(f"✅ Successfully authorized {provider}")
        click.echo(f"   Token expires: {token.expires_at}")
        click.echo(f"   Scopes: {token.scope}")
        
    except Exception as e:
        click.echo(f"❌ Authorization failed: {str(e)}")


@cli.command()
@click.option('--user-id', default='system', help='User ID to check tokens for')
def status(user_id: str):
    """Check OAuth token status for all providers"""
    asyncio.run(_status(user_id))


async def _status(user_id: str):
    """Internal status function"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    # Setup OAuth manager
    oauth_manager = OAuthManager()
    oauth_manager.set_token_storage(FileTokenStorage())
    
    # Setup providers
    await _setup_providers(oauth_manager)
    
    click.echo(f"🔍 OAuth Token Status for user: {user_id}")
    click.echo("=" * 50)
    
    any_tokens = False
    for provider_name in oauth_manager.list_providers():
        token = await oauth_manager.get_token(user_id, provider_name)
        
        if token:
            any_tokens = True
            status_icon = "✅" if not token.is_expired else "❌"
            expiry_text = token.expires_at.strftime("%Y-%m-%d %H:%M:%S") if token.expires_at else "No expiry"
            
            click.echo(f"{status_icon} {provider_name}:")
            click.echo(f"   Expires: {expiry_text}")
            click.echo(f"   Expired: {'Yes' if token.is_expired else 'No'}")
            click.echo(f"   Expires soon: {'Yes' if token.expires_soon else 'No'}")
            click.echo(f"   Has refresh token: {'Yes' if token.refresh_token else 'No'}")
            click.echo()
        else:
            click.echo(f"⚪ {provider_name}: No token stored")
    
    if not any_tokens:
        click.echo("ℹ️  No OAuth tokens found. Use 'python oauth_manager.py authorize' to get started.")


@cli.command()
@click.option('--provider', required=True, help='OAuth provider to revoke')
@click.option('--user-id', default='system', help='User ID to revoke token for')
def revoke(provider: str, user_id: str):
    """Revoke OAuth token for a provider"""
    asyncio.run(_revoke(provider, user_id))


async def _revoke(provider: str, user_id: str):
    """Internal revoke function"""
    # Setup OAuth manager
    oauth_manager = OAuthManager()
    oauth_manager.set_token_storage(FileTokenStorage())
    
    # Setup providers
    await _setup_providers(oauth_manager)
    
    success = await oauth_manager.revoke_token(user_id, provider)
    
    if success:
        click.echo(f"✅ Successfully revoked token for {provider}")
    else:
        click.echo(f"❌ Failed to revoke token for {provider} (may not exist)")


@cli.command()
@click.option('--provider', required=True, help='OAuth provider to test')
@click.option('--user-id', default='system', help='User ID to test token for')
def test(provider: str, user_id: str):
    """Test OAuth token by making an API call"""
    asyncio.run(_test(provider, user_id))


async def _test(provider: str, user_id: str):
    """Internal test function"""
    import httpx
    
    # Setup OAuth manager
    oauth_manager = OAuthManager()
    oauth_manager.set_token_storage(FileTokenStorage())
    
    # Setup providers
    await _setup_providers(oauth_manager)
    
    token = await oauth_manager.get_token(user_id, provider)
    
    if not token:
        click.echo(f"❌ No token found for {provider}")
        return
    
    if token.is_expired:
        click.echo(f"❌ Token for {provider} is expired")
        return
    
    # Test the token with a simple API call
    click.echo(f"🧪 Testing {provider} token...")
    
    try:
        if provider == "cloudflare":
            # Test Cloudflare API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.cloudflare.com/client/v4/user/tokens/verify",
                    headers={"Authorization": f"Bearer {token.access_token}"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    click.echo(f"✅ {provider} token is valid")
                    click.echo(f"   Token ID: {data.get('result', {}).get('id', 'Unknown')}")
                else:
                    click.echo(f"❌ {provider} token test failed: HTTP {response.status_code}")
                    
        elif provider == "github":
            # Test GitHub API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.github.com/user",
                    headers={"Authorization": f"Bearer {token.access_token}"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    click.echo(f"✅ {provider} token is valid")
                    click.echo(f"   User: {data.get('login', 'Unknown')}")
                else:
                    click.echo(f"❌ {provider} token test failed: HTTP {response.status_code}")
        else:
            click.echo(f"⚠️  No test implementation for {provider}")
            
    except Exception as e:
        click.echo(f"❌ Token test failed: {str(e)}")


@cli.command()
def setup():
    """Setup OAuth configuration"""
    click.echo("🚀 PyGent Factory OAuth Setup")
    click.echo("=" * 40)
    
    # Check if OAuth config exists
    if os.path.exists('.env.local'):
        click.echo("✅ Found .env.local file")
    else:
        click.echo("⚠️  No .env.local file found")
        if click.confirm("Create .env.local from oauth.env.example?"):
            import shutil
            shutil.copy('oauth.env.example', '.env.local')
            click.echo("✅ Created .env.local file")
            click.echo("📝 Please edit .env.local and add your OAuth credentials")
    
    click.echo()
    click.echo("📋 Next steps:")
    click.echo("1. Edit .env.local and add your OAuth credentials")
    click.echo("2. For Cloudflare: Visit https://dash.cloudflare.com/profile/api-tokens")
    click.echo("3. For GitHub: Visit https://github.com/settings/applications/new")
    click.echo("4. Run: python oauth_manager.py authorize --provider cloudflare")


async def _setup_providers(oauth_manager: OAuthManager):
    """Setup OAuth providers from environment"""
    base_url = os.getenv('BASE_URL', 'http://localhost:8080')
    
    # Cloudflare
    client_id = os.getenv('CLOUDFLARE_CLIENT_ID')
    client_secret = os.getenv('CLOUDFLARE_CLIENT_SECRET')
    if client_id and client_secret:
        provider = CloudflareOAuthProvider(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=f"{base_url}/api/v1/auth/callback/cloudflare"
        )
        oauth_manager.register_provider(provider)
    
    # GitHub
    client_id = os.getenv('GITHUB_CLIENT_ID')
    client_secret = os.getenv('GITHUB_CLIENT_SECRET')
    if client_id and client_secret:
        provider = GitHubOAuthProvider(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=f"{base_url}/api/v1/auth/callback/github"
        )
        oauth_manager.register_provider(provider)


if __name__ == "__main__":
    cli()
