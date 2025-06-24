"""
OAuth API endpoints for PyGent Factory

Provides REST API endpoints for OAuth authentication flows with database-backed user management.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.auth import OAuthManager, CloudflareOAuthProvider, GitHubOAuthProvider, DatabaseTokenStorage
from src.services.user_service import get_user_service

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

# OAuth Manager instance with database storage
oauth_manager = OAuthManager()

# Set up database token storage
try:
    oauth_manager.set_token_storage(DatabaseTokenStorage())
except ImportError:
    # Fallback to file storage if database not available
    from src.auth import FileTokenStorage
    oauth_manager.set_token_storage(FileTokenStorage())


def setup_oauth_providers():
    """Setup OAuth providers from configuration"""
    import os
    
    # Cloudflare OAuth
    cloudflare_client_id = os.getenv('CLOUDFLARE_CLIENT_ID')
    cloudflare_client_secret = os.getenv('CLOUDFLARE_CLIENT_SECRET')
    base_url = os.getenv('BASE_URL', 'http://localhost:8080')
    
    if cloudflare_client_id and cloudflare_client_secret:
        cloudflare_provider = CloudflareOAuthProvider(
            client_id=cloudflare_client_id,
            client_secret=cloudflare_client_secret,
            redirect_uri=f"{base_url}/api/v1/auth/callback/cloudflare"
        )
        oauth_manager.register_provider(cloudflare_provider)
    
    # GitHub OAuth
    github_client_id = os.getenv('GITHUB_CLIENT_ID')
    github_client_secret = os.getenv('GITHUB_CLIENT_SECRET')
    
    if github_client_id and github_client_secret:
        github_provider = GitHubOAuthProvider(
            client_id=github_client_id,
            client_secret=github_client_secret,
            redirect_uri=f"{base_url}/api/v1/auth/callback/github"
        )
        oauth_manager.register_provider(github_provider)


# Initialize providers
setup_oauth_providers()


class AuthorizeRequest(BaseModel):
    provider: str
    scopes: Optional[List[str]] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: Optional[int] = None
    scope: Optional[str] = None
    provider: str


class ProviderInfo(BaseModel):
    name: str
    authorization_url: str
    default_scopes: List[str]


@router.get("/providers")
async def list_providers() -> List[ProviderInfo]:
    """List available OAuth providers"""
    providers = []
    for provider_name in oauth_manager.list_providers():
        provider = oauth_manager.get_provider(provider_name)
        if provider:
            providers.append(ProviderInfo(
                name=provider.provider_name,
                authorization_url=provider.authorization_url,
                default_scopes=provider.default_scopes
            ))
    return providers


@router.post("/authorize/{provider}")
async def get_authorization_url(
    provider: str,
    request: Request,
    scopes: Optional[List[str]] = None
) -> dict:
    """Get OAuth authorization URL for a provider"""
    
    # For now, use session ID as user ID (in production, use authenticated user)
    user_id = request.session.get('user_id', 'anonymous')
    
    try:
        auth_url = await oauth_manager.get_authorization_url(
            provider_name=provider,
            user_id=user_id,
            scopes=scopes
        )
        return {"authorization_url": auth_url}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/callback/{provider}")
async def oauth_callback(
    provider: str,
    code: str,
    state: str,
    request: Request
) -> RedirectResponse:
    """Handle OAuth callback and create/link user account"""
    
    try:
        token = await oauth_manager.handle_callback(code, state)
        
        # Import here to avoid circular imports
        from src.services.user_service import get_user_service
        user_service = get_user_service()
        
        # Try to get user info from token (provider-specific)
        user_info = {}
        oauth_user_id = None
        username = None
        email = None
        
        if provider == "cloudflare":
            # For Cloudflare, we can get user info from their API
            # This would require implementing the user info endpoint call
            username = f"cf_user_{token.access_token[:8]}"
            email = f"{username}@cloudflare.oauth"
            oauth_user_id = token.access_token[:16]
        elif provider == "github":
            # For GitHub, similar approach
            username = f"gh_user_{token.access_token[:8]}"
            email = f"{username}@github.oauth"
            oauth_user_id = token.access_token[:16]
        else:
            # Generic OAuth user
            username = f"{provider}_user_{token.access_token[:8]}"
            email = f"{username}@{provider}.oauth"
            oauth_user_id = token.access_token[:16]
        
        # Try to find existing user by OAuth provider
        user = await user_service.get_user_by_oauth(provider, oauth_user_id)
        
        if not user:
            # Create new user account
            user = await user_service.create_user(
                username=username,
                email=email,
                oauth_provider=provider,
                oauth_user_id=oauth_user_id,
                oauth_user_info=user_info
            )
            
            if not user:
                # User creation failed, try with a unique username
                import time
                username = f"{provider}_user_{int(time.time())}"
                email = f"{username}@{provider}.oauth"
                user = await user_service.create_user(
                    username=username,
                    email=email,
                    oauth_provider=provider,
                    oauth_user_id=oauth_user_id,
                    oauth_user_info=user_info
                )
        
        if user:
            # Store user info in session
            request.session['user_id'] = user.id
            request.session['username'] = user.username
            request.session[f'{provider}_token'] = token.access_token
            
            # Update last login
            await user_service.update_last_login(user.id)
            
            # Redirect to success page
            return RedirectResponse(url=f"/dashboard?auth={provider}&status=success&user={user.username}")
        else:
            # User creation failed
            return RedirectResponse(url=f"/dashboard?auth={provider}&status=error&message=User+creation+failed")
        
    except HTTPException:
        raise
    except Exception as e:
        # Redirect to error page
        return RedirectResponse(url=f"/dashboard?auth={provider}&status=error&message={str(e)}")


@router.get("/token/{provider}")
async def get_token(provider: str, request: Request) -> TokenResponse:
    """Get stored token for a provider"""
    
    # For now, use session ID as user ID
    user_id = request.session.get('user_id', 'anonymous')
    
    token = await oauth_manager.get_token(user_id, provider)
    if not token:
        raise HTTPException(status_code=404, detail="No token found for provider")
    
    if token.is_expired:
        raise HTTPException(status_code=401, detail="Token is expired")
    
    return TokenResponse(
        access_token=token.access_token,
        token_type=token.token_type,
        expires_in=token.expires_in,
        scope=token.scope,
        provider=token.provider
    )


@router.delete("/token/{provider}")
async def revoke_token(provider: str, request: Request) -> dict:
    """Revoke token for a provider"""
    
    # For now, use session ID as user ID
    user_id = request.session.get('user_id', 'anonymous')
    
    success = await oauth_manager.revoke_token(user_id, provider)
    return {"revoked": success}


@router.get("/status")
async def auth_status(request: Request) -> dict:
    """Get authentication status for all providers"""
    
    # For now, use session ID as user ID
    user_id = request.session.get('user_id', 'anonymous')
    
    status = {}
    for provider_name in oauth_manager.list_providers():
        token = await oauth_manager.get_token(user_id, provider_name)
        status[provider_name] = {
            "authenticated": token is not None and not token.is_expired,
            "expires_soon": token.expires_soon if token else False,
            "expires_at": token.expires_at.isoformat() if token and token.expires_at else None
        }
    
    return status


@router.post("/refresh/{provider}")
async def refresh_token(provider: str, request: Request) -> TokenResponse:
    """Refresh token for a provider"""
    
    # For now, use session ID as user ID
    user_id = request.session.get('user_id', 'anonymous')
    
    current_token = await oauth_manager.get_token(user_id, provider)
    if not current_token or not current_token.refresh_token:
        raise HTTPException(status_code=404, detail="No refresh token available")
    
    provider_obj = oauth_manager.get_provider(provider)
    if not provider_obj:
        raise HTTPException(status_code=400, detail="Unknown provider")
    
    try:
        new_token = await provider_obj.refresh_token(current_token.refresh_token)
        new_token.user_id = user_id
        
        # Store the new token
        await oauth_manager.token_storage.store_token(user_id, provider, new_token)
        
        return TokenResponse(
            access_token=new_token.access_token,
            token_type=new_token.token_type,
            expires_in=new_token.expires_in,
            scope=new_token.scope,
            provider=new_token.provider
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token refresh failed: {str(e)}")


# Convenience endpoints for specific providers

@router.get("/cloudflare/authorize")
async def cloudflare_authorize(request: Request) -> dict:
    """Get Cloudflare OAuth authorization URL"""
    return await get_authorization_url("cloudflare", request)


@router.get("/github/authorize") 
async def github_authorize(request: Request) -> dict:
    """Get GitHub OAuth authorization URL"""
    return await get_authorization_url("github", request)


# Email/Password Authentication Models
class LoginRequest(BaseModel):
    """Login request with username/email and password"""
    username: str  # Can be username or email
    password: str

class RegisterRequest(BaseModel):
    """User registration request"""
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    """User response model"""
    id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str
    last_login: Optional[str] = None

@router.post("/login")
async def login(login_data: LoginRequest, request: Request) -> dict:
    """Traditional username/password login"""
    
    try:
        from src.services.user_service import get_user_service
        from src.security.auth import TokenManager
        from src.config.settings import get_settings
        
        user_service = get_user_service()
        settings = get_settings()
        token_manager = TokenManager(settings)
        
        user = await user_service.authenticate_user(login_data.username, login_data.password)
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Generate JWT token
        access_token = token_manager.create_access_token(user)
        
        # Store user session (backup)
        request.session['user_id'] = user.id
        request.session['username'] = user.username
        request.session['auth_method'] = 'password'
        
        return {
            "message": "Login successful",
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.security.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                role=user.role.value,
                is_active=user.is_active,
                created_at=user.created_at.isoformat(),
                last_login=user.last_login.isoformat() if user.last_login else None
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@router.post("/register")
async def register(register_data: RegisterRequest, request: Request) -> dict:
    """User registration with email/password"""
    
    try:
        from src.services.user_service import get_user_service
        from src.security.auth import UserRole
        user_service = get_user_service()
        
        user = await user_service.create_user(
            username=register_data.username,
            email=register_data.email,
            password=register_data.password,
            role=UserRole.USER
        )
        
        if not user:
            raise HTTPException(status_code=400, detail="User registration failed. Username or email may already exist.")
        
        # Store user session
        request.session['user_id'] = user.id
        request.session['username'] = user.username
        request.session['auth_method'] = 'password'
        
        return {
            "message": "Registration successful",
            "user": UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                role=user.role.value,
                is_active=user.is_active,
                created_at=user.created_at.isoformat(),
                last_login=user.last_login.isoformat() if user.last_login else None
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/logout")
async def logout(request: Request) -> dict:
    """Logout user and clear session"""
    
    try:
        # Clear all session data
        request.session.clear()
        
        return {"message": "Logout successful"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")

@router.get("/me")
async def get_current_user_info(request: Request) -> UserResponse:
    """Get current authenticated user information"""
    
    try:
        user_id = request.session.get('user_id')
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        from src.services.user_service import get_user_service
        user_service = get_user_service()
        
        user = await user_service.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
            last_login=user.last_login.isoformat() if user.last_login else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user info: {str(e)}")
