# A2A Protocol Authentication Guide

## Overview

This document provides comprehensive authentication specifications for the A2A Protocol, covering all supported authentication methods, flows, and security considerations.

## Supported Authentication Methods

The A2A Protocol supports multiple authentication methods to accommodate various enterprise and integration scenarios:

1. **OAuth 2.0** (Recommended for production)
2. **API Key Authentication** (Simple integration scenarios)
3. **Basic Authentication** (Development and testing)
4. **Bearer Token Authentication** (JWT-based authentication)
5. **Client Certificate Authentication** (High-security environments)

## OAuth 2.0 Authentication

### Overview
OAuth 2.0 is the recommended authentication method for production A2A deployments, providing secure, standardized authentication with fine-grained scope control.

### Agent Card Configuration

```json
{
  "securitySchemes": {
    "oauth2": {
      "type": "oauth2",
      "flows": {
        "authorizationCode": {
          "authorizationUrl": "https://auth.example.com/oauth/authorize",
          "tokenUrl": "https://auth.example.com/oauth/token",
          "scopes": {
            "agent.read": "Read access to agent capabilities",
            "agent.write": "Write access to agent functions",
            "agent.admin": "Administrative access to agent",
            "tools.execute": "Execute agent tools",
            "data.access": "Access agent data resources"
          }
        },
        "clientCredentials": {
          "tokenUrl": "https://auth.example.com/oauth/token",
          "scopes": {
            "agent.service": "Service-to-service access"
          }
        }
      }
    }
  },
  "security": [{"oauth2": ["agent.read", "agent.write"]}]
}
```

### Authentication Flow

#### Authorization Code Flow (Interactive)
```mermaid
sequenceDiagram
    participant Client as Client Agent
    participant Auth as Authorization Server
    participant Resource as A2A Server
    participant User as User

    Client->>Auth: 1. Request authorization
    Auth->>User: 2. Prompt for consent
    User->>Auth: 3. Grant permission
    Auth->>Client: 4. Return authorization code
    Client->>Auth: 5. Exchange code for token
    Auth->>Client: 6. Return access token
    Client->>Resource: 7. API request with token
    Resource->>Auth: 8. Validate token
    Auth->>Resource: 9. Token validation result
    Resource->>Client: 10. Protected resource response
```

#### Client Credentials Flow (Service-to-Service)
```mermaid
sequenceDiagram
    participant Client as Client Service
    participant Auth as Authorization Server
    participant Resource as A2A Server

    Client->>Auth: 1. Request token (client_id + secret)
    Auth->>Client: 2. Return access token
    Client->>Resource: 3. API request with token
    Resource->>Auth: 4. Validate token
    Auth->>Resource: 5. Token validation result
    Resource->>Client: 6. Protected resource response
```

### Token Request Examples

#### Authorization Code Token Request
```http
POST /oauth/token HTTP/1.1
Host: auth.example.com
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=AUTH_CODE_HERE&
redirect_uri=https://client.example.com/callback&
client_id=CLIENT_ID&
client_secret=CLIENT_SECRET
```

#### Client Credentials Token Request
```http
POST /oauth/token HTTP/1.1
Host: auth.example.com
Content-Type: application/x-www-form-urlencoded
Authorization: Basic Y2xpZW50X2lkOmNsaWVudF9zZWNyZXQ=

grant_type=client_credentials&
scope=agent.service
```

### Using Access Tokens

```http
POST /a2a/v1 HTTP/1.1
Host: api.example.com
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Execute analysis"}]
    }
  }
}
```

### Token Refresh

```http
POST /oauth/token HTTP/1.1
Host: auth.example.com
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token&
refresh_token=REFRESH_TOKEN_HERE&
client_id=CLIENT_ID&
client_secret=CLIENT_SECRET
```

## API Key Authentication

### Overview
API Key authentication provides a simple authentication method suitable for development, testing, and simple integration scenarios.

### Agent Card Configuration

```json
{
  "securitySchemes": {
    "apiKey": {
      "type": "apiKey",
      "in": "header",
      "name": "X-API-Key",
      "description": "API key for agent access"
    }
  },
  "security": [{"apiKey": []}]
}
```

### Alternative Configurations

#### Query Parameter API Key
```json
{
  "securitySchemes": {
    "apiKey": {
      "type": "apiKey",
      "in": "query",
      "name": "api_key"
    }
  }
}
```

#### Custom Header API Key
```json
{
  "securitySchemes": {
    "customApiKey": {
      "type": "apiKey",
      "in": "header",
      "name": "X-Agent-API-Key"
    }
  }
}
```

### Request Examples

#### Header-based API Key
```http
POST /a2a/v1 HTTP/1.1
Host: api.example.com
X-API-Key: ak_1234567890abcdef1234567890abcdef
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "capabilities/list"
}
```

#### Query Parameter API Key
```http
GET /a2a/v1/agent-card?api_key=ak_1234567890abcdef HTTP/1.1
Host: api.example.com
```

### API Key Security Best Practices

1. **Key Rotation**: Regular rotation of API keys
2. **Scope Limitation**: Limit API key permissions
3. **Secure Storage**: Store keys securely, never in code
4. **Transmission Security**: Always use HTTPS
5. **Monitoring**: Monitor API key usage patterns

```python
# API Key management example
class APIKeyManager:
    def __init__(self):
        self.keys = {}
        self.key_permissions = {}
    
    def generate_key(self, client_id: str, permissions: List[str]) -> str:
        """Generate new API key with specified permissions"""
        key = f"ak_{secrets.token_urlsafe(32)}"
        self.keys[key] = {
            "client_id": client_id,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "active": True
        }
        self.key_permissions[key] = permissions
        return key
    
    def validate_key(self, key: str, required_permission: str) -> bool:
        """Validate API key and check permissions"""
        if key not in self.keys or not self.keys[key]["active"]:
            return False
        
        permissions = self.key_permissions.get(key, [])
        return required_permission in permissions or "admin" in permissions
```

## Basic Authentication

### Overview
Basic Authentication is supported for development and testing scenarios, but should not be used in production environments.

### Agent Card Configuration

```json
{
  "securitySchemes": {
    "basic": {
      "type": "http",
      "scheme": "basic",
      "description": "Basic HTTP authentication"
    }
  },
  "security": [{"basic": []}]
}
```

### Request Example

```http
POST /a2a/v1 HTTP/1.1
Host: api.example.com
Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ=
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "message/send",
  "params": {
    "message": {
      "role": "user", 
      "parts": [{"kind": "text", "text": "Hello"}]
    }
  }
}
```

### Security Considerations

⚠️ **Security Warning**: Basic Authentication should only be used:
- In development environments
- Over HTTPS connections only
- With strong passwords
- For non-production testing

## Bearer Token Authentication

### Overview
Bearer Token authentication supports JWT and other token formats for stateless authentication.

### Agent Card Configuration

```json
{
  "securitySchemes": {
    "bearerAuth": {
      "type": "http",
      "scheme": "bearer",
      "bearerFormat": "JWT"
    }
  },
  "security": [{"bearerAuth": []}]
}
```

### JWT Token Structure

```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "iss": "https://auth.example.com",
    "sub": "agent_client_123",
    "aud": "https://api.example.com",
    "exp": 1640995200,
    "iat": 1640991600,
    "scope": "agent.read agent.write"
  }
}
```

### Request Example

```http
POST /a2a/v1 HTTP/1.1
Host: api.example.com
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/execute",
  "params": {
    "tool": "data_analyzer",
    "parameters": {"dataset": "sales_data"}
  }
}
```

## Secondary Authentication for Tools

### Overview
The A2A Protocol supports in-task authentication for scenarios where agents need additional credentials for specific tools or resources.

### Authentication Request Flow

When an agent needs additional authentication for a specific tool or resource:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "auth/request",
  "params": {
    "type": "oauth2",
    "resource": "external_api",
    "scopes": ["read", "write"],
    "reason": "Access required for data analysis tool"
  }
}
```

### Server Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "auth_url": "https://external.com/oauth/authorize?client_id=...",
    "auth_id": "auth_123456",
    "expires_in": 600
  }
}
```

### Completing Authentication

After user completes authentication:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "auth/complete",
  "params": {
    "auth_id": "auth_123456",
    "authorization_code": "auth_code_from_external_service"
  }
}
```

### Using Secondary Credentials

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/execute",
  "params": {
    "tool": "external_data_tool",
    "auth_context": "auth_123456",
    "parameters": {"query": "sales_data"}
  }
}
```

## Multi-Factor Authentication (MFA)

### Overview
For high-security environments, the A2A Protocol supports Multi-Factor Authentication as an additional security layer.

### MFA Challenge Flow

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "auth/mfa_challenge",
  "params": {
    "challenge_type": "totp",
    "user_id": "user123"
  }
}
```

### MFA Verification

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "auth/mfa_verify",
  "params": {
    "challenge_id": "mfa_challenge_456",
    "verification_code": "123456"
  }
}
```

## Authentication Error Handling

### Common Authentication Errors

#### Invalid Credentials
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32401,
    "message": "Unauthorized",
    "data": {
      "type": "authentication_failed",
      "details": "Invalid API key"
    }
  }
}
```

#### Insufficient Permissions
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32403,
    "message": "Forbidden",
    "data": {
      "type": "insufficient_permissions",
      "required_scope": "agent.admin",
      "current_scopes": ["agent.read"]
    }
  }
}
```

#### Token Expired
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32401,
    "message": "Unauthorized",
    "data": {
      "type": "token_expired",
      "details": "Access token has expired",
      "refresh_required": true
    }
  }
}
```

## Implementation Examples

### Python Implementation

```python
import requests
import jwt
from datetime import datetime, timedelta

class A2AAuthClient:
    def __init__(self, base_url: str, auth_config: dict):
        self.base_url = base_url
        self.auth_config = auth_config
        self.access_token = None
        self.token_expires = None
    
    async def authenticate_oauth2(self, client_id: str, client_secret: str):
        """Authenticate using OAuth2 client credentials flow"""
        token_url = self.auth_config["oauth2"]["token_url"]
        
        response = requests.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "agent.read agent.write"
        })
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.token_expires = datetime.utcnow() + timedelta(
                seconds=token_data["expires_in"]
            )
            return True
        return False
    
    def get_auth_headers(self) -> dict:
        """Get authentication headers for requests"""
        if self.auth_config["type"] == "oauth2":
            return {"Authorization": f"Bearer {self.access_token}"}
        elif self.auth_config["type"] == "api_key":
            return {"X-API-Key": self.auth_config["api_key"]}
        elif self.auth_config["type"] == "basic":
            import base64
            credentials = base64.b64encode(
                f"{self.auth_config['username']}:{self.auth_config['password']}".encode()
            ).decode()
            return {"Authorization": f"Basic {credentials}"}
        
        return {}
    
    async def make_authenticated_request(self, method: str, params: dict):
        """Make authenticated A2A request"""
        headers = {
            "Content-Type": "application/json",
            **self.get_auth_headers()
        }
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        response = requests.post(
            f"{self.base_url}/a2a/v1",
            json=payload,
            headers=headers
        )
        
        return response.json()
```

### JavaScript Implementation

```javascript
class A2AAuthClient {
    constructor(baseUrl, authConfig) {
        this.baseUrl = baseUrl;
        this.authConfig = authConfig;
        this.accessToken = null;
        this.tokenExpires = null;
    }
    
    async authenticateOAuth2(clientId, clientSecret) {
        const tokenUrl = this.authConfig.oauth2.tokenUrl;
        
        const response = await fetch(tokenUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                grant_type: 'client_credentials',
                client_id: clientId,
                client_secret: clientSecret,
                scope: 'agent.read agent.write'
            })
        });
        
        if (response.ok) {
            const tokenData = await response.json();
            this.accessToken = tokenData.access_token;
            this.tokenExpires = new Date(Date.now() + tokenData.expires_in * 1000);
            return true;
        }
        return false;
    }
    
    getAuthHeaders() {
        switch (this.authConfig.type) {
            case 'oauth2':
                return { 'Authorization': `Bearer ${this.accessToken}` };
            case 'api_key':
                return { 'X-API-Key': this.authConfig.apiKey };
            case 'basic':
                const credentials = btoa(`${this.authConfig.username}:${this.authConfig.password}`);
                return { 'Authorization': `Basic ${credentials}` };
            default:
                return {};
        }
    }
    
    async makeAuthenticatedRequest(method, params) {
        const response = await fetch(`${this.baseUrl}/a2a/v1`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this.getAuthHeaders()
            },
            body: JSON.stringify({
                jsonrpc: '2.0',
                id: 1,
                method: method,
                params: params
            })
        });
        
        return await response.json();
    }
}
```

## Security Best Practices

### Token Management
1. **Secure Storage**: Never store credentials in code or logs
2. **Token Rotation**: Implement regular token rotation
3. **Scope Limitation**: Use minimal required scopes
4. **Expiration Handling**: Implement proper token refresh logic

### Network Security
1. **HTTPS Only**: Always use HTTPS in production
2. **Certificate Validation**: Validate server certificates
3. **Request Signing**: Consider request signing for high-security scenarios
4. **Rate Limiting**: Implement client-side rate limiting

### Error Handling
1. **Secure Error Messages**: Don't expose sensitive information in errors
2. **Retry Logic**: Implement exponential backoff for auth failures
3. **Audit Logging**: Log authentication events for security monitoring
4. **Graceful Degradation**: Handle auth failures gracefully

## Related Documentation

- [A2A_SECURITY_OVERVIEW.md](A2A_SECURITY_OVERVIEW.md) - Overall security architecture
- [A2A_SECURITY_AUTHORIZATION.md](A2A_SECURITY_AUTHORIZATION.md) - Authorization and access control
- [A2A_PROTOCOL_IMPLEMENTATION_GUIDE.md](A2A_PROTOCOL_IMPLEMENTATION_GUIDE.md) - Implementation guidelines
