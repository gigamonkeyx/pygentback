# Cloudflare Tunnel Setup for PyGent Factory

## 🎯 Overview

This document provides the complete setup process for establishing a Cloudflare tunnel connection for the PyGent Factory deployment. The tunnel will connect the local backend services to the deployed React UI at timpayne.net/pygent.

## 📋 Prerequisites

### **System Requirements:**
- ✅ **Windows System**: Current setup on Windows with PowerShell
- ✅ **Cloudflared Installed**: Version 2025.4.2 (built 2025-04-30-1407 UTC)
- ✅ **React UI Deployed**: Successfully deployed on Cloudflare Pages
- ✅ **Domain Access**: timpayne.net domain configured in Cloudflare

### **Project Status:**
- ✅ **TypeScript Build Errors**: Resolved
- ✅ **GitHub Repository**: Complete project structure uploaded
- ✅ **Cloudflare Pages**: React UI successfully deployed
- ⏳ **Backend Services**: Need tunnel connection for API/WebSocket

---

## 🔧 Cloudflare Tunnel Setup Process

### **Step 1: Authenticate Cloudflared**

```bash
# Authenticate with Cloudflare account
cloudflared tunnel login
```

**Expected Process:**
1. Browser opens to Cloudflare dashboard
2. Sign in to Cloudflare account
3. Authorize tunnel access for domain
4. Authentication certificate downloaded

### **Step 2: Create Tunnel**

```bash
# Create a new tunnel for PyGent Factory
cloudflared tunnel create pygent-factory
```

**Expected Output:**
- Tunnel UUID generated
- Tunnel credentials file created
- Tunnel registered in Cloudflare dashboard

### **Step 3: Configure Tunnel**

Create configuration file: `config.yml`

```yaml
tunnel: <tunnel-uuid>
credentials-file: <path-to-credentials>

ingress:
  # React UI (already deployed via Cloudflare Pages)
  - hostname: timpayne.net
    path: /pygent
    service: https://pygent-factory-ui.pages.dev
  
  # Backend API endpoints
  - hostname: api.timpayne.net
    service: http://localhost:8000
  
  # WebSocket connections
  - hostname: ws.timpayne.net
    service: http://localhost:8000
    originRequest:
      noTLSVerify: true
  
  # Catch-all rule (required)
  - service: http_status:404
```

### **Step 4: Configure DNS Records**

Add DNS records in Cloudflare dashboard:

```
Type: CNAME
Name: api.timpayne.net
Target: <tunnel-uuid>.cfargotunnel.com
Proxied: Yes

Type: CNAME  
Name: ws.timpayne.net
Target: <tunnel-uuid>.cfargotunnel.com
Proxied: Yes
```

### **Step 5: Start Tunnel**

```bash
# Run tunnel with configuration
cloudflared tunnel --config config.yml run pygent-factory

# Or install as Windows service
cloudflared service install
```

---

## 🏗️ Backend Service Architecture

### **Local Services to Expose:**

#### **1. PyGent Factory API Server**
- **Port**: 8000
- **Protocol**: HTTP/HTTPS
- **Endpoints**: 
  - `/api/chat` - Chat interface
  - `/api/reasoning` - Tree of Thought reasoning
  - `/api/mcp` - MCP marketplace
  - `/api/system` - System metrics

#### **2. WebSocket Server**
- **Port**: 8000 (same as API)
- **Protocol**: WebSocket
- **Endpoints**:
  - `/ws` - Real-time communication
  - `/ws/reasoning` - Live reasoning updates
  - `/ws/metrics` - System monitoring

#### **3. MCP Servers**
- **Filesystem MCP**: File operations
- **GitHub MCP**: Repository management  
- **PostgreSQL MCP**: Data persistence
- **Cloudflare MCP**: Infrastructure management

---

## 🔐 Security Configuration

### **Authentication & Authorization:**
- **Cloudflare Access**: Protect admin endpoints
- **API Keys**: Secure backend communication
- **CORS Configuration**: Allow frontend-backend communication

### **SSL/TLS:**
- **Cloudflare SSL**: Automatic HTTPS termination
- **Origin Certificates**: Secure tunnel communication
- **HSTS Headers**: Force HTTPS connections

---

## 📊 Monitoring & Logging

### **Tunnel Health:**
```bash
# Check tunnel status
cloudflared tunnel info pygent-factory

# View tunnel logs
cloudflared tunnel --config config.yml run pygent-factory --loglevel debug
```

### **Performance Metrics:**
- **Latency**: Monitor request/response times
- **Throughput**: Track data transfer rates
- **Uptime**: Ensure 99.9% availability
- **Error Rates**: Monitor failed connections

---

## 🚨 Troubleshooting

### **Common Issues:**

#### **Authentication Failed**
```bash
# Re-authenticate
cloudflared tunnel login
```

#### **Tunnel Not Starting**
```bash
# Check configuration
cloudflared tunnel --config config.yml validate

# Check logs
cloudflared tunnel --config config.yml run pygent-factory --loglevel debug
```

#### **DNS Resolution Issues**
- Verify CNAME records in Cloudflare dashboard
- Check tunnel UUID matches DNS target
- Ensure proxy status is enabled

#### **Backend Connection Failed**
- Verify local services are running on specified ports
- Check firewall settings
- Confirm service binding (localhost vs 0.0.0.0)

---

## 🔄 Integration with React UI

### **Frontend Configuration:**

Update React UI environment variables:

```typescript
// src/config/api.ts
const API_BASE_URL = import.meta.env.PROD 
  ? 'https://api.timpayne.net'
  : 'http://localhost:8000';

const WS_BASE_URL = import.meta.env.PROD
  ? 'wss://ws.timpayne.net'
  : 'ws://localhost:8000';
```

### **WebSocket Connection:**

```typescript
// src/services/websocket.ts
const wsUrl = import.meta.env.PROD 
  ? 'wss://ws.timpayne.net/ws'
  : 'ws://localhost:8000/ws';
```

---

## 📈 Deployment Workflow

### **Complete Deployment Process:**

1. **✅ React UI**: Deployed via Cloudflare Pages
2. **⏳ Backend Services**: Start local PyGent Factory services
3. **⏳ Cloudflare Tunnel**: Establish secure connection
4. **⏳ DNS Configuration**: Route traffic through tunnel
5. **⏳ Testing**: Verify end-to-end connectivity
6. **⏳ Monitoring**: Set up health checks and alerts

### **Success Criteria:**

- ✅ **Frontend accessible**: timpayne.net/pygent loads React UI
- ⏳ **API connectivity**: Frontend can communicate with backend
- ⏳ **WebSocket functionality**: Real-time features working
- ⏳ **MCP integration**: All MCP servers accessible
- ⏳ **Performance**: Sub-100ms response times
- ⏳ **Reliability**: 99.9% uptime

---

## 🎯 Next Steps

### **Immediate Actions:**
1. **Complete cloudflared authentication** (in progress)
2. **Create PyGent Factory tunnel**
3. **Configure tunnel ingress rules**
4. **Set up DNS records**
5. **Start backend services**
6. **Test end-to-end connectivity**

### **Future Enhancements:**
- **Load balancing**: Multiple backend instances
- **Auto-scaling**: Dynamic resource allocation
- **Health checks**: Automated failover
- **Metrics dashboard**: Real-time monitoring

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-04  
**Status**: Setup in Progress  
**Next Phase**: Tunnel Authentication and Configuration