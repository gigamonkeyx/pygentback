# 🚀 PyGent Factory Deployment Guide

## Overview
This guide documents the complete deployment process for PyGent Factory, including lessons learned, best practices, and step-by-step instructions for deploying the UI to Cloudflare Pages while keeping the backend local/private.

## Architecture Summary

### Current Working Architecture
- **Frontend**: React 18 + Vite + TypeScript + Tailwind CSS
- **Backend**: FastAPI + WebSockets + MCP Servers
- **Deployment Strategy**: Frontend to Cloudflare Pages, Backend local with tunnel
- **Communication**: WebSocket for real-time, REST API for standard operations

### Key Lessons Learned

#### 1. **Separation of Concerns**
- Keep frontend (`src/ui`) and backend completely separate
- Never run npm commands from project root - always from `src/ui`
- Use different package.json files for different purposes
- Avoid mixing frontend/backend dependencies

#### 2. **Vite Configuration**
- Use correct `base` path for deployment environment
- Configure proxy correctly for development (no path rewrite needed)
- Use absolute imports with `@/` alias for cleaner code
- Enable sourcemaps for debugging

#### 3. **WebSocket Implementation**
- Centralize WebSocket connection logic in main App component
- Disable React StrictMode in development to prevent double connections
- Use dynamic host detection (`window.location.host`)
- Implement proper error handling and reconnection logic

#### 4. **MCP Integration**
- Remove all mock/fake servers - use only real MCP servers
- Configure remote Cloudflare MCP servers properly
- Validate server connections before deployment

## Deployment Process

### Prerequisites
1. **Node.js 18+** installed
2. **Python 3.11+** with virtual environment
3. **Cloudflare account** with Pages access
4. **GitHub account** with repository access
5. **Authentication tokens** ready (GitHub, Cloudflare)

### Step 1: Prepare the UI for Deployment

```powershell
# Navigate to UI directory
cd src/ui

# Install dependencies
npm install

# Run development server to test
npm run dev

# Build for production
npm run build

# Test the build locally
npm run preview
```

### Step 2: Push UI to GitHub Repository

```powershell
# The UI code needs to be pushed to gigamonkeyx/pygent repository
# This should be done from the src/ui directory contents

# Copy src/ui contents to a clean directory
# Then push to gigamonkeyx/pygent repository
```

### Step 3: Configure Cloudflare Pages

#### Access Cloudflare Dashboard
1. Go to https://dash.cloudflare.com/pages
2. Click "Create a project"
3. Select "Connect to Git"

#### Repository Configuration
```
Repository: gigamonkeyx/pygent
Branch: main (or master)
Project name: pygent-factory
```

#### Build Settings
```
Framework preset: React
Build command: npm run build
Build output directory: dist
Root directory: / (leave empty)
Node.js version: 18
```

#### Environment Variables
```
NODE_VERSION=18
VITE_API_URL=https://api.yourdomain.com
VITE_WS_URL=wss://ws.yourdomain.com
```

### Step 4: Backend Tunnel Setup (Optional)

If you want to expose the backend publicly:

```powershell
# Install cloudflared
# Follow cloudflare_pages_setup_guide.md for tunnel setup

# Run the backend
python main.py server

# In another terminal, run tunnel
cloudflared tunnel --url http://localhost:8000
```

## File Structure

### Frontend Structure (src/ui/)
```
src/ui/
├── src/
│   ├── components/     # React components
│   ├── pages/         # Page components
│   ├── services/      # API and WebSocket services
│   ├── stores/        # Zustand state management
│   ├── hooks/         # Custom React hooks
│   ├── utils/         # Utility functions
│   ├── types/         # TypeScript type definitions
│   └── globals.css    # Global styles
├── index.html         # Entry HTML
├── package.json       # Frontend dependencies
├── vite.config.ts     # Vite configuration
├── tailwind.config.js # Tailwind CSS config
└── tsconfig.json      # TypeScript config
```

### Backend Structure
```
src/api/
├── main.py           # Backend entry point
├── routes/           # API routes
├── services/         # Business logic
├── models/           # Data models
└── utils/            # Backend utilities
```

## Configuration Files

### Key Configuration Files

#### `src/ui/vite.config.ts`
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
```

#### `src/ui/package.json`
```json
{
  "name": "pygent-factory-ui",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.8.0",
    // ... other dependencies
  }
}
```

#### `mcp_server_configs.json` (Backend)
```json
{
  "servers": [
    {
      "id": "python",
      "name": "Python",
      "command": "python",
      "args": ["-m", "mcp_server_python"],
      "enabled": true
    },
    {
      "id": "context7",
      "name": "Context7",
      "command": "npx",
      "args": ["-y", "@context7/mcp-server"],
      "enabled": true
    }
  ]
}
```

## Testing and Validation

### Local Testing
```powershell
# Test backend
cd d:\mcp\pygent-factory
python main.py server

# Test frontend (in new terminal)
cd src/ui
npm run dev

# Test WebSocket connection
# Open browser to http://localhost:3000
# Check DevTools console for WebSocket messages
```

### Production Testing
```powershell
# Run deployment validation
python comprehensive_validation_report.py

# Test API endpoints
curl https://api.yourdomain.com/api/v1/health

# Test frontend
curl https://yourdomain.com/pygent
```

## Common Issues and Solutions

### Issue 1: WebSocket Connection Failures
**Symptoms**: WebSocket connects and immediately disconnects
**Solution**: 
- Disable React StrictMode in development
- Centralize WebSocket connection logic
- Implement proper error handling

### Issue 2: Frontend 404 Errors
**Symptoms**: main.tsx not found, blank page
**Solution**:
- Ensure correct script src in index.html
- Run npm commands from correct directory (src/ui)
- Check Vite configuration

### Issue 3: API 404 Errors
**Symptoms**: /api endpoints return 404
**Solution**:
- Remove path rewrite from Vite proxy config
- Ensure backend is running on correct port
- Check CORS configuration

### Issue 4: Build Failures
**Symptoms**: npm run build fails
**Solution**:
- Clear node_modules and reinstall
- Check TypeScript configuration
- Ensure all imports are correct

## Security Considerations

### CORS Configuration
```python
# In main.py
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://yourdomain.com",
    "https://*.cloudflare.com"
]
```

### Environment Variables
- Never commit API keys or secrets
- Use environment variables for configuration
- Different configs for dev/staging/production

## Performance Optimization

### Frontend
- Code splitting with manual chunks
- Image optimization
- CSS minification with Tailwind
- Bundle size monitoring

### Backend
- Async/await for I/O operations
- Connection pooling for databases
- Caching for frequently accessed data
- Rate limiting for API endpoints

## Monitoring and Logging

### Frontend Monitoring
- Browser DevTools for debugging
- Console logging for WebSocket events
- Error boundaries for React errors

### Backend Monitoring
- Structured logging with Python logging
- Health check endpoints
- Performance metrics
- Error tracking

## Backup and Recovery

### Configuration Backup
- Keep configuration files in version control
- Document environment variables
- Backup database/state files
- Test restore procedures

## Future Improvements

### Planned Enhancements
1. Automated CI/CD pipeline
2. Staging environment setup
3. Database integration
4. User authentication
5. Advanced monitoring
6. Load balancing
7. Mobile responsiveness improvements

### Architecture Evolution
- Consider microservices architecture
- Implement proper database layer
- Add caching layer (Redis)
- Implement message queuing
- Add comprehensive testing suite

## Support and Documentation

### Internal Documentation
- `cloudflare_pages_setup_guide.md` - Cloudflare specific setup
- `ARCHITECTURE.md` - System architecture overview
- `MCP_SERVERS.md` - MCP server configuration
- `CURRENT_STATUS.md` - Current implementation status

### External Resources
- [Vite Documentation](https://vitejs.dev/)
- [React Documentation](https://react.dev/)
- [Cloudflare Pages Documentation](https://developers.cloudflare.com/pages/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Last Updated**: January 2025  
**Status**: Production Ready  
**Maintainer**: PyGent Factory Team
