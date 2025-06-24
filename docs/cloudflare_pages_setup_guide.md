# 🚀 Cloudflare Pages Setup Guide - PyGent Factory

## Current Status
✅ **GitHub Repository**: Complete PyGent Factory UI deployed  
✅ **Backend Services**: API and WebSocket working perfectly  
✅ **Vite Configuration**: Fixed with correct base path `/pygent/`  
⏳ **Cloudflare Pages**: Needs manual configuration  

## Required Setup Steps

### 1. Access Cloudflare Pages Dashboard
- URL: https://dash.cloudflare.com/pages
- Click "Create a project"
- Select "Connect to Git"

### 2. Repository Configuration
```
Repository: gigamonkeyx/pygent
Branch: master (or main)
```

### 3. Build Settings
```
Project name: pygent-factory
Production branch: master
Framework preset: React
Build command: npm run build
Build output directory: dist
Root directory: / (leave empty)
Node.js version: 18
```

### 4. Environment Variables
Add these exact environment variables:
```
VITE_API_BASE_URL=https://api.timpayne.net
VITE_WS_BASE_URL=wss://ws.timpayne.net
VITE_BASE_PATH=/pygent
NODE_VERSION=18
```

### 5. Custom Domain Setup
After initial deployment:
1. Go to "Custom domains" in your Cloudflare Pages project
2. Add custom domain: `timpayne.net`
3. Configure path-based routing for `/pygent/*`

## Verification Commands

After setup, run these commands to verify:

```bash
# Test frontend loading
curl -v https://timpayne.net/pygent

# Test API connectivity  
curl -v https://api.timpayne.net/api/v1/health

# Run complete deployment test
python test_complete_deployment.py
```

## Expected Results

### Frontend (https://timpayne.net/pygent)
- Should return 200 OK
- Should contain PyGent Factory UI
- Should load React application with routing

### API (https://api.timpayne.net)
- Health endpoint should return JSON with system status
- All endpoints should be accessible

### WebSocket (wss://ws.timpayne.net)
- Should accept connections
- Should handle ping/pong messages

## Troubleshooting

### If frontend returns 404:
1. Check Cloudflare Pages project is deployed
2. Verify custom domain configuration
3. Check build logs for errors

### If frontend loads but shows wrong content:
1. Verify repository connection in Cloudflare Pages
2. Check build configuration matches above
3. Trigger manual deployment

### If API/WebSocket not working:
1. Verify tunnel is running: `cloudflared tunnel list`
2. Check DNS records point to Cloudflare
3. Test local backend: `curl http://localhost:8000/api/v1/health`

## Build Configuration Details

The repository contains:
- ✅ Complete React 18 + TypeScript application
- ✅ All required dependencies in package.json
- ✅ Correct vite.config.ts with base: '/pygent/'
- ✅ Production environment configuration
- ✅ Tailwind CSS and UI components
- ✅ WebSocket integration
- ✅ Multi-page routing

## Next Steps After Setup

1. Run deployment test: `python test_complete_deployment.py`
2. Verify all pages load correctly
3. Test agent creation and chat functionality
4. Verify WebSocket real-time updates
5. Test API endpoints through the UI

## Support

If issues persist:
1. Check Cloudflare Pages build logs
2. Verify environment variables are set correctly
3. Test individual components (frontend, API, WebSocket)
4. Review network requests in browser developer tools
