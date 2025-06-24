# Deployment Instructions

## Step 1: GitHub Repository

1. Clone repository:
   ```
   git clone https://github.com/gigamonkeyx/pygent.git
   cd pygent
   ```

2. Copy files from deployment_ready/ to repository root

3. Commit and push:
   ```
   git add .
   git commit -m "Add PyGent Factory UI"
   git push origin main
   ```

## Step 2: Cloudflare Pages

1. Go to Cloudflare Pages dashboard
2. Connect GitHub repository: gigamonkeyx/pygent
3. Configure build settings:
   - Build command: npm run build
   - Build output: dist
   - Environment variables:
     - VITE_API_BASE_URL=https://api.timpayne.net
     - VITE_WS_BASE_URL=wss://ws.timpayne.net

## Step 3: Custom Domain

1. Add custom domain: timpayne.net
2. Configure subdirectory routing: /pygent

## Step 4: Backend Tunnels

1. Install cloudflared
2. Create tunnel for backend services
3. Configure DNS routing

## Success Criteria

- UI accessible at https://timpayne.net/pygent
- WebSocket connections functional
- Real-time features working
- Agent responses displaying correctly
