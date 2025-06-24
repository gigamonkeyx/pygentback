# Cloudflare Pages Deployment Guide

## 🚀 Quick Setup for PyGent Factory UI

### Prerequisites
- ✅ GitHub repository ready: https://github.com/gigamonkeyx/pygent
- ✅ Cloudflare account with authentication keys
- ✅ UI code successfully built and tested

### Step 1: Access Cloudflare Dashboard
1. Log into your Cloudflare account
2. Navigate to **Pages** from the left sidebar
3. Click **Create a project**

### Step 2: Connect GitHub Repository
1. Select **Connect to Git**
2. Choose **GitHub** as your Git provider
3. Select the repository: `gigamonkeyx/pygent`
4. Click **Begin setup**

### Step 3: Configure Build Settings
```
Project name: pygent-factory
Production branch: main
Build command: npm run build
Build output directory: dist
Root directory: (leave empty)
```

### Step 4: Environment Variables
No environment variables are required for the static UI deployment.

### Step 5: Deploy
1. Click **Save and Deploy**
2. Cloudflare will automatically:
   - Clone the repository
   - Run `npm install` 
   - Execute `npm run build`
   - Deploy the `dist` folder contents

### Step 6: Custom Domain (Optional)
1. Go to **Custom domains** in your Pages project
2. Add custom domain: `timpayne.net` or subdomain
3. Configure DNS records as instructed by Cloudflare

## 🔧 Build Configuration Details

The repository is already configured with:
- **package.json** with correct build scripts
- **vite.config.ts** optimized for production
- **_redirects** file for SPA routing
- **.gitignore** excluding unnecessary files

## 📱 Expected Results

After deployment, you should have:
- **Live URL**: https://pygent-factory.pages.dev (or custom domain)
- **Build Time**: ~2-3 minutes for first deployment
- **Auto-deployments**: Every push to main branch triggers rebuild

## 🐛 Troubleshooting

### Build Fails
- Check build logs in Cloudflare Pages dashboard
- Ensure all dependencies are in package.json
- Verify Node.js version compatibility

### 404 Errors on Routes
- Confirm `_redirects` file is in the repository root
- File should contain: `/* /index.html 200`

### Assets Not Loading
- Check that `dist` folder is set as build output directory
- Verify asset paths in the built application

## 🔍 Monitoring

After deployment:
1. Test all routes work correctly
2. Verify WebSocket connections to backend
3. Check mobile responsiveness
4. Test error handling and fallbacks

## 🎯 Success Checklist

- [ ] Cloudflare Pages project created
- [ ] GitHub repository connected
- [ ] Build settings configured correctly
- [ ] First deployment successful
- [ ] Custom domain configured (if desired)
- [ ] All routes working correctly
- [ ] WebSocket connections functioning
- [ ] Performance optimized

## 📞 Next Steps After Deployment

1. **Test thoroughly** - All functionality should work
2. **Set up monitoring** - Track performance and errors
3. **Configure analytics** - Monitor user behavior
4. **Plan updates** - Establish deployment workflow for updates

The UI is now ready for production use! 🎉
