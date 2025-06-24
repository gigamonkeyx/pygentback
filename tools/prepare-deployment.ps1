# PyGent Factory UI Repository Preparation Script
# This script prepares the UI code for deployment to GitHub repository

Write-Host "🚀 Preparing PyGent Factory UI for GitHub Deployment" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

# Step 1: Verify current UI is working
Write-Host "`n📋 Step 1: Verifying Current UI Setup..." -ForegroundColor Cyan

Set-Location "src\ui"

# Check if node_modules exists
if (Test-Path "node_modules") {
    Write-Host "✅ Node modules found" -ForegroundColor Green
} else {
    Write-Host "❌ Node modules not found. Installing..." -ForegroundColor Red
    npm install
}

# Test build
Write-Host "🔨 Testing build process..." -ForegroundColor Yellow
try {
    npm run build
    if (Test-Path "dist") {
        Write-Host "✅ Build successful" -ForegroundColor Green
        $buildSize = (Get-ChildItem -Path "dist" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host "📊 Build size: $([math]::Round($buildSize, 2)) MB" -ForegroundColor Cyan
    } else {
        Write-Host "❌ Build failed - dist directory not created" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Build failed: $_" -ForegroundColor Red
    exit 1
}

# Step 2: Create deployment directory
Write-Host "`n📦 Step 2: Creating Deployment Directory..." -ForegroundColor Cyan

Set-Location "..\.."
$deployDir = "pygent-ui-deploy"

if (Test-Path $deployDir) {
    Write-Host "🗑️ Removing existing deployment directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $deployDir
}

New-Item -ItemType Directory -Name $deployDir
Write-Host "✅ Created deployment directory: $deployDir" -ForegroundColor Green

# Step 3: Copy UI files
Write-Host "`n📋 Step 3: Copying UI Files..." -ForegroundColor Cyan

$sourceDir = "src\ui"
$filesToCopy = @(
    "src",
    "public",
    "index.html",
    "package.json",
    "package-lock.json",
    "vite.config.ts",
    "tsconfig.json",
    "tsconfig.node.json",
    "tailwind.config.js",
    "postcss.config.js",
    "README.md"
)

foreach ($file in $filesToCopy) {
    $sourcePath = Join-Path $sourceDir $file
    $destPath = Join-Path $deployDir $file
    
    if (Test-Path $sourcePath) {
        if (Test-Path $sourcePath -PathType Container) {
            # It's a directory
            Copy-Item -Recurse $sourcePath $destPath -Force
            Write-Host "✅ Copied directory: $file" -ForegroundColor Green
        } else {
            # It's a file
            Copy-Item $sourcePath $destPath -Force
            Write-Host "✅ Copied file: $file" -ForegroundColor Green
        }
    } else {
        Write-Host "⚠️ File not found: $file" -ForegroundColor Yellow
    }
}

# Step 4: Create deployment-specific files
Write-Host "`n📝 Step 4: Creating Deployment Files..." -ForegroundColor Cyan

# Create _redirects file for SPA routing
$redirectsContent = @"
# SPA routing for React Router
/*    /index.html   200
"@

Set-Content -Path "$deployDir\_redirects" -Value $redirectsContent
Write-Host "✅ Created _redirects file for SPA routing" -ForegroundColor Green

# Create production vite.config.ts
$prodViteConfig = @"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

// Production configuration for Cloudflare Pages
export default defineConfig({
  plugins: [react()],
  base: '/',  // Deploy to root for Cloudflare Pages
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-select'],
          utils: ['zustand', '@tanstack/react-query']
        }
      }
    },
    chunkSizeWarningLimit: 1000
  },
  server: {
    port: 3000,
    host: true,
  },
})
"@

Set-Content -Path "$deployDir\vite.config.ts" -Value $prodViteConfig -Force
Write-Host "✅ Updated vite.config.ts for production" -ForegroundColor Green

# Create production README
$prodReadme = @"
# PyGent Factory UI

Modern React application for PyGent Factory - an AI-powered agent creation and management platform.

## Features

- 🤖 AI Agent Creation and Management
- 💬 Real-time Chat Interface  
- 🔧 MCP (Model Context Protocol) Integration
- 📊 Performance Analytics
- 🎨 Modern UI with Tailwind CSS

## Tech Stack

- **Frontend**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Data Fetching**: TanStack Query
- **UI Components**: Radix UI

## Development

\`\`\`bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
\`\`\`

## Deployment

This application is configured for deployment on Cloudflare Pages.

### Build Settings
- **Framework**: React
- **Build Command**: \`npm run build\`
- **Build Output**: \`dist\`
- **Node Version**: 18

### Environment Variables
\`\`\`
NODE_VERSION=18
\`\`\`

## License

MIT License
"@

Set-Content -Path "$deployDir\README.md" -Value $prodReadme -Force
Write-Host "✅ Created production README.md" -ForegroundColor Green

# Create .gitignore
$gitignoreContent = @"
# Dependencies
node_modules/
/.pnp
.pnp.js

# Production builds
/dist
/build

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*

# Cache
.cache/
.parcel-cache/
.next/

# Temporary
*.tmp
*.temp
"@

Set-Content -Path "$deployDir\.gitignore" -Value $gitignoreContent
Write-Host "✅ Created .gitignore file" -ForegroundColor Green

# Step 5: Test the deployment directory
Write-Host "`n🧪 Step 5: Testing Deployment Directory..." -ForegroundColor Cyan

Set-Location $deployDir

# Test npm install and build in deployment directory
Write-Host "📦 Installing dependencies in deployment directory..." -ForegroundColor Yellow
npm install

Write-Host "🔨 Testing build in deployment directory..." -ForegroundColor Yellow
npm run build

if (Test-Path "dist") {
    Write-Host "✅ Deployment directory build successful" -ForegroundColor Green
    $deployBuildSize = (Get-ChildItem -Path "dist" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "📊 Deployment build size: $([math]::Round($deployBuildSize, 2)) MB" -ForegroundColor Cyan
} else {
    Write-Host "❌ Deployment directory build failed" -ForegroundColor Red
    exit 1
}

Set-Location ".."

# Step 6: Git initialization instructions
Write-Host "`n📋 Step 6: Git Repository Setup Instructions" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

Write-Host "The deployment directory is ready at: $((Get-Location).Path)\$deployDir" -ForegroundColor Green
Write-Host ""
Write-Host "To push to GitHub repository, run these commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "cd $deployDir" -ForegroundColor White
Write-Host "git init" -ForegroundColor White
Write-Host "git add ." -ForegroundColor White
Write-Host "git commit -m `"Initial PyGent Factory UI deployment`"" -ForegroundColor White
Write-Host "git branch -M main" -ForegroundColor White
Write-Host "git remote add origin https://github.com/gigamonkeyx/pygent.git" -ForegroundColor White
Write-Host "git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "After pushing to GitHub:" -ForegroundColor Yellow
Write-Host "1. Go to https://dash.cloudflare.com/pages" -ForegroundColor White
Write-Host "2. Create new project and connect to gigamonkeyx/pygent" -ForegroundColor White
Write-Host "3. Use build settings from DEPLOYMENT_GUIDE.md" -ForegroundColor White
Write-Host ""

# Step 7: Summary
Write-Host "🎉 Deployment Preparation Complete!" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host "✅ UI code copied to deployment directory" -ForegroundColor Green
Write-Host "✅ Production configuration files created" -ForegroundColor Green
Write-Host "✅ Build process tested and working" -ForegroundColor Green
Write-Host "✅ Git setup instructions provided" -ForegroundColor Green
Write-Host ""
Write-Host "📁 Deployment directory: $deployDir" -ForegroundColor Cyan
Write-Host "📊 Build size: $([math]::Round($deployBuildSize, 2)) MB" -ForegroundColor Cyan
Write-Host "🚀 Ready for GitHub and Cloudflare Pages deployment" -ForegroundColor Green
