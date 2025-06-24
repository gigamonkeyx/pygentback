# PyGent Factory Frontend Deployment to Cloudflare Pages
# This script deploys the built frontend to Cloudflare Pages

Write-Host "🚀 Deploying PyGent Factory Frontend to Cloudflare Pages..." -ForegroundColor Green

# Check if wrangler is installed
try {
    $wranglerVersion = npx wrangler --version
    Write-Host "✅ Wrangler CLI found: $wranglerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Wrangler CLI not found. Installing..." -ForegroundColor Red
    npm install -g wrangler
}

# Change to UI directory
Set-Location ui

# Ensure build is up to date
Write-Host "📦 Building frontend for production..." -ForegroundColor Yellow
npm run build

# Deploy to Cloudflare Pages
Write-Host "🌐 Deploying to Cloudflare Pages..." -ForegroundColor Yellow

# Deploy using wrangler pages
npx wrangler pages deploy dist --project-name=pygent-factory --compatibility-date=2024-01-15

Write-Host "✅ Frontend deployment complete!" -ForegroundColor Green
Write-Host "🌍 Frontend should be available at: https://timpayne.net/pygent" -ForegroundColor Cyan

# Return to root directory
Set-Location ..

Write-Host "📋 Next steps:" -ForegroundColor Yellow
Write-Host "  1. Configure custom domain in Cloudflare Pages dashboard" -ForegroundColor White
Write-Host "  2. Set up subdirectory routing for /pygent" -ForegroundColor White
Write-Host "  3. Configure environment variables in Pages dashboard" -ForegroundColor White
